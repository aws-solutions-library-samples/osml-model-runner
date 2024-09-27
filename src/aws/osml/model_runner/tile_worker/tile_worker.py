#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import asyncio
import logging
from datetime import datetime, timezone
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional

import geojson
from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from shapely.affinity import translate

from aws.osml.features import Geolocator, ImagedFeaturePropertyAccessor
from aws.osml.model_runner.app_config import MetricLabels
from aws.osml.model_runner.common import ThreadingLocalContextFilter, TileState, Timer
from aws.osml.model_runner.database import FeatureTable, RegionRequestTable
from aws.osml.model_runner.inference import Detector

logger = logging.getLogger(__name__)


class TileWorker(Thread):
    def __init__(
        self,
        in_queue: Queue,
        feature_detector: Detector,
        geolocator: Optional[Geolocator],
        feature_table: FeatureTable,
        region_request_table: RegionRequestTable,
    ) -> None:
        super().__init__()
        self.in_queue = in_queue
        self.feature_detector = feature_detector
        self.geolocator = geolocator
        self.feature_table = feature_table
        self.region_request_table = region_request_table
        self.property_accessor = ImagedFeaturePropertyAccessor()
        self.failed_tile_count: int = 0

    def run(self) -> None:
        thread_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_event_loop)
        while True:
            image_info: Dict = self.in_queue.get()
            ThreadingLocalContextFilter.set_context(image_info)

            if image_info is None:
                logging.debug("All images processed. Stopping tile worker.")
                logging.debug(
                    (
                        f"Feature Detector Stats: {self.feature_detector.request_count} requests "
                        f"with {self.failed_tile_count} failed tiles."
                    )
                )
                break

            try:
                self.process_tile(image_info)
            finally:
                self.in_queue.task_done()

        try:
            thread_event_loop.stop()
            thread_event_loop.close()
        except Exception as e:
            logging.warning("Failed to stop and close the thread event loop")
            logging.exception(e)

    @metric_scope
    def process_tile(self, image_info: Dict, metrics: MetricsLogger = None) -> None:
        """
        This method handles the processing of a single tile by invoking the ML model, geolocating the detections to
        create features and finally storing the features in the database.

        :param image_info: description of the tile to be processed
        :param metrics: the current metric scope
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.TILE_PROCESSING_OPERATION,
                    MetricLabels.MODEL_NAME_DIMENSION: self.feature_detector.endpoint,
                }
            )
            metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

        try:
            with Timer(
                task_str=f"Processing Tile {image_info['image_path']}",
                metric_name=MetricLabels.DURATION,
                logger=logger,
                metrics_logger=metrics,
            ):
                with open(image_info["image_path"], mode="rb") as payload:
                    feature_collection = self.feature_detector.find_features(payload)

                features = self._refine_features(feature_collection, image_info)

                if len(features) > 0:
                    self.feature_table.add_features(features)

                self.region_request_table.add_tile(
                    image_info.get("image_id"), image_info.get("region_id"), image_info.get("region"), TileState.SUCCEEDED
                )
        except Exception as e:
            self.failed_tile_count += 1
            logging.error(f"Failed to process region tile with error: {e}", exc_info=True)
            self.region_request_table.add_tile(
                image_info.get("image_id"), image_info.get("region_id"), image_info.get("region"), TileState.FAILED
            )
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))

    @metric_scope
    def _refine_features(self, feature_collection, image_info: Dict, metrics: MetricsLogger = None) -> List[geojson.Feature]:
        """
        This method converts the detections returned by the model into geolocated features. It first updates the
        image coordinates of each detection to be in relation to the full image then it geolocates the image feature.

        :param feature_collection: the features from the ML model
        :param image_info: a description of the image tile containing the features
        :param metrics: the current metric scope
        :return: a list of GeoJSON features
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.FEATURE_REFINEMENT_OPERATION,
                }
            )

        with Timer(
            task_str=f"Refining Features for Tile:{image_info['image_path']}",
            metric_name=MetricLabels.DURATION,
            logger=logger,
            metrics_logger=metrics,
        ):
            # TODO: Consider move invocations to Timer
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

            features = []
            ulx = image_info["region"][0][1]
            uly = image_info["region"][0][0]
            if isinstance(feature_collection, dict) and "features" in feature_collection:
                logging.debug(f"SM Model returned {len(feature_collection['features'])} features")
                for feature in feature_collection["features"]:
                    # Check to see if there is a bbox defined in image coordinates. If so, update it to
                    # use full image coordinates and store the updated value in the feature properties.
                    tiled_image_bbox = self.property_accessor.get_image_bbox(feature)
                    if tiled_image_bbox is not None:
                        full_image_bbox = translate(tiled_image_bbox, xoff=ulx, yoff=uly)
                        self.property_accessor.set_image_bbox(feature, full_image_bbox)

                    # Check to see if there is a geometry defined in image coordinates. If not and
                    # there also isn't a bbox then search to see if some of the older deprecated
                    # image properties are in use and if so use the geometry from those properties.
                    # Note that this property search will be deprecated and removed in a future release.
                    tiled_image_geometry = self.property_accessor.get_image_geometry(feature)
                    if tiled_image_bbox is None and tiled_image_geometry is None:
                        logging.debug("Feature may be using deprecated attributes.")
                        tiled_image_geometry = self.property_accessor.find_image_geometry(feature)
                        if tiled_image_geometry is None:
                            logging.warning(f"There isn't a valid detection shape for feature: {feature}")

                    # If we found an image geometry update it to use full image coordinates and store the
                    # value in the feature properties.
                    if tiled_image_geometry is not None:
                        full_image_geometry = translate(tiled_image_geometry, xoff=ulx, yoff=uly)
                        self.property_accessor.set_image_geometry(feature, full_image_geometry)

                    feature["properties"]["image_id"] = image_info["image_id"]
                    feature["properties"]["inferenceTime"] = (
                        datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                    )

                    # This conversion only happens here so the rest of the system can depend on a standard
                    # set of properties. Eventually this call should be removed once the old properties are
                    # no longer used by CV models.
                    TileWorker.convert_deprecated_feature_properties(feature)

                    features.append(feature)
            logging.debug(f"# Features Created: {len(features)}")
            if len(features) > 0:
                if self.geolocator is not None:
                    # Create a geometry for each feature in the result. The geographic coordinates of these
                    # features are computed using the sensor model provided in the image metadata
                    self.geolocator.geolocate_features(features)
                    logging.debug(f"Created Geographic Coordinates for {len(features)} features")

        return features

    @staticmethod
    def convert_deprecated_feature_properties(feature: geojson.Feature) -> None:
        """
        This function converts the legacy properties produced by CV models into the feature properties used
        by OversightML.

        :param feature: the feature that needs its "properties" updated.

        :return: None
        """

        # Create an ontology based on the models returned feature_types
        if "feature_types" in feature["properties"] and "featureClasses" not in feature["properties"]:
            feature_classes = []
            for feature_type in feature["properties"]["feature_types"]:
                feature_classes.append(
                    {
                        "iri": feature_type,
                        "score": feature["properties"]["feature_types"][feature_type],
                    }
                )
            feature["properties"]["featureClasses"] = feature_classes
            feature["properties"].pop("feature_types", None)
