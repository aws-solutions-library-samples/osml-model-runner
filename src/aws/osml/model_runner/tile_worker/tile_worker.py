#  Copyright 2023 Amazon.com, Inc. or its affiliates.

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
from shapely.geometry import Polygon

from aws.osml.model_runner.app_config import MetricLabels
from aws.osml.model_runner.common import GeojsonDetectionField, ThreadingLocalContextFilter, Timer
from aws.osml.model_runner.database import FeatureTable
from aws.osml.model_runner.inference import Detector
from aws.osml.model_runner.tile_worker import FeatureRefinery

logger = logging.getLogger(__name__)


class TileWorker(Thread):
    def __init__(
        self,
        in_queue: Queue,
        feature_detector: Detector,
        feature_refinery: Optional[FeatureRefinery],
        feature_table: FeatureTable,
    ) -> None:
        super().__init__()
        self.in_queue = in_queue
        self.feature_detector = feature_detector
        self.feature_refinery = feature_refinery
        self.feature_table = feature_table

    def run(self) -> None:
        thread_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_event_loop)
        while True:
            image_info: Dict = self.in_queue.get()
            ThreadingLocalContextFilter.set_context(image_info)

            if image_info is None:
                logging.info("All images processed. Stopping tile worker.")
                logging.info(
                    "Feature Detector Stats: {} requests with {} errors".format(
                        self.feature_detector.request_count, self.feature_detector.error_count
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

        except Exception as e:
            logging.error("Failed to process region tile!")
            logging.exception(e)
            self.feature_detector.error_count += 1  # borrow the feature detector error count to tally other errors
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
                logging.info("SM Model returned {} features".format(len(feature_collection["features"])))
                for feature in feature_collection["features"]:
                    # model returned a mask
                    scaled_polygon = None
                    if GeojsonDetectionField.GEOM in feature["properties"]:
                        polygon = Polygon(feature["properties"][GeojsonDetectionField.GEOM])
                        scaled_polygon = translate(polygon, xoff=ulx, yoff=uly)
                        feature["properties"][GeojsonDetectionField.GEOM] = list(scaled_polygon.exterior.coords)
                    # model returned a bounding box
                    if GeojsonDetectionField.BOUNDS in feature["properties"]:
                        tile_bbox = feature["properties"][GeojsonDetectionField.BOUNDS]
                        scaled_bbox = translate(
                            Polygon(FeatureRefinery.imcoords_bbox_to_polygon(tile_bbox)), xoff=ulx, yoff=uly
                        )
                        feature["properties"][GeojsonDetectionField.BOUNDS] = scaled_bbox.bounds
                    elif scaled_polygon is not None:
                        feature["properties"][GeojsonDetectionField.BOUNDS] = scaled_polygon.bounds
                    else:
                        logging.warning(f"There isn't a valid detection shape for feature: {feature}")
                    feature["properties"]["image_id"] = image_info["image_id"]
                    feature["properties"]["inferenceTime"] = datetime.now(tz=timezone.utc).isoformat()
                    FeatureRefinery.feature_property_transformation(feature)
                    features.append(feature)
            logging.info("# Features Created: {}".format(len(features)))
            if len(features) > 0:
                if self.feature_refinery is not None:
                    # Create a geometry for each feature in the result. The geographic coordinates of these
                    # features are computed using the sensor model provided in the image metadata
                    self.feature_refinery.refine_features_for_tile(features)
                    logging.info("Created Geographic Coordinates for {} features".format(len(features)))

        return features
