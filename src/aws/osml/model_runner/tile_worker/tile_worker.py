#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import asyncio
import logging
from datetime import datetime, timezone
from queue import Queue
from threading import Thread
from typing import Dict, Optional

from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.unit import Unit

from aws.osml.model_runner.app_config import MetricLabels
from aws.osml.model_runner.database import FeatureTable
from aws.osml.model_runner.inference import SMDetector
from aws.osml.model_runner.tile_worker import FeatureRefinery


class TileWorker(Thread):
    def __init__(
        self,
        in_queue: Queue,
        feature_detector: SMDetector,
        feature_refinery: Optional[FeatureRefinery],
        feature_table: FeatureTable,
        metrics: MetricsLogger,
    ) -> None:
        super().__init__()
        self.in_queue = in_queue
        self.feature_detector = feature_detector
        self.feature_refinery = feature_refinery
        self.feature_table = feature_table
        self.metrics = metrics

    def run(self) -> None:
        thread_event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_event_loop)
        while True:
            image_info: Dict = self.in_queue.get()

            if image_info is None:
                logging.info("All images processed. Stopping tile worker.")
                logging.info(
                    "Feature Detector Stats: {} requests with {} errors".format(
                        self.feature_detector.request_count, self.feature_detector.error_count
                    )
                )
                break

            try:
                logging.info("Invoking SM Endpoint")
                with open(image_info["image_path"], mode="rb") as payload:
                    feature_collection = self.feature_detector.find_features(payload)

                # Convert the features to reference the full image
                features = []
                ulx = image_info["region"][0][1]
                uly = image_info["region"][0][0]
                if isinstance(feature_collection, dict) and "features" in feature_collection:
                    logging.info("SM Model returned {} features".format(len(feature_collection["features"])))
                    for feature in feature_collection["features"]:
                        tile_bbox = feature["properties"]["bounds_imcoords"]
                        full_image_bbox = [
                            tile_bbox[0] + ulx,
                            tile_bbox[1] + uly,
                            tile_bbox[2] + ulx,
                            tile_bbox[3] + uly,
                        ]
                        feature["properties"]["bounds_imcoords"] = full_image_bbox
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
                        logging.info(f"Created Geographic Coordinates for {len(features)} features")

                    self.feature_table.add_features(features)
            except Exception as e:
                logging.error("Failed to process region tile!")
                logging.exception(e)
                if self.metrics:
                    self.metrics.put_metric(MetricLabels.TILE_PROCESSING_ERROR, 1, str(Unit.COUNT.value))
                    self.metrics.put_metric(MetricLabels.TILE_CREATION_FAILURE, 1, str(Unit.COUNT.value))
            finally:
                self.in_queue.task_done()

        try:
            thread_event_loop.stop()
            thread_event_loop.close()
        except Exception as e:
            logging.warning("Failed to stop and close the thread event loop")
            logging.exception(e)
