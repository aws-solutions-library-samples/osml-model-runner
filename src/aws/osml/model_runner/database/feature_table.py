#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import time
from dataclasses import dataclass
from secrets import token_hex
from typing import Dict, List, Optional

import geojson
from _decimal import Decimal
from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from dacite import from_dict
from geojson import Feature

from aws.osml.model_runner.app_config import MetricLabels, ServiceConfig
from aws.osml.model_runner.common import ImageDimensions, Timer

from .ddb_helper import DDBHelper, DDBItem, DDBKey
from .exceptions import AddFeaturesException

logger = logging.getLogger(__name__)


@dataclass
class FeatureItem(DDBItem):
    """
    FeatureItem is a dataclass meant to represent a single item in the FeatureTable
    The data schema is defined as follows:
        hash_key: str
        range_key: str
        tile_id: str
        features: [str]
        expire_time: Optional[Decimal] = None
    """

    hash_key: str
    range_key: Optional[str] = None
    tile_id: Optional[str] = None
    features: Optional[List[str]] = None
    expire_time: Optional[Decimal] = None

    def __post_init__(self):
        self.ddb_key = DDBKey(
            hash_key="hash_key",
            hash_value=self.hash_key,
            range_key="range_key",
            range_value=self.range_key,
        )


class FeatureTable(DDBHelper):
    def __init__(self, table_name: str, tile_size: ImageDimensions, overlap: ImageDimensions) -> None:
        super().__init__(table_name)
        self.tile_size = tile_size
        self.overlap = overlap

    @metric_scope
    def add_features(self, features: List[Feature], metrics: MetricsLogger = None):
        """
        Group all the features together and add/update an item in the DDB

        :param features: List[Feature] = the list of features
        :param metrics: MetricsLogger = the metrics logger to use to report metrics.

        :return: None
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
        start_time_millisec = int(time.time() * 1000)
        # These records are temporary and will expire 24 hours after creation. Jobs should take
        # minutes to run so this time should be conservative enough to let a team debug an urgent
        # issue without leaving a ton of state leftover in the system.
        expire_time_epoch_sec = Decimal(int(start_time_millisec / 1000) + (24 * 60 * 60))
        with Timer(
            task_str="Add image features",
            metric_name=MetricLabels.FEATURE_STORE_LATENCY,
            logger=logger,
            metrics_logger=metrics,
        ):
            for key, grouped_features in self.group_features_by_key(features).items():
                try:
                    image_id, tile_id = key.split("-region-", 1)

                    logger.debug(
                        f"Starting Add Features to DDB: {len(grouped_features)} " f"features for {tile_id} {image_id}"
                    )

                    feature_count = 0
                    total_encoded_length = 0
                    encoded_features = []
                    for feature in grouped_features:
                        feature_count += 1
                        encoded_feature = geojson.dumps(feature)
                        total_encoded_length += len(encoded_feature)
                        encoded_features.append(encoded_feature)
                        # Once we exceed the 200K byte limit on our features write them to DDB. We are
                        # batching at this size because a single row in DDB only allows for 400K. We also
                        # need to make sure we are processing the last item no matter what the size is.
                        if total_encoded_length > int(ServiceConfig.ddb_max_item_size) or feature_count >= len(
                            grouped_features
                        ):
                            logger.debug(
                                f"Putting Feature Batch of {len(encoded_features)} "
                                f"features with total size of {total_encoded_length} for {tile_id} {image_id}"
                            )

                            # Build up a feature item and put it in the table
                            result = self.put_ddb_item(
                                FeatureItem(
                                    hash_key=image_id,
                                    range_key=token_hex(16),
                                    tile_id=tile_id,
                                    features=encoded_features,
                                    expire_time=expire_time_epoch_sec,
                                )
                            )

                            # Reset the batch
                            total_encoded_length = 0
                            encoded_features = []

                            # Check that we got a success response
                            status_code = result.get("ResponseMetadata", {}).get("HTTPStatusCode")
                            if status_code != 200:
                                if isinstance(metrics, MetricsLogger):
                                    metrics.put_metric(MetricLabels.FEATURE_UPDATE, 1, str(Unit.COUNT.value))
                                    metrics.put_metric(MetricLabels.FEATURE_ERROR, 1, str(Unit.COUNT.value))
                                logger.error("Unable to update feature table - HTTP Status Code: {}".format(status_code))
                except Exception as err:
                    if isinstance(metrics, MetricsLogger):
                        metrics.put_metric(MetricLabels.FEATURE_UPDATE_EXCEPTION, 1, str(Unit.COUNT.value))
                        metrics.put_metric(MetricLabels.FEATURE_ERROR, 1, str(Unit.COUNT.value))
                    logger.error("There was a problem adding features: {}".format(err))
                    raise AddFeaturesException("Failed to add features for tile!") from err

    @metric_scope
    def get_features(self, image_id: str, metrics: MetricsLogger = None) -> List[Feature]:
        """
        Query the database for all items with given image_id, the convert them into feature items, then
        go through all of the items and group the features per tile

        :param image_id: str = unique image_id for the job
        :param dedupe: Optional[bool] = remove any duplicate items
        :param metrics: MetricsLogger = the metrics logger to use to report metrics.

        :return: List[Feature] = the list of features
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
        with Timer(
            task_str="Aggregate image features",
            metric_name=MetricLabels.FEATURE_AGG_LATENCY,
            logger=logger,
            metrics_logger=metrics,
        ):
            # Query the database for all items with the given image_id (hash_key)
            rows = self.query_items(FeatureItem(image_id))

            # Convert them into feature items
            feature_items: List[FeatureItem] = []
            for row in rows:
                feature_items.append(from_dict(FeatureItem, row))

            # Combine all feature batches from the same tile together
            grouped_items = self.group_items_by_tile_id(feature_items)

            # Go through our items and group our features per tile
            features: List[Feature] = []
            for group in grouped_items:
                batch_features = []
                for item in grouped_items[group]:
                    if item.features:
                        for feature in item.features:
                            batch_features.append(geojson.loads(feature))
                    else:
                        logger.warning(f"Found FeatureTable item: {item.range_key} with no features!")
                features.extend(batch_features)
        return features

    def group_features_by_key(self, features: List[Feature]) -> Dict[str, List[Feature]]:
        """
        Group all the feature items by key

        :param features: List[Feature] = the list of features

        :return: Dict[str, List[Feature]] = dict which contains a unique key and within
                                    key contains a list of features
        """
        result: Dict[str, List[Feature]] = {}
        for feature in features:
            key = self.generate_tile_key(feature)
            result.setdefault(key, []).append(feature)
        return result

    @staticmethod
    def group_items_by_tile_id(items: List[FeatureItem]) -> Dict[str, List[FeatureItem]]:
        """
        Group all the feature items by tile id

        :param items: List[FeatureItem] = the list of feature items

        :return: Dict[str, List[FeatureItem]] = dict which contains a unique tile id and within
                                    tile id contains a list of features
        """
        grouped_items: Dict[str, List[FeatureItem]] = {}
        for item in items:
            if item.tile_id:
                grouped_items.setdefault(item.tile_id, []).append(item)
            else:
                logger.warning(f"Found FeatureTable item: {item.range_key} with no tile_id!")
        return grouped_items

    def generate_tile_key(self, feature: Feature) -> str:
        """
        Generate the tile key based on the given feature.

        :param feature: Feature = properties of a feature

        :return: str = tile key
        """
        bbox = feature["properties"]["bounds_imcoords"]

        # TODO: Check tile size to see if it is w,h or row/col
        # This is the size of the unique pixels in each tile
        stride_x = self.tile_size[0] - self.overlap[0]
        stride_y = self.tile_size[1] - self.overlap[1]

        max_x_index = int(bbox[2] / stride_x)
        max_y_index = int(bbox[3] / stride_y)

        min_x_index = int(bbox[0] / stride_x)
        min_y_index = int(bbox[1] / stride_y)
        min_x_offset = int(bbox[0]) % stride_x
        min_y_offset = int(bbox[1]) % stride_y

        if min_x_offset < self.overlap[0] and min_x_index > 0:
            min_x_index -= 1
        if min_y_offset < self.overlap[1] and min_y_index > 0:
            min_y_index -= 1

        return "{}-region-{}:{}:{}:{}".format(
            feature["properties"]["image_id"], min_x_index, max_x_index, min_y_index, max_y_index
        )
