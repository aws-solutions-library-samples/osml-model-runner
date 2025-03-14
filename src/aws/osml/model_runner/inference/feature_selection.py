#  Copyright 2023-2025 Amazon.com, Inc. or its affiliates.

from typing import List, Tuple

import numpy as np
from geojson import Feature

from aws.osml.model_runner.common import (
    FeatureDistillationAlgorithm,
    FeatureDistillationAlgorithmType,
    get_feature_image_bounds,
)
from aws.osml.model_runner.common.ensemble_boxes_nms import nms, soft_nms
from aws.osml.model_runner.inference.exceptions import FeatureDistillationException


class FeatureSelector:
    """
    The FeatureSelector class is used to select a subset of geojson features from a larger set
    using an algorith such as NMS or Soft NMS.  Parameters such as thresholds and the algorithm
    to use can be set by passing a FeatureSelectionOptions object in when the FeatureSelector
    is instantiated.
    """

    def __init__(self, options: FeatureDistillationAlgorithm = None) -> None:
        """
        Constructor for FeatureSelector class that selects an algorithm (e.g. NMS, Soft NMS, etc.) and sets parameters
        specific to each algorithm (e.g. IoU threshold).

        :param options: FeatureSelectionOptions = options to use to set the algorithm, thresholds, and parameters.
        """
        self.options = options

    def select_features(self, feature_list: List[Feature]) -> List[Feature]:
        """
        Selects a subset of features from a larger set of features using an algorithm such as NMS or Soft NMS.

        :param feature_list: a list of geojson features with a property of bounds_imcoords
        :return: the filtered list of features
        """
        if feature_list is None or not feature_list:
            return []
        if not self.options:
            return feature_list
        boxes_array, scores_array, labels_array = self._get_lists_from_features(feature_list)
        if self.options.algorithm_type == FeatureDistillationAlgorithmType.SOFT_NMS:
            boxes, scores, labels, indices = soft_nms(
                boxes=[np.array(boxes_array)],
                scores=[np.array(scores_array)],
                labels=[np.array(labels_array)],
                weights=None,
                iou_thr=self.options.iou_threshold,
                sigma=self.options.sigma,
                thresh=self.options.skip_box_threshold,
            )
        elif self.options.algorithm_type == FeatureDistillationAlgorithmType.NMS:
            boxes, scores, labels, indices = nms(
                boxes=[np.array(boxes_array)],
                scores=[np.array(scores_array)],
                labels=[np.array(labels_array)],
                weights=None,
                iou_thr=self.options.iou_threshold,
            )
        else:
            raise FeatureDistillationException(f"Invalid feature distillation algorithm: {self.options.algorithm_type}")
        return self._get_features_from_lists(feature_list, scores, labels, indices)

    def _get_lists_from_features(self, feature_list: List[Feature]) -> Tuple[np.array, np.array, np.array]:
        """
        This function converts the GeoJSON features into lists of normalized bounding boxes, scores, and label IDs
        needed by the selection algorithm implementations. As a side effect of this function various class attributes
        are set to support the inverse mapping of algorithm primitives to the full features. See
        _get_features_from_lists for the inverse function.

        :param feature_list: the input set of GeoJSON features to preprocess
        :return: tuple of lists - bounding boxes, confidence scores, category labels
        """

        n_features = len(feature_list)
        boxes = np.zeros((n_features, 4))
        scores = np.zeros(n_features)
        categories = []
        self.extents = [None, None, None, None]  # [min_x, min_y, max_x, max_y]
        self.labels_map = dict()

        for i, feature in enumerate(feature_list):
            # [min_x, min_y, max_x, max_y]
            bounds_imcoords = get_feature_image_bounds(feature)

            # This is a workaround for assumptions made by the NMS library and normalization code in this class.
            # All of that code assumes that features have bounding boxes with a non-zero area. That assumption
            # does not hold for features reported as a single point geometry or others that might simply be
            # erroneously reported with a zero width or height bbox. No matter the cause, we would like those
            # features to pass through our feature selection processing without triggering errors. Here we
            # add 0.1 of a pixel to the width or height of any bbox if it is currently zero. This does not change
            # the actual reported geometry of the feature in any way it just ensures the assumption of a non-zero
            # area is true.
            boxes[i] = (
                bounds_imcoords[0],
                bounds_imcoords[1],
                bounds_imcoords[2] + 0.1 if bounds_imcoords[0] == bounds_imcoords[2] else bounds_imcoords[2],
                bounds_imcoords[3] + 0.1 if bounds_imcoords[1] == bounds_imcoords[3] else bounds_imcoords[3],
            )

            category, score = self._get_category_and_score_from_feature(feature)
            categories.append(category)
            scores[i] = score

        # calculate data extents
        if n_features > 0:
            self.extents = [
                float(np.min(boxes[:, 0])),  # min_x
                float(np.min(boxes[:, 1])),  # min_y
                float(np.max(boxes[:, 2])),  # max_x
                float(np.max(boxes[:, 3])),  # max_y
            ]
        # Determine categories
        unique_categories = list(set(categories))
        for idx, unique_category in enumerate(unique_categories):
            self.labels_map[str(idx)] = unique_category
            self.labels_map[unique_category] = str(idx)
        labels_indexes = [int(self.labels_map.get(category, None)) for category in categories]

        # Normalize the boxes
        normalized_boxes = self._normalize_boxes(boxes)

        return normalized_boxes, scores, np.array(labels_indexes)

    def _normalize_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        This function normalizes the bounding boxes by subtracting the minimum x and y coordinates from each
        coordinate and dividing by the range of x and y coordinates. That means that all bounding boxes coordinates
        will be in the range of [0.0, 1.0] where 0.0 is the minimum of the extent and 1.0 is the maximum.

        :param boxes: bounding boxes to normalize
        :return: the normalized array of bounding boxes
        """
        # boxes: [[x1, y1, x2, y2], ...]
        if boxes.size == 0:
            return np.array([])
        x_range = self.extents[2] - self.extents[0]
        y_range = self.extents[3] - self.extents[1]
        normalized_boxes = np.zeros_like(boxes)
        normalized_boxes[:, 0] = (boxes[:, 0] - self.extents[0]) / x_range  # x1
        normalized_boxes[:, 1] = (boxes[:, 1] - self.extents[1]) / y_range  # y1
        normalized_boxes[:, 2] = (boxes[:, 2] - self.extents[0]) / x_range  # x2
        normalized_boxes[:, 3] = (boxes[:, 3] - self.extents[1]) / y_range  # y2
        return normalized_boxes

    @staticmethod
    def _get_category_and_score_from_feature(feature: Feature) -> Tuple[str, float]:
        """
        Get the feature class with the highest score from the featureClasses property.

        :return: tuple of feature class and highest score
        """
        max_score = -1.0
        max_class = ""
        for feature_class in feature.get("properties", {}).get("featureClasses", []):
            if feature_class.get("score") > max_score:
                max_score = feature_class.get("score")
                max_class = feature_class.get("iri")
        return max_class, max_score

    def _get_features_from_lists(
        self, feature_list: List[Feature], scores: np.array, labels: np.array, indices: np.array
    ) -> List[Feature]:
        """
        This function selects features from the feature_list based on the indices provided by the NMS algorithm.
        In the case when SOFT_NMS is selected, it also updates the scores as required.

        :param feature_list: the original list of GeoJSON features
        :param scores: the updated scores for each feature
        :param labels: the labels for each feature
        :param indices: the indices of the features to keep
        :return: the refined list of GeoJSON features
        """
        selected_features = [feature_list[i] for i in indices]

        # Verify selected_features, scores, and labels have the same length
        try:
            assert len(selected_features) == len(scores) == len(labels)
        except AssertionError:
            raise FeatureDistillationException(
                f"Mismatched lengths: features={len(selected_features)}, scores={len(scores)}, labels={len(labels)}"
            )

        if self.options.algorithm_type == FeatureDistillationAlgorithmType.SOFT_NMS:
            for feature, score, label in zip(selected_features, scores, labels):
                category = self.labels_map.get(str(label))
                for feature_class in feature.get("properties", {}).get("featureClasses", []):
                    if feature_class.get("iri") == category:
                        feature_class["rawScore"] = feature_class.get("score")
                        feature_class["score"] = score
        return selected_features
