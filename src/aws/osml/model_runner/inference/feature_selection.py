#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from collections import OrderedDict
from typing import List, Tuple

from ensemble_boxes import nms, soft_nms
from geojson import Feature

from aws.osml.model_runner.common import (
    FeatureDistillationAlgorithm,
    FeatureDistillationAlgorithmType,
    get_feature_image_bounds,
)
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
        boxes_list, scores_list, labels_list = self._get_lists_from_features(feature_list)
        if self.options.algorithm_type == FeatureDistillationAlgorithmType.SOFT_NMS:
            boxes, scores, labels = soft_nms(
                [boxes_list],
                [scores_list],
                [labels_list],
                weights=None,
                iou_thr=self.options.iou_threshold,
                sigma=self.options.sigma,
                thresh=self.options.skip_box_threshold,
            )
        elif self.options.algorithm_type == FeatureDistillationAlgorithmType.NMS:
            boxes, scores, labels = nms(
                [boxes_list], [scores_list], [labels_list], weights=None, iou_thr=self.options.iou_threshold
            )
        else:
            raise FeatureDistillationException(f"Invalid feature distillation algorithm: {self.options.algorithm_type}")
        return self._get_features_from_lists(boxes, scores, labels)

    def _get_lists_from_features(self, feature_list: List[Feature]) -> Tuple[List, List, List]:
        """
        This function converts the GeoJSON features into lists of normalized bounding boxes, scores, and label IDs
        needed by the selection algorithm implementations. As a side effect of this function various class attributes
        are set to support the inverse mapping of algorithm primitives to the full features. See
        _get_features_from_lists for the inverse function.

        :param feature_list: the input set of GeoJSON features to preprocess
        :return: tuple of lists - bounding boxes, confidence scores, category labels
        """

        boxes = []
        scores = []
        categories = []
        self.extents = [None, None, None, None]  # [min_x, min_y, max_x, max_y]
        self.feature_id_map = OrderedDict()
        self.labels_map = dict()

        for feature in feature_list:
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
            bounds_imcoords = (
                bounds_imcoords[0],
                bounds_imcoords[1],
                bounds_imcoords[2] + 0.1 if bounds_imcoords[0] == bounds_imcoords[2] else bounds_imcoords[2],
                bounds_imcoords[3] + 0.1 if bounds_imcoords[1] == bounds_imcoords[3] else bounds_imcoords[3],
            )

            category, score = self._get_category_and_score_from_feature(feature)
            boxes.append(bounds_imcoords)
            categories.append(category)
            scores.append(score)
            if self.extents[0] is None or self.extents[0] > bounds_imcoords[0]:
                self.extents[0] = bounds_imcoords[0]
            if self.extents[1] is None or self.extents[1] > bounds_imcoords[1]:
                self.extents[1] = bounds_imcoords[1]
            if self.extents[2] is None or self.extents[2] < bounds_imcoords[2]:
                self.extents[2] = bounds_imcoords[2]
            if self.extents[3] is None or self.extents[3] < bounds_imcoords[3]:
                self.extents[3] = bounds_imcoords[3]
            bounds_imcoords_rounded = [int(round(coord)) for coord in bounds_imcoords]
            feature_hash_id = hash(str(bounds_imcoords_rounded) + category)
            self.feature_id_map[feature_hash_id] = feature
        unique_categories = list(set(categories))
        for idx, unique_category in enumerate(unique_categories):
            self.labels_map[str(idx)] = unique_category
            self.labels_map[unique_category] = str(idx)
        labels_indexes = [int(self.labels_map.get(category, None)) for category in categories]
        normalized_boxes = self._normalize_boxes(boxes)

        return normalized_boxes, scores, labels_indexes

    def _normalize_boxes(self, boxes: List[List[int]]) -> List[List[float]]:
        """
        This function normalizes the bounding boxes by subtracting the minimum x and y coordinates from each
        coordinate and dividing by the range of x and y coordinates. That means that all bounding boxes coordinates
        will be in the range of [0.0, 1.0] where 0.0 is the minimum of the extent and 1.0 is the maximum. See
        _denormalize_boxes() to convert back to bboxes in pixel coordinates.

        :param boxes: the list of bounding boxes to normalize
        :return: the normalized list of bounding boxes
        """
        # boxes: [x1, y1, x2, y2]
        min_x = self.extents[0]
        min_y = self.extents[1]
        x_range = self.extents[2] - self.extents[0]
        y_range = self.extents[3] - self.extents[1]
        normalized_boxes = []
        for box in boxes:
            x1_norm = (box[0] - min_x) / x_range
            y1_norm = (box[1] - min_y) / y_range
            x2_norm = (box[2] - min_x) / x_range
            y2_norm = (box[3] - min_y) / y_range
            normalized_boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
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

    def _get_features_from_lists(self, boxes: List, scores: List, labels: List) -> List[Feature]:
        """
        This function consolidates the lists of bounding boxes, scores, and labels into the GeoJSON features.
        This happens by finding the feature with a matching bounding box and category in the original feature list
        and updating the score if necessary. Any features that were in the original feature list that were not
        in the input lists end up filtered out of the result.

        :param boxes: the normalized bounding boxes for the features
        :param scores: the updated scores for each bounding box
        :param labels: the labels for each bounding box
        :return: the refined list of GeoJSON features
        """
        features = []
        im_boxes = self._denormalize_boxes(boxes)
        for box, score, label in zip(im_boxes, scores, labels):
            category = self.labels_map.get(str(label))
            feature_hash_id = hash(str(box) + category)
            feature = self.feature_id_map.get(feature_hash_id)
            if feature:
                if self.options.algorithm_type == FeatureDistillationAlgorithmType.SOFT_NMS:
                    for feature_class in feature.get("properties", {}).get("featureClasses", []):
                        if feature_class.get("iri") == category:
                            feature_class["rawScore"] = feature_class.get("score")
                            feature_class["score"] = score
                features.append(feature)
        return features

    def _denormalize_boxes(self, boxes: List[List[float]]) -> List[List[int]]:
        """
        This function denormalizes the bounding boxes by multiplying each coordinate by the width or height
        of the extent and then adding in the extent minimums. That puts all bounding boxes back into the
        image coordinate space. This is the inverse of _normalize_boxes().

        :param boxes: the list of bounding boxes to denormalize
        :return: the denormalized list of bounding boxes
        """
        # boxes: [x1, y1, x2, y2]
        min_x = self.extents[0]
        min_y = self.extents[1]
        x_range = self.extents[2] - self.extents[0]
        y_range = self.extents[3] - self.extents[1]
        denormalized_boxes = []
        for box in boxes:
            x1 = int(round(box[0] * x_range + min_x))
            y1 = int(round(box[1] * y_range + min_y))
            x2 = int(round(box[2] * x_range + min_x))
            y2 = int(round(box[3] * y_range + min_y))
            denormalized_boxes.append([x1, y1, x2, y2])
        return denormalized_boxes
