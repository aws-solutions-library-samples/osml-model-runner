#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from collections import OrderedDict
from typing import List, Tuple

from ensemble_boxes import nms, soft_nms
from geojson import Feature

from aws.osml.model_runner.common import (
    FeatureDistillationAlgorithm,
    FeatureDistillationAlgorithmType,
    GeojsonDetectionField,
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
        :param options: FeatureSelectionOptions = options to use to set the algorithm, thresholds, and parameters.
        """
        self.options = options

    def select_features(self, feature_list: List[Feature]) -> List[Feature]:
        """
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
        :param feature_list:
        :return: tuple of lists - bounding boxes, confidence scores, category labels
        """

        boxes = []
        scores = []
        categories = []
        self.extents = [None, None, None, None]  # [min_x, min_y, max_x, max_y]
        self.feature_id_map = OrderedDict()
        self.labels_map = dict()
        for feature in feature_list:
            # imcoords: [x1, y1, x2, y2]
            bounds_imcoords = feature.get("properties", {}).get(GeojsonDetectionField.BOUNDS)
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
        feature_dict = feature.get("properties", {}).get("feature_types")
        if feature_dict:
            return max(feature_dict.items(), key=lambda x: x[1])
        else:
            return "", 1.0

    def _get_features_from_lists(self, boxes: List, scores: List, labels: List) -> List[Feature]:
        """
        :param boxes:
        :param scores:
        :param labels:
        :return:
        """
        features = []
        im_boxes = self._denormalize_boxes(boxes)
        for box, score, label in zip(im_boxes, scores, labels):
            category = self.labels_map.get(str(label))
            feature_hash_id = hash(str(box) + category)
            feature = self.feature_id_map.get(feature_hash_id)
            if feature:
                if self.options.algorithm_type == FeatureDistillationAlgorithmType.SOFT_NMS and score != feature.get(
                    "properties", {}
                ).get("detection_score"):
                    feature["properties"]["adjusted_feature_types"] = {category: score}
                    feature_ontology_list = feature.get("properties", {}).get("detection", {}).get("ontology")
                    if feature_ontology_list:
                        adjusted_feature_ontology_list = []
                        for iri in feature_ontology_list:
                            if iri.get("iri") == category:
                                iri["adjustedDetectionScore"] = score
                            adjusted_feature_ontology_list.append(iri)
                        feature["properties"]["detection"]["ontology"] = adjusted_feature_ontology_list
                features.append(feature)
        return features

    def _denormalize_boxes(self, boxes: List[List[float]]) -> List[List[int]]:
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
