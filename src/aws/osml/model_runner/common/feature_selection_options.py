#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from dataclasses import dataclass
from typing import Dict, List

from aws.osml.model_runner.common.feature_selection_algorithm import FeatureSelectionAlgorithm


@dataclass
class FeatureSelectionOptions:
    """
    :property iou_threshold: float = intersection over union threshold
                                    - if greater than this value boxes are considered the same
    :property algorithm: FeatureSelectionAlgorithm = algorithm to use to combine object detections
    :property skip_box_threshold: float = boxes with a confidence below this threshold value are skipped
    :property sigma: float = value - only applies to Soft NMS
    """

    algorithm: FeatureSelectionAlgorithm = FeatureSelectionAlgorithm.NMS
    iou_threshold: float = 0.75
    skip_box_threshold: float = 0.0001
    sigma: float = 0.1


def feature_selection_options_factory(properties_list: List) -> Dict:
    """
    Factory to facilitate converting the dataclass to a dict.
    Used with asdict(my_feature_selection_object, dict_factory=feature_selection_options_factory)
    :param properties_list: list of properties
    :return: dictionary representation of dataclass
    """
    return {k: (v.name if isinstance(v, FeatureSelectionAlgorithm) else v) for k, v in properties_list}
