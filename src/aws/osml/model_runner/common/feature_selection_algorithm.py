#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from enum import auto

from aws.osml.model_runner.common import AutoStringEnum


class FeatureSelectionAlgorithm(str, AutoStringEnum):
    """
    Enum defining the entity selection algorithm used in combining bounding boxes.
    """

    NONE = auto()
    NMS = auto()
    SOFT_NMS = auto()  # gaussian Soft-NMS (as opposed to linear)
