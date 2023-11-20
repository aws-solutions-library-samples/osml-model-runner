#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, EnumMeta, auto
from typing import Dict, List

from .auto_string_enum import AutoStringEnum


class ABCEnumMeta(ABCMeta, EnumMeta):
    """
    A metaclass combining ABCMeta and EnumMeta, used as a base for creating
    enums for MRPostProcessing algorithm types. It ensures that the derived enums
    are both Enums and ABCs, enabling type checks and abstract method enforcement.
    """

    def __new__(mcls, *args, **kw):
        cls = super().__new__(mcls, *args, **kw)
        if issubclass(cls, Enum) and getattr(cls, "__abstractmethods__", None):
            raise TypeError("...")
        return cls

    def __call__(cls, *args, **kw):
        if getattr(cls, "__abstractmethods__", None):
            raise TypeError(
                f"Can't instantiate abstract class {cls.__name__} " f"with frozen methods {set(cls.__abstractmethods__)}"
            )
        return super().__call__(*args, **kw)


class MRPostProcessingAlgorithmType(AutoStringEnum, metaclass=ABCEnumMeta):
    pass


class FeatureDistillationAlgorithmType(str, MRPostProcessingAlgorithmType):
    """
    Enum for defining different feature distillation algorithms used in post-processing.
    Each member represents a specific algorithm for entity selection or fusion.
    NMS: Non-maximum Suppression
    SOFT_NMS: Variant of NMS (https://arxiv.org/abs/1704.04503). This implementation is gaussian Soft-NMS,
                as opposed to linear.
    NMW: Non-maximum weighted
    WBF: Weighted boxes fusion
    """

    NMS = auto()
    SOFT_NMS = auto()


class MRPostprocessingStep(str, AutoStringEnum):
    """
    Enum defining available steps for MR post-processing.
    """

    FEATURE_DISTILLATION = auto()


@dataclass(frozen=True)
class MRPostProcessingAlgorithm(ABC):
    """
    Abstract base class representing a generic MR post-processing algorithm.
    :param algorithm_type: The type of the algorithm, represented by a specific enum.
    """

    algorithm_type: MRPostProcessingAlgorithmType


@dataclass(frozen=True)
class FeatureDistillationAlgorithm(MRPostProcessingAlgorithm):
    """
    Generic FeatureDistillationAlgorithm class that is designed to be extended for each specific algorithm.
    :property algorithm_type: FeatureSelectionAlgorithmType = algorithm to use to combine object detections
    :property iou_threshold: float = intersection over union threshold
                                    - if greater than this value boxes are considered the same
    """

    algorithm_type: FeatureDistillationAlgorithmType
    iou_threshold: float


@dataclass(frozen=True)
class FeatureDistillationNMS(FeatureDistillationAlgorithm):
    """
    :property algorithm_type: FeatureSelectionAlgorithmType = algorithm to use to combine object detections
    :property iou_threshold: float = intersection over union threshold
                                    - if greater than this value boxes are considered the same
    """

    algorithm_type: FeatureDistillationAlgorithmType = field(default=FeatureDistillationAlgorithmType.NMS)
    iou_threshold: float = field(default=0.75)


@dataclass(frozen=True)
class FeatureDistillationSoftNMS(FeatureDistillationAlgorithm):
    """
    :property algorithm_type: FeatureSelectionAlgorithmType = algorithm to use to combine object detections
    :property iou_threshold: float = intersection over union threshold
                                    - if greater than this value boxes are considered the same
    :property skip_box_threshold: float = boxes with a confidence below this threshold value are skipped
    :property sigma: float = value - only applies to Soft NMS
    """

    algorithm_type: FeatureDistillationAlgorithmType = field(default=FeatureDistillationAlgorithmType.SOFT_NMS)
    iou_threshold: float = field(default=0.75)
    skip_box_threshold: float = field(default=0.0001)
    sigma: float = field(default=0.1)


@dataclass(frozen=True)
class MRPostProcessing:
    """
    Represents an operation for MR post-processing.
    :param step: The post-processing step (MRPostprocessingStep).
    :param algorithm: The algorithm used in this step (MRPostProcessingAlgorithm).
    """

    step: MRPostprocessingStep
    algorithm: MRPostProcessingAlgorithm


class PostProcessingDeserializer(ABC):
    """
    Abstract post-processing deserializer that defines a deserialize method.
    """

    @staticmethod
    @abstractmethod
    def deserialize(post_processing_algorithm: dict) -> MRPostProcessingAlgorithm:
        pass


class FeatureDistillationDeserializer(PostProcessingDeserializer):
    """
    Provides static methods for deserializing feature distillation algorithms from a dictionary.
    """

    @staticmethod
    def deserialize(post_processing_algorithm: dict) -> FeatureDistillationAlgorithm:
        if "algorithm_type" in post_processing_algorithm:
            post_processing_algorithm["algorithm_type"] = FeatureDistillationAlgorithmType(
                post_processing_algorithm["algorithm_type"]
            )
            if post_processing_algorithm["algorithm_type"] == FeatureDistillationAlgorithmType.NMS:
                return FeatureDistillationNMS(**post_processing_algorithm)
            elif post_processing_algorithm["algorithm_type"] == FeatureDistillationAlgorithmType.SOFT_NMS:
                return FeatureDistillationSoftNMS(**post_processing_algorithm)

        raise ValueError(
            f"Failed to deserialize. {post_processing_algorithm} is not a valid feature distillation algorithm object."
        )


class MRPostProcessingDeserializer:
    """
    Provides static method for obtaining the appropriate deserializer based on the post-processing step.
    """

    @staticmethod
    def get_deserializer(step: MRPostprocessingStep) -> PostProcessingDeserializer:
        if step == MRPostprocessingStep.FEATURE_DISTILLATION:
            return FeatureDistillationDeserializer()
        else:
            raise ValueError("Failed to get deserializer. MR post-processing step not valid.")


def deserialize_post_processing_list(mr_processing_list: List) -> List[MRPostProcessing]:
    """
    Deserializes a list of MR post-processing steps and algorithms from a list of dictionaries.
    :param mr_processing_list: A list of dictionaries representing MR post-processing objects.
    :return: A list of MRPostProcessing objects.
    """
    try:
        post_processing_list: List[MRPostProcessing] = []
        for mr_processing_dict in mr_processing_list:
            step = MRPostprocessingStep(mr_processing_dict["step"])
            algorithm_dict = mr_processing_dict["algorithm"]
            deserializer = MRPostProcessingDeserializer().get_deserializer(step)
            post_processing_list.append(MRPostProcessing(step=step, algorithm=deserializer.deserialize(algorithm_dict)))
        return post_processing_list

    except Exception as err:
        raise err


def mr_post_processing_options_factory(properties_list: List) -> Dict:
    """
    Factory function for converting dataclass instances to dictionaries.
    Specifically used with MR post-processing related dataclasses.
    :param properties_list: A list of properties (key-value pairs) of a dataclass.
    :return: A dictionary representation of the dataclass.
    """
    return {k: (v.name if isinstance(v, MRPostProcessingAlgorithmType) else v) for k, v in properties_list}
