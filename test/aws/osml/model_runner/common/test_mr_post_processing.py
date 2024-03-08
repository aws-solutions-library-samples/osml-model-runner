#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from dataclasses import asdict
from json import dumps, loads
from unittest import TestCase


class TestMRPostProcessing(TestCase):
    def test_feature_distillation_deserializer_nms(self):
        from aws.osml.model_runner.common import (
            FeatureDistillationAlgorithmType,
            FeatureDistillationDeserializer,
            FeatureDistillationNMS,
            mr_post_processing_options_factory,
        )

        feature_selection_option = FeatureDistillationNMS()

        feature_selection_options_json = dumps(
            asdict(feature_selection_option, dict_factory=mr_post_processing_options_factory)
        )
        expected_json = '{"algorithm_type": "NMS", "iou_threshold": 0.75}'
        assert feature_selection_options_json == expected_json

        deserializer = FeatureDistillationDeserializer()
        new_feature_selection_option = deserializer.deserialize(loads(feature_selection_options_json))

        assert isinstance(new_feature_selection_option, FeatureDistillationNMS)
        assert isinstance(new_feature_selection_option.algorithm_type, FeatureDistillationAlgorithmType)

    def test_feature_distillation_deserializer_soft_nms(self):
        from aws.osml.model_runner.common import (
            FeatureDistillationAlgorithmType,
            FeatureDistillationDeserializer,
            FeatureDistillationSoftNMS,
            mr_post_processing_options_factory,
        )

        feature_selection_option = FeatureDistillationSoftNMS()

        feature_selection_options_json = dumps(
            asdict(feature_selection_option, dict_factory=mr_post_processing_options_factory)
        )
        expected_json = '{"algorithm_type": "SOFT_NMS", "iou_threshold": 0.75, "skip_box_threshold": 0.0001, "sigma": 0.1}'
        assert feature_selection_options_json == expected_json

        deserializer = FeatureDistillationDeserializer()
        new_feature_selection_option = deserializer.deserialize(loads(feature_selection_options_json))

        assert isinstance(new_feature_selection_option, FeatureDistillationSoftNMS)
        assert isinstance(new_feature_selection_option.algorithm_type, FeatureDistillationAlgorithmType)

    def test_feature_distillation_deserializer_missing_algorithm_type(self):
        from aws.osml.model_runner.common import FeatureDistillationDeserializer

        invalid_json = '{"iou_threshold": 0.75, "skip_box_threshold": 0.0001}'

        deserializer = FeatureDistillationDeserializer()
        with self.assertRaises(ValueError):
            deserializer.deserialize(loads(invalid_json))

    def test_feature_distillation_deserializer_invalid_algorithm_type(self):
        from aws.osml.model_runner.common import FeatureDistillationDeserializer

        invalid_json = '{"algorithm_type": "MAGIC", "iou_threshold": 0.75, "skip_box_threshold": 0.0001}'

        deserializer = FeatureDistillationDeserializer()
        with self.assertRaises(ValueError):
            deserializer.deserialize(loads(invalid_json))

    def test_mr_post_processing_deserializer_feature_distillation_step(self):
        from aws.osml.model_runner.common import (
            FeatureDistillationDeserializer,
            MRPostProcessingDeserializer,
            MRPostprocessingStep,
        )

        step = MRPostprocessingStep.FEATURE_DISTILLATION

        deserializer = MRPostProcessingDeserializer.get_deserializer(step)
        assert isinstance(deserializer, FeatureDistillationDeserializer)

    def test_mr_post_processing_deserializer_invalid_step(self):
        from enum import auto

        from aws.osml.model_runner.common import MRPostProcessingDeserializer

        class MRPostprocessingStep:
            MAGIC = auto()

        step = MRPostprocessingStep.MAGIC
        with self.assertRaises(ValueError):
            MRPostProcessingDeserializer.get_deserializer(step)

    def test_deserialize_post_processing_list(self):
        from aws.osml.model_runner.common import MRPostProcessing, deserialize_post_processing_list

        test_list = [
            {"step": "FEATURE_DISTILLATION", "algorithm": {"algorithm_type": "NMS", "iou_threshold": 0.75}},
            {
                "step": "FEATURE_DISTILLATION",
                "algorithm": {"algorithm_type": "SOFT_NMS", "iou_threshold": 0.75, "skip_box_threshold": 0.0001},
            },
        ]
        deserialized_list = deserialize_post_processing_list(test_list)

        assert isinstance(deserialized_list[0], MRPostProcessing)
        assert isinstance(deserialized_list[1], MRPostProcessing)

    def test_deserialize_post_processing_list_exception(self):
        from aws.osml.model_runner.common import deserialize_post_processing_list

        bad_test_list = [
            {"step": "FEATURE_DISTILLATION", "algorithm": {"algorithm_type": "NMS", "iou_threshold": 0.75}},
            {"step": "FIND_ALL_THE_OBJECTS", "algorithm": {"algorithm_type": "MAGIC"}},
        ]
        with self.assertRaises(Exception):
            deserialize_post_processing_list(bad_test_list)
