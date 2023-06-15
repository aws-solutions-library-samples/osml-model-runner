#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from dataclasses import asdict
from json import dumps, loads
from unittest import TestCase

from dacite import Config, from_dict


class TestFeatureSelectionOptions(TestCase):
    def test_feature_feature_selection_options_serialization(self):
        from aws.osml.model_runner.common import (
            FeatureSelectionAlgorithm,
            FeatureSelectionOptions,
            feature_selection_options_factory,
        )

        feature_selection_options = FeatureSelectionOptions()

        feature_selection_options_json = dumps(
            asdict(feature_selection_options, dict_factory=feature_selection_options_factory)
        )
        expected_json = '{"algorithm": "NMS", "iou_threshold": 0.75, "skip_box_threshold": 0.0001, "sigma": 0.1}'
        assert feature_selection_options_json == expected_json

        new_feature_selection_options = from_dict(
            data_class=FeatureSelectionOptions,
            data=loads(feature_selection_options_json),
            config=Config(cast=[FeatureSelectionAlgorithm]),
        )
        assert isinstance(new_feature_selection_options, FeatureSelectionOptions)
        assert isinstance(new_feature_selection_options.algorithm, FeatureSelectionAlgorithm)
