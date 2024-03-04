#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest


class TestFeatureDetectorFactory(unittest.TestCase):
    def test_sm_detector_generation(self):
        from aws.osml.model_runner.api.inference import ModelInvokeMode
        from aws.osml.model_runner.inference import FeatureDetectorFactory, SMDetector

        feature_detector = FeatureDetectorFactory(
            endpoint="test",
            endpoint_mode=ModelInvokeMode.SM_ENDPOINT,
        ).build()
        assert isinstance(feature_detector, SMDetector)
        assert feature_detector.mode == ModelInvokeMode.SM_ENDPOINT

    def test_http_detector_generation(self):
        from aws.osml.model_runner.api.inference import ModelInvokeMode
        from aws.osml.model_runner.inference import FeatureDetectorFactory, HTTPDetector

        feature_detector = FeatureDetectorFactory(
            endpoint="test",
            endpoint_mode=ModelInvokeMode.HTTP_ENDPOINT,
        ).build()
        assert isinstance(feature_detector, HTTPDetector)
        assert feature_detector.mode == ModelInvokeMode.HTTP_ENDPOINT
