#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest


class TestFeatureDetectorFactory(unittest.TestCase):
    def test_sm_detector_generation(self):
        """
        Test that the FeatureDetectorFactory correctly creates an SMDetector
        when the endpoint mode is set to ModelInvokeMode.SM_ENDPOINT.
        """
        from aws.osml.model_runner.api.inference import ModelInvokeMode
        from aws.osml.model_runner.inference import FeatureDetectorFactory, SMDetector

        feature_detector = FeatureDetectorFactory(
            endpoint="test",
            endpoint_mode=ModelInvokeMode.SM_ENDPOINT,
        ).build()

        # Verify that the detector is an instance of SMDetector and has the correct mode
        assert isinstance(feature_detector, SMDetector)
        assert feature_detector.mode == ModelInvokeMode.SM_ENDPOINT

    def test_http_detector_generation(self):
        """
        Test that the FeatureDetectorFactory correctly creates an HTTPDetector
        when the endpoint mode is set to ModelInvokeMode.HTTP_ENDPOINT.
        """
        from aws.osml.model_runner.api.inference import ModelInvokeMode
        from aws.osml.model_runner.inference import FeatureDetectorFactory, HTTPDetector

        feature_detector = FeatureDetectorFactory(
            endpoint="test",
            endpoint_mode=ModelInvokeMode.HTTP_ENDPOINT,
        ).build()

        # Verify that the detector is an instance of HTTPDetector and has the correct mode
        assert isinstance(feature_detector, HTTPDetector)
        assert feature_detector.mode == ModelInvokeMode.HTTP_ENDPOINT


if __name__ == "__main__":
    unittest.main()
