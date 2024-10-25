#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import json
import unittest
from json import JSONDecodeError
from unittest import TestCase
from unittest.mock import patch

import pytest
from urllib3.response import HTTPResponse

# Mock response simulating a successful HTTP response with valid JSON feature collection
MOCK_RESPONSE = HTTPResponse(
    body=json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "1cc5e6d6-e12f-430d-adf0-8d2276ce8c5a",
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    "properties": {
                        "bounds_imcoords": [429, 553, 440, 561],
                        "feature_types": {"ground_motor_passenger_vehicle": 0.2961518168449402},
                        "detection_score": 0.2961518168449402,
                        "image_id": "test-image-id",
                    },
                }
            ],
        }
    ).encode(),
    status=200,
)

# Mock response simulating an HTTP response with invalid JSON
MOCK_BAD_JSON_RESPONSE = HTTPResponse(body="Not a json string".encode(), status=200)


class TestHTTPDetector(TestCase):
    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features(self, mock_pool_manager):
        """
        Test the find_features method to verify that the HTTPDetector correctly processes
        a valid HTTP response and returns a feature collection.
        """
        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        mock_name = "test"
        feature_detector = HTTPDetector(endpoint=mock_endpoint, name=mock_name)

        # Verify that the detector is correctly initialized
        assert feature_detector.name == mock_name

        # Mock the HTTP response
        mock_pool_manager.return_value.request.return_value = MOCK_RESPONSE

        with open("./test/data/small.ntf", "rb") as image_file:
            # Call the method and verify the response
            feature_collection = feature_detector.find_features(image_file)
            assert feature_collection["type"] == "FeatureCollection"
            assert len(feature_collection["features"]) == 1

    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features_RetryError(self, mock_pool_manager):
        """
        Test that find_features raises a RetryError when a retry issue occurs during the HTTP request.
        """
        from requests.exceptions import RetryError

        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        feature_detector = HTTPDetector(endpoint=mock_endpoint)

        # Simulate a retry error during the HTTP request
        mock_pool_manager.return_value.request.side_effect = RetryError("test RetryError")

        with open("./test/data/small.ntf", "rb") as image_file:
            # Expecting the function to raise a RetryError
            with pytest.raises(RetryError):
                feature_detector.find_features(image_file)

    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features_MaxRetryError(self, mock_pool_manager):
        """
        Test that find_features raises a MaxRetryError when maximum retries are exceeded during the HTTP request.
        """
        from urllib3.exceptions import MaxRetryError

        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        feature_detector = HTTPDetector(endpoint=mock_endpoint)

        # Simulate a maximum retry error during the HTTP request
        mock_pool_manager.return_value.request.side_effect = MaxRetryError("test MaxRetryError", url=mock_endpoint)

        with open("./test/data/small.ntf", "rb") as image_file:
            # Expecting the function to raise a MaxRetryError
            with pytest.raises(MaxRetryError):
                feature_detector.find_features(image_file)

    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features_JSONDecodeError(self, mock_pool_manager):
        """
        Test that find_features raises a JSONDecodeError when the HTTP response contains invalid JSON data.
        """
        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        feature_detector = HTTPDetector(endpoint=mock_endpoint)

        # Simulate an HTTP response with invalid JSON content
        mock_pool_manager.return_value.request.return_value = MOCK_BAD_JSON_RESPONSE

        with open("./test/data/small.ntf", "rb") as image_file:
            # Expecting the function to raise a JSONDecodeError
            with pytest.raises(JSONDecodeError):
                feature_detector.find_features(image_file)


if __name__ == "__main__":
    unittest.main()
