import json
from unittest import TestCase
from unittest.mock import patch

from urllib3.response import HTTPResponse

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

MOCK_BAD_JSON_RESPONSE = HTTPResponse(body="Not a json string".encode(), status=200)


class TestSMDetector(TestCase):
    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features(self, mock_pool_manager):
        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        mock_name = "test"
        feature_detector = HTTPDetector(endpoint=mock_endpoint, name=mock_name)
        assert feature_detector.name == mock_name

        mock_pool_manager.return_value.request.return_value = MOCK_RESPONSE

        with open("./test/data/small.ntf", "rb") as image_file:
            feature_collection = feature_detector.find_features(image_file)
            assert feature_collection["type"] == "FeatureCollection"
            assert len(feature_collection["features"]) == 1

    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features_RetryError(self, mock_pool_manager):
        from requests.exceptions import RetryError

        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        feature_detector = HTTPDetector(endpoint=mock_endpoint)

        mock_pool_manager.return_value.request.side_effect = RetryError("test RetryError")

        with open("./test/data/small.ntf", "rb") as image_file:
            feature_detector.find_features(image_file)
            assert feature_detector.error_count == 1

    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features_MaxRetryError(self, mock_pool_manager):
        from urllib3.exceptions import MaxRetryError

        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        feature_detector = HTTPDetector(endpoint=mock_endpoint)

        mock_pool_manager.return_value.request.side_effect = MaxRetryError("test MaxRetryError", url=mock_endpoint)

        with open("./test/data/small.ntf", "rb") as image_file:
            feature_detector.find_features(image_file)
            assert feature_detector.error_count == 1

    @patch("aws.osml.model_runner.inference.http_detector.urllib3.PoolManager", autospec=True)
    def test_find_features_JSONDecodeError(self, mock_pool_manager):
        from aws.osml.model_runner.inference import HTTPDetector

        mock_endpoint = "http://dummy/endpoint"
        feature_detector = HTTPDetector(endpoint=mock_endpoint)

        mock_pool_manager.return_value.request.return_value = MOCK_BAD_JSON_RESPONSE

        with open("./test/data/small.ntf", "rb") as image_file:
            feature_detector.find_features(image_file)
            assert feature_detector.error_count == 1
