#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import datetime
import io
import json
from json import JSONDecodeError
from unittest import TestCase
from unittest.mock import Mock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

MOCK_MODEL_RESPONSE = {
    "Body": io.StringIO(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "1cc5e6d6-e12f-430d-adf0-8d2276ce8c5a",
                        "geometry": {"type": "Point", "coordinates": [-43.679691, -22.941953]},
                        "properties": {
                            "bounds_imcoords": [429, 553, 440, 561],
                            "geom_imcoords": [[429, 553], [429, 561], [440, 561], [440, 553], [429, 553]],
                            "featureClasses": [{"iri": "ground_motor_passenger_vehicle", "score": 0.2961518168449402}],
                            "detection_score": 0.2961518168449402,
                            "image_id": "2pp5e6d6-e12f-430d-adf0-8d2276ceadf0",
                        },
                    }
                ],
            }
        )
    )
}


class TestSMDetector(TestCase):
    def test_construct_with_execution_role(self):
        from aws.osml.model_runner.inference import SMDetector

        sm_client = boto3.client("sagemaker-runtime")
        sm_client_stub = Stubber(sm_client)
        sm_client_stub.activate()
        aws_credentials = {
            "AccessKeyId": "FAKE-ACCESS-KEY-ID",
            "SecretAccessKey": "FAKE-ACCESS-KEY",
            "SessionToken": "FAKE-SESSION-TOKEN",
            "Expiration": datetime.datetime.now(),
        }
        with patch("aws.osml.model_runner.inference.sm_detector.boto3") as mock_boto3:
            mock_boto3.client.return_value = sm_client
            SMDetector("test-endpoint", aws_credentials)
            mock_boto3.client.assert_called_once_with(
                "sagemaker-runtime",
                aws_access_key_id="FAKE-ACCESS-KEY-ID",
                aws_secret_access_key="FAKE-ACCESS-KEY",
                aws_session_token="FAKE-SESSION-TOKEN",
                config=ANY,
            )

    def test_find_features(self):
        from aws.osml.model_runner.inference import SMDetector

        feature_detector = SMDetector("test-endpoint")
        sm_runtime_stub = Stubber(feature_detector.sm_client)
        sm_runtime_stub.add_response(
            "invoke_endpoint",
            expected_params={"EndpointName": "test-endpoint", "Body": ANY},
            service_response=MOCK_MODEL_RESPONSE,
        )
        sm_runtime_stub.activate()

        with open("./test/data/GeogToWGS84GeoKey5.tif", "rb") as image_file:
            encoded_image = image_file.read()
            feature_collection = feature_detector.find_features(encoded_image)
            sm_runtime_stub.assert_no_pending_responses()
            assert feature_collection["type"] == "FeatureCollection"
            assert len(feature_collection["features"]) == 1

    def test_find_features_throw_json_exception(self):
        from aws.osml.model_runner.inference import SMDetector

        feature_detector = SMDetector("test-endpoint")
        sm_runtime_stub = Stubber(feature_detector.sm_client)
        sm_runtime_stub.add_response(
            "invoke_endpoint",
            expected_params={"EndpointName": "test-endpoint", "Body": ANY},
            service_response=MOCK_MODEL_RESPONSE,
        )
        sm_runtime_stub.add_client_error(str(JSONDecodeError))
        sm_runtime_stub.activate()

        with open("./test/data/GeogToWGS84GeoKey5.tif", "rb") as image_file:
            with pytest.raises(JSONDecodeError):
                feature_detector.find_features(image_file)

    def test_find_features_throw_client_exception(self):
        from aws.osml.model_runner.inference import SMDetector

        sm_client = boto3.client("sagemaker-runtime")
        sm_client_stub = Stubber(sm_client)
        feature_detector = SMDetector("test-endpoint")
        feature_detector.sm_client = sm_client
        sm_client_stub.add_response(
            "invoke_endpoint",
            expected_params={"EndpointName": "test-endpoint", "Body": ANY},
            service_response=MOCK_MODEL_RESPONSE,
        )
        sm_client_stub.add_client_error(str(ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item")))
        feature_detector.sm_client.invoke_endpoint = Mock(
            side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "send_message")
        )
        sm_client_stub.activate()

        with open("./test/data/GeogToWGS84GeoKey5.tif", "rb") as image_file:
            with pytest.raises(ClientError):
                feature_detector.find_features(image_file)

    def test_sm_name_generation(self):
        from aws.osml.model_runner.api.inference import ModelInvokeMode
        from aws.osml.model_runner.inference import HTTPDetector, SMDetector

        sm_name = "sm-test"
        sm_detector = SMDetector(endpoint=sm_name)
        assert sm_detector.mode == ModelInvokeMode.SM_ENDPOINT
        assert sm_detector.endpoint == sm_name

        http_name = "http-test"
        http_detector = HTTPDetector(endpoint=http_name)
        assert http_detector.mode == ModelInvokeMode.HTTP_ENDPOINT
        assert http_detector.endpoint == http_name
