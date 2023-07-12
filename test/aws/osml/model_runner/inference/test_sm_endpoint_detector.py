#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import datetime
import io
import json
import unittest
from json import JSONDecodeError
from unittest.mock import Mock

import boto3
import botocore
import mock
from botocore.stub import ANY, Stubber

MOCK_RESPONSE = {
    "Body": io.StringIO(
        json.dumps(
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
        )
    )
}


class TestSMDetector(unittest.TestCase):
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
        with mock.patch("aws.osml.model_runner.inference.sm_endpoint_detector.boto3") as mock_boto3:
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
            service_response=MOCK_RESPONSE,
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
            service_response=MOCK_RESPONSE,
        )
        sm_runtime_stub.add_client_error(str(JSONDecodeError))
        sm_runtime_stub.activate()

        with open("./test/data/GeogToWGS84GeoKey5.tif", "rb") as image_file:
            encoded_image = image_file.read()
            feature_collection = feature_detector.find_features(encoded_image)
            assert feature_collection["type"] == "FeatureCollection"
            assert len(feature_collection) == 2
            assert len(feature_collection["features"]) == 0

    def test_find_features_throw_client_exception(self):
        from aws.osml.model_runner.inference import SMDetector

        sm_client = boto3.client("sagemaker-runtime")
        sm_client_stub = Stubber(sm_client)
        feature_detector = SMDetector("test-endpoint")
        feature_detector.sm_client = sm_client
        sm_client_stub.add_response(
            "invoke_endpoint",
            expected_params={"EndpointName": "test-endpoint", "Body": ANY},
            service_response=MOCK_RESPONSE,
        )
        sm_client_stub.add_client_error(
            botocore.exceptions.ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item")
        )
        feature_detector.sm_client.invoke_endpoint = Mock(
            side_effect=botocore.exceptions.ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "send_message")
        )
        sm_client_stub.activate()

        with open("./test/data/GeogToWGS84GeoKey5.tif", "rb") as image_file:
            encoded_image = image_file.read()
            feature_collection = feature_detector.find_features(encoded_image)
            assert feature_collection["type"] == "FeatureCollection"
            assert len(feature_collection) == 2
            assert len(feature_collection["features"]) == 0

    def test_sm_name_generation(self):
        from aws.osml.model_runner.api.inference import ModelInvokeMode
        from aws.osml.model_runner.inference import SMDetector

        feature_detector = SMDetector("test-endpoint")
        assert feature_detector.mode == ModelInvokeMode.SM_ENDPOINT
