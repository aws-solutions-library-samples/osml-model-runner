#  Copyright 2023 Amazon.com, Inc. or its affiliates.
import unittest
from typing import List
from unittest import mock

import boto3
import geojson
from botocore.stub import ANY, Stubber

TEST_PREFIX = "folder"
TEST_RESULTS_BUCKET = "test-results-bucket"
TEST_IMAGE_ID = "test-image-id"
MOCK_S3_PUT_OBJECT_RESPONSE = {
    "ResponseMetadata": {
        "RequestId": "5994D680BF127CE3",
        "HTTPStatusCode": 200,
        "RetryAttempts": 1,
    },
    "ETag": '"6299528715bad0e3510d1e4c4952ee7e"',
}

MOCK_S3_BUCKETS_RESPONSE = {
    "ResponseMetadata": {
        "RequestId": "5994D680BF127CE3",
        "HTTPStatusCode": 200,
        "RetryAttempts": 1,
    },
}


class TestS3Sink(unittest.TestCase):
    def setUp(self):
        self.sample_feature_list = self.build_feature_list()

    def tearDown(self):
        self.sample_feature_list = None

    def test_write_features_default_credentials(self):
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        s3_sink = S3Sink(TEST_RESULTS_BUCKET, TEST_PREFIX)
        s3_client_stub = Stubber(s3_sink.s3_client)
        s3_client_stub.activate()
        s3_client_stub.add_response(
            "head_bucket",
            MOCK_S3_BUCKETS_RESPONSE,
            {"Bucket": TEST_RESULTS_BUCKET},
        )
        s3_client_stub.add_response(
            "put_object",
            MOCK_S3_PUT_OBJECT_RESPONSE,
            {
                "ACL": "bucket-owner-full-control",
                "Bucket": TEST_RESULTS_BUCKET,
                "Key": "{}/{}.geojson".format(TEST_PREFIX, TEST_IMAGE_ID),
                "Body": ANY,
            },
        )
        s3_sink.write(TEST_IMAGE_ID, self.sample_feature_list)
        s3_client_stub.assert_no_pending_responses()

    def test_write_features_default_credentials_image_id_with_slash(self):
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        image_id_with_slashes = "fake/image/123"

        s3_sink = S3Sink(TEST_RESULTS_BUCKET, TEST_PREFIX)
        s3_client_stub = Stubber(s3_sink.s3_client)
        s3_client_stub.activate()
        s3_client_stub.add_response(
            "head_bucket",
            MOCK_S3_BUCKETS_RESPONSE,
            {"Bucket": TEST_RESULTS_BUCKET},
        )
        s3_client_stub.add_response(
            "put_object",
            MOCK_S3_PUT_OBJECT_RESPONSE,
            {
                "ACL": "bucket-owner-full-control",
                "Bucket": TEST_RESULTS_BUCKET,
                "Key": "{}/123.geojson".format(TEST_PREFIX),
                "Body": ANY,
            },
        )
        s3_sink.write(image_id_with_slashes, self.sample_feature_list)
        s3_client_stub.assert_no_pending_responses()

    def test_s3_bucket_404_failure(self):
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        s3_sink = S3Sink(TEST_RESULTS_BUCKET, TEST_PREFIX)
        s3_client_stub = Stubber(s3_sink.s3_client)
        s3_client_stub.activate()
        s3_client_stub.add_client_error(
            "head_bucket",
            service_error_code="404",
            service_message="Not Found",
            expected_params={"Bucket": "test-results-bucket"},
        )
        s3_sink.write(TEST_IMAGE_ID, self.sample_feature_list)
        s3_client_stub.assert_no_pending_responses()

    def test_s3_bucket_403_failure(self):
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        s3_sink = S3Sink(TEST_RESULTS_BUCKET, TEST_PREFIX)
        s3_client_stub = Stubber(s3_sink.s3_client)
        s3_client_stub.activate()
        s3_client_stub.add_client_error(
            "head_bucket",
            service_error_code="403",
            service_message="Forbidden",
            expected_params={"Bucket": "test-results-bucket"},
        )
        s3_sink.write(TEST_IMAGE_ID, self.sample_feature_list)
        s3_client_stub.assert_no_pending_responses()

    @mock.patch("aws.osml.model_runner.common.credentials_utils.sts_client")
    def test_assumed_credentials(self, mock_sts):
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        test_access_key_id = "123456789"
        test_secret_access_key = "987654321"
        test_secret_token = "SecretToken123"

        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": test_access_key_id,
                "SecretAccessKey": test_secret_access_key,
                "SessionToken": test_secret_token,
            }
        }

        session_patch = mock.patch("boto3.Session", autospec=True)
        test_session = session_patch.start()
        boto3.DEFAULT_SESSION = test_session

        S3Sink(TEST_RESULTS_BUCKET, TEST_PREFIX, assumed_role="OSMLS3Writer")

        boto3.DEFAULT_SESSION.client.assert_called_with(
            "s3",
            aws_access_key_id=test_access_key_id,
            aws_secret_access_key=test_secret_access_key,
            aws_session_token=test_secret_token,
            config=ANY,
        )
        boto3.DEFAULT_SESSION = None
        session_patch.stop()

    def test_return_name(self):
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        s3_sink = S3Sink(TEST_RESULTS_BUCKET, TEST_PREFIX)
        assert "S3" == s3_sink.name()

    def test_return_mode(self):
        from aws.osml.model_runner.api.sink import SinkMode
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        s3_sink = S3Sink(TEST_RESULTS_BUCKET, TEST_PREFIX)
        assert SinkMode.AGGREGATE == s3_sink.mode

    @staticmethod
    def build_feature_list() -> List[geojson.Feature]:
        with open("./test/data/detections.geojson", "r") as geojson_file:
            sample_features = geojson.load(geojson_file)["features"]
        return sample_features


if __name__ == "__main__":
    unittest.main()
