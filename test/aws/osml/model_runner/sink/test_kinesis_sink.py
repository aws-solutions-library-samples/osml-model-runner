#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import datetime
import unittest
from typing import List
from unittest import mock

import boto3
import geojson
import pytest
from botocore.stub import ANY, Stubber
from geojson import FeatureCollection

TEST_JOB_ID = "test-job-id"
TEST_RESULTS_STREAM = "test-results-stream"
MOCK_KINESIS_RESPONSE = {
    "FailedRecordCount": 1,  # Bug where this has to be set to min-value of 1: https://github.com/boto/botocore/issues/2063
    "Records": [
        {
            "ShardId": "shardId-000000000000",
            "SequenceNumber": "49632155903354096944077309979289188168053675801607929858",
        }
    ],
}

MOCK_KINESIS_DESCRIBE_STREAM_RESPONSE = {
    "StreamDescription": {
        "StreamName": TEST_RESULTS_STREAM,
        "StreamARN": "arn:aws:kinesis:us-west-2:012345678910:stream/test-stream-012345678910",
        "StreamStatus": "ACTIVE",
        "StreamModeDetails": {"StreamMode": "PROVISIONED"},
        "Shards": [
            {
                "ShardId": "shardId-000000000000",
                "HashKeyRange": {
                    "StartingHashKey": "0",
                    "EndingHashKey": "340282366920938463463374607431768211455",
                },
                "SequenceNumberRange": {
                    "StartingSequenceNumber": "StartingSequenceNumber",
                    "EndingSequenceNumber": "49632155903354096944077309979289188168053675801607929858",
                },
            }
        ],
        "HasMoreShards": True,
        "RetentionPeriodHours": 123,
        "StreamCreationTimestamp": datetime.datetime(2022, 1, 1),
        "EnhancedMonitoring": [
            {
                "ShardLevelMetrics": ["ALL"],
            }
        ],
    },
}

MOCK_KINESIS_BAD_DESCRIBE_STREAM_RESPONSE = {
    "StreamDescription": {
        "StreamName": TEST_RESULTS_STREAM,
        "StreamARN": "arn:aws:kinesis:us-west-2:012345678910:stream/test-stream-012345678910",
        "StreamStatus": "DELETING",
        "StreamModeDetails": {"StreamMode": "PROVISIONED"},
        "Shards": [
            {
                "ShardId": "shardId-000000000000",
                "HashKeyRange": {
                    "StartingHashKey": "0",
                    "EndingHashKey": "340282366920938463463374607431768211455",
                },
                "SequenceNumberRange": {
                    "StartingSequenceNumber": "StartingSequenceNumber",
                    "EndingSequenceNumber": "49632155903354096944077309979289188168053675801607929858",
                },
            }
        ],
        "HasMoreShards": True,
        "RetentionPeriodHours": 123,
        "StreamCreationTimestamp": datetime.datetime(2022, 1, 1),
        "EnhancedMonitoring": [
            {
                "ShardLevelMetrics": ["ALL"],
            }
        ],
    },
}


class TestKinesisSink(unittest.TestCase):
    def setUp(self):
        self.test_feature_list = self.build_feature_list()

    def tearDown(self):
        self.test_feature_list = None

    def test_write_features_default_credentials(self):
        """
        Write features to Kinesis using default credentials.
        Ensures that the `write` method can send records correctly when default
        credentials are used.
        """
        from aws.osml.model_runner.sink.kinesis_sink import KinesisSink

        kinesis_sink = KinesisSink(TEST_RESULTS_STREAM)
        kinesis_client_stub = Stubber(kinesis_sink.kinesis_client)
        kinesis_client_stub.activate()
        kinesis_client_stub.add_response(
            "describe_stream",
            MOCK_KINESIS_DESCRIBE_STREAM_RESPONSE,
            {"StreamName": TEST_RESULTS_STREAM},
        )

        records = [
            {"Data": geojson.dumps(FeatureCollection([feature])), "PartitionKey": TEST_JOB_ID}
            for feature in self.test_feature_list
        ]

        kinesis_client_stub.add_response(
            "put_records",
            MOCK_KINESIS_RESPONSE,
            {"StreamName": TEST_RESULTS_STREAM, "Records": records},
        )

        kinesis_sink.write(TEST_JOB_ID, self.test_feature_list)
        kinesis_client_stub.assert_no_pending_responses()

    def test_write_oversized_record(self):
        """
        Attempt to write oversized record to Kinesis.
        This should trigger an InvalidKinesisStreamException as the data exceeds Kinesis size limits.
        """
        from aws.osml.model_runner.sink.exceptions import InvalidKinesisStreamException
        from aws.osml.model_runner.sink.kinesis_sink import KinesisSink

        kinesis_sink = KinesisSink(TEST_RESULTS_STREAM)
        kinesis_client_stub = Stubber(kinesis_sink.kinesis_client)
        kinesis_client_stub.activate()

        kinesis_client_stub.add_response(
            "describe_stream",
            MOCK_KINESIS_DESCRIBE_STREAM_RESPONSE,
            {"StreamName": TEST_RESULTS_STREAM},
        )

        records = [
            {"Data": geojson.dumps(FeatureCollection([feature])), "PartitionKey": TEST_JOB_ID}
            for feature in self.test_feature_list
        ]

        kinesis_client_stub.add_client_error(
            "put_records",
            service_error_code="ValidationException",
            service_message="Member must have length less than or equal to 1048576.",
            expected_params={"StreamName": TEST_RESULTS_STREAM, "Records": records},
        )
        with pytest.raises(InvalidKinesisStreamException):
            kinesis_sink.write(TEST_JOB_ID, self.test_feature_list)

        kinesis_client_stub.assert_no_pending_responses()

    def test_bad_kinesis_stream_failure(self):
        """
        Attempt to write to a Kinesis stream with a bad status (e.g., DELETING).
        Validates that the `write` method does not send records and returns False.
        """
        from aws.osml.model_runner.sink.kinesis_sink import KinesisSink

        kinesis_sink = KinesisSink(TEST_RESULTS_STREAM)
        kinesis_client_stub = Stubber(kinesis_sink.kinesis_client)
        kinesis_client_stub.activate()
        kinesis_client_stub.add_response(
            "describe_stream",
            MOCK_KINESIS_BAD_DESCRIBE_STREAM_RESPONSE,
            {"StreamName": TEST_RESULTS_STREAM},
        )
        assert not kinesis_sink.write(TEST_JOB_ID, self.test_feature_list)
        kinesis_client_stub.assert_no_pending_responses()

    @mock.patch("aws.osml.model_runner.common.credentials_utils.sts_client")
    def test_assumed_credentials(self, mock_sts):
        """
        Initialize KinesisSink with assumed role credentials.
        Ensures that the Kinesis client is correctly configured with the assumed role's credentials.
        """
        from aws.osml.model_runner.sink.kinesis_sink import KinesisSink

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

        KinesisSink(stream=TEST_RESULTS_STREAM, assumed_role="OSMLKinesisWriter")

        boto3.DEFAULT_SESSION.client.assert_called_with(
            "kinesis",
            aws_access_key_id=test_access_key_id,
            aws_secret_access_key=test_secret_access_key,
            aws_session_token=test_secret_token,
            config=ANY,
        )
        boto3.DEFAULT_SESSION = None
        session_patch.stop()

    def test_empty_features(self):
        """
        Attempt to write an empty list of features.
        Validates that no records are sent to the Kinesis stream.
        """
        from aws.osml.model_runner.sink.kinesis_sink import KinesisSink

        kinesis_sink = KinesisSink(TEST_RESULTS_STREAM)
        kinesis_client_stub = Stubber(kinesis_sink.kinesis_client)
        kinesis_client_stub.activate()

        kinesis_client_stub.add_response(
            "describe_stream",
            MOCK_KINESIS_DESCRIBE_STREAM_RESPONSE,
            {"StreamName": TEST_RESULTS_STREAM},
        )

        # Attempt to write an empty list should succeed without flushing any records.
        assert kinesis_sink.write(TEST_JOB_ID, [])
        kinesis_client_stub.assert_no_pending_responses()

    @staticmethod
    def build_feature_list() -> List[geojson.Feature]:
        """
        Builds a known list of testing features from a data file.

        :return: A list of 6 different GeoJSON Features.
        """
        with open("./test/data/detections.geojson", "r") as geojson_file:
            sample_features = geojson.load(geojson_file)["features"]
        return sample_features


if __name__ == "__main__":
    unittest.main()
