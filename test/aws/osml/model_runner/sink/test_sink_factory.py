#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import json
import unittest
from typing import List
from unittest import mock

from geojson import Feature

from aws.osml.model_runner.api import InvalidImageRequestException
from aws.osml.model_runner.sink.sink_factory import SinkFactory


class TestSinkFactory(unittest.TestCase):
    def setUp(self):
        """
        Setup mock features and destinations for testing.
        """
        self.sample_feature_list = self.build_feature_list()
        self.s3_destination = json.dumps([{"type": "S3", "bucket": "test-bucket", "prefix": "test-prefix"}])
        self.kinesis_destination = json.dumps([{"type": "Kinesis", "stream": "test-stream"}])
        self.mixed_destinations = json.dumps(
            [{"type": "S3", "bucket": "test-bucket", "prefix": "test-prefix"}, {"type": "Kinesis", "stream": "test-stream"}]
        )

    def tearDown(self):
        """
        Clean up any resources after tests.
        """
        self.sample_feature_list = None

    @mock.patch("aws.osml.model_runner.sink.s3_sink.S3Sink.write", return_value=True)
    def test_s3_sink(self, mock_write):
        """
        Test sink features writing to S3 sink.
        Ensures that `sink_features` writes correctly to S3 using the SinkFactory.
        """
        result = SinkFactory.sink_features("test-job-id", self.s3_destination, self.sample_feature_list)
        self.assertTrue(result)
        mock_write.assert_called_once()

    @mock.patch("aws.osml.model_runner.sink.kinesis_sink.KinesisSink.write", return_value=True)
    def test_kinesis_sink(self, mock_write):
        """
        Test sink features writing to Kinesis sink.
        Ensures that `sink_features` writes correctly to Kinesis using the SinkFactory.
        """
        result = SinkFactory.sink_features("test-job-id", self.kinesis_destination, self.sample_feature_list)
        self.assertTrue(result)
        mock_write.assert_called_once()

    @mock.patch("aws.osml.model_runner.sink.kinesis_sink.KinesisSink.write", return_value=True)
    @mock.patch("aws.osml.model_runner.sink.s3_sink.S3Sink.write", return_value=True)
    def test_mixed_sinks_success(self, mock_s3_write, mock_kinesis_write):
        """
        Test sink features with mixed S3 and Kinesis sinks.
        Ensures that `sink_features` can handle both sinks and reports success when both write correctly.
        """
        result = SinkFactory.sink_features("test-job-id", self.mixed_destinations, self.sample_feature_list)
        self.assertTrue(result)
        mock_s3_write.assert_called_once()
        mock_kinesis_write.assert_called_once()

    @mock.patch("aws.osml.model_runner.sink.s3_sink.S3Sink.write", return_value=False)
    @mock.patch("aws.osml.model_runner.sink.kinesis_sink.KinesisSink.write", return_value=True)
    def test_mixed_sinks_partial_success(self, mock_s3_write, mock_kinesis_write):
        """
        Test sink features with one successful and one failed sink.
        Ensures that `sink_features` continues when one sink fails but another succeeds.
        """
        result = SinkFactory.sink_features("test-job-id", self.mixed_destinations, self.sample_feature_list)
        self.assertTrue(result)
        mock_s3_write.assert_called_once()
        mock_kinesis_write.assert_called_once()

    @mock.patch("aws.osml.model_runner.sink.s3_sink.S3Sink.write", return_value=False)
    @mock.patch("aws.osml.model_runner.sink.kinesis_sink.KinesisSink.write", return_value=False)
    def test_mixed_sinks_failure(self, mock_s3_write, mock_kinesis_write):
        """
        Test sink features when both S3 and Kinesis write operations fail.
        Ensures that `sink_features` reports failure when neither sink can write.
        """
        result = SinkFactory.sink_features("test-job-id", self.mixed_destinations, self.sample_feature_list)
        self.assertFalse(result)
        mock_s3_write.assert_called_once()
        mock_kinesis_write.assert_called_once()

    def test_invalid_sink_type(self):
        """
        Test outputs_to_sinks with an invalid sink type.
        Ensures that the method raises an InvalidImageRequestException for unknown sink types.
        """
        invalid_destination = json.dumps([{"type": "InvalidType", "bucket": "test-bucket", "prefix": "test-prefix"}])
        with self.assertRaises(InvalidImageRequestException):
            SinkFactory.outputs_to_sinks(json.loads(invalid_destination))

    def test_no_outputs_defined(self):
        """
        Test sink_features with no output destinations.
        Ensures that the method raises an InvalidImageRequestException when no destinations are provided.
        """
        with self.assertRaises(InvalidImageRequestException):
            SinkFactory.sink_features("test-job-id", "", self.sample_feature_list)

    @staticmethod
    def build_feature_list() -> List[Feature]:
        """
        Builds a sample list of geojson features for testing.
        """
        return [Feature(geometry={"type": "Point", "coordinates": [102.0, 0.5]}, properties={"prop0": "value0"})]


if __name__ == "__main__":
    unittest.main()
