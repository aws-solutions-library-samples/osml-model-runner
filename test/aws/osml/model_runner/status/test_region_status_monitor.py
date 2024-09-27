#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
import unittest

import boto3
from moto import mock_aws

from aws.osml.model_runner.app_config import BotoConfig, ServiceConfig
from aws.osml.model_runner.common import RequestStatus
from aws.osml.model_runner.database.region_request_table import RegionRequestItem
from aws.osml.model_runner.status.exceptions import StatusMonitorException
from aws.osml.model_runner.status.region_status_monitor import RegionStatusMonitor


@mock_aws
class TestRegionStatusMonitor(unittest.TestCase):
    def setUp(self):
        """Sets up the SNS mock and test data."""
        # Mock the SNS topic creation
        self.sns = boto3.client("sns", config=BotoConfig.default)
        sns_response = self.sns.create_topic(Name=os.environ["REGION_STATUS_TOPIC"])

        # Create an instance of RegionStatusMonitor for testing
        self.monitor = RegionStatusMonitor(sns_response.get("TopicArn"))

        # Set up test region request item
        self.test_request_item = RegionRequestItem(
            job_id="test-job",
            image_id="test-image",
            region_id="test-region",
            processing_duration=1000,
            failed_tile_count=0,
            failed_tiles=[],
            succeeded_tile_count=0,
            succeeded_tiles=[],
            total_tiles=10,
        )

    def test_process_event_success(self):
        """Tests process_event for a successful region request item."""
        status = RequestStatus.SUCCESS
        message = "Processing completed successfully."

        # No exception should be raised for a valid region request item
        try:
            self.monitor.process_event(self.test_request_item, status, message)
        except StatusMonitorException:
            self.fail("process_event raised StatusMonitorException unexpectedly!")

        # Check if message was published to SNS
        response = self.sns.list_topics()
        assert ServiceConfig.region_status_topic in response["Topics"][0]["TopicArn"]

    def test_process_event_failure(self):
        """Tests process_event for a failed region request item with missing fields."""
        invalid_request_item = RegionRequestItem(
            job_id=None,  # Required field
            image_id="test-image",
            region_id="test-region",
            processing_duration=None,  # Required field
            failed_tiles=[],
            total_tiles=10,
        )
        status = RequestStatus.FAILED
        message = "Processing failed."

        with self.assertRaises(StatusMonitorException):
            self.monitor.process_event(invalid_request_item, status, message)

    def test_get_status_success(self):
        """Tests get_status for a successful region request."""
        status = self.monitor.get_status(self.test_request_item)
        self.assertEqual(status, RequestStatus.SUCCESS)

    def test_get_status_partial(self):
        """Tests get_status for a partial region request."""
        self.test_request_item.failed_tile_count = 3  # Some tiles failed
        status = self.monitor.get_status(self.test_request_item)
        self.assertEqual(status, RequestStatus.PARTIAL)

    def test_get_status_failed(self):
        """Tests get_status for a failed region request."""
        self.test_request_item.failed_tile_count = 10  # All tiles failed
        status = self.monitor.get_status(self.test_request_item)
        self.assertEqual(status, RequestStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
