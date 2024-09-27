#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
import unittest

import boto3
from moto import mock_aws

from aws.osml.model_runner.app_config import BotoConfig, ServiceConfig
from aws.osml.model_runner.common import RequestStatus
from aws.osml.model_runner.database.job_table import JobItem
from aws.osml.model_runner.status.exceptions import StatusMonitorException
from aws.osml.model_runner.status.image_status_monitor import ImageStatusMonitor


@mock_aws
class TestImageStatusMonitor(unittest.TestCase):
    def setUp(self):
        """Sets up the SNS mock and test data."""
        # Mock the SNS topic creation
        self.sns = boto3.client("sns", config=BotoConfig.default)
        sns_response = self.sns.create_topic(Name=os.environ["IMAGE_STATUS_TOPIC"])

        # Create an instance of ImageStatusMonitor for testing
        self.monitor = ImageStatusMonitor(sns_response.get("TopicArn"))

        # Set up test job item
        self.test_job_item = JobItem(
            job_id="test-job",
            image_id="test-image",
            processing_duration=1000,
            region_success=5,
            region_error=0,
            region_count=5,
        )

    def test_process_event_success(self):
        """Tests process_event for a successful image request item."""
        status = RequestStatus.SUCCESS
        message = "Processing completed successfully."

        # No exception should be raised for a valid job item
        try:
            self.monitor.process_event(self.test_job_item, status, message)
        except StatusMonitorException:
            self.fail("process_event raised StatusMonitorException unexpectedly!")

        # Check if message was published to SNS
        response = self.sns.list_topics()
        assert ServiceConfig.image_status_topic in response["Topics"][0]["TopicArn"]

    def test_process_event_failure(self):
        """Tests process_event for a failed image request item with missing fields."""
        invalid_job_item = JobItem(
            job_id=None,
            image_id="test-image",
            processing_duration=None,
            region_success=0,
            region_error=5,
            region_count=5,
        )
        status = RequestStatus.FAILED
        message = "Processing failed."

        with self.assertRaises(StatusMonitorException):
            self.monitor.process_event(invalid_job_item, status, message)

    def test_get_status_success(self):
        """Tests get_status for a successful image request."""
        status = self.monitor.get_status(self.test_job_item)
        self.assertEqual(status, RequestStatus.SUCCESS)

    def test_get_status_partial(self):
        """Tests get_status for a partial image request."""
        self.test_job_item.region_success = 3
        self.test_job_item.region_error = 2
        status = self.monitor.get_status(self.test_job_item)
        self.assertEqual(status, RequestStatus.PARTIAL)

    def test_get_status_failed(self):
        """Tests get_status for a failed image request."""
        self.test_job_item.region_success = 0
        self.test_job_item.region_error = 5  # All regions failed
        status = self.monitor.get_status(self.test_job_item)
        self.assertEqual(status, RequestStatus.FAILED)

    def test_get_status_in_progress(self):
        """Tests get_status for an in-progress image request."""
        self.test_job_item.region_success = 2
        self.test_job_item.region_error = 1
        self.test_job_item.region_count = 5  # Still in progress
        status = self.monitor.get_status(self.test_job_item)
        self.assertEqual(status, RequestStatus.IN_PROGRESS)


if __name__ == "__main__":
    unittest.main()
