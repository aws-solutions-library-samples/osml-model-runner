#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import json
import os
from decimal import Decimal
from unittest import TestCase

import boto3
from moto import mock_sns, mock_sqs


@mock_sqs
@mock_sns
class TestStatusMonitor(TestCase):
    def setUp(self) -> None:
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.job_table import JobItem
        from aws.osml.model_runner.status.status_monitor import StatusMonitor

        self.status_monitor = StatusMonitor()
        self.job_item = JobItem(image_id="test-image-id")

        # create mock topic and queue
        sns_helper = self.status_monitor.image_status_sns
        sns = sns_helper.sns_client
        sns_response = sns.create_topic(Name=os.environ["IMAGE_STATUS_TOPIC"])
        mock_topic_arn = sns_response.get("TopicArn")
        self.status_monitor.image_status_sns.topic_arn = mock_topic_arn
        self.sqs = boto3.client("sqs", config=BotoConfig.default)
        sqs_response = self.sqs.create_queue(QueueName="mock_queue")
        self.mock_queue_url = sqs_response.get("QueueUrl")
        queue_attributes = self.sqs.get_queue_attributes(QueueUrl=self.mock_queue_url, AttributeNames=["QueueArn"])
        queue_arn = queue_attributes.get("Attributes").get("QueueArn")

        sns.subscribe(TopicArn=mock_topic_arn, Protocol="sqs", Endpoint=queue_arn)

    def tearDown(self) -> None:
        self.status_monitor = None
        self.job_item = None
        self.sqs = None

    def test_process_event(self):
        from aws.osml.model_runner.common.typing import ImageRequestStatus

        self.job_item.job_id = "1234"
        self.job_item.job_arn = "arn"
        self.job_item.processing_time = Decimal(1)
        mock_message = "test message 1"
        mock_status = ImageRequestStatus.SUCCESS
        expected_attributes = {
            "image_status": {"Type": "String", "Value": mock_status.value},
            "job_id": {"Type": "String", "Value": self.job_item.job_id},
            "job_arn": {"Type": "String", "Value": self.job_item.job_arn},
            "image_id": {"Type": "String", "Value": self.job_item.image_id},
            "processing_duration": {"Type": "String", "Value": str(self.job_item.processing_time)},
        }
        expected_message = f"StatusMonitor update: {mock_status} {self.job_item.job_id}: {mock_message}"
        self.status_monitor.process_event(self.job_item, mock_status, mock_message)

        messages = self.sqs.receive_message(
            QueueUrl=self.mock_queue_url,
            MessageAttributeNames=[
                "image_status",
                "job_id",
                "job_arn",
                "image_id",
                "processing_duration",
            ],
        ).get("Messages")
        assert len(messages) == 1
        message_body = json.loads(messages[0].get("Body"))
        assert message_body.get("Message") == expected_message
        assert message_body.get("MessageAttributes") == expected_attributes

    def test_process_event_sns_failure(self):
        from aws.osml.model_runner.common.typing import ImageRequestStatus
        from aws.osml.model_runner.status.status_monitor import StatusMonitorException

        self.job_item.job_id = "1234"
        self.job_item.job_arn = "arn"
        self.job_item.processing_time = Decimal(1)
        self.status_monitor.image_status_sns.topic_arn = "topic:arn:that:does:not:exist"
        with self.assertRaises(StatusMonitorException):
            self.status_monitor.process_event(self.job_item, ImageRequestStatus.SUCCESS, "test")

    def test_process_event_missing_attributes(self):
        from aws.osml.model_runner.common.typing import ImageRequestStatus
        from aws.osml.model_runner.status.status_monitor import StatusMonitorException

        with self.assertRaises(StatusMonitorException):
            self.status_monitor.process_event(self.job_item, ImageRequestStatus.SUCCESS, "test")
        self.job_item.job_id = "1234"
        with self.assertRaises(StatusMonitorException):
            self.status_monitor.process_event(self.job_item, ImageRequestStatus.SUCCESS, "test")
        self.job_item.job_arn = "arn"
        with self.assertRaises(StatusMonitorException):
            self.status_monitor.process_event(self.job_item, ImageRequestStatus.SUCCESS, "test")
        self.job_item.job_arn = None
        self.job_item.processing_time = Decimal(1)
        with self.assertRaises(StatusMonitorException):
            self.status_monitor.process_event(self.job_item, ImageRequestStatus.SUCCESS, "test")

    def test_get_image_request_status_success(self):
        from aws.osml.model_runner.common.typing import ImageRequestStatus

        self.job_item.region_count = 2
        self.job_item.region_success = 2
        self.job_item.region_error = 0
        image_status = self.status_monitor.get_image_request_status(self.job_item)
        assert image_status == ImageRequestStatus.SUCCESS

    def test_get_image_request_status_partial(self):
        from aws.osml.model_runner.common.typing import ImageRequestStatus

        self.job_item.region_count = 2
        self.job_item.region_success = 1
        self.job_item.region_error = 1
        image_status = self.status_monitor.get_image_request_status(self.job_item)
        assert image_status == ImageRequestStatus.PARTIAL

    def test_get_image_request_status_failed(self):
        from aws.osml.model_runner.common.typing import ImageRequestStatus

        self.job_item.region_count = 2
        self.job_item.region_success = 0
        self.job_item.region_error = 2
        image_status = self.status_monitor.get_image_request_status(self.job_item)
        assert image_status == ImageRequestStatus.FAILED

    def test_get_image_request_status_in_progress(self):
        from aws.osml.model_runner.common.typing import ImageRequestStatus

        self.job_item.region_count = 2
        self.job_item.region_success = 1
        self.job_item.region_error = 0
        image_status = self.status_monitor.get_image_request_status(self.job_item)
        assert image_status == ImageRequestStatus.IN_PROGRESS

    def test_get_image_request_status_missing_attributes(self):
        from aws.osml.model_runner.status.status_monitor import StatusMonitorException

        with self.assertRaises(StatusMonitorException):
            self.status_monitor.get_image_request_status(self.job_item)
        self.job_item.region_count = 2
        with self.assertRaises(StatusMonitorException):
            self.status_monitor.get_image_request_status(self.job_item)
        self.job_item.region_success = 2
        with self.assertRaises(StatusMonitorException):
            self.status_monitor.get_image_request_status(self.job_item)
        self.job_item.region_success = None
        self.job_item.region_error = 1
        with self.assertRaises(StatusMonitorException):
            self.status_monitor.get_image_request_status(self.job_item)
