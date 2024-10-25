#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import json
import os
import unittest
from unittest import TestCase
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

# Mock to simulate a ClientError when trying to publish to SNS
TEST_MOCK_PUBLISH_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "publish"))


@mock_aws
class TestSnsHelper(TestCase):
    def setUp(self):
        """
        Set up the mock AWS environment. This includes creating an SNS topic and an SQS queue,
        subscribing the queue to the SNS topic, and initializing the SNSHelper instance.
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.status.sns_helper import SNSHelper

        # Initialize mock SNS and SQS clients
        self.sns = boto3.client("sns", config=BotoConfig.default)
        sns_response = self.sns.create_topic(Name=os.environ["IMAGE_STATUS_TOPIC"])
        self.mock_topic_arn = sns_response.get("TopicArn")

        self.sqs = boto3.client("sqs", config=BotoConfig.default)
        sqs_response = self.sqs.create_queue(QueueName="mock_queue")
        self.mock_queue_url = sqs_response.get("QueueUrl")
        queue_attributes = self.sqs.get_queue_attributes(QueueUrl=self.mock_queue_url, AttributeNames=["QueueArn"])
        queue_arn = queue_attributes.get("Attributes").get("QueueArn")

        # Subscribe the mock SQS queue to the SNS topic
        self.sns.subscribe(TopicArn=self.mock_topic_arn, Protocol="sqs", Endpoint=queue_arn)

        # Initialize SNSHelper with the mock topic ARN
        self.image_status_sns = SNSHelper(self.mock_topic_arn)

    def tearDown(self):
        """
        Clean up by setting all instance variables to None.
        """
        self.sns = None
        self.mock_topic_arn = None
        self.sqs = None
        self.mock_queue_url = None
        self.image_status_sns = None

    def test_publish_message_success(self):
        """
        Test that a valid message with string and binary attributes is successfully published to SNS
        and received by the subscribed SQS queue. Verify that message attributes are correctly transformed.
        """
        mock_message = "test message 1"
        mock_attributes = {"key1": "string data", "bin1": b"binary data"}
        expected_attributes = {
            "key1": {"Type": "String", "Value": "string data"},
            "bin1": {"Type": "Binary", "Value": "YmluYXJ5IGRhdGE="},  # Base64 encoded binary data
        }

        # Publish the message
        self.image_status_sns.publish_message(mock_message, mock_attributes)

        # Retrieve and verify the message from the mock SQS queue
        messages = self.sqs.receive_message(QueueUrl=self.mock_queue_url, MessageAttributeNames=["key1", "bin1"]).get(
            "Messages"
        )
        assert len(messages) == 1
        message_body = json.loads(messages[0].get("Body"))
        assert message_body.get("Message") == mock_message
        assert message_body.get("MessageAttributes") == expected_attributes

    def test_publish_message_success_drop_invalid_types(self):
        """
        Test that a message with invalid attribute types (e.g., integer) drops the invalid attributes
        and only publishes valid string and binary attributes.
        """
        mock_message = "test invalid data gets removed"
        mock_attributes = {"key1": "string data", "bin1": b"binary data", "invalid_int_data": 1}
        expected_attributes = {
            "key1": {"Type": "String", "Value": "string data"},
            "bin1": {"Type": "Binary", "Value": "YmluYXJ5IGRhdGE="},  # Base64 encoded binary data
        }

        # Publish the message
        self.image_status_sns.publish_message(mock_message, mock_attributes)

        # Retrieve and verify the message from the mock SQS queue
        messages = self.sqs.receive_message(QueueUrl=self.mock_queue_url, MessageAttributeNames=["key1", "bin1"]).get(
            "Messages"
        )
        assert len(messages) == 1
        message_body = json.loads(messages[0].get("Body"))
        assert message_body.get("Message") == mock_message
        assert message_body.get("MessageAttributes") == expected_attributes

    def test_publish_message_failure(self):
        """
        Test that when the SNS publish operation fails, an SNSPublishException is raised.
        """
        from aws.osml.model_runner.status.exceptions import SNSPublishException

        # Mock the publish method to simulate a failure
        self.image_status_sns.sns_client.publish = TEST_MOCK_PUBLISH_EXCEPTION
        mock_message = "test message 1"
        mock_attributes = {"key1": "string data", "bin1": b"binary data"}

        # Expecting SNSPublishException due to the mock exception
        with self.assertRaises(SNSPublishException):
            self.image_status_sns.publish_message(mock_message, mock_attributes)

    def test_publish_message_no_topic(self):
        """
        Test that if no SNS topic is configured, the publish_message method should gracefully return None
        and not attempt to send a message.
        """
        from aws.osml.model_runner.status.sns_helper import SNSHelper

        # Initialize SNSHelper with no topic
        image_status_sns = SNSHelper(None)
        mock_message = "test message 1"
        mock_attributes = {"key1": "string data", "bin1": b"binary data"}

        # Verify that publishing returns None when no topic is configured
        response = image_status_sns.publish_message(mock_message, mock_attributes)
        assert response is None


if __name__ == "__main__":
    unittest.main()
