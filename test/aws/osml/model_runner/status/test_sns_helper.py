#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import json
import os
import unittest
from unittest import TestCase
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

TEST_MOCK_PUBLISH_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "publish"))


@mock_aws
class TestSnsHelper(TestCase):
    def setUp(self):
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.status.sns_helper import SNSHelper

        self.sns = boto3.client("sns", config=BotoConfig.default)
        sns_response = self.sns.create_topic(Name=os.environ["IMAGE_STATUS_TOPIC"])
        self.mock_topic_arn = sns_response.get("TopicArn")

        self.sqs = boto3.client("sqs", config=BotoConfig.default)
        sqs_response = self.sqs.create_queue(QueueName="mock_queue")
        self.mock_queue_url = sqs_response.get("QueueUrl")
        queue_attributes = self.sqs.get_queue_attributes(QueueUrl=self.mock_queue_url, AttributeNames=["QueueArn"])
        queue_arn = queue_attributes.get("Attributes").get("QueueArn")

        self.sns.subscribe(TopicArn=self.mock_topic_arn, Protocol="sqs", Endpoint=queue_arn)

        self.image_status_sns = SNSHelper(self.mock_topic_arn)

    def tearDown(self):
        self.sns = None
        self.mock_topic_arn = None
        self.sqs = None
        self.mock_queue_url = None
        self.image_status_sns = None

    def test_publish_message_success(self):
        mock_message = "test message 1"
        mock_attributes = {"key1": "string data", "bin1": b"binary data"}
        expected_attributes = {
            "key1": {"Type": "String", "Value": "string data"},
            "bin1": {"Type": "Binary", "Value": "YmluYXJ5IGRhdGE="},
        }
        self.image_status_sns.publish_message(mock_message, mock_attributes)
        messages = self.sqs.receive_message(QueueUrl=self.mock_queue_url, MessageAttributeNames=["key1", "bin1"]).get(
            "Messages"
        )
        assert len(messages) == 1
        message_body = json.loads(messages[0].get("Body"))
        assert message_body.get("Message") == mock_message
        assert message_body.get("MessageAttributes") == expected_attributes

    def test_publish_message_success_drop_invalid_types(self):
        mock_message = "test invalid data gets removed"
        mock_attributes = {"key1": "string data", "bin1": b"binary data", "invalid_int_data": 1}
        expected_attributes = {
            "key1": {"Type": "String", "Value": "string data"},
            "bin1": {"Type": "Binary", "Value": "YmluYXJ5IGRhdGE="},
        }
        self.image_status_sns.publish_message(mock_message, mock_attributes)
        messages = self.sqs.receive_message(QueueUrl=self.mock_queue_url, MessageAttributeNames=["key1", "bin1"]).get(
            "Messages"
        )
        assert len(messages) == 1
        message_body = json.loads(messages[0].get("Body"))
        assert message_body.get("Message") == mock_message
        assert message_body.get("MessageAttributes") == expected_attributes

    def test_publish_message_failure(self):
        from aws.osml.model_runner.status.exceptions import SNSPublishException

        self.image_status_sns.sns_client.publish = TEST_MOCK_PUBLISH_EXCEPTION
        mock_message = "test message 1"
        mock_attributes = {"key1": "string data", "bin1": b"binary data"}
        with self.assertRaises(SNSPublishException):
            self.image_status_sns.publish_message(mock_message, mock_attributes)

    def test_publish_message_no_topic(self):
        from aws.osml.model_runner.status.sns_helper import SNSHelper

        image_status_sns = SNSHelper(None)
        mock_message = "test message 1"
        mock_attributes = {"key1": "string data", "bin1": b"binary data"}
        response = image_status_sns.publish_message(mock_message, mock_attributes)
        assert response is None


if __name__ == "__main__":
    unittest.main()
