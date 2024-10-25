#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from unittest import TestCase
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

# Mock messages for testing
TEST_MOCK_MESSAGE = {
    "Type": "Notification",
    "MessageId": "63077f04-26ef-5d12-824f-23059ffa2596",
    "TopicArn": "arn:aws:sns:us-west-2:012345678910:user-SNS-ImageStatusTopic9DE4DAE6-LFLJincx1Hka",
    "Message": "StatusMonitor update: IN_PROGRESS 07ac729f-a43d-4c96-952b-8126366fb298: Processing regions",
    "Timestamp": "2022-11-30T20:02:29.796Z",
    "MessageAttributes": {
        "job_id": {"Type": "String", "Value": "0"},
        "processing_duration": {"Type": "String", "Value": "0.290897369384765625"},
        "image_status": {"Type": "String", "Value": "IN_PROGRESS"},
        "image_id": {"Type": "String", "Value": "0:s3://test-images-012345678910/images/small.ntf"},
    },
}

TEST_MOCK_INVALID_MESSAGE = {"BAD_MESSAGE": "INVALID_MESSAGE"}

# Mock exception to simulate client errors
TEST_MOCK_CLIENT_EXCEPTION = Mock(
    side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "send_message")
)


@mock_aws
class TestRequestQueue(TestCase):
    def setUp(self):
        """
        Sets up the test environment by creating a mock SQS queue
        and initializing the RequestQueue object.
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.queue.request_queue import RequestQueue

        # Set up SQS resource and client
        self.sqs = boto3.resource("sqs", config=BotoConfig.default)
        self.sqs_client = boto3.client("sqs", config=BotoConfig.default)
        self.sqs_response = self.sqs.create_queue(QueueName="mock_queue")
        self.mock_queue_url = self.sqs_response.url
        self.request_queue = RequestQueue(queue_url=self.mock_queue_url)

    def tearDown(self):
        """
        Cleans up the test environment.
        """
        self.sqs = None
        self.sqs_client = None
        self.sqs_response = None
        self.mock_queue_url = None
        self.request_queue = None

    def test_send_request_succeed(self):
        """
        Test that a message can be successfully sent to the SQS queue.
        """
        self.request_queue.send_request(TEST_MOCK_MESSAGE)

        # Verify that the message was added to the queue
        sqs_messages = self.sqs_response.receive_messages()
        assert sqs_messages[0].receipt_handle is not None

    def test_reset_request_succeed(self):
        """
        Test that a message visibility timeout can be successfully reset.
        """
        self.request_queue.send_request(TEST_MOCK_MESSAGE)
        sqs_messages = self.sqs_response.receive_messages()
        receipt_handle = sqs_messages[0].receipt_handle

        # Reset the visibility timeout
        self.request_queue.reset_request(receipt_handle)
        sqs_messages = self.sqs_response.receive_messages()
        assert sqs_messages[0].receipt_handle is not None

    def test_finish_request_succeed(self):
        """
        Test that a message can be successfully deleted from the SQS queue.
        """
        self.request_queue.send_request(TEST_MOCK_MESSAGE)
        sqs_messages = self.sqs_response.receive_messages()
        receipt_handle = sqs_messages[0].receipt_handle

        # Delete the message
        self.request_queue.finish_request(receipt_handle)
        sqs_messages = self.sqs_response.receive_messages()
        assert len(sqs_messages) == 0

    def test_send_request_failure(self):
        """
        Test that the send_request method handles client errors gracefully.
        """
        self.request_queue.sqs_client.send_message = TEST_MOCK_CLIENT_EXCEPTION
        # Should not raise an exception
        self.request_queue.send_request(TEST_MOCK_MESSAGE)

    def test_reset_request_failure(self):
        """
        Test that the reset_request method handles client errors gracefully.
        """
        self.request_queue.send_request(TEST_MOCK_MESSAGE)
        sqs_messages = self.sqs_response.receive_messages()
        receipt_handle = sqs_messages[0].receipt_handle

        self.request_queue.sqs_client.change_message_visibility = TEST_MOCK_CLIENT_EXCEPTION
        # Should not raise an exception
        self.request_queue.reset_request(receipt_handle)

    def test_finish_request_failure(self):
        """
        Test that the finish_request method handles client errors gracefully.
        """
        self.request_queue.send_request(TEST_MOCK_MESSAGE)
        sqs_messages = self.sqs_response.receive_messages()
        receipt_handle = sqs_messages[0].receipt_handle

        self.request_queue.sqs_client.delete_message = TEST_MOCK_CLIENT_EXCEPTION
        # Should not raise an exception
        self.request_queue.finish_request(receipt_handle)

    def test_iter_request_queue(self):
        """
        Test that the RequestQueue iterator correctly retrieves messages.
        """
        request_queue_iter = iter(self.request_queue)

        # Check if there's no pending request in the queue
        receipt_handle, request_message = next(request_queue_iter)
        assert receipt_handle is None
        assert request_message is None

        # Add a pending request to the queue
        self.request_queue.send_request(TEST_MOCK_MESSAGE)
        receipt_handle, request_message = next(request_queue_iter)
        assert receipt_handle is not None
        assert request_message is not None

    def test_iter_request_queue_exception(self):
        """
        Test that the iterator handles exceptions when receiving messages.
        """
        request_queue_iter = iter(self.request_queue)

        # Simulate client exception
        self.request_queue.sqs_client.receive_message = TEST_MOCK_CLIENT_EXCEPTION
        # Should not raise an exception
        receipt_handle, request_message = next(request_queue_iter)
        assert receipt_handle is None
        assert request_message is None


if __name__ == "__main__":
    unittest.main()
