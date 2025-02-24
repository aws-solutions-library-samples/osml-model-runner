#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import json
import time
import unittest

import boto3
from moto import mock_aws

from aws.osml.model_runner.api import ImageRequest
from aws.osml.model_runner.database.requested_jobs_table import RequestedJobsTable
from aws.osml.model_runner.queue.buffered_image_request_queue import BufferedImageRequestQueue


@mock_aws
class TestBufferedImageRequestQueue(unittest.TestCase):
    """Test cases for BufferedImageRequestQueue"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set up mock AWS resources
        self.sqs = boto3.client("sqs")
        self.dynamodb = boto3.resource("dynamodb")
        self.cloudwatch = boto3.client("cloudwatch")

        # Create SQS queues
        self.queue_url = self.sqs.create_queue(QueueName="test-image-queue")["QueueUrl"]
        self.dlq_url = self.sqs.create_queue(QueueName="test-image-dlq")["QueueUrl"]

        # Create DynamoDB table
        self.table_name = "test-requested-jobs"
        self.dynamodb.create_table(
            TableName=self.table_name,
            KeySchema=[{"AttributeName": "endpoint_id", "KeyType": "HASH"}, {"AttributeName": "job_id", "KeyType": "RANGE"}],
            AttributeDefinitions=[
                {"AttributeName": "endpoint_id", "AttributeType": "S"},
                {"AttributeName": "job_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Create RequestedJobsTable instance
        self.jobs_table = RequestedJobsTable(self.table_name)

        # Create BufferedImageRequestQueue instance
        self.queue = BufferedImageRequestQueue(
            image_queue_url=self.queue_url,
            image_dlq_url=self.dlq_url,
            requested_jobs_table=self.jobs_table,
            max_jobs_lookahead=10,
            retry_time=60,
            max_retry_attempts=2,
        )

    def create_sample_image_request_message_body(self, job_name: str = "test-job") -> dict:
        """Helper method to create a sample image request message"""
        return {
            "jobName": job_name,
            "jobId": f"{job_name}-id",
            "imageUrls": ["s3://test-bucket/test.nitf"],
            "outputs": [
                {"type": "S3", "bucket": "test-bucket", "prefix": "results"},
                {"type": "Kinesis", "stream": "test-stream", "batchSize": 1000},
            ],
            "imageProcessor": {"name": "test-model", "type": "SM_ENDPOINT"},
            "imageProcessorTileSize": 2048,
            "imageProcessorTileOverlap": 50,
        }

    def test_get_outstanding_requests_empty_queue(self):
        """Test getting outstanding requests when queue is empty"""
        requests = self.queue.get_outstanding_requests()
        self.assertEqual(len(requests), 0)

    def test_get_outstanding_requests_with_valid_messages(self):
        """Test getting outstanding requests with valid messages in queue"""
        # Add messages to queue
        for i in range(3):
            message = self.create_sample_image_request_message_body(f"job-{i}")
            self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=json.dumps(message))

        # Get outstanding requests
        requests = self.queue.get_outstanding_requests()
        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].job_id, "job-0-id")

    def test_handle_invalid_message(self):
        """Test handling of invalid messages"""
        # Send invalid message to queue
        self.sqs.send_message(QueueUrl=self.queue_url, MessageBody="invalid-json")

        # Process messages
        requests = self.queue.get_outstanding_requests()

        # Verify invalid message was moved to DLQ
        dlq_messages = self.sqs.receive_message(QueueUrl=self.dlq_url, MaxNumberOfMessages=10).get("Messages", [])

        self.assertEqual(len(requests), 0)
        self.assertEqual(len(dlq_messages), 1)
        self.assertEqual(dlq_messages[0]["Body"], "invalid-json")

    def test_retry_failed_requests(self):
        """Test retry mechanism for failed requests"""
        # Create a request and add it to the table
        request_data = self.create_sample_image_request_message_body()
        image_request = ImageRequest.from_external_message(request_data)
        status_record = self.jobs_table.add_new_request(image_request)

        # Force update of the item to look like it has already been run at sometime
        # in the past (longer than the retry timeout)
        self.jobs_table.table.update_item(
            Key={"endpoint_id": status_record.endpoint_id, "job_id": status_record.job_id},
            UpdateExpression="SET last_attempt = :time, num_attempts = num_attempts + :inc",
            ExpressionAttributeValues={":time": int(time.time()) - (self.queue.retry_time + 5), ":inc": 1},
            ReturnValues="UPDATED_NEW",
        )

        # Get outstanding requests, the request should be returned
        requests = self.queue.get_outstanding_requests()
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].num_attempts, 1)

    def test_purge_completed_requests(self):
        """Test purging of completed requests"""
        # Create a completed request
        request_data = self.create_sample_image_request_message_body()
        image_request = ImageRequest.from_external_message(request_data)

        self.jobs_table.add_new_request(image_request)
        self.jobs_table.update_request_details(image_request, region_count=1)
        self.jobs_table.complete_region(image_request, "region1")

        # Get outstanding requests (should purge completed)
        requests = self.queue.get_outstanding_requests()
        self.assertEqual(len(requests), 0)

    def test_max_retry_attempts_exceeded(self):
        """Test handling of requests that exceed max retry attempts"""
        # Create a request and add it to the table
        request_data = self.create_sample_image_request_message_body()
        image_request = ImageRequest.from_external_message(request_data)
        status_record = self.jobs_table.add_new_request(image_request)

        # Use up all of the retries
        for i in range(self.queue.max_retry_attempts + 1):
            self.jobs_table.start_next_attempt(status_record)
            status_record.num_attempts += 1

        # Get outstanding requests and make sure the attempt is not among them
        requests = self.queue.get_outstanding_requests()
        self.assertEqual(len(requests), 0)

        # Verify message was moved to DLQ
        dlq_messages = self.sqs.receive_message(QueueUrl=self.dlq_url, MaxNumberOfMessages=10).get("Messages", [])

        self.assertEqual(len(dlq_messages), 1)

    def test_respect_max_jobs_lookahead(self):
        """Test that the queue respects the max_jobs_lookahead limit"""
        # Add more messages than max_jobs_lookahead
        for i in range(self.queue.max_jobs_lookahead + 5):
            message = self.create_sample_image_request_message_body(f"job-{i}")
            self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=json.dumps(message))

        # Get outstanding requests
        requests = self.queue.get_outstanding_requests()
        self.assertEqual(len(requests), self.queue.max_jobs_lookahead)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Clean up DynamoDB table
        self.dynamodb.Table(self.table_name).delete()

        # Clean up SQS queues
        self.sqs.delete_queue(QueueUrl=self.queue_url)
        self.sqs.delete_queue(QueueUrl=self.dlq_url)


if __name__ == "__main__":
    unittest.main()
