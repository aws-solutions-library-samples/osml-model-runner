#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import unittest

from aws.osml.model_runner.api.image_request import ImageRequest
from aws.osml.model_runner.api.inference import ModelInvokeMode
from aws.osml.model_runner.scheduler.fifo_image_scheduler import FIFOImageScheduler


class MockRequestQueue:
    def __init__(self):
        self.messages = []
        self.finished_receipts = set()
        self.reset_receipts = set()

    def add_message(self, receipt_handle, message):
        self.messages.append((receipt_handle, message))

    def finish_request(self, receipt_handle):
        self.finished_receipts.add(receipt_handle)

    def reset_request(self, receipt_handle, visibility_timeout=0):
        self.reset_receipts.add(receipt_handle)

    def __iter__(self):
        return iter(self.messages)


class TestFIFOImageScheduler(unittest.TestCase):
    def setUp(self):
        self.mock_queue = MockRequestQueue()
        self.scheduler = FIFOImageScheduler(self.mock_queue)

    def test_get_next_scheduled_request_success(self):
        """Test successful retrieval of next scheduled request"""
        # Setup
        test_receipt_handle = "receipt-123"
        test_message = {
            "jobName": "test-job-name",
            "jobId": "job-123",
            "imageUrls": ["test-image-url"],
            "outputs": [
                {"type": "S3", "bucket": "test-bucket", "prefix": "test-bucket-prefix"},
                {"type": "Kinesis", "stream": "test-stream", "batchSize": 1000},
            ],
            "imageProcessor": {"name": "test-model", "type": "SM_ENDPOINT"},
            "imageProcessorTileSize": 1024,
            "imageProcessorTileOverlap": 50,
        }
        self.mock_queue.add_message(test_receipt_handle, test_message)

        # Execute
        result = self.scheduler.get_next_scheduled_request()

        # Assert
        self.assertIsInstance(result, ImageRequest)
        self.assertTrue(result.is_valid())
        self.assertEqual(result.job_id, "job-123")

    def test_get_next_scheduled_request_empty_queue(self):
        """Test behavior when queue is empty"""
        # Execute
        result = self.scheduler.get_next_scheduled_request()

        # Assert
        self.assertIsNone(result)

    def test_get_next_scheduled_request_invalid_request(self):
        """Test handling of invalid image request"""
        # Setup
        test_receipt_handle = "receipt-123"
        test_message = {
            "jobId": "job-123",
            # Missing required fields to make it invalid
        }
        self.mock_queue.add_message(test_receipt_handle, test_message)

        # Execute
        result = self.scheduler.get_next_scheduled_request()

        # Assert
        self.assertIsNone(result)
        self.assertIn(test_receipt_handle, self.mock_queue.finished_receipts)

    def test_finish_request_success(self):
        """Test successful completion of request"""
        # Setup
        test_receipt_handle = "receipt-123"
        test_image_request = ImageRequest(
            job_id="job-123",
            image_id="image-123",
            image_url="s3://bucket/image.tif",
            image_read_role="arn:aws:iam::123456789012:role/read-role",
            outputs=[{"sink_type": "s3", "url": "s3://bucket/output/"}],
            model_name="test-model",
            model_invoke_mode=ModelInvokeMode.SM_ENDPOINT,
            model_invocation_role="arn:aws:iam::123456789012:role/invoke-role",
        )
        self.scheduler.job_id_to_message_handle["job-123"] = test_receipt_handle

        # Execute
        self.scheduler.finish_request(test_image_request)

        # Assert
        self.assertIn(test_receipt_handle, self.mock_queue.finished_receipts)
        self.assertNotIn("job-123", self.scheduler.job_id_to_message_handle)

    def test_finish_request_with_retry(self):
        """Test finishing request with retry flag"""
        # Setup
        test_receipt_handle = "receipt-123"
        test_image_request = ImageRequest(
            job_id="job-123",
            image_id="image-123",
            image_url="s3://bucket/image.tif",
            image_read_role="arn:aws:iam::123456789012:role/read-role",
            outputs=[{"sink_type": "s3", "url": "s3://bucket/output/"}],
            model_name="test-model",
            model_invoke_mode=ModelInvokeMode.SM_ENDPOINT,
            model_invocation_role="arn:aws:iam::123456789012:role/invoke-role",
        )
        self.scheduler.job_id_to_message_handle["job-123"] = test_receipt_handle

        # Execute
        self.scheduler.finish_request(test_image_request, should_retry=True)

        # Assert
        self.assertIn(test_receipt_handle, self.mock_queue.reset_receipts)
        self.assertNotIn("job-123", self.scheduler.job_id_to_message_handle)

    def test_multiple_requests_in_queue(self):
        """Test handling multiple requests in the queue"""
        # Setup
        test_messages = [
            (
                "receipt-1",
                {
                    "jobName": "test-job-name",
                    "jobId": "job-1",
                    "imageUrls": ["test-image-url"],
                    "outputs": [
                        {"type": "S3", "bucket": "test-bucket", "prefix": "test-bucket-prefix"},
                        {"type": "Kinesis", "stream": "test-stream", "batchSize": 1000},
                    ],
                    "imageProcessor": {"name": "test-model", "type": "SM_ENDPOINT"},
                    "imageProcessorTileSize": 1024,
                    "imageProcessorTileOverlap": 50,
                },
            ),
            (
                "receipt-2",
                {
                    "jobName": "test-job-name",
                    "jobId": "job-2",
                    "imageUrls": ["test-image-url"],
                    "outputs": [
                        {"type": "S3", "bucket": "test-bucket", "prefix": "test-bucket-prefix"},
                        {"type": "Kinesis", "stream": "test-stream", "batchSize": 1000},
                    ],
                    "imageProcessor": {"name": "test-model", "type": "SM_ENDPOINT"},
                    "imageProcessorTileSize": 1024,
                    "imageProcessorTileOverlap": 50,
                },
            ),
        ]

        for receipt, message in test_messages:
            self.mock_queue.add_message(receipt, message)

        # Execute and Assert first request
        first_request = self.scheduler.get_next_scheduled_request()
        self.assertIsInstance(first_request, ImageRequest)
        self.assertEqual(first_request.job_id, "job-1")
        self.assertEqual(self.scheduler.job_id_to_message_handle["job-1"], "receipt-1")

        # Execute and Assert second request
        second_request = self.scheduler.get_next_scheduled_request()
        self.assertIsInstance(second_request, ImageRequest)
        self.assertEqual(second_request.job_id, "job-2")
        self.assertEqual(self.scheduler.job_id_to_message_handle["job-2"], "receipt-2")


if __name__ == "__main__":
    unittest.main()
