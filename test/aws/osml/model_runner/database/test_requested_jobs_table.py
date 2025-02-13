#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import unittest

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

from aws.osml.model_runner.api import ImageRequest
from aws.osml.model_runner.database.requested_jobs_table import ImageRequestStatusRecord, RequestedJobsTable


class TestRequestedJobsTable(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.table_name = "test-requested-jobs"
        self.mock_aws = mock_aws()
        self.mock_aws.start()

        # Create the mock DynamoDB table
        self.ddb = boto3.resource("dynamodb")
        self.ddb.create_table(
            TableName=self.table_name,
            KeySchema=[{"AttributeName": "endpoint_id", "KeyType": "HASH"}, {"AttributeName": "job_id", "KeyType": "RANGE"}],
            AttributeDefinitions=[
                {"AttributeName": "endpoint_id", "AttributeType": "S"},
                {"AttributeName": "job_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        self.table = RequestedJobsTable(self.table_name)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.mock_aws.stop()

    def create_sample_image_request(self, job_name: str = "test-job") -> ImageRequest:
        """Helper method to create a sample ImageRequest"""
        return ImageRequest.from_external_message(
            {
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
        )

    def test_add_new_request(self):
        """Test adding a new request to the table"""
        image_request = self.create_sample_image_request()
        self.table.add_new_request(image_request)

        # Verify the item was added correctly
        response = self.ddb.Table(self.table_name).get_item(
            Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id}
        )
        item = response["Item"]

        self.assertEqual(item["endpoint_id"], image_request.model_name)
        self.assertEqual(item["job_id"], image_request.job_id)
        self.assertEqual(item["num_attempts"], 0)
        self.assertEqual(item["regions_complete"], [])

    def test_get_outstanding_requests(self):
        """Test retrieving outstanding requests from the table"""
        # Add multiple requests
        requests = [
            self.create_sample_image_request(job_name="test-job-1"),
            self.create_sample_image_request(job_name="test-job-2"),
        ]

        for request in requests:
            self.table.add_new_request(request)

        # Retrieve outstanding requests
        outstanding = self.table.get_outstanding_requests()

        self.assertEqual(len(outstanding), 2)
        self.assertIsInstance(outstanding[0], ImageRequestStatusRecord)

    def test_start_next_attempt(self):
        """Test starting the next attempt for a request"""
        image_request = self.create_sample_image_request()
        self.table.add_new_request(image_request)

        # Get the record and try to start next attempt
        records = self.table.get_outstanding_requests()
        self.assertEqual(len(records), 1)

        # First attempt should succeed
        success = self.table.start_next_attempt(records[0])
        self.assertTrue(success)

        # Verify the attempt was recorded
        response = self.ddb.Table(self.table_name).get_item(
            Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id}
        )
        item = response["Item"]
        self.assertEqual(item["num_attempts"], 1)

        # Trying to start next attempt with old record should fail
        success = self.table.start_next_attempt(records[0])
        self.assertFalse(success)

    def test_complete_request(self):
        """Test completing and removing a request"""
        image_request = self.create_sample_image_request()
        self.table.add_new_request(image_request)

        # Verify the item exists
        response = self.ddb.Table(self.table_name).get_item(
            Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id}
        )
        self.assertIn("Item", response)

        # Complete the request
        self.table.complete_request(image_request)

        # Verify the item was deleted
        response = self.ddb.Table(self.table_name).get_item(
            Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id}
        )
        self.assertNotIn("Item", response)

    def test_get_outstanding_requests_pagination(self):
        """Test that get_outstanding_requests handles pagination correctly"""
        # Add enough items to trigger pagination (DynamoDB default limit is 1MB)
        for i in range(10):
            request = self.create_sample_image_request(job_name=f"test-job-{i}")
            self.table.add_new_request(request)

        # Retrieve all items
        outstanding = self.table.get_outstanding_requests()

        self.assertEqual(len(outstanding), 10)
        self.assertIsInstance(outstanding[0], ImageRequestStatusRecord)
        self.assertEqual(len({r.job_id for r in outstanding}), 10)

    def test_complete_region_multiple(self):
        """Test completing multiple different regions."""
        # Arrange
        image_request = self.create_sample_image_request()
        self.table.add_new_request(image_request)

        # Act
        result1 = self.table.complete_region(image_request, "region1")
        result2 = self.table.complete_region(image_request, "region2")
        result3 = self.table.complete_region(image_request, "region1")  # duplicate

        # Assert
        self.assertTrue(result1)
        self.assertTrue(result2)
        self.assertFalse(result3)

        # Verify final state
        response = self.ddb.Table(self.table_name).get_item(
            Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id}
        )
        completed_regions = response["Item"]["regions_complete"]
        self.assertEqual(len(completed_regions), 2)
        self.assertIn("region1", completed_regions)
        self.assertIn("region2", completed_regions)

    def test_complete_region_nonexistent_record(self):
        """Test completing a region for a non-existent record."""
        # Arrange
        image_request = self.create_sample_image_request()

        # Act/Assert
        with self.assertRaises(ClientError):
            self.table.complete_region(image_request, "region1")

    def test_update_request_details_success(self):
        """Test successfully updating region count for a request."""
        # Arrange
        image_request = self.create_sample_image_request()
        self.table.add_new_request(image_request)
        region_count = 5

        # Act
        self.table.update_request_details(image_request, region_count)

        # Verify the region count was updated
        response = self.ddb.Table(self.table_name).get_item(
            Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id}
        )
        self.assertEqual(response["Item"]["region_count"], region_count)

    def test_update_request_details_nonexistent_record(self):
        """Test updating region count for a non-existent record."""
        # Arrange
        image_request = self.create_sample_image_request("missing-request-details-job")
        region_count = 5

        # Act/Assert
        with self.assertRaises(ClientError):
            self.table.update_request_details(image_request, region_count)


if __name__ == "__main__":
    unittest.main()
