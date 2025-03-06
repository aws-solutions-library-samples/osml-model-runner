#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import time
import unittest
from typing import List, Optional
from unittest.mock import Mock

import boto3
from moto import mock_aws

from aws.osml.model_runner.api import ImageRequest
from aws.osml.model_runner.database import ImageRequestStatusRecord
from aws.osml.model_runner.scheduler.endpoint_load_image_scheduler import (
    EndpointLoadImageScheduler,
    EndpointUtilizationSummary,
)


@mock_aws
class TestEndpointLoadImageScheduler(unittest.TestCase):
    """Test cases for EndpointLoadImageScheduler"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set up mock AWS resources
        self.sagemaker = boto3.client("sagemaker")

        # Create mock endpoints in SageMaker
        self.endpoints = {
            "endpoint1": {"InstanceCount": 2},
            "endpoint2": {"InstanceCount": 1},
            "endpoint3": {"InstanceCount": 3},
        }

        for endpoint_id, config in self.endpoints.items():
            self.sagemaker.create_model(
                ModelName=f"{endpoint_id}-model", PrimaryContainer={"Image": "test-model-container-image"}
            )
            self.sagemaker.create_endpoint_config(
                EndpointConfigName=f"{endpoint_id}-config",
                ProductionVariants=[
                    {
                        "InstanceType": "ml.m5.xlarge",
                        "InitialInstanceCount": config["InstanceCount"],
                        "VariantName": "AllTraffic",
                        "ModelName": f"{endpoint_id}-model",
                    }
                ],
            )

            self.sagemaker.create_endpoint(EndpointName=f"{endpoint_id}-model", EndpointConfigName=f"{endpoint_id}-config")

        # Create mock BufferedImageRequestQueue
        self.mock_queue = Mock()
        self.mock_queue.retry_time = 600

        # Create scheduler
        self.scheduler = EndpointLoadImageScheduler(image_request_queue=self.mock_queue)

    def create_sample_image_request(self, job_name: str = "test-job", model_name: str = "endpoint1-model") -> ImageRequest:
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
                "imageProcessor": {"name": model_name, "type": "SM_ENDPOINT"},
                "imageProcessorTileSize": 2048,
                "imageProcessorTileOverlap": 50,
            }
        )

    def create_status_record(
        self,
        job_name: str,
        model_name: str,
        request_time: Optional[int] = None,
        last_attempt: Optional[int] = None,
        num_attempts: Optional[int] = None,
        regions_complete: Optional[List[str]] = None,
        region_count: Optional[int] = None,
    ) -> ImageRequestStatusRecord:
        """Helper method to create a status record"""
        image_request = self.create_sample_image_request(job_name, model_name)
        image_status_record = ImageRequestStatusRecord.new_from_request(image_request)
        if request_time is not None:
            image_status_record.request_time = request_time
        if last_attempt is not None:
            image_status_record.last_attempt = last_attempt
        if num_attempts is not None:
            image_status_record.num_attempts = num_attempts
        if regions_complete is not None:
            image_status_record.regions_complete = regions_complete
        if region_count is not None:
            image_status_record.region_count = region_count
        return image_status_record

    def test_get_next_scheduled_request_no_requests(self):
        """Test scheduling when there are no requests"""
        self.mock_queue.get_outstanding_requests.return_value = []
        result = self.scheduler.get_next_scheduled_request()
        self.assertIsNone(result)

    def test_get_next_scheduled_request_single_endpoint(self):
        """Test scheduling with requests for a single endpoint"""
        time_in_past = int(time.time() - 5)
        status_records = [
            self.create_status_record("job1", "endpoint1-model", request_time=time_in_past),
            self.create_status_record("job2", "endpoint1-model", request_time=time_in_past + 1),
        ]

        self.mock_queue.get_outstanding_requests.return_value = status_records
        self.mock_queue.requested_jobs_table.start_next_attempt.return_value = True

        result = self.scheduler.get_next_scheduled_request()
        self.assertIsNotNone(result)
        self.assertEqual(result.job_id, "job1-id")

    def test_get_next_scheduled_request_multiple_endpoints(self):
        """Test scheduling with requests across multiple endpoints"""
        time_in_past = int(time.time() - 10)
        status_records = [
            self.create_status_record("job1", "endpoint1-model", request_time=time_in_past + 1),
            self.create_status_record("job2", "endpoint2-model", request_time=time_in_past),
            self.create_status_record("job3", "endpoint3-model", request_time=time_in_past + 2),
        ]

        self.mock_queue.get_outstanding_requests.return_value = status_records
        self.mock_queue.requested_jobs_table.start_next_attempt.return_value = True

        result = self.scheduler.get_next_scheduled_request()
        self.assertIsNotNone(result)
        # Should choose job2 because all 3 endpoints have no load and it was submitted first
        self.assertEqual(result.job_id, "job2-id")

    def test_get_next_scheduled_request_with_existing_load(self):
        """Test scheduling considering existing endpoint load"""
        status_records = [
            # endpoint1 (2 instances) has 3 running jobs
            self.create_status_record("job1", "endpoint1-model", region_count=1),
            self.create_status_record("job2", "endpoint1-model", region_count=1),
            self.create_status_record("job3", "endpoint1-model", region_count=1),
            # endpoint2 (1 instance) has 1 running job
            self.create_status_record("job4", "endpoint2-model", region_count=1),
            # endpoint3 (3 instances) has no running jobs
            self.create_status_record("job5", "endpoint3-model"),
        ]

        self.mock_queue.get_outstanding_requests.return_value = status_records
        self.mock_queue.requested_jobs_table.start_next_attempt.return_value = True

        result = self.scheduler.get_next_scheduled_request()
        self.assertIsNotNone(result)
        # Should choose endpoint3 as it has lowest load factor (0/3)
        self.assertEqual(result.job_id, "job5-id")

    def test_get_next_scheduled_request_start_attempt_failure(self):
        """Test scheduling when start_next_attempt fails"""
        status_records = [self.create_status_record("job1", "endpoint1-model")]

        self.mock_queue.get_outstanding_requests.return_value = status_records
        self.mock_queue.requested_jobs_table.start_next_attempt.return_value = False

        result = self.scheduler.get_next_scheduled_request()
        self.assertIsNone(result)

    def test_get_next_scheduled_request_sagemaker_error(self):
        """Test handling of SageMaker API errors"""
        status_records = [
            self.create_status_record("job1", "nonexistent-endpoint", region_count=1),
            self.create_status_record("job2", "endpoint3-model", region_count=2),
        ]

        self.mock_queue.get_outstanding_requests.return_value = status_records
        self.mock_queue.requested_jobs_table.start_next_attempt.return_value = True

        result = self.scheduler.get_next_scheduled_request()
        self.assertIsNotNone(result)
        # Should choose endpoint2 as it has lowest load factor (2/3) assuming the unknown endpoint
        # defaulted to 1 instance
        self.assertEqual(result.job_id, "job2-id")

    def test_endpoint_utilization_summary(self):
        """Test EndpointUtilizationSummary calculations"""
        summary = EndpointUtilizationSummary(endpoint_id="test-endpoint", instance_count=2, current_load=4, requests=[])
        self.assertEqual(summary.load_factor, 2)
