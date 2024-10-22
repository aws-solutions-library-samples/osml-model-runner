#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
import time
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

TEST_IMAGE_ID = "test-image-id"
MOCK_PUT_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "put_item"))
MOCK_UPDATE_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item"))


@mock_aws
class TestJobTable(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.job_table import JobItem, JobTable

        # Create virtual DDB table for testing
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table = self.ddb.create_table(
            TableName=os.environ["JOB_TABLE"],
            KeySchema=[{"AttributeName": "image_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "image_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        self.job_table = JobTable(os.environ["JOB_TABLE"])
        self.job_item = JobItem(image_id=TEST_IMAGE_ID)

    def tearDown(self):
        """
        Delete virtual DDB resources/tables after each test
        """
        self.table.delete()
        self.ddb = None
        self.job_table = None
        self.job_item = None

    def test_image_started_success(self):
        """
        Validate we can start an image, and it gets created in the table.
        """
        self.job_table.start_image_request(self.job_item)
        resulting_job_item = self.job_table.get_image_request(TEST_IMAGE_ID)
        assert resulting_job_item.image_id == TEST_IMAGE_ID

    def test_region_complete_success_count(self):
        """
        Validate that when we complete a region successfully, it updates the DDB item.
        """
        self.job_table.start_image_request(self.job_item)
        self.job_table.complete_region_request(TEST_IMAGE_ID, False)
        resulting_job_item = self.job_table.get_image_request(TEST_IMAGE_ID)
        assert resulting_job_item.region_success == Decimal(1)
        assert resulting_job_item.region_error == Decimal(0)

    def test_region_complete_error_count(self):
        """
        Validate that when we fail to complete a region, it updates the DDB item.
        """
        self.job_table.start_image_request(self.job_item)
        self.job_item.region_count = Decimal(0)
        self.job_item.region_success = Decimal(0)
        self.job_item.region_error = Decimal(0)
        self.job_table.update_ddb_item(self.job_item)
        self.job_table.complete_region_request(TEST_IMAGE_ID, True)
        resulting_job_item = self.job_table.get_image_request(TEST_IMAGE_ID)
        assert resulting_job_item.region_error == Decimal(1)
        assert resulting_job_item.region_success == Decimal(0)

    def test_is_image_complete_success(self):
        """
        Validate that we can successfully determine when an image has been completed.
        """
        self.job_table.start_image_request(self.job_item)
        self.job_item.region_count = Decimal(1)
        self.job_item.region_success = Decimal(1)
        self.job_item.region_error = Decimal(0)
        self.job_table.update_ddb_item(self.job_item)
        assert self.job_table.is_image_request_complete(self.job_item)

    def test_region_ended_success(self):
        """
        Validate that we can successfully end an image's processing by setting its end time.
        """
        self.job_table.start_image_request(self.job_item)
        self.job_table.end_image_request(TEST_IMAGE_ID)
        resulting_job_item = self.job_table.get_image_request(TEST_IMAGE_ID)
        assert resulting_job_item.end_time is not None

    @patch("aws.osml.model_runner.database.job_table.JobTable.put_ddb_item", MOCK_PUT_EXCEPTION)
    def test_start_image_failure(self):
        """
        Validate that we throw the correct StartImageFailed exception.
        """
        from aws.osml.model_runner.database.exceptions import StartImageException

        with self.assertRaises(StartImageException):
            self.job_table.start_image_request(self.job_item)

    @patch("aws.osml.model_runner.database.job_table.JobTable.update_ddb_item", MOCK_PUT_EXCEPTION)
    def test_complete_region_failure(self):
        """
        Validate that we throw the correct CompleteRegionException when region completion fails.
        """
        from aws.osml.model_runner.database.exceptions import CompleteRegionException

        self.job_table.start_image_request(self.job_item)
        with self.assertRaises(CompleteRegionException):
            self.job_table.complete_region_request(TEST_IMAGE_ID, False)

    @patch("aws.osml.model_runner.database.job_table.JobTable.update_ddb_item", MOCK_UPDATE_EXCEPTION)
    def test_end_image_failure(self):
        """
        Validate that we throw the correct EndImageException when ending image processing fails.
        """
        from aws.osml.model_runner.database.exceptions import EndImageException

        self.job_table.start_image_request(self.job_item)
        with self.assertRaises(EndImageException):
            self.job_table.end_image_request(TEST_IMAGE_ID)

    @patch(
        "aws.osml.model_runner.database.job_table.JobTable.get_ddb_item",
        Mock(side_effect=ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "get_item")),
    )
    def test_get_image_request_failure(self):
        """
        Validate that we throw the correct GetImageRequestItemException when an image request can't be found.
        """
        from aws.osml.model_runner.database.exceptions import GetImageRequestItemException

        with self.assertRaises(GetImageRequestItemException):
            self.job_table.get_image_request("DOES-NOT-EXIST-IMAGE-ID")

    def test_is_image_request_complete_failure(self):
        """
        Validate that we throw the correct IsImageCompleteException when checking if an image is complete fails.
        """
        from aws.osml.model_runner.database.exceptions import IsImageCompleteException

        self.job_table.start_image_request(self.job_item)
        self.job_item.region_count = None
        self.job_item.region_success = None
        self.job_item.region_error = None

        with self.assertRaises(IsImageCompleteException):
            self.job_table.is_image_request_complete(self.job_item)

    def test_get_processing_duration(self):
        """
        Validate that `get_processing_duration` correctly calculates the processing time in seconds.
        """
        from aws.osml.model_runner.database.job_table import JobTable

        start_time = int(time.time() * 1000) - 5000
        duration = JobTable.get_processing_duration(start_time)
        # Processing duration should be at least 5 seconds
        assert duration >= 5


if __name__ == "__main__":
    unittest.main()
