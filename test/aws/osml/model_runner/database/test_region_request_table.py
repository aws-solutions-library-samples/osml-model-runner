#  Copyright 2023 Amazon.com, Inc. or its affiliates.
import os
import unittest
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from moto import mock_dynamodb

TEST_IMAGE_ID = "test-image-id"
TEST_REGION_ID = "test-region-id"
TEST_JOB_ID = "test-job-id"
TEST_MOCK_PUT_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "put_item"))
TEST_MOCK_UPDATE_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item"))
TEST_REGION_REQUEST_TABLE_KEY_SCHEMA = [
    {"AttributeName": "region_id", "KeyType": "HASH"},
    {"AttributeName": "image_id", "KeyType": "RANGE"},
]
TEST_REGION_REQUEST_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "region_id", "AttributeType": "S"},
    {"AttributeName": "image_id", "AttributeType": "S"},
]


@mock_dynamodb
class TestRegionRequestTable(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.region_request_table import RegionRequestItem, RegionRequestTable

        # Prepare something ahead of all tests
        # Create virtual DDB table to write test data into
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table = self.ddb.create_table(
            TableName=os.environ["REGION_REQUEST_TABLE"],
            KeySchema=TEST_REGION_REQUEST_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_REGION_REQUEST_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )
        self.region_request_table = RegionRequestTable(os.environ["REGION_REQUEST_TABLE"])
        self.region_request_item = RegionRequestItem(region_id=TEST_REGION_ID, image_id=TEST_IMAGE_ID, job_id=TEST_JOB_ID)

    def tearDown(self):
        """
        Delete virtual DDB resources/tables after each test
        """

        self.table.delete()
        self.ddb = None
        self.region_request_table = None
        self.region_request_item = None

    def test_region_started_success(self):
        """
        Validate we can start a region, and it gets created in the table
        """
        from aws.osml.model_runner.database.region_request_table import RegionRequestStatus

        self.region_request_table.start_region_request(self.region_request_item)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)
        assert resulting_region_request_item.image_id == TEST_IMAGE_ID
        assert resulting_region_request_item.region_id == TEST_REGION_ID
        assert resulting_region_request_item.job_id == TEST_JOB_ID
        assert resulting_region_request_item.region_status == RegionRequestStatus.STARTING

    def test_region_complete_success(self):
        """
        Validate that when we complete a region successfully it updates the ddb item
        """
        from aws.osml.model_runner.database.region_request_table import RegionRequestStatus

        self.region_request_table.start_region_request(self.region_request_item)
        self.region_request_table.complete_region_request(self.region_request_item)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)

        assert resulting_region_request_item.region_status == RegionRequestStatus.SUCCESS
        assert resulting_region_request_item.last_updated_time is not None
        assert resulting_region_request_item.end_time is not None

    def test_region_updated_success(self):
        """
        Validate that when we can update the item in region request ddb
        """
        self.region_request_table.start_region_request(self.region_request_item)
        self.region_request_item.total_tiles = 1
        self.region_request_table.update_region_request(self.region_request_item)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)

        assert resulting_region_request_item.total_tiles == 1
        assert resulting_region_request_item.last_updated_time is not None

    def test_region_complete_failed(self):
        """
        Validate that when we complete a region successfully it updates the ddb item
        """
        from aws.osml.model_runner.database.region_request_table import RegionRequestStatus

        self.region_request_table.start_region_request(self.region_request_item)
        self.region_request_table.complete_region_request(self.region_request_item, True)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)

        assert resulting_region_request_item.region_status == RegionRequestStatus.FAILED
        assert resulting_region_request_item.last_updated_time is not None
        assert resulting_region_request_item.end_time is not None

    def test_start_region_failure_exception(self):
        """
        Validate that throw the correct StartRegionFailed exception
        """
        from aws.osml.model_runner.database.exceptions import StartRegionException

        self.region_request_table.table.put_item = TEST_MOCK_PUT_EXCEPTION
        with self.assertRaises(StartRegionException):
            self.region_request_table.start_region_request(self.region_request_item)

    def test_complete_region_failure_exception(self):
        """
        Validate that throw the correct CompleteRegionFailed exception
        """
        from aws.osml.model_runner.database.exceptions import CompleteRegionException

        self.region_request_table.table.update_item = TEST_MOCK_UPDATE_EXCEPTION
        self.region_request_table.start_region_request(self.region_request_item)
        with self.assertRaises(CompleteRegionException):
            self.region_request_table.complete_region_request(TEST_IMAGE_ID, TEST_REGION_ID)

    def test_region_updated_failure_exception(self):
        """
        Validate that throw the correct UpdateRegionFailed exception
        """
        from aws.osml.model_runner.database.exceptions import UpdateRegionException

        self.region_request_table.table.update_item = TEST_MOCK_UPDATE_EXCEPTION
        self.region_request_table.start_region_request(self.region_request_item)
        with self.assertRaises(UpdateRegionException):
            self.region_request_table.update_region_request(self.region_request_item)

    def test_get_region_request_none(self):
        """
        Validate that throw the correct GetRegionFailed exception
        """
        self.region_request_table.table.update_item = TEST_MOCK_UPDATE_EXCEPTION
        self.region_request_table.start_region_request(self.region_request_item)

        resulting_region_request = self.region_request_table.get_region_request(
            "DOES-NOT-EXIST-REGION-ID", "DOES-NOT-EXIST-IMAGE-ID"
        )
        assert resulting_region_request is None


if __name__ == "__main__":
    unittest.main()
