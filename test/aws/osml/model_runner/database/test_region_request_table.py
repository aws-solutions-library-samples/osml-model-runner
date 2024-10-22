#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

TEST_TABLE_NAME = "test-region-request-table"
TEST_IMAGE_ID = "test-image-id"
TEST_REGION_ID = "test-region-id"
TEST_JOB_ID = "test-job-id"
MOCK_PUT_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "put_item"))
MOCK_UPDATE_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item"))


@mock_aws
class TestRegionRequestTable(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use.
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.region_request_table import RegionRequestItem, RegionRequestTable

        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table = self.ddb.create_table(
            TableName=TEST_TABLE_NAME,
            KeySchema=[
                {"AttributeName": "region_id", "KeyType": "HASH"},
                {"AttributeName": "image_id", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "region_id", "AttributeType": "S"},
                {"AttributeName": "image_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        self.region_request_table = RegionRequestTable(TEST_TABLE_NAME)
        self.region_request_item = RegionRequestItem(TEST_REGION_ID, TEST_IMAGE_ID, TEST_JOB_ID)

    def tearDown(self):
        """
        Delete virtual DDB resources/tables after each test.
        """
        self.table.delete()
        self.ddb = None
        self.region_request_table = None
        self.region_request_item = None

    def test_region_started_success(self):
        """
        Validate that starting a region request successfully stores it in the table.
        """
        from aws.osml.model_runner.common import RequestStatus

        self.region_request_table.start_region_request(self.region_request_item)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)
        assert resulting_region_request_item.image_id == TEST_IMAGE_ID
        assert resulting_region_request_item.region_id == TEST_REGION_ID
        assert resulting_region_request_item.job_id == TEST_JOB_ID
        assert resulting_region_request_item.region_status == RequestStatus.STARTED

    def test_region_complete_success(self):
        """
        Validate that completing a region request updates the DDB item successfully.
        """
        from aws.osml.model_runner.common import RequestStatus

        self.region_request_table.start_region_request(self.region_request_item)
        self.region_request_table.complete_region_request(self.region_request_item, RequestStatus.SUCCESS)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)

        assert resulting_region_request_item.region_status == RequestStatus.SUCCESS
        assert resulting_region_request_item.last_updated_time is not None
        assert resulting_region_request_item.end_time is not None

    def test_region_updated_success(self):
        """
        Validate that updating an item in the region request table works as expected.
        """
        self.region_request_table.start_region_request(self.region_request_item)
        self.region_request_item.total_tiles = 1
        self.region_request_table.update_region_request(self.region_request_item)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)

        assert resulting_region_request_item.total_tiles == 1
        assert resulting_region_request_item.last_updated_time is not None

    def test_region_complete_failed(self):
        """
        Validate that marking a region request as failed updates the DDB item.
        """
        from aws.osml.model_runner.common import RequestStatus

        self.region_request_table.start_region_request(self.region_request_item)
        self.region_request_table.complete_region_request(self.region_request_item, RequestStatus.FAILED)
        resulting_region_request_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)

        assert resulting_region_request_item.region_status == RequestStatus.FAILED
        assert resulting_region_request_item.last_updated_time is not None
        assert resulting_region_request_item.end_time is not None

    def test_start_region_failure_exception(self):
        """
        Validate that a StartRegionException is raised when starting a region fails.
        """
        from aws.osml.model_runner.database.exceptions import StartRegionException

        self.region_request_table.table.put_item = MOCK_PUT_EXCEPTION
        with self.assertRaises(StartRegionException):
            self.region_request_table.start_region_request(self.region_request_item)

    def test_complete_region_failure_exception(self):
        """
        Validate that a CompleteRegionException is raised when completing a region fails.
        """
        from aws.osml.model_runner.database.exceptions import CompleteRegionException

        self.region_request_table.table.update_item = MOCK_UPDATE_EXCEPTION
        self.region_request_table.start_region_request(self.region_request_item)
        with self.assertRaises(CompleteRegionException):
            self.region_request_table.complete_region_request(self.region_request_item, "FAILED")

    def test_region_updated_failure_exception(self):
        """
        Validate that an UpdateRegionException is raised when updating a region request fails.
        """
        from aws.osml.model_runner.database.exceptions import UpdateRegionException

        self.region_request_table.table.update_item = MOCK_UPDATE_EXCEPTION
        self.region_request_table.start_region_request(self.region_request_item)
        with self.assertRaises(UpdateRegionException):
            self.region_request_table.update_region_request(self.region_request_item)

    def test_get_region_request_none(self):
        """
        Validate that getting a non-existent region request returns None.
        """
        self.region_request_table.table.update_item = MOCK_UPDATE_EXCEPTION
        self.region_request_table.start_region_request(self.region_request_item)

        resulting_region_request = self.region_request_table.get_region_request(
            "DOES-NOT-EXIST-REGION-ID", "DOES-NOT-EXIST-IMAGE-ID"
        )
        assert resulting_region_request is None

    def test_add_tile_success(self):
        """
        Validate that tiles can be added as succeeded to the region request item.
        """
        from aws.osml.model_runner.common import TileState

        self.region_request_table.start_region_request(self.region_request_item)
        tile = ((0, 0), (256, 256))
        self.region_request_table.add_tile(TEST_IMAGE_ID, TEST_REGION_ID, tile, TileState.SUCCEEDED)

        success_tile_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)
        assert len(success_tile_item.succeeded_tiles) == 1

    def test_add_tile_failed(self):
        """
        Validate that tiles can be added as failed to the region request item.
        """
        from aws.osml.model_runner.common import TileState

        self.region_request_table.start_region_request(self.region_request_item)
        tile = ((0, 0), (256, 256))
        self.region_request_table.add_tile(TEST_IMAGE_ID, TEST_REGION_ID, tile, TileState.FAILED)
        failed_tile_item = self.region_request_table.get_region_request(TEST_REGION_ID, TEST_IMAGE_ID)

        assert len(failed_tile_item.failed_tiles) == 1

    def test_add_tile_invalid_format(self):
        """
        Validate that adding a tile with an invalid format raises UpdateRegionException.
        """
        from aws.osml.model_runner.common import TileState
        from aws.osml.model_runner.database.exceptions import UpdateRegionException

        self.region_request_table.start_region_request(self.region_request_item)

        with self.assertRaises(UpdateRegionException):
            self.region_request_table.add_tile(TEST_IMAGE_ID, TEST_REGION_ID, "bad_format", TileState.SUCCEEDED)

    def test_from_region_request_with_partial_data(self):
        """
        Validate that from_region_request handles partial data in the RegionRequest.
        """
        from aws.osml.model_runner.api import RegionRequest
        from aws.osml.model_runner.database import RegionRequestItem

        region_request = RegionRequest(
            region_id=TEST_REGION_ID,
            image_id=TEST_IMAGE_ID,
            job_id=None,
            region_bounds=[[0, 0], [256, 256]],
            tile_size=[256, 256],
            tile_overlap=[0, 0],
            tile_format="tif",
            tile_compression="LZW",
        )

        region_request_item = RegionRequestItem.from_region_request(region_request)
        assert region_request_item.region_id == TEST_REGION_ID
        assert region_request_item.image_id == TEST_IMAGE_ID
        assert region_request_item.job_id is None
        assert region_request_item.tile_format == "tif"


if __name__ == "__main__":
    unittest.main()
