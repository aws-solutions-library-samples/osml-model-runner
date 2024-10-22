#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
import unittest
from unittest.mock import Mock, patch

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

MOCK_PUT_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "put_item"))
MOCK_PUT_CONDITIONAL_EXCEPTION = Mock(
    side_effect=ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": "ClientError"}}, "put_item")
)
MOCK_UPDATE_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item"))


@mock_aws
class TestEndpointStatisticsTable(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DynamoDB resources/tables for each test to use.
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.endpoint_statistics_table import EndpointStatisticsItem, EndpointStatisticsTable

        # Create virtual DynamoDB table for testing
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table = self.ddb.create_table(
            TableName=os.environ["ENDPOINT_TABLE"],
            KeySchema=[{"AttributeName": "endpoint", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "endpoint", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        self.endpoint_statistics_table = EndpointStatisticsTable(os.environ["ENDPOINT_TABLE"])
        self.endpoint_statistics_item = EndpointStatisticsItem(endpoint="test-endpoint")

    def tearDown(self):
        """
        Delete virtual DynamoDB resources/tables after each test.
        """
        self.table.delete()
        self.ddb = None
        self.endpoint_statistics_table = None
        self.endpoint_statistics_item = None

    def test_upsert_endpoint_creates_new_entry(self):
        """
        Test that the `upsert_endpoint` method correctly creates a new entry if the endpoint does not exist.
        """
        self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=5)
        result = self.endpoint_statistics_table.get_ddb_item(self.endpoint_statistics_item)
        assert result == {"endpoint": "test-endpoint", "max_regions": 5, "regions_in_progress": 0}

    @patch(
        "aws.osml.model_runner.database.endpoint_statistics_table.EndpointStatisticsTable.put_ddb_item",
        MOCK_PUT_CONDITIONAL_EXCEPTION,
    )
    def test_upsert_endpoint_updates_existing_entry(self):
        """
        Test that the `upsert_endpoint` method updates an existing entry's `max_regions` if it already exists.
        """
        # First upsert to create the item
        self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=5)
        # Mock the conditional exception to force the update path
        self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=10)
        result = self.endpoint_statistics_table.get_ddb_item(self.endpoint_statistics_item)
        assert result["max_regions"] == 10

    def test_increment_region_count(self):
        """
        Test that the `increment_region_count` method correctly increases the in-progress region count by 1.
        """
        self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=5)
        self.endpoint_statistics_table.increment_region_count("test-endpoint")
        result = self.endpoint_statistics_table.get_ddb_item(self.endpoint_statistics_item)
        assert result["regions_in_progress"] == 1

    def test_decrement_region_count(self):
        """
        Test that the `decrement_region_count` method correctly decreases the in-progress region count by 1.
        """
        self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=5)
        self.endpoint_statistics_table.increment_region_count("test-endpoint")
        self.endpoint_statistics_table.increment_region_count("test-endpoint")
        self.endpoint_statistics_table.decrement_region_count("test-endpoint")
        result = self.endpoint_statistics_table.get_ddb_item(self.endpoint_statistics_item)
        assert result["regions_in_progress"] == 1

    @patch(
        "aws.osml.model_runner.database.endpoint_statistics_table.EndpointStatisticsTable.update_ddb_item",
        MOCK_UPDATE_EXCEPTION,
    )
    def test_increment_region_count_failure(self):
        """
        Test that `increment_region_count` handles exceptions properly when the update fails.
        """
        self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=5)
        with self.assertRaises(ClientError):
            self.endpoint_statistics_table.increment_region_count("test-endpoint")

    def test_current_in_progress_regions(self):
        """
        Test that the `current_in_progress_regions` method correctly retrieves the in-progress region count.
        """
        self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=5)
        self.endpoint_statistics_table.increment_region_count("test-endpoint")
        in_progress = self.endpoint_statistics_table.current_in_progress_regions("test-endpoint")
        assert in_progress == 1

    @patch(
        "aws.osml.model_runner.database.endpoint_statistics_table.EndpointStatisticsTable.put_ddb_item", MOCK_PUT_EXCEPTION
    )
    def test_upsert_endpoint_failure(self):
        """
        Test that `upsert_endpoint` handles a non-conditional client error and does not create or update the endpoint.
        """
        with self.assertRaises(ClientError):
            self.endpoint_statistics_table.upsert_endpoint("test-endpoint", max_regions=5)
        result = self.endpoint_statistics_table.get_ddb_item(self.endpoint_statistics_item)
        assert result == {}, "Expected no entry in table due to failure"


if __name__ == "__main__":
    unittest.main()
