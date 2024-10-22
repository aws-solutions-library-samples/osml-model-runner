#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
from unittest import TestCase

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

TEST_IMAGE_ID = "test-image-id"


@mock_aws
class TestDDBHelper(TestCase):
    def setUp(self):
        """
        Set up the DynamoDB mock environment, including creating a test table and initializing DDB items.
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.ddb_helper import DDBItem, DDBKey
        from aws.osml.model_runner.database.job_table import JobItem

        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table_name = os.environ["JOB_TABLE"]
        self.table = self.ddb.create_table(
            TableName=self.table_name,
            KeySchema=[{"AttributeName": "image_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "image_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        # Initialize DDB items for testing
        self.job_item = JobItem(image_id=TEST_IMAGE_ID)
        ddb_key = DDBKey(hash_key="image_id", hash_value="12345")
        self.ddb_item = DDBItem()
        self.ddb_item.ddb_key = ddb_key
        ddb_range_key = DDBKey(hash_key="image_id", hash_value="12345", range_key="other_field", range_value="range")
        self.range_job_item = JobItem(image_id="test-image-id")
        self.range_job_item.ddb_key = ddb_range_key

    def tearDown(self):
        """
        Clean up the mock environment by deleting the test table and resetting instance variables.
        """
        self.table.delete()
        self.table = None
        self.ddb = None
        self.table_name = None
        self.ddb_item = None
        self.job_item = None

    def test_ddb_item_to_put(self):
        """
        Test that the `to_put` method correctly prepares an item for a DynamoDB put operation.
        """
        data_to_put = self.ddb_item.to_put()
        assert data_to_put == {}, "Expected empty dictionary for default DDBItem"

    def test_ddb_item_to_update(self):
        """
        Test that the `to_update` method correctly prepares an item for a DynamoDB update operation.
        """
        data_to_update = self.ddb_item.to_update()
        assert data_to_update == {}, "Expected empty dictionary for default DDBItem"

    def test_ddb_helper_put_ddb_item(self):
        """
        Test that the `put_ddb_item` method correctly puts an item into DynamoDB and handles conditions.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        res = helper.put_ddb_item(self.job_item)
        assert res["ResponseMetadata"]["HTTPStatusCode"] == 200, "Expected successful HTTP status code"

        with self.assertRaises(ClientError):
            # Test conditional put that should fail due to existing item
            helper.put_ddb_item(self.job_item, condition_expression="attribute_not_exists(image_id)")

    def test_ddb_helper_get_ddb_item(self):
        """
        Test that the `get_ddb_item` method correctly retrieves an item from DynamoDB.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        returned_item_dict = helper.get_ddb_item(self.job_item)
        assert returned_item_dict == {"image_id": TEST_IMAGE_ID}, "Expected item to match inserted job item"

    def test_ddb_helper_delete_ddb_item(self):
        """
        Test that the `delete_ddb_item` method correctly deletes an item from DynamoDB.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        res = helper.delete_ddb_item(self.job_item)
        assert res["ResponseMetadata"]["HTTPStatusCode"] == 200, "Expected successful deletion response"

    def test_ddb_helper_update_ddb_item(self):
        """
        Test that the `update_ddb_item` method correctly updates an item in DynamoDB.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        self.job_item.model_name = "noop"
        results = helper.update_ddb_item(self.job_item)
        assert results == {"image_id": "test-image-id", "model_name": "noop"}, "Expected updated item attributes"

    def test_ddb_helper_update_ddb_item_invalid_params(self):
        """
        Test that the `update_ddb_item` method raises an exception when invalid parameters are provided.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper
        from aws.osml.model_runner.database.exceptions import DDBUpdateException

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        self.job_item.model_name = "noop"
        with self.assertRaises(DDBUpdateException):
            # Expect failure when only update attributes are provided without an expression
            helper.update_ddb_item(self.job_item, update_attr={":model_name": "noop"})

    def test_ddb_helper_query_items(self):
        """
        Test that the `query_items` method correctly queries and retrieves items based on a hash key.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)

        retrieved_items = helper.query_items(self.job_item)
        assert len(retrieved_items) == 1, "Expected one item to be retrieved from query"

    def test_ddb_helper_get_update_params(self):
        """
        Test the utility method `get_update_params` to ensure it generates correct update expressions and attributes.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        self.job_item.model_name = "noop"
        update_expression, update_attributes = helper.get_update_params(self.job_item.to_update(), self.job_item)
        assert update_expression == "SET  model_name = :model_name", "Expected correct update expression"
        assert update_attributes == {":model_name": "noop"}, "Expected correct update attributes"

        self.range_job_item.model_name = "noop_2"
        update_dict = self.range_job_item.to_update()
        update_dict["other_field"] = "value"
        range_update_expression, range_update_attributes = helper.get_update_params(update_dict, self.range_job_item)
        assert range_update_expression == "SET  model_name = :model_name", "Expected range-based update expression"
        assert range_update_attributes == {":model_name": "noop_2"}, "Expected range-based update attributes"

    def test_ddb_helper_get_keys(self):
        """
        Test the utility method `get_keys` to ensure it retrieves keys correctly from DDBItem instances.
        """
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        keys = helper.get_keys(self.range_job_item)
        assert keys == {"image_id": "12345", "other_field": "range"}, "Expected hash and range keys to be correct"
