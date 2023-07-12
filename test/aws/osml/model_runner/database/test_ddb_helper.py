#  Copyright 2023 Amazon.com, Inc. or its affiliates.
import os
from unittest import TestCase

import boto3
from botocore.exceptions import ClientError
from moto import mock_dynamodb

TEST_JOB_TABLE_KEY_SCHEMA = [{"AttributeName": "image_id", "KeyType": "HASH"}]
TEST_JOB_TABLE_ATTRIBUTE_DEFINITIONS = [{"AttributeName": "image_id", "AttributeType": "S"}]
TEST_IMAGE_ID = "test-image-id"


@mock_dynamodb
class TestDDBHelper(TestCase):
    def setUp(self):
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.ddb_helper import DDBItem, DDBKey
        from aws.osml.model_runner.database.job_table import JobItem

        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table_name = os.environ["JOB_TABLE"]
        self.table = self.ddb.create_table(
            TableName=self.table_name,
            KeySchema=TEST_JOB_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_JOB_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )

        self.job_item = JobItem(image_id=TEST_IMAGE_ID)
        ddb_key = DDBKey(hash_key="image_id", hash_value="12345")
        self.ddb_item = DDBItem()
        self.ddb_item.ddb_key = ddb_key
        ddb_range_key = DDBKey(hash_key="image_id", hash_value="12345", range_key="other_field", range_value="range")
        self.range_job_item = JobItem(image_id=TEST_IMAGE_ID)
        self.range_job_item.ddb_key = ddb_range_key

    def tearDown(self):
        self.table.delete()
        self.table = None
        self.ddb = None
        self.table_name = None
        self.ddb_item = None
        self.job_item = None

    def test_ddb_item_to_put(self):
        data_to_put = self.ddb_item.to_put()

        assert data_to_put == {}

    def test_ddb_item_to_update(self):
        data_to_update = self.ddb_item.to_update()

        assert data_to_update == {}

    def test_ddb_helper_put_ddb_item(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        res = helper.put_ddb_item(self.job_item)
        assert res["ResponseMetadata"]["HTTPStatusCode"] == 200

        with self.assertRaises(ClientError):
            helper.put_ddb_item(self.job_item, condition_expression="attribute_not_exists(image_id)")

    def test_ddb_helper_get_ddb_item(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        returned_item_dict = helper.get_ddb_item(self.job_item)
        assert returned_item_dict == {"image_id": TEST_IMAGE_ID}

    def test_ddb_helper_delete_ddb_item(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        res = helper.delete_ddb_item(self.job_item)
        assert res["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_ddb_helper_update_ddb_item(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        self.job_item.model_name = "noop"
        results = helper.update_ddb_item(self.job_item)
        assert results == {"image_id": "test-image-id", "model_name": "noop"}

    def test_ddb_helper_update_ddb_item_invalid_params(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper
        from aws.osml.model_runner.database.exceptions import DDBUpdateException

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)
        self.job_item.model_name = "noop"
        with self.assertRaises(DDBUpdateException):
            helper.update_ddb_item(self.job_item, update_attr={":model_name": "noop"})

    def test_ddb_helper_query_items(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        helper.put_ddb_item(self.job_item)

        retrieved_items = helper.query_items(self.job_item)
        assert len(retrieved_items) == 1

    def test_ddb_helper_get_update_params(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        self.job_item.model_name = "noop"
        update_expression, update_attributes = helper.get_update_params(self.job_item.to_update(), self.job_item)
        assert update_expression == "SET  model_name = :model_name"
        assert update_attributes == {":model_name": "noop"}

        self.range_job_item.model_name = "noop_2"
        update_dict = self.range_job_item.to_update()
        update_dict["other_field"] = "value"
        range_update_expression, range_update_attributes = helper.get_update_params(update_dict, self.range_job_item)
        assert range_update_expression == "SET  model_name = :model_name"
        assert range_update_attributes == {":model_name": "noop_2"}

    def test_ddb_helper_get_keys(self):
        from aws.osml.model_runner.database.ddb_helper import DDBHelper

        helper = DDBHelper(self.table_name)
        keys = helper.get_keys(self.range_job_item)
        assert keys == {"image_id": "12345", "other_field": "range"}
