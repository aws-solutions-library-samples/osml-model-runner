#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
import unittest
from unittest.mock import Mock

import boto3
from botocore.exceptions import ClientError
from moto import mock_aws

TEST_ENDPOINT_TABLE_KEY_SCHEMA = [{"AttributeName": "endpoint", "KeyType": "HASH"}]
TEST_ENDPOINT_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "endpoint", "AttributeType": "S"},
]
TEST_MOCK_PUT_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "put_item"))
TEST_MOCK_PUT_CONDITIONAL_EXCEPTION = Mock(
    side_effect=ClientError({"Error": {"Code": "ConditionalCheckFailedException", "Message": "ClientError"}}, "put_item")
)
TEST_MOCK_UPDATE_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item"))


@mock_aws
class TestEndpointStatisticsTable(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.endpoint_statistics_table import EndpointStatisticsItem, EndpointStatisticsTable

        # Prepare something ahead of all tests
        # Create virtual DDB table to write test data into
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table = self.ddb.create_table(
            TableName=os.environ["ENDPOINT_TABLE"],
            KeySchema=TEST_ENDPOINT_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_ENDPOINT_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )
        self.endpoint_statistics_table = EndpointStatisticsTable(os.environ["ENDPOINT_TABLE"])
        self.endpoint_statistics_item = EndpointStatisticsItem(endpoint="test-endpoint")

    def tearDown(self):
        """
        Delete virtual DDB resources/tables after each test
        """

        self.table.delete()
        self.ddb = None
        self.endpoint_statistics_table = None
        self.endpoint_statistics_item = None

    # def test_upsert_endpoint_failure(self):
    #     self.endpoint_statistics_table.table.put_item = TEST_MOCK_PUT_CONDITIONAL_EXCEPTION
    #     self.endpoint_statistics_table.upsert_endpoint(self.endpoint_statistics_item, 0)
    #     # resulting_job_item = self.job_table.get_image_request(TEST_IMAGE_ID)
    #     # assert resulting_job_item.image_id == TEST_IMAGE_ID

    #     pass


if __name__ == "__main__":
    unittest.main()
