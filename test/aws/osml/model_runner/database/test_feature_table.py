#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
import unittest
from typing import List
from unittest.mock import Mock

import boto3
import geojson
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber
from moto import mock_aws

image_id = (
    "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif"  # noqa
)
TEST_FEATURE_TABLE_KEY_SCHEMA = [
    {"AttributeName": "hash_key", "KeyType": "HASH"},
    {"AttributeName": "range_key", "KeyType": "RANGE"},
]
TEST_FEATURE_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "hash_key", "AttributeType": "S"},
    {"AttributeName": "range_key", "AttributeType": "S"},
]

TEST_FEATURE_1 = {
    "hash_key": {"S": image_id},
    "range_key": {"S": "0:0:0:0-1"},
    "tile_id": {"S": "0:0:0:0"},
    "features": {
        "L": [
            {
                "S": '{"type": "Feature", "id": "96128a11-2e46-47b8-a33b-55ce8150a455", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"bounds_imcoords": [429, 553, 440, 561], "feature_types": {"ground_motor_passenger_vehicle": 0.2961518466472626}, "detection_score": 0.2961518466472626, "image_id": "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif"}}'  # noqa: E501
            },
            {
                "S": '{"type": "Feature", "id": "fb033f8b-aefe-40a1-a56c-dc42e494477b", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"bounds_imcoords": [414, 505, 423, 515], "feature_types": {"ground_motor_passenger_vehicle": 0.2887503504753113}, "detection_score": 0.2887503504753113, "image_id": "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif"}}'  # noqa: E501
            },
            {
                "S": '{"type": "Feature", "id": "0c4970ba-228d-487b-a29a-71ed97adbd89", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"bounds_imcoords": [664, 597, 674, 607], "feature_types": {"ground_motor_passenger_vehicle": 0.27162906527519226}, "detection_score": 0.27162906527519226, "image_id": "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif"}}'  # noqa: E501
            },
        ],
    },
}

TEST_FEATURE_2 = {
    "hash_key": {
        "S": image_id
        # noqa: E501
    },
    "range_key": {"S": "0:0:1:1-1"},
    "tile_id": {"S": "0:0:1:1"},
    "features": {
        "L": [
            {
                "S": '{"type": "Feature", "id": "26c28104-4d3f-4595-b252-cb2af1dfff4b", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"bounds_imcoords": [646, 1649, 654, 1658], "feature_types": {"ground_motor_passenger_vehicle": 0.25180014967918396}, "detection_score": 0.25180014967918396, "image_id": "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif"}}'  # noqa: E501
            }
        ]
    },
}

TEST_MOCK_BATCH_WRITE_EXCEPTION = Mock(
    side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "batch_write_item")
)


@mock_aws
class TestFeatureTable(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.feature_table import FeatureTable

        # Prepare something ahead of all tests
        # Create virtual DDB table to write test data into
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table = self.ddb.create_table(
            TableName=os.environ["FEATURE_TABLE"],
            KeySchema=TEST_FEATURE_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_FEATURE_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )
        self.feature_table = FeatureTable(os.environ["FEATURE_TABLE"], (2048, 2048), (50, 50))
        self.feature_table.hash_salt = 1

    def tearDown(self):
        """
        Delete virtual DDB resources/tables after each test
        """

        self.table.delete()
        self.ddb = None
        self.feature_table = None

    def test_get_all_features_paginated(self):
        with Stubber(self.feature_table.table.meta.client) as ddb_stubber:
            page_1_params = {
                "ConsistentRead": True,
                "TableName": os.environ["FEATURE_TABLE"],
                "KeyConditionExpression": ANY,
            }
            page_1_response = {
                "Items": [TEST_FEATURE_1],
                "LastEvaluatedKey": {
                    "hash_key": TEST_FEATURE_1["hash_key"],
                    "range_key": TEST_FEATURE_1["range_key"],
                    "tile_id": TEST_FEATURE_1["tile_id"],
                },
            }

            page_2_params = {
                "ConsistentRead": True,
                "TableName": os.environ["FEATURE_TABLE"],
                "KeyConditionExpression": ANY,
                "ExclusiveStartKey": ANY,
            }
            page_2_response = {"Items": [TEST_FEATURE_2]}

            ddb_stubber.add_response("query", page_1_response, page_1_params)
            ddb_stubber.add_response("query", page_2_response, page_2_params)

            result = self.feature_table.get_features(image_id)
            ddb_stubber.assert_no_pending_responses()
            assert len(result) == 4

    def test_group_features_by_key(self):
        features_dict = self.feature_table.group_features_by_key(self.get_feature_list())
        assert len(features_dict) == 1

    def test_generate_tile_key(self):
        result = self.feature_table.generate_tile_key(self.get_feature_list()[0])
        tile_key = "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif-region-0:0:0:0"  # noqa
        assert result == tile_key

    def test_add_and_get_features(self):
        features = self.get_feature_list()
        self.feature_table.add_features(features)
        ddb_features = self.feature_table.get_features(image_id)

        assert len(ddb_features) == len(features)

    def test_add_features_throw_exceptions(self):
        from aws.osml.model_runner.database.exceptions import AddFeaturesException

        features = self.get_feature_list()
        self.feature_table.batch_write_items = TEST_MOCK_BATCH_WRITE_EXCEPTION
        with self.assertRaises(AddFeaturesException):
            self.feature_table.add_features(features)

    @staticmethod
    def get_feature_list() -> List[geojson.Feature]:
        with open("./test/data/detections.geojson", "r") as geojson_file:
            sample_features = geojson.load(geojson_file)["features"]
        return sample_features
