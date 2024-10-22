#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
import unittest
from typing import List
from unittest.mock import Mock, patch

import boto3
import geojson
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber
from moto import mock_aws

TEST_IMAGE_ID = (
    "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif"
    # noqa
)

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

        # Create virtual DDB table for testing
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)
        self.table = self.ddb.create_table(
            TableName=os.environ["FEATURE_TABLE"],
            KeySchema=[
                {"AttributeName": "hash_key", "KeyType": "HASH"},
                {"AttributeName": "range_key", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "hash_key", "AttributeType": "S"},
                {"AttributeName": "range_key", "AttributeType": "S"},
            ],
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
        """
        Test that the `get_features` method correctly retrieves all features across paginated results.
        """
        # Grab a couple features to test with
        feature_1 = {
            "hash_key": {"S": TEST_IMAGE_ID},
            "range_key": {"S": "0:0:0:0-1"},
            "tile_id": {"S": "0:0:0:0"},
            "features": {
                "L": [
                    {
                        "S": '{"type": "Feature", "id": "96128a11-2e46-47b8-a33b-55ce8150a455", "geometry": {"type": "Point", "coordinates": [0.0, 0.0]}, "properties": {"bounds_imcoords": [429, 553, 440, 561], "feature_types": {"ground_motor_passenger_vehicle": 0.2961518466472626}, "detection_score": 0.2961518466472626, "image_id": "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif"}} '  # noqa: E501
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

        feature_2 = {
            "hash_key": {"S": TEST_IMAGE_ID},
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

        with Stubber(self.feature_table.table.meta.client) as ddb_stubber:
            page_1_params = {
                "ConsistentRead": True,
                "TableName": os.environ["FEATURE_TABLE"],
                "KeyConditionExpression": ANY,
            }
            page_1_response = {
                "Items": [feature_1],
                "LastEvaluatedKey": {
                    "hash_key": feature_1["hash_key"],
                    "range_key": feature_1["range_key"],
                    "tile_id": feature_1["tile_id"],
                },
            }

            page_2_params = {
                "ConsistentRead": True,
                "TableName": os.environ["FEATURE_TABLE"],
                "KeyConditionExpression": ANY,
                "ExclusiveStartKey": ANY,
            }
            page_2_response = {"Items": [feature_2]}

            ddb_stubber.add_response("query", page_1_response, page_1_params)
            ddb_stubber.add_response("query", page_2_response, page_2_params)

            result = self.feature_table.get_features(TEST_IMAGE_ID)
            ddb_stubber.assert_no_pending_responses()
            assert len(result) == 4, "Expected to retrieve 4 features from paginated results"

    def test_group_features_by_key(self):
        """
        Test that `group_features_by_key` correctly groups features by their generated keys.
        """
        features_dict = self.feature_table.group_features_by_key(self.get_feature_list())
        assert len(features_dict) == 1, "Expected features to be grouped under a single key"

    def test_generate_tile_key(self):
        """
        Test that `generate_tile_key` correctly creates the expected tile key for a feature.
        """
        result = self.feature_table.generate_tile_key(self.get_feature_list()[0])
        expected_tile_key = "7db12549-3bcb-49c8-acba-25d46ef5cbf3:s3://spacenet-dataset/AOIs/AOI_1_Rio/srcData/mosaic_3band/013022223131.tif-region-0:0:0:0"  # noqa
        assert result == expected_tile_key, "Expected tile key did not match generated tile key"

    def test_add_and_get_features(self):
        """
        Test that `add_features` correctly adds features to the table and `get_features` retrieves them.
        """
        features = self.get_feature_list()
        self.feature_table.add_features(features)
        ddb_features = self.feature_table.get_features(TEST_IMAGE_ID)
        assert len(ddb_features) == len(features), "Mismatch between added and retrieved features"

    @patch("aws.osml.model_runner.database.feature_table.FeatureTable.batch_write_items", TEST_MOCK_BATCH_WRITE_EXCEPTION)
    def test_add_features_throw_exceptions(self):
        """
        Test that `add_features` raises an `AddFeaturesException` when a batch write operation fails.
        """
        from aws.osml.model_runner.database.exceptions import AddFeaturesException

        features = self.get_feature_list()
        with self.assertRaises(AddFeaturesException):
            self.feature_table.add_features(features)

    def test_aggregate_features(self):
        """
        Test that `aggregate_features` correctly aggregates features across an entire image request.
        """
        from aws.osml.model_runner.database.job_table import JobItem

        features = self.get_feature_list()
        self.feature_table.add_features(features)

        job_item = JobItem(image_id=TEST_IMAGE_ID)
        aggregated_features = self.feature_table.aggregate_features(job_item)
        assert len(aggregated_features) == len(features), "Expected features to be aggregated correctly"

    def test_generate_tile_key_bad_bbox(self):
        """
        Test `generate_tile_key` with edge case of an invalid BBOX to ensure it raises a ValueError correctly.
        """
        # Feature with out-of-bound coordinates
        feature = geojson.Feature(
            geometry={"type": "Point", "coordinates": [5000, 5000]}, properties={"image_id": TEST_IMAGE_ID}
        )
        with self.assertRaises(ValueError):
            self.feature_table.generate_tile_key(feature)

    @staticmethod
    def get_feature_list() -> List[geojson.Feature]:
        """
        Utility function to read sample features from a GeoJSON file.
        """
        with open("./test/data/detections.geojson", "r") as geojson_file:
            sample_features = geojson.load(geojson_file)["features"]
        return sample_features
