#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
import io
import json
from unittest.mock import Mock

from botocore.exceptions import ClientError

MOCK_MODEL_RESPONSE = {
    "Body": io.StringIO(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": "1cc5e6d6-e12f-430d-adf0-8d2276ce8c5a",
                        "geometry": {"type": "Point", "coordinates": [-43.679691, -22.941953]},
                        "properties": {
                            "bounds_imcoords": [429, 553, 440, 561],
                            "geom_imcoords": [[429, 553], [429, 561], [440, 561], [440, 553], [429, 553]],
                            "featureClasses": [{"iri": "ground_motor_passenger_vehicle", "score": 0.2961518168449402}],
                            "detection_score": 0.2961518168449402,
                            "image_id": "2pp5e6d6-e12f-430d-adf0-8d2276ceadf0",
                        },
                    }
                ],
            }
        )
    )
}


TEST_CONFIG = {
    "ACCOUNT_ID": "123456789123",
    "ELEVATION_DATA_LOCATION": "s3://test-bucket/elevation-data",
    "ENDPOINT_PRODUCTION_VARIANTS": [
        {"VariantName": "Primary", "ModelName": "TestModel", "InitialInstanceCount": 1, "InstanceType": "ml.m5.12xlarge"}
    ],
    "ENDPOINT_TABLE_ATTRIBUTE_DEFINITIONS": [{"AttributeName": "endpoint", "AttributeType": "S"}],
    "ENDPOINT_TABLE_KEY_SCHEMA": [{"AttributeName": "endpoint", "KeyType": "HASH"}],
    "FEATURE_TABLE_ATTRIBUTE_DEFINITIONS": [
        {"AttributeName": "hash_key", "AttributeType": "S"},
        {"AttributeName": "range_key", "AttributeType": "S"},
    ],
    "FEATURE_TABLE_KEY_SCHEMA": [
        {"AttributeName": "hash_key", "KeyType": "HASH"},
        {"AttributeName": "range_key", "KeyType": "RANGE"},
    ],
    "IMAGE_BUCKET": "test-image-bucket",
    "IMAGE_EXTENSION": "NITF",
    "IMAGE_FILE": "./test/data/small.ntf",
    "IMAGE_ID": "test-image-id",
    "IMAGE_KEY": "small.ntf",
    "JOB_ID": "test-job-id",
    "JOB_NAME": "test-job-name",
    "JOB_TABLE_ATTRIBUTE_DEFINITIONS": [{"AttributeName": "image_id", "AttributeType": "S"}],
    "JOB_TABLE_KEY_SCHEMA": [{"AttributeName": "image_id", "KeyType": "HASH"}],
    "MOCK_PUT_EXCEPTION": Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "put_item")),
    "MOCK_UPDATE_EXCEPTION": Mock(
        side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item")
    ),
    "MODEL_ENDPOINT": "TestEndpoint",
    "MODEL_NAME": "TestModel",
    "REGION_ID": "test-region-id",
    "REGION_REQUEST_TABLE_ATTRIBUTE_DEFINITIONS": [
        {"AttributeName": "region_id", "AttributeType": "S"},
        {"AttributeName": "image_id", "AttributeType": "S"},
    ],
    "REGION_REQUEST_TABLE_KEY_SCHEMA": [
        {"AttributeName": "region_id", "KeyType": "HASH"},
        {"AttributeName": "image_id", "KeyType": "RANGE"},
    ],
    "RESULTS_BUCKET": "test-results-bucket",
    "RESULTS_STREAM": "test-results-stream",
    "S3_FULL_BUCKET_PATH": "s3://test-results-bucket/test/data/small.ntf",
    "SM_MODEL_CONTAINER": {
        "Image": "123456789123.dkr.ecr.us-east-1.amazonaws.com/test:1",
        "ModelDataUrl": "s3://test-bucket/model.tar.gz",
    },
}
