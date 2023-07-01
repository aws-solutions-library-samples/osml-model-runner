#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import os
from typing import List, Optional

import boto3
import geojson
from botocore.exceptions import ClientError
from geojson import Feature, FeatureCollection

from aws.osml.model_runner.api import SinkMode, SinkType
from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.common import get_credentials_for_assumed_role

from .sink import Sink

logger = logging.getLogger(__name__)


class S3Sink(Sink):
    def __init__(
        self,
        bucket: str,
        prefix: str,
        assumed_role: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        if assumed_role:
            assumed_credentials = get_credentials_for_assumed_role(assumed_role)
            # Here we will be writing to S3 using an IAM role other than the one for this process.
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=assumed_credentials["AccessKeyId"],
                aws_secret_access_key=assumed_credentials["SecretAccessKey"],
                aws_session_token=assumed_credentials["SessionToken"],
                config=BotoConfig.default,
            )
        else:
            # If no invocation role is provided the assumption is that the default role for this
            # container will be sufficient to write to the S3 bucket.
            self.s3_client = boto3.client("s3", config=BotoConfig.default)

    @property
    def mode(self) -> SinkMode:
        return SinkMode.AGGREGATE

    def write(self, image_id: str, features: List[Feature]) -> bool:
        features_collection = FeatureCollection(features)

        # validate if S3 bucket exists and accessible
        if self.validate_s3_bucket():
            # image_id is the concatenation of the job id and source image url in s3. We just
            # want to base our key off of the original image file name so split by '/' and use
            # the last element
            object_key = os.path.join(self.prefix, image_id.split("/")[-1] + ".geojson")
            # Add the aggregated features to a feature collection and encode the full set of features
            # as a GeoJSON output.
            self.s3_client.put_object(
                Body=str(geojson.dumps(features_collection)),
                Bucket=self.bucket,
                Key=object_key,
                ACL="bucket-owner-full-control",
            )
            logger.info(
                "Wrote aggregate feature collection for Image '{}' to s3://{}/{}".format(image_id, self.bucket, object_key)
            )
            return True
        else:
            return False

    def validate_s3_bucket(self) -> bool:
        """
        Check if Output S3 bucket exists and can be read/written to it

        :return: bool = True if bucket exist and can be read/written to it
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as err:
            if err.response["Error"]["Code"] == "404":  # Does not exist
                logging.error("This S3 Bucket({}) does not exist".format(self.bucket))
            elif err.response["Error"]["Code"] == "403":  # Forbidden access
                logging.error("Do not have permission to read/write this S3 Bucket({})".format(self.bucket))
            logging.error("Cannot read/write S3 Bucket ({})".format(self.bucket))
            return False

    @staticmethod
    def name() -> str:
        return str(SinkType.S3.value)
