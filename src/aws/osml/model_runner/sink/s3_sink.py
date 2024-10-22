#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import os
import tempfile
from typing import List, Optional

import boto3
import geojson
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from geojson import Feature, FeatureCollection

from aws.osml.model_runner.api import SinkMode, SinkType
from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.common import get_credentials_for_assumed_role

from .sink import Sink

logger = logging.getLogger(__name__)


class S3Sink(Sink):
    """
    A sink for writing aggregated GeoJSON feature collections to an S3 bucket.

    This class handles uploading GeoJSON data to a specified S3 bucket, optionally using an
    assumed IAM role for accessing the bucket. It validates the bucket's accessibility and writes
    the aggregated features to S3.

    :param bucket: The name of the S3 bucket.
    :param prefix: The prefix within the bucket where the files will be stored.
    :param assumed_role: Optional IAM role ARN to assume for accessing the bucket.
    """

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
        """
        The mode of the sink, which is aggregate.

        :return: The `SinkMode` enumeration value representing the aggregate mode.
        """
        return SinkMode.AGGREGATE

    def write(self, image_id: str, features: List[Feature]) -> bool:
        """
        Write aggregated GeoJSON feature collection to the S3 bucket.

        Validates if the S3 bucket is accessible and uploads a temporary file containing
        the aggregated features. The object key is derived from the `image_id`.

        :param image_id: The identifier for the image, used to generate the S3 object key.
        :param features: A list of GeoJSON features to be aggregated and written to S3.
        :return: `True` if the upload was successful, `False` otherwise.

        :raises ClientError: If there are errors while uploading the file to S3.
        """
        features_collection = FeatureCollection(features)

        # validate if S3 bucket exists and accessible
        if self.validate_s3_bucket():
            # image_id is the concatenation of the job id and source image url in s3. We just
            # want to base our key off of the original image file name so split by '/' and use
            # the last element
            object_key = os.path.join(self.prefix, image_id.split("/")[-1] + ".geojson")

            # Create a temporary file to store aggregated features as a GeoJSON data
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                with open(temp_file.name, "w") as f:
                    f.write(geojson.dumps(features_collection))

                # Use upload_file to upload the file to S3
                self.s3_client.upload_file(
                    Filename=temp_file.name,
                    Bucket=self.bucket,
                    Key=object_key,
                    Config=TransferConfig(
                        multipart_threshold=64 * 1024**2,  # 64 MB
                        max_concurrency=10,
                        multipart_chunksize=128 * 1024**2,  # 128 MB
                        use_threads=True,
                    ),
                    ExtraArgs={"ACL": "bucket-owner-full-control"},
                )

            logger.debug(f"Wrote aggregate feature collection for '{image_id}' to s3://{self.bucket}/{object_key}")
            return True
        else:
            return False

    def validate_s3_bucket(self) -> bool:
        """
        Check if the output S3 bucket exists and can be read/written to.

        :return: `True` if the bucket exists and can be accessed, `False` otherwise.

        :raises ClientError: If there are issues accessing the S3 bucket.
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as err:
            if err.response["Error"]["Code"] == "404":  # Does not exist
                logging.error(f"This S3 Bucket({self.bucket}) does not exist")
            elif err.response["Error"]["Code"] == "403":  # Forbidden access
                logging.error(f"Do not have permission to read/write this S3 Bucket({self.bucket})")
            logging.error(f"Cannot read/write S3 Bucket ({self.bucket})")
            return False

    @staticmethod
    def name() -> str:
        """
        Retrieve the name of the sink type.

        :return: The string representation of the sink type.
        """
        return str(SinkType.S3.value)
