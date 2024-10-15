#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import boto3

from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.common import VALID_IMAGE_COMPRESSION, VALID_IMAGE_FORMATS, get_credentials_for_assumed_role

from .exceptions import InvalidS3ObjectException
from .inference import VALID_MODEL_HOSTING_OPTIONS


def shared_properties_are_valid(request) -> bool:
    """
    There are some attributes that are shared between ImageRequests and RegionRequests. This
    function exists to validate if ImageRequests/RegionRequests have all the metadata info
    in order to process it

    :param request: an object of either ImageRequests or RegionRequests

    :return: bool = True if the request contains all the mandatory attributes with acceptable values,
                 False otherwise
    """
    if not request.image_id or not request.image_url:
        return False

    if not request.model_name:
        return False

    if not request.model_invoke_mode or request.model_invoke_mode not in VALID_MODEL_HOSTING_OPTIONS:
        return False

    if not request.tile_size or len(request.tile_size) != 2:
        return False

    if request.tile_size[0] <= 0 or request.tile_size[1] <= 0:
        return False

    if not request.tile_overlap or len(request.tile_overlap) != 2:
        return False

    if (
        request.tile_overlap[0] < 0
        or request.tile_overlap[0] >= request.tile_size[0]
        or request.tile_overlap[1] < 0
        or request.tile_overlap[1] >= request.tile_size[1]
    ):
        return False

    if not request.tile_format or request.tile_format not in VALID_IMAGE_FORMATS:
        return False

    if request.tile_compression and request.tile_compression not in VALID_IMAGE_COMPRESSION:
        return False

    if request.image_read_role and not request.image_read_role.startswith("arn:"):
        return False

    if request.model_invocation_role and not request.model_invocation_role.startswith("arn:"):
        return False

    return True


def get_image_path(image_url: str, assumed_role: str) -> str:
    """
    Returns the formatted image path for GDAL to read the image, either from S3 or a local file.

    If the image URL points to an S3 path, this method validates the image's existence in S3
    and reformats the path to use GDAL's /vsis3/ driver. Otherwise, it returns the local or
    network image path.

    :param image_url: str = formatted image path to S3 bucket
    :param assumed_role: str = containing a formatted arn role

    :return: The formatted image path.
    """
    if "s3:/" in image_url:
        validate_image_path(image_url, assumed_role)
        return image_url.replace("s3:/", "/vsis3", 1)
    return image_url


def validate_image_path(image_url: str, assumed_role: str) -> bool:
    """
    Validate if an image exists in S3 bucket

    :param image_url: str = formatted image path to S3 bucket
    :param assumed_role: str = containing a formatted arn role

    :return: bool
    """
    bucket, key = image_url.replace("s3://", "").split("/", 1)
    if assumed_role:
        assumed_credentials = get_credentials_for_assumed_role(assumed_role)
        # Here we will be writing to S3 using an IAM role other than the one for this process.
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=assumed_credentials["AccessKeyId"],
            aws_secret_access_key=assumed_credentials["SecretAccessKey"],
            aws_session_token=assumed_credentials["SessionToken"],
            config=BotoConfig.default,
        )
    else:
        # If no invocation role is provided the assumption is that the default role for this
        # container will be sufficient to read/write to the S3 bucket.
        s3_client = boto3.client("s3", config=BotoConfig.default)

    try:
        # head_object is the fastest approach to determine if it exists in S3
        # also its less expensive to do the head_object approach
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as err:
        raise InvalidS3ObjectException("This image does not exist!") from err
