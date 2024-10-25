#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.
import logging

import boto3

from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.common import VALID_IMAGE_COMPRESSION, VALID_IMAGE_FORMATS, get_credentials_for_assumed_role

from .exceptions import InvalidS3ObjectException
from .inference import VALID_MODEL_HOSTING_OPTIONS

logger = logging.getLogger(__name__)


def shared_properties_are_valid(request) -> bool:
    """
    Validates that the request contains all mandatory attributes with acceptable values.

    This function checks attributes shared between ImageRequests and RegionRequests, ensuring
    they contain all required metadata for processing.

    :param request: An object of either ImageRequests or RegionRequests.
    :return: True if the request is valid; False otherwise. Logs warnings for each failed validation.
    """
    if not request.image_id or not request.image_url:
        logger.error("Validation failed: `image_id` or `image_url` is missing.")
        return False

    if not request.model_name:
        logger.error("Validation failed: `model_name` is missing.")
        return False

    if not request.model_invoke_mode or request.model_invoke_mode not in VALID_MODEL_HOSTING_OPTIONS:
        logger.error(
            f"Validation failed: `model_invoke_mode` is either missing or invalid. "
            f"Expected one of {VALID_MODEL_HOSTING_OPTIONS}, but got '{request.model_invoke_mode}'."
        )
        return False

    if not request.tile_size or len(request.tile_size) != 2:
        logger.error("Validation failed: `tile_size` is missing or does not contain two dimensions.")
        return False

    if request.tile_size[0] <= 0 or request.tile_size[1] <= 0:
        logger.error("Validation failed: `tile_size` dimensions must be positive values.")
        return False

    if not request.tile_overlap or len(request.tile_overlap) != 2:
        logger.error("Validation failed: `tile_overlap` is missing or does not contain two dimensions.")
        return False

    if (
        request.tile_overlap[0] < 0
        or request.tile_overlap[0] >= request.tile_size[0]
        or request.tile_overlap[1] < 0
        or request.tile_overlap[1] >= request.tile_size[1]
    ):
        logger.error("Validation failed: `tile_overlap` values must be non-negative and less than `tile_size` dimensions.")
        return False

    if not request.tile_format or request.tile_format not in VALID_IMAGE_FORMATS:
        logger.error(
            f"Validation failed: `tile_format` is either missing or invalid. "
            f"Expected one of {VALID_IMAGE_FORMATS}, but got '{request.tile_format}'."
        )
        return False

    if request.tile_compression and request.tile_compression not in VALID_IMAGE_COMPRESSION:
        logger.error(
            f"Validation failed: `tile_compression` is invalid. "
            f"Expected one of {VALID_IMAGE_COMPRESSION}, but got '{request.tile_compression}'."
        )
        return False

    if request.image_read_role and not request.image_read_role.startswith("arn:"):
        logger.error("Validation failed: `image_read_role` does not start with 'arn:'.")
        return False

    if request.model_invocation_role and not request.model_invocation_role.startswith("arn:"):
        logger.error("Validation failed: `model_invocation_role` does not start with 'arn:'.")
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


def validate_image_path(image_url: str, assumed_role: str = None) -> bool:
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
