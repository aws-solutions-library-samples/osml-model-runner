#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
from json import dumps, loads
from typing import Any, Dict, List, Optional

import boto3
import shapely.geometry
import shapely.wkt
from shapely.geometry.base import BaseGeometry

from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.common import (
    FeatureDistillationAlgorithm,
    FeatureDistillationNMS,
    ImageCompression,
    ImageDimensions,
    ImageFormats,
    MRPostProcessing,
    MRPostprocessingStep,
    deserialize_post_processing_list,
    get_credentials_for_assumed_role,
)

from .exceptions import InvalidS3ObjectException
from .inference import ModelInvokeMode
from .request_utils import shared_properties_are_valid
from .sink import SinkType

logger = logging.getLogger(__name__)


class ImageRequest(object):
    """
    Request for the Model Runner to process an image.

    This class contains the attributes that make up an image processing request along with
    constructors and factory methods used to create these requests from common constructs.
    """

    def __init__(self, *initial_data: Dict[str, Any], **kwargs: Any):
        """
        This constructor allows users to create these objects using a combination of dictionaries
        and keyword arguments.

        :param initial_data: Dict[str, Any] = dictionaries that contain attributes/values that map to this class's
                             attributes
        :param kwargs: Any = keyword arguments provided on the constructor to set specific attributes
        """
        default_post_processing = [
            MRPostProcessing(step=MRPostprocessingStep.FEATURE_DISTILLATION, algorithm=FeatureDistillationNMS())
        ]

        self.job_id: str = ""
        self.job_arn: str = ""
        self.image_id: str = ""
        self.image_url: str = ""
        self.image_read_role: str = ""
        self.outputs: List[dict] = []
        self.model_name: str = ""
        self.model_invoke_mode: ModelInvokeMode = ModelInvokeMode.NONE
        self.tile_size: ImageDimensions = (1024, 1024)
        self.tile_overlap: ImageDimensions = (50, 50)
        self.tile_format: ImageFormats = ImageFormats.NITF
        self.tile_compression: ImageCompression = ImageCompression.NONE
        self.model_invocation_role: str = ""
        self.feature_properties: List[dict] = []
        self.roi: Optional[BaseGeometry] = None
        self.post_processing: List[MRPostProcessing] = default_post_processing

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @staticmethod
    def from_external_message(image_request: Dict[str, Any]):
        """
        This method is used to construct an ImageRequest given a dictionary reconstructed from the
        JSON representation of a request that appears on the Image Job Queue. The structure of
        that message is generally governed by AWS API best practices and may evolve over time as
        the public APIs for this service mature.

        :param image_request: Dict[str, Any] = dictionary of values from the decoded JSON request

        :return: the ImageRequest
        """
        properties: Dict[str, Any] = {}
        if "imageProcessorTileSize" in image_request:
            tile_dimension = int(image_request["imageProcessorTileSize"])
            properties["tile_size"] = (tile_dimension, tile_dimension)

        if "imageProcessorTileOverlap" in image_request:
            overlap_dimension = int(image_request["imageProcessorTileOverlap"])
            properties["tile_overlap"] = (overlap_dimension, overlap_dimension)

        if "imageProcessorTileFormat" in image_request:
            properties["tile_format"] = image_request["imageProcessorTileFormat"]

        if "imageProcessorTileCompression" in image_request:
            properties["tile_compression"] = image_request["imageProcessorTileCompression"]

        properties["job_arn"] = image_request["jobArn"]
        properties["job_id"] = image_request["jobId"]

        properties["image_url"] = image_request["imageUrls"][0]
        properties["image_id"] = image_request["jobId"] + ":" + properties["image_url"]
        if "imageReadRole" in image_request:
            properties["image_read_role"] = image_request["imageReadRole"]

        properties["model_name"] = image_request["imageProcessor"]["name"]
        properties["model_invoke_mode"] = image_request["imageProcessor"]["type"]
        if "assumedRole" in image_request["imageProcessor"]:
            properties["model_invocation_role"] = image_request["imageProcessor"]["assumedRole"]

        if "regionOfInterest" in image_request:
            properties["roi"] = shapely.wkt.loads(image_request["regionOfInterest"])

        # Support explicit outputs
        if image_request.get("outputs"):
            properties["outputs"] = image_request["outputs"]
        # Support legacy image request
        elif image_request.get("outputBucket") and image_request.get("outputPrefix"):
            properties["outputs"] = [
                {
                    "type": SinkType.S3.value,
                    "bucket": image_request["outputBucket"],
                    "prefix": image_request["outputPrefix"],
                }
            ]
        if image_request.get("featureProperties"):
            properties["feature_properties"] = image_request["featureProperties"]
        if image_request.get("postProcessing"):
            image_request["postProcessing"] = loads(
                dumps(image_request["postProcessing"])
                .replace("algorithmType", "algorithm_type")
                .replace("iouThreshold", "iou_threshold")
                .replace("skipBoxThreshold", "skip_box_threshold")
            )
            properties["post_processing"] = deserialize_post_processing_list(image_request.get("postProcessing"))

        return ImageRequest(properties)

    def is_valid(self) -> bool:
        """
        Check to see if this request contains required attributes and meaningful values

        :return: bool = True if the request contains all the mandatory attributes with acceptable values,
                 False otherwise
        """
        if not shared_properties_are_valid(self):
            logger.error("Invalid shared properties in ImageRequest")
            return False

        if not self.job_arn or not self.job_id or not self.outputs:
            logger.error("Missing job arn, job id, or outputs properties in ImageRequest")
            return False

        num_feature_detection_options = len(self.get_feature_distillation_option())
        if num_feature_detection_options > 1:
            logger.error("{} feature distillation options in ImageRequest".format(num_feature_detection_options))
            return False

        return True

    def get_shared_values(self) -> Dict[str, Any]:
        """
        Returns a formatted dict that contains the properties of an image

        :return: Dict[str, Any] = the properties of an image
        """
        return {
            "image_id": self.image_id,
            "job_id": self.job_id,
            "image_url": self.image_url,
            "image_read_role": self.image_read_role,
            "model_name": self.model_name,
            "model_invoke_mode": self.model_invoke_mode,
            "model_invocation_role": self.model_invocation_role,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "tile_format": self.tile_format,
            "tile_compression": self.tile_compression,
        }

    def get_feature_distillation_option(self) -> List[FeatureDistillationAlgorithm]:
        """
        Parses the post-processing property and extracts the relevant feature distillation selection, if present
        :return:
        """
        return [
            op.algorithm
            for op in self.post_processing
            if op.step == MRPostprocessingStep.FEATURE_DISTILLATION
            and isinstance(op.algorithm, FeatureDistillationAlgorithm)
        ]

    @staticmethod
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
            # head_object is a fastest approach to determine if it exists in S3
            # also its less expensive to do the head_object approach
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as err:
            raise InvalidS3ObjectException("This image does not exist!") from err
