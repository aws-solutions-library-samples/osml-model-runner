#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
from dataclasses import dataclass, field
from json import dumps, loads
from typing import Any, Dict, List, Optional

import shapely.geometry
import shapely.wkt
from dacite import from_dict
from shapely.geometry.base import BaseGeometry

from aws.osml.model_runner.common import (
    FeatureDistillationAlgorithm,
    FeatureDistillationNMS,
    ImageCompression,
    ImageDimensions,
    ImageFormats,
    MRPostProcessing,
    MRPostprocessingStep,
    deserialize_post_processing_list,
)

from .inference import ModelInvokeMode
from .request_utils import shared_properties_are_valid
from .sink import VALID_SYNC_TYPES, SinkType

logger = logging.getLogger(__name__)


@dataclass
class ImageRequest:
    """
    Request for the Model Runner to process an image.

    This class contains the attributes that make up an image processing request, along with
    constructors and factory methods used to create these requests from common constructs.

    Attributes:
        job_id: The unique identifier for the image processing job.
        image_id: A combined identifier for the image, usually composed of the job ID and image URL.
        image_url: The URL location of the image to be processed.
        image_read_role: The IAM role used to read the image from the provided URL.
        outputs: A list of output configurations where results should be stored.
        model_name: The name of the model to use for image processing.
        model_invoke_mode: The mode in which the model is invoked, such as synchronous or asynchronous.
        tile_size: Dimensions of the tiles into which the image is split for processing.
        tile_overlap: Overlap between tiles, defined in dimensions.
        tile_format: The format of the tiles (e.g., NITF, GeoTIFF).
        tile_compression: Compression type to use for the tiles (e.g., None, JPEG).
        model_invocation_role: IAM role assumed for invoking the model.
        feature_properties: Additional properties to include in the feature processing.
        roi: Region of interest within the image, defined as a geometric shape.
        post_processing: List of post-processing steps to apply to the features detected.
    """

    job_id: str = ""
    image_id: str = ""
    image_url: str = ""
    image_read_role: str = ""
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = ""
    model_invoke_mode: ModelInvokeMode = ModelInvokeMode.NONE
    tile_size: ImageDimensions = (1024, 1024)
    tile_overlap: ImageDimensions = (50, 50)
    tile_format: str = ImageFormats.NITF.value
    tile_compression: str = ImageCompression.NONE.value
    model_invocation_role: str = ""
    feature_properties: List[Dict[str, Any]] = field(default_factory=list)
    roi: Optional[BaseGeometry] = None
    post_processing: List[MRPostProcessing] = field(
        default_factory=lambda: [
            MRPostProcessing(step=MRPostprocessingStep.FEATURE_DISTILLATION, algorithm=FeatureDistillationNMS())
        ]
    )

    @staticmethod
    def from_external_message(image_request: Dict[str, Any]) -> "ImageRequest":
        """
        Constructs an ImageRequest from a dictionary that represents an external message.

        :param image_request: Dictionary of values from the decoded JSON request.
        :return: ImageRequest instance.
        """
        properties: Dict[str, Any] = {
            "job_id": image_request.get("jobId", ""),
            "image_url": image_request.get("imageUrls", [""])[0],
            "image_id": f"{image_request.get('jobId', '')}:{image_request.get('imageUrls', [''])[0]}",
            "image_read_role": image_request.get("imageReadRole", ""),
            "model_name": image_request["imageProcessor"]["name"],
            "model_invoke_mode": ImageRequest._parse_model_invoke_mode(image_request["imageProcessor"].get("type")),
            "model_invocation_role": image_request["imageProcessor"].get("assumedRole", ""),
            "tile_size": ImageRequest._parse_tile_dimension(image_request.get("imageProcessorTileSize")),
            "tile_overlap": ImageRequest._parse_tile_dimension(image_request.get("imageProcessorTileOverlap")),
            "tile_format": ImageRequest._parse_tile_format(image_request.get("imageProcessorTileFormat")),
            "tile_compression": ImageRequest._parse_tile_compression(image_request.get("imageProcessorTileCompression")),
            "roi": ImageRequest._parse_roi(image_request.get("regionOfInterest")),
            "outputs": ImageRequest._parse_outputs(image_request),
            "feature_properties": image_request.get("featureProperties", []),
            "post_processing": ImageRequest._parse_post_processing(image_request.get("postProcessing")),
        }
        return from_dict(ImageRequest, properties)

    @staticmethod
    def _parse_tile_dimension(value: Optional[str]) -> ImageDimensions:
        """
        Converts a string value to a tuple of integers representing tile dimensions.

        :param value: String value representing tile dimension.
        :return: Tuple of integers as tile dimensions.
        """
        return (int(value), int(value)) if value else None

    @staticmethod
    def _parse_roi(roi: Optional[str]) -> Optional[BaseGeometry]:
        """
        Parses the region of interest from a WKT string.

        :param roi: WKT string representing the region of interest.
        :return: Parsed BaseGeometry object or None.
        """
        return shapely.wkt.loads(roi) if roi else None

    @staticmethod
    def _parse_tile_format(tile_format: Optional[str]) -> Optional[ImageFormats]:
        """
        Parses the region desired tile format to use for processing.

        :param tile_format: String representing the tile format to use.
        :return: Parsed ImageFormats object or ImageFormats.NITF.
        """
        return ImageFormats[tile_format].value if tile_format else ImageFormats.NITF.value

    @staticmethod
    def _parse_tile_compression(tile_compression: Optional[str]) -> Optional[ImageCompression]:
        """
        Parses the region desired tile compression format to use for processing.

        :param tile_compression: String representing the tile compression format to use.
        :return: Parsed ImageFormats object or ImageCompression.NONE.
        """
        return ImageCompression[tile_compression].value if tile_compression else ImageCompression.NONE.value

    @staticmethod
    def _parse_model_invoke_mode(model_invoke_mode: Optional[str]) -> Optional[ModelInvokeMode]:
        """
        Parses the region desired tile compression format to use for processing.

        :param model_invoke_mode: String representing the tile compression format to use.
        :return: Parsed ModelInvokeMode object or ModelInvokeMode.SM_ENDPOINT.
        """
        return ModelInvokeMode[model_invoke_mode] if model_invoke_mode else ModelInvokeMode.SM_ENDPOINT

    @staticmethod
    def _parse_outputs(image_request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Parses the output configuration from the image request, including support for legacy inputs.

        :param image_request: Dictionary of image request attributes.
        :return: List of output configurations.
        """
        if image_request.get("outputs"):
            return image_request["outputs"]

        # Support legacy image request fields: outputBucket and outputPrefix
        if image_request.get("outputBucket") and image_request.get("outputPrefix"):
            return [
                {
                    "type": SinkType.S3.value,
                    "bucket": image_request["outputBucket"],
                    "prefix": image_request["outputPrefix"],
                }
            ]
        # No outputs were defined in the request
        logger.warning("No output syncs were present in this request.")
        return []

    @staticmethod
    def _parse_post_processing(post_processing: Optional[Dict[str, Any]]) -> List[MRPostProcessing]:
        """
        Deserializes and cleans up post-processing data.

        :param post_processing: Dictionary of post-processing configurations.
        :return: List of MRPostProcessing instances.
        """
        if not post_processing:
            return [MRPostProcessing(step=MRPostprocessingStep.FEATURE_DISTILLATION, algorithm=FeatureDistillationNMS())]
        cleaned_post_processing = loads(
            dumps(post_processing)
            .replace("algorithmType", "algorithm_type")
            .replace("iouThreshold", "iou_threshold")
            .replace("skipBoxThreshold", "skip_box_threshold")
        )
        return deserialize_post_processing_list(cleaned_post_processing)

    def is_valid(self) -> bool:
        """
        Validates whether the ImageRequest instance has all required attributes.

        :return: True if valid, False otherwise.
        """
        if not shared_properties_are_valid(self):
            logger.error("Invalid shared properties in ImageRequest")
            return False
        if not self.job_id:
            logger.error("Missing job id in ImageRequest")
            return False
        if len(self.get_feature_distillation_option()) > 1:
            logger.error("Multiple feature distillation options in ImageRequest")
            return False
        if len(self.outputs) > 0:
            for output in self.outputs:
                sink_type = output.get("type")
                if sink_type not in VALID_SYNC_TYPES:
                    logger.error(f"Invalid sink type '{sink_type}' in ImageRequest")
                    return False
        return True

    def get_shared_values(self) -> Dict[str, Any]:
        """
        Retrieves a dictionary of shared values related to the image.

        :return: Dictionary of shared image properties.
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
        Extracts the feature distillation options from the post-processing configuration.

        :return: List of FeatureDistillationAlgorithm instances.
        """
        return [
            op.algorithm
            for op in self.post_processing
            if op.step == MRPostprocessingStep.FEATURE_DISTILLATION
            and isinstance(op.algorithm, FeatureDistillationAlgorithm)
        ]
