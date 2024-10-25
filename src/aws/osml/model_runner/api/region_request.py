#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
from dataclasses import dataclass
from typing import Any, Dict

from aws.osml.model_runner.common import ImageCompression, ImageDimensions, ImageFormats, ImageRegion

from .inference import ModelInvokeMode
from .request_utils import shared_properties_are_valid

logger = logging.getLogger(__name__)


@dataclass
class RegionRequest:
    """
    Request for the Model Runner to process a region of an image.

    This class contains the attributes that make up a region processing request, along with
    constructors used to create these requests from common constructs.

    Attributes:
        region_id: The unique identifier for the region being processed.
        image_id: The identifier for the source image from which the region is derived.
        image_extension: The file extension of the image (e.g., .tif, .ntf).
        job_id: The unique identifier for the image processing job.
        image_url: The URL location of the image to be processed.
        image_read_role: The IAM role used to read the image from the provided URL.
        model_name: The name of the model to use for region processing.
        model_invoke_mode: The mode in which the model is invoked, such as synchronous or asynchronous.
        model_invocation_role: IAM role assumed for invoking the model.
        tile_size: Dimensions of the tiles into which the region is split for processing.
        tile_overlap: Overlap between tiles, defined in dimensions.
        tile_format: The format of the tiles (e.g., NITF, GeoTIFF).
        tile_compression: Compression type to use for the tiles (e.g., None, JPEG).
        region_bounds: Bounds of the region within the image, defined as upper-left corner coordinates and dimensions.
    """

    region_id: str = ""
    image_id: str = ""
    image_extension: str = ""
    job_id: str = ""
    image_url: str = ""
    image_read_role: str = ""
    model_name: str = ""
    model_invoke_mode: ModelInvokeMode = ModelInvokeMode.NONE
    model_invocation_role: str = ""
    tile_size: ImageDimensions = (1024, 1024)
    tile_overlap: ImageDimensions = (50, 50)
    tile_format: ImageFormats = ImageFormats.NITF
    tile_compression: ImageCompression = ImageCompression.NONE
    region_bounds: ImageRegion = ((0, 0), (0, 0))

    def __init__(self, *initial_data: Dict[str, Any], **kwargs: Any):
        """
        Initialize a RegionRequest instance using a combination of dictionaries and keyword arguments.

        :param initial_data: dictionaries containing attributes/values that map to this class's attributes.
        :param kwargs: keyword arguments to set specific attributes.
        """
        self._set_attributes_from_data(*initial_data, **kwargs)

    def _set_attributes_from_data(self, *initial_data: Dict[str, Any], **kwargs: Any):
        """
        Sets attributes on the instance from initial data dictionaries and keyword arguments.

        :param initial_data: dictionaries of attribute values.
        :param kwargs: keyword arguments to set attributes.
        """
        for data in initial_data:
            self._set_attributes_from_dict(data)
        self._set_attributes_from_dict(kwargs)

    def _set_attributes_from_dict(self, data: Dict[str, Any]):
        """
        Helper method to set instance attributes based on a dictionary.

        :param data: dictionary containing attribute-value pairs to set.
        """
        for key, value in data.items():
            setattr(self, key, value)

    def is_valid(self) -> bool:
        """
        Check to see if this request contains required attributes and meaningful values.

        :return: True if the request contains all the mandatory attributes with acceptable values, False otherwise.
        """
        if not shared_properties_are_valid(self):
            return False

        if not self.region_bounds or self.region_bounds == ((0, 0), (0, 0)):
            logger.error("Invalid bounds in RegionRequest")
            return False

        return True
