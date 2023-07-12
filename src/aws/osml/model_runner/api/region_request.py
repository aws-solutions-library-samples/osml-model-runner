#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from typing import Any, Dict

from aws.osml.model_runner.common import ImageCompression, ImageDimensions, ImageFormats, ImageRegion

from .inference import ModelInvokeMode
from .request_utils import shared_properties_are_valid


class RegionRequest(object):
    """
    Request for the Model Runner to process a region of an image.

    This class contains the attributes that make up a region processing request along with
    constructors used to create these requests from common constructs.
    """

    def __init__(self, *initial_data: Dict[str, Any], **kwargs: Any):
        """
        This constructor allows users to create these objects using a combination of dictionaries
        and keyword arguments.

        :param initial_data: dictionaries that contain attributes/values that map to this class's
                             attributes
        :param kwargs: keyword arguments provided on the constructor to set specific attributes
        """
        self.region_id: str = ""
        self.image_id: str = ""
        self.image_extension: str = ""
        self.job_id: str = ""
        self.image_url: str = ""
        self.image_read_role: str = ""
        self.model_name: str = ""
        self.model_invoke_mode: ModelInvokeMode = ModelInvokeMode.NONE
        self.model_invocation_role: str = ""
        self.tile_size: ImageDimensions = (1024, 1024)
        self.tile_overlap: ImageDimensions = (50, 50)
        self.tile_format: ImageFormats = ImageFormats.NITF
        self.tile_compression: ImageCompression = ImageCompression.NONE
        # Bounds are: UL corner (row, column) , dimensions (w, h)
        self.region_bounds: ImageRegion = ((0, 0), (0, 0))

        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def is_valid(self) -> bool:
        """
        Check to see if this request contains required attributes and meaningful values

        :return: bool = True if the request contains all the mandatory attributes with acceptable values,
                 False otherwise
        """
        if not shared_properties_are_valid(self):
            return False

        if not self.region_bounds:
            return False

        return True
