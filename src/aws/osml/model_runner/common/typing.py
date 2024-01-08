#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from enum import Enum, auto
from typing import Tuple

from aws.osml.model_runner.common import AutoStringEnum

# Pixel coordinate (row, column)
ImageCoord = Tuple[int, int]
# 2D shape (w, h)
ImageDimensions = Tuple[int, int]
# UL corner (row, column) , dimensions (w, h)
ImageRegion = Tuple[ImageCoord, ImageDimensions]


class ImageCompression(str, AutoStringEnum):
    """
    Enumeration defining compression algorithms for image.
    """

    NONE = auto()
    JPEG = auto()
    J2K = auto()
    LZW = auto()


class ImageFormats(str, AutoStringEnum):
    """
    Enumeration defining image encodings.
    """

    NITF = auto()
    JPEG = auto()
    PNG = auto()
    GTIFF = auto()


class ImageRequestStatus(str, AutoStringEnum):
    """
    Enumeration defining the image request status
    """

    STARTED = auto()
    IN_PROGRESS = auto()
    PARTIAL = auto()
    SUCCESS = auto()
    FAILED = auto()


class RegionRequestStatus(str, AutoStringEnum):
    """
    Enumeration defining status for region
    """

    STARTING = auto()
    PARTIAL = auto()
    IN_PROGRESS = auto()
    SUCCESS = auto()
    FAILED = auto()


class GeojsonDetectionField(str, Enum):
    """
    Enumeration defining the model geojson field to index depending on the shape
    """

    BOUNDS = "bounds_imcoords"
    GEOM = "geom_imcoords"


VALID_IMAGE_COMPRESSION = [item.value for item in ImageCompression]
VALID_IMAGE_FORMATS = [item.value for item in ImageFormats]
