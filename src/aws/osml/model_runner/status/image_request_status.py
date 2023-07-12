#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from dataclasses import dataclass, field, fields
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from shapely.geometry.base import BaseGeometry

from aws.osml.model_runner.api import ModelInvokeMode
from aws.osml.model_runner.common import ImageCompression, ImageDimensions, ImageFormats, ImageRegion, ImageRequestStatus
from aws.osml.model_runner.sink import Sink


@dataclass
class ImageRequestStatusMessage:
    image_status: ImageRequestStatus
    job_id: str
    job_arn: Optional[str] = None
    image_id: Optional[str] = None
    image_url: Optional[str] = None
    image_read_role: Optional[str] = None
    outputs: Optional[List[Sink]] = field(default=None, repr=False)
    model_name: Optional[str] = None
    processing_duration: Optional[Decimal] = None
    model_invoke_mode: Optional[ModelInvokeMode] = None
    model_invocation_role: Optional[str] = None
    tile_size: Optional[ImageDimensions] = None
    tile_overlap: Optional[ImageDimensions] = None
    tile_format: Optional[ImageFormats] = None
    tile_compression: Optional[ImageCompression] = None
    roi: Optional[BaseGeometry] = None
    region_bounds: Optional[ImageRegion] = None

    def asdict_str_values(self) -> Dict[str, str]:
        string_dict: Dict[str, str] = dict()
        for k, v in self.asdict().items():
            if isinstance(v, Enum):
                string_dict[k] = str(v.value)
            elif isinstance(v, List):
                string_dict[k] = str([str(item) for item in v])
            else:
                string_dict[k] = str(v)
        return string_dict

    def asdict(self) -> dict:
        return dict((fld.name, getattr(self, fld.name)) for fld in fields(self) if getattr(self, fld.name))
