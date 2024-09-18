#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, List, Optional

from aws.osml.model_runner.common import RequestStatus


@dataclass
class StatusMessage:
    status: RequestStatus
    job_id: str
    image_status: Optional[RequestStatus] = None
    image_id: Optional[str] = None
    region_id: Optional[str] = None
    processing_duration: Optional[int] = None
    failed_tiles: Optional[List[List[List[int]]]] = None

    def asdict_str_values(self) -> Dict[str, str]:
        string_dict: Dict[str, str] = dict()
        for k, v in self.asdict().items():
            if isinstance(v, Enum):
                string_dict[k] = str(v.value)
            elif k == "failed_tiles":
                # Format the failed tiles more cleanly
                string_dict[k] = str(
                    [{f"{i + 1}": [[int(coord) for coord in point] for point in tile]} for i, tile in enumerate(v)]
                )
            else:
                string_dict[k] = str(v)
        return string_dict

    def asdict(self) -> dict:
        return dict((fld.name, getattr(self, fld.name)) for fld in fields(self) if getattr(self, fld.name))
