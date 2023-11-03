#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from dacite import from_dict

from aws.osml.model_runner.common import RegionRequestStatus

from .ddb_helper import DDBHelper, DDBItem, DDBKey
from .exceptions import CompleteRegionException, GetRegionRequestItemException, StartRegionException, UpdateRegionException

logger = logging.getLogger(__name__)


@dataclass
class RegionRequestItem(DDBItem):
    """
    RegionRequestItem is a dataclass meant to represent a single item in the Region table

    The data schema is defined as follows:
    region_id: str = primary key - formatted as region (pixel bounds) + "-" + unique_identifier
    image_id: str = secondary key - image_id for the job
    start_time: Optional[Decimal] = time in epoch seconds when the job started
    last_updated_time: Optional[Decimal] = time in epoch seconds when job is processing (periodically)
    end_time: Optional[Decimal] = time in epoch seconds when the job ended
    message: Optional[str] = information about the region job
    status: Optional[str] = region job status - PROCESSING, COMPLETED, FAILED
    region_retry_count: Optional[Decimal] = total count of regions expected for this image
    region_pixel_bounds: = Region pixel bounds
    """

    region_id: str
    image_id: str
    job_id: Optional[str] = None
    start_time: Optional[Decimal] = None
    last_updated_time: Optional[Decimal] = None
    end_time: Optional[Decimal] = None
    message: Optional[str] = None
    region_status: Optional[str] = None
    total_tiles: Optional[Decimal] = None
    region_retry_count: Optional[Decimal] = None
    region_pixel_bounds: Optional[str] = None

    def __post_init__(self):
        self.ddb_key = DDBKey(
            hash_key="image_id",
            hash_value=self.image_id,
            range_key="region_id",
            range_value=self.region_id,
        )


class RegionRequestTable(DDBHelper):
    """
    RegionRequestTable is a class meant to help OSML with accessing and interacting with the region processing jobs we
    track as part of the region table. It extends the DDBHelper class and provides its own item data class for use when
    working with items from the table. It also sets the key for which we index on this table in the constructor.

    :param table_name: str = the name of the table to interact with

    :return: None
    """

    def __init__(self, table_name: str) -> None:
        super().__init__(table_name)

    def start_region_request(self, region_request_item: RegionRequestItem) -> RegionRequestItem:
        """
        Start a region processing request for given region pixel bounds, this should be the first record
        for this region in the table.

        :param region_request_item: RegionRequestItem = the unique identifier for the region we want to add to ddb

        :return: RegionRequestItem = Updated region request item
        """

        try:
            start_time_millisec = Decimal(time.time() * 1000)

            # Update the job item to have the correct start parameters
            region_request_item.start_time = start_time_millisec
            region_request_item.region_status = RegionRequestStatus.STARTING
            region_request_item.region_retry_count = Decimal(0)

            # Put the item into the table
            self.put_ddb_item(region_request_item)

            return region_request_item
        except Exception as err:
            raise StartRegionException("Failed to add region request to the table!") from err

    def complete_region_request(self, region_request_item: RegionRequestItem, region_status: RegionRequestStatus):
        """
        Update the region job to reflect that a region has succeeded or failed.

        :param region_request_item: RegionRequestItem = the unique identifier for the region we want to update
        :param region_status: RegionRequestStatus = Status of region at completion (FAILURE, PARTIAL, SUCCESS, etc.)

        :return: RegionRequestItem = Updated region request item
        """
        try:
            region_request_item.last_updated_time = Decimal(time.time() * 1000)
            region_request_item.region_status = region_status
            region_request_item.end_time = Decimal(time.time() * 1000)

            return from_dict(
                RegionRequestItem,
                self.update_ddb_item(region_request_item),
            )
        except Exception as e:
            raise CompleteRegionException("Failed to complete region!") from e

    def update_region_request(self, region_request_item: RegionRequestItem) -> RegionRequestItem:
        """
        Update the region info in the ddb

        :param region_request_item: RegionRequestItem = the unique identifier for the region we want to update

        :return: RegionRequestItem = Updated region request item
        """
        try:
            region_request_item.last_updated_time = Decimal(time.time() * 1000)

            return from_dict(
                RegionRequestItem,
                self.update_ddb_item(region_request_item),
            )
        except Exception as e:
            raise UpdateRegionException("Failed to update region!") from e

    def get_region_request(self, region_id: str, image_id: str) -> Optional[RegionRequestItem]:
        """
        Get a RegionRequestItem object from the table based on the region_id and image_id provided

        :param region_id: str = the unique identifier for the region we want to start processing
        :param image_id: str = the unique identifier for the image (range key)

        :return: Optional[RegionRequestItem] = region request item
        """
        try:
            # Retrieve job item from our table and set to expected RegionRequestItem class
            return from_dict(
                RegionRequestItem,
                self.get_ddb_item(RegionRequestItem(region_id=region_id, image_id=image_id)),
            )
        except Exception as e:
            logger.warning(GetRegionRequestItemException("Failed to get RegionRequestItem! {0}".format(e)))
            return None
