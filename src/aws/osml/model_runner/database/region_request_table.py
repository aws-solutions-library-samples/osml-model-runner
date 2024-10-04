#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from dacite import from_dict

from aws.osml.model_runner.api import RegionRequest
from aws.osml.model_runner.common import ImageRegion, RequestStatus, TileState

from .ddb_helper import DDBHelper, DDBItem, DDBKey
from .exceptions import CompleteRegionException, GetRegionRequestItemException, StartRegionException, UpdateRegionException

logger = logging.getLogger(__name__)


@dataclass
class RegionRequestItem(DDBItem):
    """
    RegionRequestItem is a dataclass meant to represent a single item in the Region table.

    The data schema is defined as follows:
    region_id: str = primary key - formatted as region (pixel bounds) + "-" + unique_identifier
    image_id: str = secondary key - image_id for the job
    job_id: Optional[str] = job identifier for tracking
    start_time: Optional[int] = time in epoch seconds when the job started
    last_updated_time: Optional[int] = time in epoch seconds when the job is processing (periodically updated)
    end_time: Optional[int] = time in epoch seconds when the job ended
    expire_time: Optional[int] = time in epoch seconds when the item will expire from the table
    image_read_role: Optional[str] = IAM role to read the image for processing
    processing_duration: Optional[int] = time in seconds to complete region processing
    message: Optional[str] = additional information about the region job
    region_status: Optional[str] = region job status - PROCESSING, COMPLETED, FAILED
    total_tiles: Optional[int] = total number of tiles to be processed for the region
    failed_tiles: Optional[List] = list of tiles that failed processing
    failed_tile_count: Optional[int] = count of failed tiles that failed to process
    succeeded_tiles: Optional[List] = list of tiles that succeeded processing
    succeeded_tile_count: Optional[int] = count of successfully processed tiles
    region_bounds: Optional[List[List[int]]] = list of pixel bounds that define the region
    region_retry_count: Optional[int] = number of times the region processing has been retried
    tile_compression: Optional[str] = compression type of tiles for the region (e.g., 'LZW', 'JPEG')
    tile_format: Optional[str] = file format of the tiles (e.g., 'tif', 'ntf')
    tile_overlap: Optional[List[int]] = overlap dimensions for the tiles in the region
    tile_size: Optional[List[int]] = size dimensions of the tiles in the region
    """

    region_id: str
    image_id: str
    job_id: Optional[str] = None
    start_time: Optional[int] = None
    last_updated_time: Optional[int] = None
    end_time: Optional[int] = None
    expire_time: Optional[int] = None
    image_read_role: Optional[str] = None
    processing_duration: Optional[int] = None
    message: Optional[str] = None
    region_status: Optional[str] = None
    total_tiles: Optional[int] = None
    failed_tiles: Optional[List] = None
    failed_tile_count: Optional[int] = None
    succeeded_tiles: Optional[List] = None
    succeeded_tile_count: Optional[int] = None
    region_bounds: Optional[List[List[int]]] = None
    region_retry_count: Optional[int] = None
    tile_compression: Optional[str] = None
    tile_format: Optional[str] = None
    tile_overlap: Optional[List[int]] = None
    tile_size: Optional[List[int]] = None

    def __post_init__(self):
        self.ddb_key = DDBKey(
            hash_key="image_id",
            hash_value=self.image_id,
            range_key="region_id",
            range_value=self.region_id,
        )

    @classmethod
    def from_region_request(cls, region_request: RegionRequest) -> "RegionRequestItem":
        """
        Helper method to create a RegionRequestItem from a RegionRequest object.

        :param region_request: A RegionRequest object.
        :return: A RegionRequestItem instance with the relevant fields populated.
        """
        return cls(
            region_id=region_request.region_id,
            image_id=region_request.image_id,
            job_id=region_request.job_id,
            image_read_role=region_request.image_read_role,
            region_bounds=[
                [int(region_request.region_bounds[0][0]), int(region_request.region_bounds[0][1])],
                [int(region_request.region_bounds[1][0]), int(region_request.region_bounds[1][1])],
            ],
            tile_size=[int(region_request.tile_size[0]), int(region_request.tile_size[1])],
            tile_overlap=[int(region_request.tile_overlap[0]), int(region_request.tile_overlap[1])],
            tile_format=str(region_request.tile_format),
            tile_compression=str(region_request.tile_compression),
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
            start_time_millisec = int(time.time() * 1000)

            # Update the job item to have the correct start parameters
            region_request_item.start_time = start_time_millisec
            region_request_item.region_status = RequestStatus.STARTED
            region_request_item.region_retry_count = 0
            region_request_item.succeeded_tile_count = 0
            region_request_item.failed_tile_count = 0
            region_request_item.processing_duration = 0
            region_request_item.expire_time = int((start_time_millisec / 1000) + (24 * 60 * 60))

            # Put the item into the table
            self.put_ddb_item(region_request_item)

            return region_request_item
        except Exception as err:
            raise StartRegionException("Failed to add region request to the table!") from err

    def complete_region_request(self, region_request_item: RegionRequestItem, region_status: RequestStatus):
        """
        Update the region job to reflect that a region has succeeded or failed.

        :param region_request_item: RegionRequestItem = the unique identifier for the region we want to update
        :param region_status: RegionRequestStatus = Status of region at completion (FAILURE, PARTIAL, SUCCESS, etc.)

        :return: RegionRequestItem = Updated region request item
        """
        try:
            region_request_item.last_updated_time = int(time.time() * 1000)
            region_request_item.region_status = region_status
            region_request_item.end_time = int(time.time() * 1000)
            region_request_item.processing_duration = region_request_item.end_time - region_request_item.start_time

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
            region_request_item.last_updated_time = int(time.time() * 1000)

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
        except Exception as err:
            logger.warning(GetRegionRequestItemException(f"Failed to get RegionRequestItem! {err}"))
            return None

    def add_tile(self, image_id: str, region_id: str, tile: ImageRegion, state: TileState) -> RegionRequestItem:
        """
        Append tile to the with the associated state to associated RegionRequestItem in the table.

        :param image_id: str = the id of the image request we want to update
        :param region_id: str = the id of the region request we want to update
        :param tile: ImageRegion = list of values to append to the 'succeeded_tiles' property
        :param state: str = state of the tile to add, i.e. succeeded or failed
        :return: The new updated DDB item.
        """
        # Validate the tile is a tuple of tuples (ImageRegion format)
        if not (isinstance(tile, tuple) and isinstance(tile[0], tuple) and isinstance(tile[1], tuple)):
            raise UpdateRegionException(f"Invalid tile format. Expected a tuple of tuples, got {type(tile)}")

        try:
            # Build the update expression using list_append to append a value
            update_expr = (
                f"SET {state.value}_tiles = list_append(if_not_exists({state.value}_tiles, :empty_list), " f":new_values)"
            )
            update_attr = {":new_values": [[list(coord) for coord in tile]], ":empty_list": []}

            # Perform the update on DynamoDB
            new_item = self.update_ddb_item(RegionRequestItem(region_id, image_id), update_expr, update_attr)

            # Return the updated item
            logger.debug(f"Successfully appended {tile} to item with image_id={image_id}, region_id={region_id}.")
            return from_dict(
                RegionRequestItem,
                new_item,
            )
        except Exception as err:
            logger.error(f"Failed to append {state.value} {tile} to item region_id={region_id}: {str(err)}")
            raise UpdateRegionException(f"Failed to append {state.value} {tile} to item region_id={region_id}.") from err
