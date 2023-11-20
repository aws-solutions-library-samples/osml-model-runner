#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from dacite import from_dict

from .ddb_helper import DDBHelper, DDBItem, DDBKey
from .exceptions import (
    CompleteRegionException,
    EndImageException,
    GetImageRequestItemException,
    IsImageCompleteException,
    StartImageException,
)


@dataclass
class JobItem(DDBItem):
    """
    JobItem is a dataclass meant to represent a single item in the JobStatus table

    The data schema is defined as follows:
    image_id: str = unique image_id for the job
    start_time: Optional[Decimal] = time in epoch milliseconds when the job started
    expire_time: Optional[Decimal] = time in epoch seconds when the job will expire
    end_time: Optional[Decimal] = time in epoch milliseconds when the job ended
    region_success: Optional[Decimal] = current count of regions that have succeeded for this image
    region_error: Optional[Decimal] = current count of regions that have errored for this image
    region_count: Optional[Decimal] = total count of regions expected for this image
    width: Optional[Decimal] = width of the image
    height: Optional[Decimal] = height of the image
    feature_distillation_options: Optional[str] = the options used in selecting features (NMS/SOFT_NMS, thresholds, etc.)
    """

    image_id: str
    job_id: Optional[str] = None
    job_arn: Optional[str] = None
    image_url: Optional[str] = None
    image_read_role: Optional[str] = None
    model_invoke_mode: Optional[str] = None
    start_time: Optional[Decimal] = None
    expire_time: Optional[Decimal] = None
    end_time: Optional[Decimal] = None
    region_success: Optional[Decimal] = None
    region_error: Optional[Decimal] = None
    region_count: Optional[Decimal] = None
    width: Optional[Decimal] = None
    height: Optional[Decimal] = None
    extents: Optional[str] = None
    tile_size: Optional[str] = None
    tile_overlap: Optional[str] = None
    model_name: Optional[str] = None
    outputs: Optional[str] = None
    processing_time: Optional[Decimal] = None
    feature_properties: Optional[str] = None
    image_security_classification: Optional[str] = None
    feature_distillation_option: Optional[str] = None

    def __post_init__(self):
        self.ddb_key = DDBKey(hash_key="image_id", hash_value=self.image_id)


class JobTable(DDBHelper):
    """
    JobTable is a class meant to help OSML with accessing and interacting with the image processing jobs we track
    as part of the job status table. It extends the DDBHelper class and provides its own item data class for use when
    working with items from the table. It also  sets the key for which we index on this table in the constructor.

    :param table_name: str = the name of the table to interact with

    :return: None
    """

    def __init__(self, table_name: str) -> None:
        super().__init__(table_name)

    def start_image_request(self, image_request_item: JobItem) -> JobItem:
        """
        Start an image processing request for given image_id, this should be the first record for this image in the
        table.

        :param image_request_item: the unique identifier for the image we want to add to ddb

        :return: JobItem = response from ddb
        """

        try:
            # These records are temporary and will expire 24 hours after creation. Jobs should take
            # minutes to run so this time should be conservative enough to let a team debug an urgent
            # issue without leaving a ton of state leftover in the system.
            start_time_millisec = Decimal(time.time() * 1000)
            expire_time_epoch_sec = Decimal(int(start_time_millisec / 1000) + (24 * 60 * 60))

            # Update the job item to have the correct start parameters
            image_request_item.start_time = start_time_millisec
            image_request_item.processing_time = Decimal(0)
            image_request_item.expire_time = expire_time_epoch_sec
            image_request_item.region_success = Decimal(0)
            image_request_item.region_error = Decimal(0)

            # Put the item into the table
            self.put_ddb_item(image_request_item)

            # Return the updated image request
            return image_request_item
        except Exception as err:
            raise StartImageException("Failed to start image processing!") from err

    def complete_region_request(self, image_id: str, error: bool) -> JobItem:
        """
        Update the image job to reflect that a region has succeeded or failed.

        :param image_id: str = the unique identifier for the image we want to update
        :param error: bool = if there was an error processing the region, is true else false

        :return: None
        """
        try:
            # Determine if we increment the success or error counts
            if error:
                # Build custom update expression for updating region_error in DDB
                update_exp = "SET region_error = region_error + :error_count"
                # Build custom update attributes for updating region_error in DDB
                update_attr = {":error_count": Decimal(1)}
            else:
                # Build custom update expression for updating region_error in DDB
                update_exp = "SET region_success = region_success + :success_count"
                # Build custom update attributes for updating region_error in DDB
                update_attr = {":success_count": Decimal(1)}

            # Update item in the table and translate to a JobItem
            return from_dict(
                JobItem,
                self.update_ddb_item(
                    ddb_item=JobItem(image_id=image_id),
                    update_exp=update_exp,
                    update_attr=update_attr,
                ),
            )

        except Exception as err:
            raise CompleteRegionException("Failed to complete region!") from err

    @staticmethod
    def is_image_request_complete(image_request_item: JobItem) -> bool:
        """
        Read the table for a ddb item and determine if the image_id associated with the job has completed processing all
        regions associated with that image.

        :param image_request_item: JobItem = the unique identifier for the image we want to check if the image is completed

        :return: bool
        """
        # Check that the image request has valid properties
        if (
            image_request_item.region_count is not None
            and image_request_item.region_success is not None
            and image_request_item.region_error is not None
        ):
            # Determine if we have completed all regions
            completed_regions = image_request_item.region_success + image_request_item.region_error
            return image_request_item.region_count == completed_regions
        else:
            raise IsImageCompleteException("Failed to check if image is complete!")

    def end_image_request(self, image_id: str) -> JobItem:
        """
        Stop an image processing job for given image_id and record the time the job ended, this should be the last
        record for this image in the table.

        :param image_id: str = the unique identifier for the image we want to stop processing

        :return: None
        """
        try:
            # Get the latest item
            image_request_item = self.get_image_request(image_id)

            # Give it an end time
            image_request_item.end_time = Decimal(time.time() * 1000)

            # Update the item in the table
            return self.update_image_request(image_request_item)

        except Exception as e:
            raise EndImageException("Failed to end image!") from e

    def get_image_request(self, image_id: str) -> JobItem:
        """
        Get a JobItem object from the table based on the image_id provided

        :param image_id: str = the unique identifier for the image we want to start processing

        :return: JobItem = updated image request item from ddb
        """
        try:
            # Retrieve job item from our table and set to expected JobItem class
            return from_dict(JobItem, self.get_ddb_item(JobItem(image_id=image_id)))
        except Exception as e:
            raise GetImageRequestItemException("Failed to get ImageRequestItem!") from e

    def update_image_request(self, image_request_item: JobItem) -> JobItem:
        """
        Get a JobItem object from the table based on the image_id provided

        :param image_request_item: JobItem =

        :return: ImageRequestItem = updated image request item from ddb
        """
        # Update the processing time on our message
        if image_request_item.start_time is not None:
            image_request_item.processing_time = self.get_processing_time(image_request_item.start_time)

        return from_dict(JobItem, self.update_ddb_item(image_request_item))

    @staticmethod
    def get_processing_time(start_time: Decimal) -> Decimal:
        return Decimal(time.time() - (float(start_time) / float(1000.0)))
