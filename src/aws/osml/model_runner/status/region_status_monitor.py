#  Copyright 2024 Amazon.com, Inc. or its affiliates.

import logging

from aws.osml.model_runner.common import RequestStatus
from aws.osml.model_runner.database.region_request_table import RegionRequestItem

from .base_status_monitor import BaseStatusMonitor
from .exceptions import StatusMonitorException
from .status_message import StatusMessage


class RegionStatusMonitor(BaseStatusMonitor):
    """
    RegionStatusMonitor is responsible for monitoring and publishing the status of region processing requests.

    This class uses Amazon SNS to publish updates on the status of region requests, including success, partial success,
    and failure. It interacts with RegionRequestItem data to determine the current status of the request and logs
    important status changes.
    """

    def __init__(self, region_status_topic: str) -> None:
        """
        Initializes the RegionStatusMonitor with the specified SNS topic for publishing region status updates.

        :param region_status_topic: str = The SNS topic to which status updates will be published.
        """
        super().__init__(region_status_topic)

    def process_event(self, region_request_item: RegionRequestItem, status: RequestStatus, message: str) -> None:
        """
        Publishes the status message via SNS for region requests.

        This method publishes status updates for a specific region request by creating an SNS message and logging
        the event. If the required fields (job_id, image_id, region_id, processing_duration) are present in the request item,
        it sends a message using the StatusMessage class.

        :param region_request_item: RegionRequestItem = The region request for which the status is being updated.
        :param status: RequestStatus = The current status of the region request.
        :param message: str = A message describing the reason for the status update.

        :return: None

        :raises StatusMonitorException: Raised if the request item is missing required fields or if SNS publication fails.
        """
        if (
            region_request_item.job_id is not None
            and region_request_item.image_id is not None
            and region_request_item.region_id is not None
            and region_request_item.processing_duration is not None
        ):
            try:
                logging.debug(
                    "RegionStatusMonitorUpdate",
                    extra={
                        "reason": message,
                        "status": status,
                        "request": region_request_item.__dict__,
                    },
                )

                sns_message_attributes = StatusMessage(
                    status=status,
                    job_id=region_request_item.job_id,
                    image_id=region_request_item.image_id,
                    processing_duration=region_request_item.processing_duration,
                    region_id=region_request_item.region_id,
                    failed_tiles=region_request_item.failed_tiles,
                )

                status_message = f"StatusMonitor update: {status} {region_request_item.job_id}: {message}"
                self.sns_helper.publish_message(
                    status_message,
                    sns_message_attributes.asdict_str_values(),
                )
            except Exception as err:
                raise StatusMonitorException(f"StatusMonitor failed: {status} {region_request_item.job_id}: {err}")
        else:
            raise StatusMonitorException(f"StatusMonitor failed: {status} {region_request_item.job_id}")

    def get_status(self, request_item: RegionRequestItem) -> RequestStatus:
        """
        Determines the current status of a region request.

        This method evaluates the total tiles and failed tile counts for regions in the request and returns the
        appropriate status: SUCCESS, PARTIAL, or FAILED.

        :param request_item: RegionRequestItem = The region request item containing tile processing information.

        :return: RequestStatus = The status of the region request based on the tile counts.
        """
        region_status = RequestStatus.SUCCESS
        if request_item.total_tiles == request_item.failed_tile_count:
            region_status = RequestStatus.FAILED
        if 0 < request_item.failed_tile_count < request_item.total_tiles:
            region_status = RequestStatus.PARTIAL
        return region_status
