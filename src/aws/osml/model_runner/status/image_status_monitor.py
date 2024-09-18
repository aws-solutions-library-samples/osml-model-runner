#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging

from aws.osml.model_runner.common import RequestStatus
from aws.osml.model_runner.database.job_table import JobItem

from .base_status_monitor import BaseStatusMonitor
from .exceptions import StatusMonitorException
from .status_message import StatusMessage


class ImageStatusMonitor(BaseStatusMonitor):
    """
    ImageStatusMonitor is responsible for monitoring and publishing the status of image processing requests.

    This class uses Amazon SNS to publish updates on image request status, including success, partial success,
    failure, and progress. It interacts with JobItem data to determine the current status of the request and logs
    important status changes.
    """

    def __init__(self, image_status_topic: str) -> None:
        """
        Initializes the ImageStatusMonitor with the specified SNS topic for publishing image status updates.

        :param image_status_topic: str = The SNS topic to which status updates will be published.
        """
        super().__init__(image_status_topic)

    def process_event(self, image_request_item: JobItem, status: RequestStatus, message: str) -> None:
        """
        Publishes the status message via SNS for image requests.

        This method publishes status updates for a specific image request by creating an SNS message and logging
        the event. If the required fields (job_id, image_id, processing_duration) are present in the request item,
        it sends a message using the StatusMessage class.

        :param image_request_item: JobItem = The image request for which the status is being updated.
        :param status: RequestStatus = The current status of the image request.
        :param message: str = A message describing the reason for the status update.

        :return: None

        :raises StatusMonitorException: Raised if the request item is missing required fields or if SNS publication fails.
        """
        if (
            image_request_item.job_id is not None
            and image_request_item.image_id is not None
            and image_request_item.processing_duration is not None
        ):
            try:
                logging.info(
                    "ImageStatusMonitorUpdate",
                    extra={
                        "reason": message,
                        "status": status,
                        "request": image_request_item.__dict__,
                    },
                )

                sns_message_attributes = StatusMessage(
                    status=status,
                    image_status=status,
                    job_id=image_request_item.job_id,
                    image_id=image_request_item.image_id,
                    processing_duration=image_request_item.processing_duration,
                )

                status_message = f"StatusMonitor update: {status} {image_request_item.job_id}: {message}"
                self.sns_helper.publish_message(
                    status_message,
                    sns_message_attributes.asdict_str_values(),
                )
            except Exception as err:
                raise StatusMonitorException(f"StatusMonitor failed: {status} {image_request_item.job_id}: {err}")
        else:
            raise StatusMonitorException(f"StatusMonitor failed: {status} {image_request_item.job_id}")

    def get_status(self, request_item: JobItem) -> RequestStatus:
        """
        Determines the current status of an image request.

        This method evaluates the success and error counts for regions in the image request and returns the
        appropriate status: SUCCESS, PARTIAL, FAILED, or IN_PROGRESS.

        :param request_item: JobItem = The image request item containing region processing information.

        :return: RequestStatus = The status of the image request based on the region counts.
        """
        if request_item.region_success == request_item.region_count:
            return RequestStatus.SUCCESS
        elif (
            request_item.region_success + request_item.region_error == request_item.region_count
            and request_item.region_success > 0
        ):
            return RequestStatus.PARTIAL
        elif request_item.region_error == request_item.region_count:
            return RequestStatus.FAILED
        else:
            return RequestStatus.IN_PROGRESS
