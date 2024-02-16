#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging

from aws.osml.model_runner.app_config import ServiceConfig
from aws.osml.model_runner.common import ImageRequestStatus
from aws.osml.model_runner.database.job_table import JobItem

from .exceptions import StatusMonitorException
from .image_request_status import ImageRequestStatusMessage
from .sns_helper import SNSHelper


class StatusMonitor:
    def __init__(self) -> None:
        self.image_status_sns = SNSHelper(ServiceConfig.image_status_topic)

    def process_event(self, image_request_item: JobItem, status: ImageRequestStatus, message: str) -> None:
        """
        Publishes the message via SNS

        :param image_request_item: JobItem = the image request item
        :param status: ImageRequestStatus = the status of an image request
        :param message: str = the message to notify the user

        :return: None
        """
        # Check that the image request has valid properties
        if (
            image_request_item.job_id is not None
            and image_request_item.job_arn is not None
            and image_request_item.image_id is not None
            and image_request_item.processing_time is not None
        ):
            try:
                logging.info(
                    "StatusMonitorUpdate",
                    extra={
                        "reason": message,
                        "status": status,
                        "request": image_request_item.__dict__,
                    },
                )

                sns_message_attributes = ImageRequestStatusMessage(
                    image_status=status,
                    job_id=image_request_item.job_id,
                    job_arn=image_request_item.job_arn,
                    image_id=image_request_item.image_id,
                    processing_duration=image_request_item.processing_time,
                )

                status_message = f"StatusMonitor update: {status} {image_request_item.job_id}: {message}"
                self.image_status_sns.publish_message(
                    status_message,
                    sns_message_attributes.asdict_str_values(),
                )
            except Exception as err:
                raise StatusMonitorException(
                    "StatusMonitor failed: {} {}: {}".format(status, image_request_item.job_id, err)
                )
        else:
            raise StatusMonitorException("StatusMonitor failed: {} {}".format(status, image_request_item.job_id))

    @staticmethod
    def get_image_request_status(image_request_item: JobItem) -> ImageRequestStatus:
        """
        Produce a image request status from a given image request

        :param image_request_item: JobItem = the image request item

        :return: ImageRequestStatus = the current status of an image request
        """
        # Check that the image request has valid properties
        if (
            image_request_item.region_count is not None
            and image_request_item.region_success is not None
            and image_request_item.region_error is not None
        ):
            if image_request_item.region_success == image_request_item.region_count:
                return ImageRequestStatus.SUCCESS
            elif (
                image_request_item.region_success + image_request_item.region_error == image_request_item.region_count
                and image_request_item.region_success > 0
            ):
                return ImageRequestStatus.PARTIAL
            elif image_request_item.region_error == image_request_item.region_count:
                return ImageRequestStatus.FAILED
            else:
                return ImageRequestStatus.IN_PROGRESS
        else:
            raise StatusMonitorException("Failed get status for given image request!")
