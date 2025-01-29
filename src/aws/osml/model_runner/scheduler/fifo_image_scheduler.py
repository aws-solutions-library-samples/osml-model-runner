#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import logging
from typing import Optional

from aws.osml.model_runner.api import ImageRequest, InvalidImageRequestException
from aws.osml.model_runner.queue import RequestQueue
from aws.osml.model_runner.scheduler.image_scheduler import ImageScheduler

logger = logging.getLogger(__name__)


class FIFOImageScheduler(ImageScheduler):
    """
    This first in first out (FIFO) scheduler is just a pass through to a request queue.
    """

    def __init__(self, image_request_queue: RequestQueue):
        self.image_request_queue = image_request_queue
        self.image_requests_iter = iter(self.image_request_queue)
        self.job_id_to_message_handle = {}

    def get_next_scheduled_request(self) -> Optional[ImageRequest]:
        """
        Return the next image request to be processed. This implementation retrieves the next message on the queue
        and returns the image request created from that message.

        :return: the next image request, None if there is not a request pending execution
        """
        logger.debug("FIFO image scheduler checking work queue for images to process...")
        try:
            receipt_handle, image_request_message = next(self.image_requests_iter)
            if image_request_message:
                try:
                    image_request = ImageRequest.from_external_message(image_request_message)
                    if not image_request.is_valid():
                        raise InvalidImageRequestException(f"Invalid image request: {image_request_message}")

                    self.job_id_to_message_handle[image_request.job_id] = receipt_handle
                    return image_request
                except Exception:
                    logger.error("Failed to parse image request", exc_info=True)
                    self.image_request_queue.finish_request(receipt_handle)
        except Exception:
            logger.error("Unable to retrieve an image request from the queue", exc_info=True)

        return None

    def finish_request(self, image_request: ImageRequest, should_retry: bool = False) -> None:
        """
        Mark the given image request as finished.

        :param image_request: the image request
        :param should_retry: true if this request was not complete and can be retried immediately
        """
        logger.debug(f"Finished processing image request: {image_request}")
        receipt_handle = self.job_id_to_message_handle[image_request.job_id]
        if should_retry:
            self.image_request_queue.reset_request(receipt_handle, visibility_timeout=0)
        else:
            self.image_request_queue.finish_request(receipt_handle)
        del self.job_id_to_message_handle[image_request.job_id]
