#  Copyright 2025 Amazon.com, Inc. or its affiliates.

from abc import ABC, abstractmethod
from typing import Optional

from aws.osml.model_runner.api import ImageRequest


class ImageScheduler(ABC):
    """
    ImageSchedule defines an abstract base for classes that determine how to schedule images for processing.
    """

    @abstractmethod
    def get_next_scheduled_request(self) -> Optional[ImageRequest]:
        """
        Return the next image request to be processed.

        :return: the image reqeust
        """

    @abstractmethod
    def finish_request(self, image_request: ImageRequest, should_retry: bool = False) -> None:
        """
        Mark the given image request as finished.

        :param image_request: the image request
        :param should_retry: true if this request was not complete and can be retried immediately
        """
