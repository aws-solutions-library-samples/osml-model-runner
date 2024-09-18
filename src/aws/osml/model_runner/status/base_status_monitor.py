#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from abc import ABC, abstractmethod

from aws.osml.model_runner.common import RequestStatus
from aws.osml.model_runner.database import DDBItem

from .sns_helper import SNSHelper


class BaseStatusMonitor(ABC):
    def __init__(self, sns_topic: str) -> None:
        self.sns_helper = SNSHelper(sns_topic)

    @abstractmethod
    def process_event(self, request_item: DDBItem, status: RequestStatus, message: str) -> None:
        """
        Publishes the message via SNS. Must be implemented by subclasses.

        :param request_item: DDBItem = the request item (image/region)
        :param status: RequestStatus = the status of a request
        :param message: str = the message to notify the user

        :return: None
        """
        pass

    @abstractmethod
    def get_status(self, request_item: DDBItem) -> RequestStatus:
        """
        Produce a request status from a given request item. Must be implemented by subclasses.

        :param request_item: DDBItem = the request item (image/region)

        :return: RequestStatus = the current status of the request
        """
        pass
