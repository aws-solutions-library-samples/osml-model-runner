#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import abc
from typing import List

from geojson import Feature

from aws.osml.model_runner.api import SinkMode


class Sink(abc.ABC):
    """
    The mechanism by which detected features are sent to their destination.
    """

    def __str__(self) -> str:
        return f"{self.name()} {self.mode.value}"

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """
        The name of the sink.

        :return: str = the output name
        """

    @property
    @abc.abstractmethod
    def mode(self) -> SinkMode:
        """
        The write mode of the sink. Either Streaming (per tile results)
        or Aggregate (per image results).

        :return: SinkMode = the type of write mode of the sink
        """

    @abc.abstractmethod
    def write(self, image_id: str, features: List[Feature]) -> bool:
        """
        Write feature list for given image id to the sink.

        :param image_id: str = the unique identifier for the image
        :param features: List[Feature] = the list of features

        :return: bool = if it has been written/output successfully
        """
