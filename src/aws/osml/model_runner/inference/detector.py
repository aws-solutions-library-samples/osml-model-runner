#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import abc
from io import BufferedReader

from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from geojson import FeatureCollection

from aws.osml.model_runner.api import ModelInvokeMode


class Detector(abc.ABC):
    """
    The mechanism by which detected features are sent to their destination.
    """

    def __init__(self, endpoint: str) -> None:
        """
        Endpoint Detector base class.

        :param endpoint: str = the endpoint that will be invoked
        """
        self.endpoint = endpoint
        self.request_count = 0
        self.error_count = 0

    @property
    @abc.abstractmethod
    def mode(self) -> ModelInvokeMode:
        """
        The mode of the detector.
        """

    @abc.abstractmethod
    @metric_scope
    def find_features(self, payload: BufferedReader, metrics: MetricsLogger) -> FeatureCollection:
        """
        Query the established endpoint mode to find features based on a payload

        :param payload: BufferedReader = the BufferedReader object that holds the
                                    data that will be  sent to the feature generator
        :param metrics: MetricsLogger = the metrics logger object to capture the log data on the system

        :return: FeatureCollection = a feature collection containing the center point of a tile
        """
