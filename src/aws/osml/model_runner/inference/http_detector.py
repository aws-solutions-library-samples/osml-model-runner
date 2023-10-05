import logging
from io import BufferedReader
from json import JSONDecodeError
from typing import Optional

import geojson
import urllib3
from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from geojson import FeatureCollection

from aws.osml.model_runner.api import ModelInvokeMode
from aws.osml.model_runner.app_config import MetricLabels, ServiceConfig
from aws.osml.model_runner.common import Timer

from .detector import Detector
from .endpoint_builder import FeatureEndpointBuilder
from .feature_utils import create_mock_feature_collection

logger = logging.getLogger(__name__)


class HTTPDetector(Detector):
    def __init__(self, endpoint: str) -> None:
        """
        A HTTP model endpoint invoking object, intended to query sagemaker endpoints.

        :param endpoint: str = the full URL to invoke the model

        :return: None
        """
        self.http_pool = urllib3.PoolManager(cert_reqs="CERT_NONE")
        super().__init__(endpoint=endpoint)

    @property
    def mode(self) -> ModelInvokeMode:
        return ModelInvokeMode.HTTP_ENDPOINT

    @metric_scope
    def find_features(self, payload: BufferedReader, metrics: MetricsLogger) -> FeatureCollection:
        """
        Query the established endpoint mode to find features based on a payload

        :param payload: BufferedReader = the BufferedReader object that holds the
                                    data that will be sent to the feature generator
        :param metrics: MetricsLogger = the metrics logger object to capture the log data on the system

        :return: FeatureCollection = a feature collection containing the center point of a tile
        """
        retry_count = 0
        logger.info("Invoking HTTP Endpoint: {}".format(self.endpoint))
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions({"HTTPModelEndpoint": self.endpoint})

        try:
            self.request_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.MODEL_INVOCATION, 1, str(Unit.COUNT.value))

            with Timer(
                task_str="Invoke HTTP Endpoint",
                metric_name=MetricLabels.ENDPOINT_LATENCY,
                logger=logger,
                metrics_logger=metrics,
            ):
                # If we are not running against a real model
                if self.endpoint == ServiceConfig.noop_model_name:
                    return create_mock_feature_collection(payload)
                else:
                    response = self.http_pool.request(
                        method="POST",
                        url=self.endpoint,
                        body=payload,
                    )
                    self.request_count = 1
                    if isinstance(metrics, MetricsLogger):
                        metrics.put_metric(MetricLabels.ENDPOINT_RETRY_COUNT, retry_count, str(Unit.COUNT.value))
                    return geojson.loads(response.data.decode("utf-8"))
        except JSONDecodeError as err:
            self.error_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.FEATURE_DECODE, 1, str(Unit.COUNT.value))
                metrics.put_metric(MetricLabels.MODEL_ERROR, 1, str(Unit.COUNT.value))
            logger.error(
                "Unable to decode response from model. URL: {}, Status: {}, Headers: {}, Response: {}".format(
                    self.endpoint, response.status, response.info(), response.data
                )
            )
            logger.exception(err)
            self.error_count += 1

        # Return an empty feature collection if the process errored out
        return FeatureCollection([])


class HTTPDetectorBuilder(FeatureEndpointBuilder):
    def __init__(
        self,
        endpoint: str,
    ):
        super().__init__()
        self.endpoint = endpoint

    def build(self) -> Optional[Detector]:
        return HTTPDetector(
            endpoint=self.endpoint,
        )
