#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

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
from requests.exceptions import RetryError
from urllib3.exceptions import MaxRetryError
from urllib3.util.retry import Retry

from aws.osml.model_runner.api import ModelInvokeMode
from aws.osml.model_runner.app_config import MetricLabels
from aws.osml.model_runner.common import Timer

from .detector import Detector
from .endpoint_builder import FeatureEndpointBuilder

logger = logging.getLogger(__name__)


class CountingRetry(urllib3.Retry):
    """
    A custom Retry class that counts the number of retries during HTTP requests.
    Inherits from urllib3's Retry class to implement retry logic with an additional retry count.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the CountingRetry class with retry settings.

        :return: None
        """
        super(CountingRetry, self).__init__(*args, **kwargs)
        self.retry_counts = 0

    def increment(self, *args, **kwargs) -> Retry:
        """
        Increments the retry count and calls the parent class's increment method.

        :return: Retry = A Retry object with updated retry count.
        """
        result = super(CountingRetry, self).increment(*args, **kwargs)
        result.retry_counts = self.retry_counts + 1

        return result

    @classmethod
    def from_retry(cls, retry_instance: Retry) -> "CountingRetry":
        """
        Creates a CountingRetry instance from an existing Retry instance.

        :param retry_instance: Retry = The Retry instance to convert.
        :return: CountingRetry = A new CountingRetry object with the same configurations as the provided Retry instance.
        """
        if isinstance(retry_instance, cls):
            return retry_instance

        return cls(
            total=retry_instance.total,
            connect=retry_instance.connect,
            read=retry_instance.read,
            redirect=retry_instance.redirect,
            status=retry_instance.status,
            other=retry_instance.other,
            allowed_methods=retry_instance.allowed_methods,
            status_forcelist=retry_instance.status_forcelist,
            backoff_factor=retry_instance.backoff_factor,
            raise_on_redirect=retry_instance.raise_on_redirect,
            raise_on_status=retry_instance.raise_on_status,
            history=retry_instance.history,
            respect_retry_after_header=retry_instance.respect_retry_after_header,
            remove_headers_on_redirect=retry_instance.remove_headers_on_redirect,
        )


class HTTPDetector(Detector):
    """
    HTTPDetector is responsible for invoking HTTP-based model endpoints to run model inference.

    This class interacts with a model endpoint over HTTP to send a payload for feature detection and retrieves
    geojson-formatted feature detection results. It supports retry logic with exponential backoff for network-related
    issues.
    """

    def __init__(self, endpoint: str, name: Optional[str] = None, retry: Optional[urllib3.Retry] = None) -> None:
        """
        Initializes the HTTPDetector with the model endpoint URL, optional name, and retry policy.

        :param endpoint: str = The full URL of the model endpoint to invoke.
        :param name: Optional[str] = A name for the model endpoint.
        :param retry: Optional[Retry] = Retry policy for network requests.
        """
        if retry is None:
            self.retry = CountingRetry(total=8, backoff_factor=1, raise_on_status=True)
        else:
            self.retry = CountingRetry.from_retry(retry)
        self.http_pool = urllib3.PoolManager(cert_reqs="CERT_NONE", retries=self.retry)
        self.name = name or "http"
        super().__init__(endpoint=endpoint)

    @property
    def mode(self) -> ModelInvokeMode:
        """
        Defines the invocation mode for the detector as HTTP endpoint.

        :return: ModelInvokeMode.HTTP_ENDPOINT
        """
        return ModelInvokeMode.HTTP_ENDPOINT

    @metric_scope
    def find_features(self, payload: BufferedReader, metrics: MetricsLogger) -> FeatureCollection:
        """
        Invokes the HTTP model endpoint to detect features from the given payload.

        This method sends a payload to the HTTP model endpoint and retrieves feature detection results
        in the form of a geojson FeatureCollection. If configured, it logs metrics about the invocation process.

        :param payload: BufferedReader = The data to be sent to the HTTP model for feature detection.
        :param metrics: MetricsLogger = The metrics logger to capture system performance and log metrics.

        :return: FeatureCollection = A geojson FeatureCollection containing the detected features.

        :raises RetryError: Raised if the request fails after retries.
        :raises MaxRetryError: Raised if the maximum retry attempts are reached.
        :raises JSONDecodeError: Raised if there is an error decoding the model's response.
        """
        logger.debug(f"Invoking Model: {self.name}")
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.MODEL_INVOCATION_OPERATION,
                    MetricLabels.MODEL_NAME_DIMENSION: self.name,
                }
            )

        try:
            self.request_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

            with Timer(
                task_str="Invoke HTTP Endpoint",
                metric_name=MetricLabels.DURATION,
                logger=logger,
                metrics_logger=metrics,
            ):
                response = self.http_pool.request(
                    method="POST",
                    url=self.endpoint,
                    body=payload,
                )
                retry_count = self.retry.retry_counts
                if isinstance(metrics, MetricsLogger):
                    metrics.put_metric(MetricLabels.RETRIES, retry_count, str(Unit.COUNT.value))

                return geojson.loads(response.data.decode("utf-8"))

        except RetryError as err:
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error(f"Retry failed - failed due to {err}")
            logger.exception(err)
            raise err
        except MaxRetryError as err:
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error(f"Max retries reached - failed due to {err.reason}")
            logger.exception(err)
            raise err
        except JSONDecodeError as err:
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error(
                (
                    f"Unable to decode response from model. URL: {self.endpoint}, Status: {response.status}, "
                    f"Headers: {response.info()}, Response: {response.data}"
                )
            )
            logger.exception(err)
            raise err


class HTTPDetectorBuilder(FeatureEndpointBuilder):
    """
    HTTPDetectorBuilder is responsible for building an HTTPDetector configured with an HTTP model endpoint.

    This builder constructs an HTTPDetector instance that can send payloads to HTTP-based model endpoints for feature
    detection.
    """

    def __init__(self, endpoint: str):
        """
        Initializes the HTTPDetectorBuilder with the model endpoint URL.

        :param endpoint: str = The full URL of the model endpoint to be used.
        """
        super().__init__()
        self.endpoint = endpoint

    def build(self) -> Optional[Detector]:
        """
        Builds and returns an HTTPDetector based on the configured parameters.

        :return: Optional[Detector] = An HTTPDetector instance configured for the specified HTTP model endpoint.
        """
        return HTTPDetector(
            endpoint=self.endpoint,
        )
