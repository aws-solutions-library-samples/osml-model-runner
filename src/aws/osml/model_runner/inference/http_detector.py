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
from aws.osml.model_runner.app_config import MetricLabels, ServiceConfig
from aws.osml.model_runner.common import Timer

from .detector import Detector
from .endpoint_builder import FeatureEndpointBuilder
from .feature_utils import create_mock_feature_collection

logger = logging.getLogger(__name__)


class CountingRetry(urllib3.Retry):
    def __init__(self, *args, **kwargs):
        """
        Retry class implementation that counts the number of retries.

        :return: None
        """
        super(CountingRetry, self).__init__(*args, **kwargs)
        self.retry_counts = 0

    def increment(self, *args, **kwargs) -> Retry:
        # Call the parent's increment function
        result = super(CountingRetry, self).increment(*args, **kwargs)
        result.retry_counts = self.retry_counts + 1

        return result

    @classmethod
    def from_retry(cls, retry_instance: Retry) -> "CountingRetry":
        """Create a CountingRetry instance from a Retry instance."""
        if isinstance(retry_instance, cls):
            return retry_instance  # No conversion needed if it's already a CountingRetry instance

        # Create a CountingRetry instance with the same configurations
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
    def __init__(self, endpoint: str, name: Optional[str] = None, retry: Optional[urllib3.Retry] = None) -> None:
        """
        An HTTP model endpoint interface object, intended to query sagemaker endpoints.

        :param endpoint: Full url to invoke the model
        :param name: Name to give the model endpoint
        :param retry: Retry policy to use when invoking the model

        :return: None
        """
        # Setup Retry with exponential backoff
        # - We will retry for a maximum of eight times.
        # - We start with a backoff of 1 second.
        # - We will double the backoff after each failed retry attempt.
        # - We cap the maximum backoff time to 255 seconds.
        # - We can adjust these values as required.
        if retry is None:
            self.retry = CountingRetry(total=8, backoff_factor=1, raise_on_status=True)
        else:
            self.retry = CountingRetry.from_retry(retry)
        self.http_pool = urllib3.PoolManager(cert_reqs="CERT_NONE", retries=self.retry)
        if name:
            self.name = name
        else:
            self.name = "http"
        super().__init__(endpoint=endpoint)

    @property
    def mode(self) -> ModelInvokeMode:
        return ModelInvokeMode.HTTP_ENDPOINT

    @metric_scope
    def find_features(self, payload: BufferedReader, metrics: MetricsLogger) -> FeatureCollection:
        """
        Query the established endpoint mode to find features based on a payload

        :param payload: BufferedReader object that holds the data that will be sent to the feature generator
        :param metrics: Metrics logger object to capture the log data on the system

        :return: GeoJSON FeatureCollection containing the center point of a tile
        """
        logger.info("Invoking Model: {}".format(self.name))
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
                # If we are not running against a real model
                if self.endpoint == ServiceConfig.noop_geom_model_name:
                    return create_mock_feature_collection(payload, geom=True)
                elif self.endpoint == ServiceConfig.noop_bounds_model_name:
                    return create_mock_feature_collection(payload)
                else:
                    response = self.http_pool.request(
                        method="POST",
                        url=self.endpoint,
                        body=payload,
                    )
                    # get the history of retries and count them
                    retry_count = self.retry.retry_counts
                    if isinstance(metrics, MetricsLogger):
                        metrics.put_metric(MetricLabels.RETRIES, retry_count, str(Unit.COUNT.value))
                    return geojson.loads(response.data.decode("utf-8"))
        except RetryError as err:
            self.error_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error("Retry failed - failed due to {}".format(err))
            logger.exception(err)
        except MaxRetryError as err:
            self.error_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error("Max retries reached - failed due to {}".format(err.reason))
            logger.exception(err)
        except JSONDecodeError as err:
            self.error_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error(
                "Unable to decode response from model. URL: {}, Status: {}, Headers: {}, Response: {}".format(
                    self.endpoint, response.status, response.info(), response.data
                )
            )
            logger.exception(err)

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
