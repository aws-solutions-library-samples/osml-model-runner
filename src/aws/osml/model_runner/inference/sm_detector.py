#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
from io import BufferedReader
from json import JSONDecodeError
from typing import Dict, Optional

import boto3
import geojson
from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from botocore.exceptions import ClientError
from geojson import FeatureCollection

from aws.osml.model_runner.api import ModelInvokeMode
from aws.osml.model_runner.app_config import BotoConfig, MetricLabels
from aws.osml.model_runner.common import Timer

from .detector import Detector
from .endpoint_builder import FeatureEndpointBuilder

logger = logging.getLogger(__name__)


class SMDetector(Detector):
    """
    SMDetector is responsible for invoking SageMaker endpoints to run model inference for feature detection.

    This class interacts with SageMaker runtime to send a payload for model inference and receive geojson-formatted
    feature detection results. It supports both managed AWS endpoints and cross-account invocation using
    provided IAM credentials.
    """

    def __init__(self, endpoint: str, assumed_credentials: Dict[str, str] = None) -> None:
        """
        Initializes the SMDetector with the SageMaker endpoint and optional credentials.

        :param endpoint: str = The name of the SageMaker endpoint to invoke.
        :param assumed_credentials: Dict[str, str] = Optional credentials for invoking the SageMaker model.
        """
        if assumed_credentials is not None:
            # Use the provided credentials to invoke SageMaker endpoints in another AWS account.
            self.sm_client = boto3.client(
                "sagemaker-runtime",
                config=BotoConfig.sagemaker,
                aws_access_key_id=assumed_credentials.get("AccessKeyId"),
                aws_secret_access_key=assumed_credentials.get("SecretAccessKey"),
                aws_session_token=assumed_credentials.get("SessionToken"),
            )
        else:
            # Use the default role for this container if no specific credentials are provided.
            self.sm_client = boto3.client("sagemaker-runtime", config=BotoConfig.sagemaker)
        super().__init__(endpoint=endpoint)

    @property
    def mode(self) -> ModelInvokeMode:
        """
        Defines the invocation mode for the detector as SageMaker endpoint.

        :return: ModelInvokeMode.SM_ENDPOINT
        """
        return ModelInvokeMode.SM_ENDPOINT

    @metric_scope
    def find_features(self, payload: BufferedReader, metrics: MetricsLogger) -> FeatureCollection:
        """
        Invokes the SageMaker model endpoint to detect features from the given payload.

        This method sends a payload to the SageMaker model endpoint and retrieves feature detection results
        in the form of a geojson FeatureCollection. If configured, it logs metrics about the invocation process.

        :param payload: BufferedReader = The data to be sent to the SageMaker model for feature detection.
        :param metrics: MetricsLogger = The metrics logger to capture system performance and log metrics.

        :return: FeatureCollection = A geojson FeatureCollection containing the detected features.

        :raises ClientError: Raised if there is an error while invoking the SageMaker endpoint.
        :raises JSONDecodeError: Raised if there is an error decoding the model's response.
        """
        logger.debug(f"Invoking Model: {self.endpoint}")
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.MODEL_INVOCATION_OPERATION,
                    MetricLabels.MODEL_NAME_DIMENSION: self.endpoint,
                }
            )

        try:
            self.request_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

            with Timer(
                task_str="Invoke SM Endpoint",
                metric_name=MetricLabels.DURATION,
                logger=logger,
                metrics_logger=metrics,
            ):
                # Invoke the real SageMaker model endpoint
                model_response = self.sm_client.invoke_endpoint(EndpointName=self.endpoint, Body=payload)
                retry_count = model_response.get("ResponseMetadata", {}).get("RetryAttempts", 0)
                if isinstance(metrics, MetricsLogger):
                    metrics.put_metric(MetricLabels.RETRIES, retry_count, str(Unit.COUNT.value))

                # Parse the model's response as a geojson FeatureCollection
                return geojson.loads(model_response.get("Body").read())

        except ClientError as ce:
            error_code = ce.response.get("Error", {}).get("Code")
            http_status_code = ce.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error(
                f"Unable to get detections from model - HTTP Status Code: {http_status_code}, Error Code: {error_code}"
            )
            logger.exception(ce)
            raise ce
        except JSONDecodeError as de:
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error("Unable to decode response from model.")
            logger.exception(de)
            raise de


class SMDetectorBuilder(FeatureEndpointBuilder):
    """
    SMDetectorBuilder is responsible for building an SMDetector configured with a SageMaker endpoint.

    This builder constructs an SMDetector instance with an optional set of assumed credentials for cross-account
    invocation of SageMaker models.
    """

    def __init__(self, endpoint: str, assumed_credentials: Dict[str, str] = None):
        """
        Initializes the SMDetectorBuilder with the SageMaker endpoint and optional credentials.

        :param endpoint: str = The name of the SageMaker endpoint to be used.
        :param assumed_credentials: Dict[str, str] = Optional credentials to use with the SageMaker endpoint.
        """
        super().__init__()
        self.endpoint = endpoint
        self.assumed_credentials = assumed_credentials

    def build(self) -> Optional[Detector]:
        """
        Builds and returns an SMDetector based on the configured parameters.

        :return: Optional[Detector] = An SMDetector instance configured for the specified SageMaker endpoint.
        """
        return SMDetector(
            endpoint=self.endpoint,
            assumed_credentials=self.assumed_credentials,
        )
