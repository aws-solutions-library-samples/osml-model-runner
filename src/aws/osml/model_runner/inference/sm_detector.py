#  Copyright 2023 Amazon.com, Inc. or its affiliates.

#  Copyright 2023 Amazon.com, Inc. or its affiliates.

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
from aws.osml.model_runner.app_config import BotoConfig, MetricLabels, ServiceConfig
from aws.osml.model_runner.common import Timer

from .detector import Detector
from .endpoint_builder import FeatureEndpointBuilder
from .feature_utils import create_mock_feature_collection

logger = logging.getLogger(__name__)


class SMDetector(Detector):
    def __init__(self, endpoint: str, assumed_credentials: Dict[str, str] = None) -> None:
        """
        Sagemaker model endpoint invoking object, intended to query sagemaker endpoints.

        :param endpoint: str = the name of the sagemaker endpoint that will be invoked
        :param assumed_credentials: Dict[str, str] = Optional credentials to invoke the sagemaker model

        :return: None
        """
        if assumed_credentials is not None:
            # Here we will be invoking the SageMaker endpoints using an IAM role other than the
            # one for this process. Use those credentials when creating the Boto3 SageMaker client.
            # This is the typical case when the SageMaker endpoints do not reside in the same AWS
            # account as the model runner.
            self.sm_client = boto3.client(
                "sagemaker-runtime",
                config=BotoConfig.sagemaker,
                aws_access_key_id=assumed_credentials.get("AccessKeyId"),
                aws_secret_access_key=assumed_credentials.get("SecretAccessKey"),
                aws_session_token=assumed_credentials.get("SessionToken"),
            )
        else:
            # If no invocation role is provided, the assumption is that the default role for this
            # container will be sufficient to invoke the SageMaker endpoints. This will typically
            # be the case for AWS managed models running in the same account as the model runner.
            self.sm_client = boto3.client("sagemaker-runtime", config=BotoConfig.sagemaker)
        super().__init__(endpoint=endpoint)

    @property
    def mode(self) -> ModelInvokeMode:
        return ModelInvokeMode.SM_ENDPOINT

    @metric_scope
    def find_features(self, payload: BufferedReader, metrics: MetricsLogger) -> FeatureCollection:
        """
        Query the established endpoint mode to find features based on a payload

        :param payload: BufferedReader = the BufferedReader object that holds the
                                    data that will be sent to the feature generator
        :param metrics: MetricsLogger = the metrics logger object to capture the log data on the system

        :return: FeatureCollection = a feature collection containing the center point of a tile
        """
        logger.info("Invoking Model: {}".format(self.endpoint))
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
                # If we are not running against a real model
                if self.endpoint == ServiceConfig.noop_bounds_model_name:
                    # We are expecting the body of the message to contain a geojson FeatureCollection
                    return create_mock_feature_collection(payload)
                elif self.endpoint == ServiceConfig.noop_geom_model_name:
                    return create_mock_feature_collection(payload, geom=True)
                else:
                    # Use the sagemaker model endpoint to invoke the model and return detection points
                    # as a geojson FeatureCollection
                    model_response = self.sm_client.invoke_endpoint(EndpointName=self.endpoint, Body=payload)
                    retry_count = model_response.get("ResponseMetadata", {}).get("RetryAttempts", 0)
                    if isinstance(metrics, MetricsLogger):
                        metrics.put_metric(MetricLabels.RETRIES, retry_count, str(Unit.COUNT.value))

                    # We are expecting the body of the message to contain a geojson FeatureCollection
                    return geojson.loads(model_response.get("Body").read())
        # If there was an error with the boto calls
        except ClientError as ce:
            self.error_count += 1
            error_code = ce.response.get("Error", {}).get("Code")
            http_status_code = ce.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error(
                "Unable to get detections from model - HTTP Status Code: {}, Error Code: {}".format(
                    http_status_code, error_code
                )
            )
            logger.exception(ce)
        # If there was an error parsing the models output
        except JSONDecodeError as de:
            self.error_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
            logger.error("Unable to decode response from model.")
            logger.exception(de)

        # Return an empty feature collection if the process errored out
        return FeatureCollection([])


class SMDetectorBuilder(FeatureEndpointBuilder):
    def __init__(self, endpoint: str, assumed_credentials: Dict[str, str] = None):
        """
        :param endpoint: The URL to the SageMaker endpoint
        :param assumed_credentials: The credentials to use with the SageMaker endpoint
        """
        super().__init__()
        self.endpoint = endpoint
        self.assumed_credentials = assumed_credentials

    def build(self) -> Optional[Detector]:
        """
        :return: a SageMaker detector based on the parameters defined during initialization
        """
        return SMDetector(
            endpoint=self.endpoint,
            assumed_credentials=self.assumed_credentials,
        )
