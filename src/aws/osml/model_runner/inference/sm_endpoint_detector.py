#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
from io import BufferedReader
from json import JSONDecodeError
from typing import Dict

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
from .feature_utils import create_mock_feature_collection

logger = logging.getLogger(__name__)


class SMDetector(Detector):
    def __init__(self, model_name: str, assumed_credentials: Dict[str, str] = None) -> None:
        """
        A sagemaker model endpoint invoking object, intended to query sagemaker endpoints.

        :param model_name: str = the name of the sagemaker endpoint that will be invoked
        :param assumed_credentials: Dict[str, str] = Optional credentials to invoke the sagemaker model

        :return: None
        """
        super().__init__()
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
            # If no invocation role is provided the assumption is that the default role for this
            # container will be sufficient to invoke the SageMaker endpoints. This will typically
            # be the case for AWS managed models running in the same account as the model runner.
            self.sm_client = boto3.client("sagemaker-runtime", config=BotoConfig.sagemaker)
        self.model_name = model_name
        self.request_count = 0
        self.error_count = 0

    @property
    def mode(self) -> ModelInvokeMode:
        return ModelInvokeMode.SM_ENDPOINT

    @metric_scope
    def find_features(self, payload: BufferedReader, metrics: MetricsLogger) -> FeatureCollection:
        """
        Query the established endpoint mode to find features based on a payload

        :param payload: BufferedReader = the BufferedReader object that holds the
                                    data that will be  sent to the feature generator
        :param metrics: MetricsLogger = the metrics logger object to capture the log data on the system

        :return: FeatureCollection = a feature collection containing the center point of a tile
        """
        logger.info("Invoking Model: {}".format(self.model_name))
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions({"ModelName": self.model_name})

        try:
            self.request_count += 1
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.MODEL_INVOCATION, 1, str(Unit.COUNT.value))

            with Timer(
                task_str="Invoke SM Endpoint",
                metric_name=MetricLabels.ENDPOINT_LATENCY,
                logger=logger,
                metrics_logger=metrics,
            ):
                # If we are not running against a real model
                if self.model_name == ServiceConfig.noop_model_name:
                    # We are expecting the body of the message to contain a geojson FeatureCollection
                    return create_mock_feature_collection(payload)
                else:
                    # Use the sagemaker model endpoint to invoke the model and return detection points
                    # as a geojson FeatureCollection
                    model_response = self.sm_client.invoke_endpoint(EndpointName=self.model_name, Body=payload)
                    retry_count = model_response.get("ResponseMetadata", {}).get("RetryAttempts", 0)
                    if isinstance(metrics, MetricsLogger):
                        metrics.put_metric(MetricLabels.ENDPOINT_RETRY_COUNT, retry_count, str(Unit.COUNT.value))

                    # We are expecting the body of the message to contain a geojson FeatureCollection
                    return geojson.loads(model_response.get("Body").read())
        # If there was an error with the boto calls
        except ClientError as ce:
            self.error_count += 1
            error_code = ce.response.get("Error", {}).get("Code")
            http_status_code = ce.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if isinstance(metrics, MetricsLogger):
                metrics.put_metric(MetricLabels.MODEL_ERROR, 1, str(Unit.COUNT.value))
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
                metrics.put_metric(MetricLabels.FEATURE_DECODE, 1, str(Unit.COUNT.value))
                metrics.put_metric(MetricLabels.MODEL_ERROR, 1, str(Unit.COUNT.value))
            logger.error("Unable to decode response from model.")
            logger.exception(de)

        # Return an empty feature collection if the process errored out
        return FeatureCollection([])
