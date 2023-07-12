#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import sys
from typing import List, Optional

import boto3
import geojson
from geojson import Feature, FeatureCollection

from aws.osml.model_runner.api import SinkMode, SinkType
from aws.osml.model_runner.app_config import BotoConfig, ServiceConfig
from aws.osml.model_runner.common import get_credentials_for_assumed_role

from .sink import Sink

logger = logging.getLogger(__name__)


class KinesisSink(Sink):
    def __init__(
        self,
        stream: str,
        batch_size: int = None,
        assumed_role: Optional[str] = None,
    ) -> None:
        self.stream = stream
        self.batch_size = batch_size
        if assumed_role:
            assumed_credentials = get_credentials_for_assumed_role(assumed_role)
            # Here we will be writing to Kinesis using an IAM role other than the one for this process.
            self.kinesis_client = boto3.client(
                "kinesis",
                aws_access_key_id=assumed_credentials["AccessKeyId"],
                aws_secret_access_key=assumed_credentials["SecretAccessKey"],
                aws_session_token=assumed_credentials["SessionToken"],
                config=BotoConfig.default,
            )
        else:
            # If no invocation role is provided the assumption is that the default role for this
            # container will be sufficient to write to the Kinesis stream.
            self.kinesis_client = boto3.client("kinesis", config=BotoConfig.default)

    def _flush_stream(self, partition_key: str, features: List[Feature]) -> None:
        record = geojson.dumps(FeatureCollection(features))
        self.kinesis_client.put_record(
            StreamName=self.stream,
            PartitionKey=partition_key,
            Data=record,
        )

    @property
    def mode(self) -> SinkMode:
        # Only aggregate mode is supported at the moment
        return SinkMode.AGGREGATE

    def write(self, job_id: str, features: List[Feature]) -> bool:
        pending_features: List[Feature] = []
        pending_features_size: int = 0

        if self.validate_kinesis_stream():
            for feature in features:
                if self.batch_size == 1:
                    self._flush_stream(job_id, [feature])
                else:
                    feature_size = sys.getsizeof(geojson.dumps(feature))
                    if (
                        self.batch_size and pending_features and len(pending_features) % self.batch_size == 0
                    ) or pending_features_size + feature_size > (int(ServiceConfig.kinesis_max_record_size)):
                        self._flush_stream(job_id, pending_features)
                        pending_features = []
                        pending_features_size = 0

                    pending_features.append(feature)
                    pending_features_size += feature_size

            # Flush any remaining features
            if pending_features:
                self._flush_stream(job_id, pending_features)
            logger.info("Wrote {} features for job '{}' to Kinesis Stream '{}'".format(len(features), job_id, self.stream))
            return True
        else:
            logger.error(
                "Cannot write {} features for job '{}' to Kinesis Stream '{}'".format(len(features), job_id, self.stream)
            )
            return False

    def validate_kinesis_stream(self) -> bool:
        """
        Ensure output Kinesis stream exists/can be written to

        :return: bool = True if kinesis stream exist and can be read/written to it
        """
        try:
            describe_stream_response = self.kinesis_client.describe_stream(StreamName=self.stream)

            # check if Stream is ACTIVE
            stream_status = describe_stream_response["StreamDescription"]["StreamStatus"]
            if stream_status == "ACTIVE" or stream_status == "UPDATING":
                # reason to include UPDATING is that Kinesis Stream functions during these operations
                return True
            else:
                logging.error(
                    "{} current status is: {}. It is not in ACTIVE or UPDATING state.".format(self.stream, stream_status)
                )
                return False
        except Exception as e:
            logger.error("Failed to fetch Kinesis stream - {}. {}".format(self.stream, e))
            return False

    @staticmethod
    def name() -> str:
        return str(SinkType.KINESIS.value)
