#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import sys
from typing import List, Optional

import boto3
import geojson
from geojson import Feature, FeatureCollection

from aws.osml.model_runner.api import SinkMode, SinkType
from aws.osml.model_runner.app_config import BotoConfig, ServiceConfig
from aws.osml.model_runner.common import get_credentials_for_assumed_role

from .exceptions import InvalidKinesisStreamException
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

    def _flush_stream(self, records: List[dict]) -> None:
        """
        Flushes a batch of records to the Kinesis stream.

        :param records: A list of records to be sent to the Kinesis stream.
        :returns: None
        """
        try:
            self.kinesis_client.put_records(StreamName=self.stream, Records=records)
        except Exception as err:
            raise InvalidKinesisStreamException(f"Failed to write records to Kinesis stream '{self.stream}': {err}")

    @property
    def mode(self) -> SinkMode:
        # Only aggregate mode is supported at the moment
        return SinkMode.AGGREGATE

    def write(self, job_id: str, features: List[Feature]) -> bool:
        """
        Writes a list of features to the Kinesis stream. Each feature is serialized and sent
        as a record. If the batch of records exceeds the 5 MB limit, the current batch is flushed.

        :param job_id: The ID of the job associated with the features.
        :param features: A list of features to be written to the stream.

        :returns: True if the features were successfully written, False otherwise.
        """
        pending_features: List[dict] = []
        pending_features_size: int = 0

        if self.validate_kinesis_stream():
            for feature in features:
                # Serialize feature data to JSON
                record_data = geojson.dumps(FeatureCollection([feature]))

                # Create the record dict
                record = {"Data": record_data, "PartitionKey": job_id}

                # Calculate size of the entire record (Data + PartitionKey)
                record_size = sys.getsizeof(geojson.dumps(record))

                # If adding the next record would exceed the 5 MB batch limit, flush the current batch
                if pending_features_size + record_size > int(ServiceConfig.kinesis_max_record_size_batch) or len(
                    pending_features
                ) >= int(ServiceConfig.kinesis_max_record_per_batch):
                    self._flush_stream(pending_features)
                    pending_features = []
                    pending_features_size = 0

                pending_features.append(record)
                pending_features_size += record_size

            # Flush any remaining records
            if pending_features:
                self._flush_stream(pending_features)

            logger.info(f"Wrote {len(features)} features for job '{job_id}' to Kinesis Stream '{self.stream}'")
            return True
        else:
            logger.error(f"Cannot write {len(features)} features for job '{job_id}' to Kinesis Stream '{self.stream}'")
            return False

    def validate_kinesis_stream(self) -> bool:
        """
        Ensure output Kinesis stream exists/can be written to

        :return: True if kinesis stream exist and can be read/written to it
        """
        try:
            describe_stream_response = self.kinesis_client.describe_stream(StreamName=self.stream)

            # check if Stream is ACTIVE
            stream_status = describe_stream_response["StreamDescription"]["StreamStatus"]
            if stream_status == "ACTIVE" or stream_status == "UPDATING":
                # reason to include UPDATING is that Kinesis Stream functions during these operations
                return True
            else:
                logging.error(f"{self.stream} current status is: {stream_status}. It is not in ACTIVE or UPDATING state.")
                return False
        except Exception as err:
            logger.error(f"Failed to fetch Kinesis stream - {self.stream}. {err}")
            return False

    @staticmethod
    def name() -> str:
        """
        Ensure output Kinesis stream exists/can be written to

        :return: The name of the instantiated Sink.
        """
        return str(SinkType.KINESIS.value)
