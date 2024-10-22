#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import json
import logging
from typing import Any, Dict, List

from geojson import Feature

from aws.osml.model_runner.api import InvalidImageRequestException, SinkMode
from aws.osml.model_runner.sink import KinesisSink, S3Sink, Sink

logger = logging.getLogger(__name__)


class SinkFactory:
    """
    placeholder class as sink options grow to auto select and generator sinks
    """

    @staticmethod
    def outputs_to_sinks(destinations: List[Dict[str, Any]]) -> List[Sink]:
        """
        Generated a list of sink objects based on the data structure injected into
        from an Image Request item.

        :param destinations: List[Dict[str, Any]] = destination section of an Image Request as defined
                                in the data plane api.

        :return: List[Sink] = List of Sinks objects that can be iterated on based on an image object
        """
        outputs: List[Sink] = []
        for destination in destinations:
            sink_type = destination["type"]
            if sink_type == S3Sink.name():
                outputs.append(
                    S3Sink(
                        destination["bucket"],
                        destination["prefix"],
                        destination.get("role"),
                    )
                )
            elif sink_type == KinesisSink.name():
                outputs.append(
                    KinesisSink(
                        destination["stream"],
                        destination.get("batchSize"),
                        destination.get("assumedRole"),
                    )
                )
            else:
                error = f"Invalid Image Request! Unrecognized output destination specified, '{sink_type}'"
                logger.error(error)
                raise InvalidImageRequestException(error)
        return outputs

    @staticmethod
    def sink_features(job_id: str, outputs: str, features: List[Feature]) -> bool:
        """
        Writing the features output to S3 and/or Kinesis Stream

        :param job_id: str = unique identifier for the job
        :param outputs: str = details about the job output syncs
        :param features: List[Features] = the list of features to update

        :return: bool = if it has successfully written to an output sink
        """
        tracking_output_sinks = {
            "S3": False,
            "Kinesis": False,
        }  # format: job_id = {"s3": true, "kinesis": true}

        # Ensure we have outputs defined for where to dump our features
        if outputs:
            logging.debug(f"Writing aggregate feature for job '{job_id}'")
            for sink in SinkFactory.outputs_to_sinks(json.loads(outputs)):
                if sink.mode == SinkMode.AGGREGATE and job_id:
                    is_write_output_succeeded = sink.write(job_id, features)
                    tracking_output_sinks[sink.name()] = is_write_output_succeeded

            # Log them let them know if both written to both outputs (S3 and Kinesis) or one in another
            # If both couldn't write to either stream because both were down, return False. Otherwise True
            if tracking_output_sinks["S3"] and not tracking_output_sinks["Kinesis"]:
                logging.debug("ModelRunner was able to write the features to S3 but not Kinesis. Continuing...")
                return True
            elif not tracking_output_sinks["S3"] and tracking_output_sinks["Kinesis"]:
                logging.debug("ModelRunner was able to write the features to Kinesis but not S3. Continuing...")
                return True
            elif tracking_output_sinks["S3"] and tracking_output_sinks["Kinesis"]:
                logging.debug("ModelRunner was able to write the features to both S3 and Kinesis. Continuing...")
                return True
            else:
                logging.error("ModelRunner was not able to write the features to either S3 or Kinesis. Failing...")
                return False
        else:
            raise InvalidImageRequestException("No output destinations were defined for this image request!")
