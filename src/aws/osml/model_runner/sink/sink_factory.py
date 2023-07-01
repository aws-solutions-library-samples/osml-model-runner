#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
from typing import Any, Dict, List

from aws.osml.model_runner.api import InvalidImageRequestException
from aws.osml.model_runner.sink import KinesisSink, S3Sink, Sink

logger = logging.getLogger(__name__)


class SinkFactory(Sink):
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
