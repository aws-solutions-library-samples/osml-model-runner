#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from enum import auto

from aws.osml.model_runner.common import AutoStringEnum


class SinkMode(str, AutoStringEnum):
    """
    Enumeration defining different sink output modes.
    """

    AGGREGATE = auto()
    STREAMING = auto()


class SinkType(str, AutoStringEnum):
    """
    Enumeration defining different sink output modes.
    """

    S3 = auto()
    # Mode not set to auto due to contract having been set as "Kinesis" already
    KINESIS = "Kinesis"
