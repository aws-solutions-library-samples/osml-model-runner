#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from enum import auto

from aws.osml.model_runner.common import AutoStringEnum


class ModelInvokeMode(str, AutoStringEnum):
    """
    Enumeration defining the hosting options for CV models.
    """

    NONE = auto()
    SM_ENDPOINT = auto()
    HTTP_ENDPOINT = auto()


VALID_MODEL_HOSTING_OPTIONS = [item.value for item in ModelInvokeMode]
