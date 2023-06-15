#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from typing import Dict

import boto3

from aws.osml.model_runner.app_config import BotoConfig

from .exceptions import InvalidAssumedRoleException

sts_client = boto3.client("sts", config=BotoConfig.default)


def get_credentials_for_assumed_role(assumed_role: str) -> Dict[str, str]:
    """
    Get the credential access based on the assumed role

    :param assumed_role: str = containing a formatted arn role

    :return: Dict[str, str] = a dict that contains access key id and various of credential info
    """
    try:
        assumed_invocation_role = sts_client.assume_role(RoleArn=assumed_role, RoleSessionName="AWSOversightMLModelRunner")
        return assumed_invocation_role["Credentials"]
    except Exception as err:
        raise InvalidAssumedRoleException(f"Cannot assume role based on provided ARN {assumed_role}") from err
