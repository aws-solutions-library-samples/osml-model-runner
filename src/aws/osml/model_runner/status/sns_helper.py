#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
from typing import Any, Dict, Optional, Union

import boto3

from aws.osml.model_runner.app_config import BotoConfig

from .exceptions import SNSPublishException

logger = logging.getLogger(__name__)


class SNSHelper:
    """Encapsulates Amazon SNS topic functions."""

    def __init__(self, topic_arn: Optional[str]) -> None:
        """
        :param topic_arn: Optional[str] = the topic arn to publish to.

        :return: None
        """
        self.sns_client = boto3.client("sns", config=BotoConfig.default) if topic_arn is not None else None
        self.topic_arn = topic_arn

    def publish_message(self, message: str, attributes: Dict[str, Any]) -> Optional[str]:
        """
        Publishes a message, with attributes, to a topic. Subscriptions can be filtered
        based on message attributes so that a subscription receives messages only
        when specified attributes are present.

        :param message: str = the message to publish.
        :param attributes: Dict[str, Any] = the key-value attributes to attach to the message. Values
                           must be either `str` or `bytes`.

        :return: Optional[str] The ID of the message.
        """
        att_dict: Dict[str, Dict[str, Union[str, bytes]]] = dict()
        for key, value in attributes.items():
            if isinstance(value, str):
                att_dict[key] = {"DataType": "String", "StringValue": value}
            elif isinstance(value, bytes):
                att_dict[key] = {"DataType": "Binary", "BinaryValue": value}
        if self.sns_client is not None:
            try:
                response = self.sns_client.publish(TopicArn=self.topic_arn, Message=message, MessageAttributes=att_dict)
                logger.info("Published message with attributes %s to topic %s.", attributes, self.topic_arn)
                return response.get("MessageId")
            except Exception as err:
                raise SNSPublishException(f"Failed to publish message {message} to SNS topic {self.topic_arn}") from err
        else:
            logger.info("SNS disabled! Did not send message with attributes %s.", attributes)
            return None
