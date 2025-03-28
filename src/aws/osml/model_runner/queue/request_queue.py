#  Copyright 2023-2025 Amazon.com, Inc. or its affiliates.

import json
import logging
from typing import Dict

import boto3
from botocore.exceptions import ClientError

from aws.osml.model_runner.app_config import BotoConfig

# Set up logging configuration
logger = logging.getLogger(__name__)


class RequestQueue:
    def __init__(
        self,
        queue_url: str,
        wait_seconds: int = 20,
        num_messages: int = 1,
    ) -> None:
        self.sqs_client = boto3.client("sqs", config=BotoConfig.default)
        self.queue_url = queue_url
        self.wait_seconds = wait_seconds
        self.num_messages = num_messages

    def __iter__(self):
        while True:
            try:
                queue_response = self.sqs_client.receive_message(
                    QueueUrl=self.queue_url,
                    AttributeNames=["All"],
                    MessageAttributeNames=["All"],
                    MaxNumberOfMessages=self.num_messages,
                    WaitTimeSeconds=self.wait_seconds,
                )

                logger.debug(f"Dequeued processing request {queue_response}")

                if "Messages" in queue_response:
                    for message in queue_response["Messages"]:
                        message_body = message["Body"]
                        logger.debug(f"Message Body {message_body}")

                        try:
                            work_request = json.loads(message_body)

                            yield message["ReceiptHandle"], work_request

                        except json.JSONDecodeError:
                            logger.warning(f"Skipping message that is not valid JSON: {message_body}")
                            yield None, None
                else:
                    yield None, None

            except ClientError as err:
                logger.error(f"Unable to retrieve message from queue: {err}")
                yield None, None

    def finish_request(self, receipt_handle: str) -> None:
        """
        Delete the message from the SQS

        :param receipt_handle: str = unique identifier of a message

        :return: None
        """
        try:
            # Remove the message from the queue since it has been successfully processed
            self.sqs_client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
        except ClientError as err:
            logger.error(f"Unable to remove message from queue: {err}")

    def reset_request(self, receipt_handle: str, visibility_timeout: int = 0) -> None:
        """
        Reset the message in SQS

        :param receipt_handle: str = unique identifier of a message
        :param visibility_timeout: int = period of time which SQS prevents other consumers from receiving
                                        and processing the message

        :return: None
        """
        try:
            self.sqs_client.change_message_visibility(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout,
            )
        except ClientError as err:
            logger.error(f"Unable to reset message visibility: {err}")

    def send_request(self, request: Dict) -> None:
        """
        Send the message via SQS

        :param request: Dict = unique identifier of a message

        :return: None
        """
        try:
            self.sqs_client.send_message(QueueUrl=self.queue_url, MessageBody=json.dumps(request))
        except ClientError as err:
            logger.error(f"Unable to send message visibility: {err}")
