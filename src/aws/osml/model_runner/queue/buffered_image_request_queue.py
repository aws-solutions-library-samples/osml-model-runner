#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import dataclasses
import json
import logging
import time
from typing import List

import boto3
from botocore.exceptions import ClientError

from aws.osml.model_runner.api import ImageRequest
from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.database.requested_jobs_table import ImageRequestStatusRecord, RequestedJobsTable

logger = logging.getLogger(__name__)


class BufferedImageRequestQueue:
    """
    A queue that buffers image requests from SQS and manages them in DynamoDB.
    """

    def __init__(
        self,
        image_queue_url: str,
        image_dlq_url: str,
        requested_jobs_table: RequestedJobsTable,
        max_jobs_lookahead: int = 500,
        retry_time: int = 600,
        max_retry_attempts: int = 1,
    ):
        """
        Initialize the buffered image request queue.

        :param image_queue_url: name of the SQS queue containing image requests
        :param image_dlq_url: name of the SQS dead letter queue for problematic image requests
        :param requested_jobs_table: DynamoDB table for tracking request status
        :param max_jobs_lookahead: Maximum number of jobs to look ahead in the queue
        :param retry_time: Time in seconds before retrying failed requests
        :param max_retry_attempts: Maximum number of retry attempts for failed requests
        """
        self.requested_jobs_table = requested_jobs_table
        self.max_jobs_lookahead = max_jobs_lookahead
        self.retry_time = retry_time
        self.max_retry_attempts = max_retry_attempts

        self.sqs_client = boto3.client("sqs", config=BotoConfig.default)
        self.image_queue_url = image_queue_url
        self.image_dlq_url = image_dlq_url

        self.cw_client = boto3.client("cloudwatch", config=BotoConfig.default)

    def get_outstanding_requests(self) -> List[ImageRequestStatusRecord]:
        """
        Get the list of outstanding image requests by combining requests from the SQS queue
        and existing requests in the DynamoDB table.

        :return: List of outstanding image request status records
        """
        try:
            # Load the outstanding requests that have already been pulled off the queue
            outstanding_requests = self.requested_jobs_table.get_outstanding_requests()

            # Clean up any completed or failed requests
            outstanding_requests = self._purge_finished_requests(outstanding_requests)

            # If our buffer isn't at full capacity pull new messages from the queue
            # and store them in the buffer.
            if len(outstanding_requests) < self.max_jobs_lookahead:
                outstanding_requests.extend(
                    self._fetch_new_requests(max_messages_to_fetch=self.max_jobs_lookahead - len(outstanding_requests))
                )

            current_time = int(time.time())
            visible_requests = [
                request for request in outstanding_requests if request.last_attempt + self.retry_time < current_time
            ]

            # Output custom CW metric with size of outstanding requests list.
            self._emit_buffered_queue_metrics(
                num_buffered_requests=len(outstanding_requests), num_visible_requests=len(visible_requests)
            )

            return visible_requests
        except Exception as e:
            logger.error(f"Error getting outstanding requests: {e}")
            return []

    def _fetch_new_requests(self, max_messages_to_fetch: int) -> List[ImageRequestStatusRecord]:
        """
        Fetch new requests from the SQS queue and add them to DynamoDB.

        :return: List of new image request status records
        """
        outstanding_requests = []
        messages_to_fetch = max_messages_to_fetch
        while messages_to_fetch > 0:
            try:
                # Use the SQS batch read function to retrieve multiple image requests from the queue if possible.
                response = self.sqs_client.receive_message(
                    QueueUrl=self.image_queue_url,
                    MaxNumberOfMessages=min(10, messages_to_fetch),
                    WaitTimeSeconds=1,  # Use short polling for batch requests
                    AttributeNames=["All"],
                    MessageAttributeNames=["All"],
                )

                # If there are no messages available exit the loop
                messages = response.get("Messages", [])
                if not messages:
                    break

                for message in messages:
                    try:
                        # Attempt to create a valid ImageRequest from the queue body
                        message_body = message["Body"]
                        json_body = json.loads(message_body)
                        image_request = ImageRequest.from_external_message(json_body)
                        if not image_request.is_valid():
                            raise ValueError(f"Invalid image request: {message_body}")

                        # If we have a valid request move it from the SQS queue into our DDB table. The order of these
                        # operations is important to ensure we don't lose any requests. (i.e. ensure they are added
                        # to the table before we delete them from the queue).
                        request_status_record = self.requested_jobs_table.add_new_request(image_request)
                        self.sqs_client.delete_message(QueueUrl=self.image_queue_url, ReceiptHandle=message["ReceiptHandle"])
                        outstanding_requests.append(request_status_record)
                        messages_to_fetch -= 1

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.info(f"Invalid message received. Moving to DLQ. {e}")
                        self._handle_invalid_message(message)
                    except ClientError:
                        # In this case we were unable to either add the request to the DDB table or remove it from the
                        # image queue. The table add happens before the delete so if the request is not recorded it
                        # will be left on the SQS queue for a retry attempt. If the delete attempt failed it will appear
                        # just like a duplicate SQS message.
                        logger.error("Unable to move valid image request from input queue to DDB.", exc_info=True)

            except ClientError as e:
                logger.error(f"Error receiving messages from SQS: {e}")
                break

        return outstanding_requests

    def _handle_invalid_message(self, message: dict) -> None:
        """
        This is the error handling for an invalid or improperly formatted image request.
        We attempt to move it directly to the DLQ to avoid any unnecessary retries.

        :param message: The invalid SQS message
        """
        try:
            self.sqs_client.send_message(QueueUrl=self.image_dlq_url, MessageBody=message["Body"])
            self.sqs_client.delete_message(QueueUrl=self.image_queue_url, ReceiptHandle=message["ReceiptHandle"])
        except ClientError as ce:
            logger.error("Unable to move invalid image request from input queue to DLQ.")
            logger.exception(ce)

    def _purge_finished_requests(
        self, outstanding_requests: List[ImageRequestStatusRecord]
    ) -> List[ImageRequestStatusRecord]:
        """
        Remove completed requests and move failed requests to DLQ.

        :param outstanding_requests: List of current outstanding requests
        :return: Updated list of outstanding requests
        """
        current_outstanding_requests = []

        for request in outstanding_requests:
            try:
                if request.num_attempts >= self.max_retry_attempts:
                    # Move to DLQ if max retries exceeded
                    self.sqs_client.send_message(
                        QueueUrl=self.image_dlq_url, MessageBody=json.dumps(dataclasses.asdict(request.request_payload))
                    )
                    self.requested_jobs_table.complete_request(request.request_payload)
                elif request.region_count == len(request.regions_complete):
                    # Complete the request if all regions are processed
                    self.requested_jobs_table.complete_request(request.request_payload)
                else:
                    current_outstanding_requests.append(request)
            except ClientError:
                logger.error(f"Unable to cleanup outstanding request {request.job_id}", exc_info=True)

        return current_outstanding_requests

    def _emit_buffered_queue_metrics(self, num_buffered_requests: int, num_visible_requests: int) -> None:
        """
        Emit metrics about the number of buffered requests to CloudWatch.

        :param num_buffered_requests: The current number of requests in the buffer
        :param num_visible_requests: The current number of requests that are waiting to be processed
        """
        try:
            self.cw_client.put_metric_data(
                MetricData=[
                    {"MetricName": "ApproximateNumberOfRequestsBuffered", "Unit": "Count", "Value": num_buffered_requests},
                    {"MetricName": "ApproximateNumberOfRequestsVisible", "Unit": "Count", "Value": num_visible_requests},
                ],
                Namespace="OSML/ModelRunner/ImageRequestsQueue",
            )
        except ClientError as ce:
            error_code = ce.response["Error"]["Code"]
            error_message = ce.response["Error"]["Message"]
            logger.error(
                "Failed to emit CloudWatch metrics for buffered requests. "
                f"Error code: {error_code}, Message: {error_message}"
            )
        except Exception as e:
            logger.error(f"Unexpected error while emitting CloudWatch metrics for buffered requests: {e}", exc_info=True)
