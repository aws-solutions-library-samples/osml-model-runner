#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from aws.osml.model_runner.api import ImageRequest
from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.database.dataclass_ddb_mixin import DataclassDDBMixin

logger = logging.getLogger(__name__)


@dataclass
class ImageRequestStatusRecord(DataclassDDBMixin):
    """
    Represents the status of an image processing request.

    This class tracks the state of image processing jobs, including their progress,
    attempt counts, and completion status. It includes behaviors from the DataclassDDBMixin
    allowing it to be easily converted to and from an Item dictionary compatible with
    DynamoDB.

    :param endpoint_id: The identifier of the model endpoint processing the request
    :param job_id: The unique identifier for this processing job
    :param request_time: Unix timestamp when the request was created
    :param request_payload: The original image processing request
    :param last_attempt: Unix timestamp of the last processing attempt
    :param num_attempts: Number of times this job has been attempted
    :param regions_complete: List of regions that have completed processing
    :param region_count: Total number of regions to process (optional)
    """

    endpoint_id: str
    job_id: str
    request_time: int
    request_payload: ImageRequest
    last_attempt: int
    num_attempts: int
    regions_complete: List[str]
    region_count: Optional[int] = None

    @classmethod
    def new_from_request(cls, image_request: ImageRequest) -> "ImageRequestStatusRecord":
        """
        Create a new status record from an image request.

        :param image_request: The image processing request to create a status record for
        :return: A new ImageRequestStatusRecord instance
        """
        return cls(
            endpoint_id=image_request.model_name,
            job_id=image_request.job_id,
            request_time=int(time.time()),
            request_payload=image_request,
            last_attempt=0,
            num_attempts=0,
            regions_complete=[],
            region_count=None,
        )


class RequestedJobsTable:
    """
    Manages a collection of outstanding image processing job requests in DynamoDB.

    This class provides methods to track and manage the lifecycle of image processing jobs, including creation,
    updates, and completion. It is intended to only keep requests that are either pending execution or in progress.
    It does not keep a history of past requests as they are removed once completed (either successfully or with errors).

    Note that some operations on this class, e.g. get_outstanding_requests, perform a full table scan. It is
    essential that users of this table ensure there is some reasonable limit to the number of requests stored.
    The expectation is that the system is only working on some relatively small number of images at any one time
    (likely 10s, at most 1000 requests). If that assumption doesn't hold true in the future we will need to expand
    this class to have a secondary index based on request time and offer protections to only scan through a limited
    number of requests.
    """

    def __init__(self, table_name: str):
        """
        Initialize the RequestedJobsTable.

        :param table_name: Name of the DynamoDB table to used to track outstanding requests
        """
        self.table_name = table_name
        self.client = boto3.resource("dynamodb", config=BotoConfig.ddb)
        self.table = self.client.Table(table_name)

    def add_new_request(self, image_request: ImageRequest) -> ImageRequestStatusRecord:
        """
        Add a new status record to the table based on the image request.

        :param image_request: The image processing request to add
        :return: The status record for the image processing request
        :raises ClientError: If there is an error adding the request to DynamoDB
        """
        logger.debug(f"Adding ImageRequest for {image_request.job_id} to job status table.")
        try:
            request_status_record = ImageRequestStatusRecord.new_from_request(image_request)
            self.table.put_item(Item=request_status_record.to_ddb_item())
            return request_status_record
        except ClientError as ce:
            logger.error(f"Unable to add ImageRequest {image_request.job_id} to job status table.")
            logger.exception(ce)
            raise

    def update_request_details(self, image_request: ImageRequest, region_count: int) -> None:
        """
        Update the region count for an image request.

        :param image_request: The image processing request to update
        :param region_count: The total number of regions to process
        :raises ClientError: If there is an unexpected error updating DynamoDB
        """
        logger.debug(f"Updating region count to {region_count} for job {image_request.job_id}")
        try:
            self.table.update_item(
                Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id},
                UpdateExpression="SET region_count = :count",
                ConditionExpression="attribute_exists(job_id)",
                ExpressionAttributeValues={":count": region_count},
                ReturnValues="UPDATED_NEW",
            )
            logger.debug(f"Successfully updated region count for job {image_request.job_id}")
        except ClientError as ce:
            logger.error(
                f"Unable to update region count in job status table for {image_request.job_id}. "
                f"Failed to set count to {region_count}"
            )
            logger.exception(ce)
            raise

    def get_outstanding_requests(self) -> List[ImageRequestStatusRecord]:
        """
        Retrieve all outstanding image processing requests.

        :return: List of all incomplete image processing requests
        :raises ClientError: If there is an error querying DynamoDB
        """
        logger.debug("Scanning job status table for outstanding ImageRequests.")
        try:
            response = self.table.scan(ConsistentRead=True)
            items = response.get("Items", [])

            # Handle pagination if there are more items
            while "LastEvaluatedKey" in response:
                response = self.table.scan(ConsistentRead=True, ExclusiveStartKey=response["LastEvaluatedKey"])
                items.extend(response.get("Items", []))
            logger.debug(f"Found {len(items)} outstanding requests.")

            # Convert DynamoDB items back to ImageRequestStatusRecord objects
            return [ImageRequestStatusRecord.from_ddb_item(item) for item in items]
        except ClientError as ce:
            logger.error("Unable to scan job status table for outstanding image requests.")
            logger.exception(ce)
            raise

    def start_next_attempt(self, request_status_record: ImageRequestStatusRecord) -> bool:
        """
        Start the next processing attempt for a request.

        Updates the attempt counter and timestamp for the given request. Uses conditional update to ensure only one
        worker processes the request.

        :param request_status_record: The request status record to update
        :return: True if the attempt was successfully started, False if another worker has already started this job
        :raises ClientError: If there is an unexpected error updating DynamoDB
        """
        logger.debug(f"Updating job status table for new attempt of {request_status_record.job_id}")
        try:
            current_time = int(time.time())
            self.table.update_item(
                Key={"endpoint_id": request_status_record.endpoint_id, "job_id": request_status_record.job_id},
                UpdateExpression="SET last_attempt = :time, num_attempts = num_attempts + :inc",
                ConditionExpression="num_attempts = :current_attempts",
                ExpressionAttributeValues={
                    ":time": current_time,
                    ":inc": 1,
                    ":current_attempts": request_status_record.num_attempts,
                },
                ReturnValues="UPDATED_NEW",
            )
            logger.debug(f"Successfully recorded new attempt for {request_status_record.job_id}")
            return True
        except ClientError as ce:
            if ce.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.debug(
                    f"Unable to update job status table for {request_status_record.job_id}. "
                    "Another worker got to it first."
                )
                return False
            else:
                logger.error(
                    f"Unable to update job status table for {request_status_record.job_id}. " "Unexpected DynamoDB Error"
                )
                logger.exception(ce)
                raise

    def complete_request(self, image_request: ImageRequest):
        """
        Remove an image processing request from the table.

        :param image_request: The completed image request to remove
        :raises ClientError: If there is an error removing the item from DynamoDB
        """
        logger.debug(f"Removing {image_request.job_id} from job status table")
        try:
            self.table.delete_item(Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id})
        except ClientError as ce:
            logger.error(f"Unable to remove {image_request.job_id} from job status table.")
            logger.exception(ce)
            raise

    def complete_region(self, image_request: ImageRequest, region_id: str) -> bool:
        """
        Update the status record to mark a region as complete.
        Only adds the region if it's not already in the regions_complete list.

        :param image_request: The image processing request being updated
        :param region_id: The identifier of the completed region
        :return: True if the region was added, False if it was already present
        :raises ClientError: If there is an unexpected error updating DynamoDB
        """
        logger.debug(f"Adding completed region {region_id} for job {image_request.job_id}")
        try:
            self.table.update_item(
                Key={"endpoint_id": image_request.model_name, "job_id": image_request.job_id},
                UpdateExpression="SET regions_complete = list_append(regions_complete, :region)",
                ConditionExpression="NOT contains(regions_complete, :region_value)",
                ExpressionAttributeValues={":region": [region_id], ":region_value": region_id},
                ReturnValues="UPDATED_NEW",
            )
            logger.debug(f"Successfully recorded completed region {region_id} for job {image_request.job_id}")
            return True
        except ClientError as ce:
            if ce.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.debug(f"Region {region_id} was already marked as complete for job {image_request.job_id}")
                return False
            logger.error(
                f"Unable to update job status table for {image_request.job_id}. "
                f"Failed to add completed region {region_id}"
            )
            logger.exception(ce)
            raise
