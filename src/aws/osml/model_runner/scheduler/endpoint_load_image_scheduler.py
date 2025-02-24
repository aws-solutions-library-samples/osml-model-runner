#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import logging
from dataclasses import dataclass
from itertools import groupby
from operator import attrgetter
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from aws.osml.model_runner.api import ImageRequest
from aws.osml.model_runner.app_config import BotoConfig
from aws.osml.model_runner.database import ImageRequestStatusRecord
from aws.osml.model_runner.queue import BufferedImageRequestQueue
from aws.osml.model_runner.scheduler.image_scheduler import ImageScheduler

logger = logging.getLogger(__name__)


@dataclass
class EndpointUtilizationSummary:
    """
    Tracks the load information for a SageMaker endpoint.

    :param endpoint_id: The identifier of the endpoint
    :param instance_count: Number of instances backing the endpoint
    :param current_load: Number of images currently being processed
    :param requests: List of pending requests for this endpoint
    """

    endpoint_id: str
    instance_count: int
    current_load: int
    requests: List[ImageRequestStatusRecord]

    @property
    def load_factor(self) -> float:
        """
        Calculate the load factor for this endpoint which is just the ratio of load per endpoint instance.

        :return: The load factor (current_load / instance_count)
        :rtype: float
        """
        return self.current_load / max(1, self.instance_count)


class EndpointLoadImageScheduler(ImageScheduler):
    """
    This class prioritizes image jobs that will make requests against the least utilized model endpoint.

    It does this by using a buffered request queue that will allow us to look ahead some number of requests
    and then pick the oldest request for the endpoint currently processing the fewest number of regions.
    """

    def __init__(self, image_request_queue: BufferedImageRequestQueue):
        """
        Initialize the load based image scheduler.

        :param image_request_queue: a request queue that buffers messages to enable lookahead
        """
        self.image_request_queue = image_request_queue
        self.sm_client = boto3.client("sagemaker", config=BotoConfig.default)

    def get_next_scheduled_request(self) -> Optional[ImageRequest]:
        """
        Get the image request for the endpoint with the lowest load.

        :return: The next image request to process, if any
        """
        try:
            logger.debug("Starting image processing request selection process")
            outstanding_requests = self.image_request_queue.get_outstanding_requests()
            if not outstanding_requests:
                logger.debug("No image processing request available to start")
                return None
            logger.debug(f"Retrieved {len(outstanding_requests)} image processing requests from the buffered queue")

            # Group requests by endpoint and calculate loads
            grouped_requests = self._group_requests_by_endpoint(outstanding_requests)
            endpoint_utilization = self._calculate_endpoint_utilization(grouped_requests)

            if logger.isEnabledFor(logging.DEBUG):
                utilization_rows = []
                for endpoint_summary in endpoint_utilization:
                    utilization_rows.append(
                        f"\t{endpoint_summary.endpoint_id}"
                        f"\t{endpoint_summary.instance_count}"
                        f"\t{len(endpoint_summary.requests)}"
                        f"\t{endpoint_summary.current_load}"
                    )
                utilization_str = "\n".join(utilization_rows)
                logger.debug(
                    "Current Endpoint Utilization:\n"
                    "  \tEndpoint\tInstance Count\tRequest Count\tCurrent Load\n"
                    f"  {utilization_str}"
                )

            # Find next eligible request
            next_request = self._select_next_eligible_request(endpoint_utilization)
            if not next_request:
                logger.debug("No outstanding requests are eligible to start. Stopping scheduling cycle.")
                return None
            logger.debug(f"Selected job {next_request.job_id} requested at {next_request.request_time} for processing.")

            # Try to start the next attempt. If the attempt can't be started that usually means the conditional
            # update of the record failed because another worker started the same request. In that case we
            # do not return an image processing request because we want this worker to go check the region
            # queue before starting a new image.
            if self.image_request_queue.requested_jobs_table.start_next_attempt(next_request):
                logger.debug(f"Started selected job {next_request.job_id}. Attempt # {next_request.num_attempts +1}")
                return next_request.request_payload

            logger.debug(
                f"Unable to start selected job {next_request.job_id}. "
                "Request was likely started by another worker. Stopping scheduling cycle."
            )
            return None

        except Exception as e:
            logger.error(f"Error getting next scheduled request: {e}", exc_info=True)
            return None

    def finish_request(self, image_request: ImageRequest, should_retry: bool = False) -> None:
        """
        Complete processing of an image request.

        :param image_request: The completed image request
        :param should_retry: Whether the request should be retried
        """
        # Nothing to do here. The requests are fully managed by the buffered queue and do not need manual cleanup.
        # This is just a noop placeholder for the abstract method defined on the base class.
        pass

    def _get_endpoint_instance_count(self, endpoint_name: str) -> int:
        """
        Get the number of instances backing a SageMaker endpoint.

        :param endpoint_name: Name of the SageMaker endpoint
        :return: Number of instances backing the endpoint
        """
        try:
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            production_variant = response["ProductionVariants"][0]
            return production_variant.get("CurrentInstanceCount", 1)
        except ClientError as e:
            logger.error(f"Error describing endpoint {endpoint_name}: {e}")
            return 1  # Default to 1 instance if we can't get the count

    def _group_requests_by_endpoint(
        self, requests: List[ImageRequestStatusRecord]
    ) -> Dict[str, List[ImageRequestStatusRecord]]:
        """
        Group requests by their endpoint ID.

        :param requests: List of request records to group
        :return: Dictionary mapping endpoint IDs to lists of requests
        """
        sorted_requests = sorted(requests, key=attrgetter("endpoint_id"))
        return {endpoint_id: list(group) for endpoint_id, group in groupby(sorted_requests, key=attrgetter("endpoint_id"))}

    def _calculate_endpoint_utilization(
        self, grouped_requests: Dict[str, List[ImageRequestStatusRecord]]
    ) -> List[EndpointUtilizationSummary]:
        """
        Calculate load information for each endpoint.

        The load of an endpoint is estimated by counting up the total number of regions that still need to be
        processed for running requests against that endpoint. This is approximate because there is still some
        variance in size for regions but overall this heuristic recognizes that the whole image sizes will vary
        widely and larger images will place a more substantial load on the endpoints.

        :param grouped_requests: Requests grouped by endpoint ID
        :return: List of endpoint load information
        """
        endpoint_loads = []
        for endpoint_id, requests in grouped_requests.items():
            instance_count = self._get_endpoint_instance_count(endpoint_id)

            current_load = 0
            for r in requests:
                if r.region_count is None:
                    if r.last_attempt > 0:
                        # Attempt has started but we don't have a count of regions yet. Assume it is 1
                        current_load += 1
                else:
                    current_load += r.region_count - len(r.regions_complete)

            endpoint_loads.append(
                EndpointUtilizationSummary(
                    endpoint_id=endpoint_id, instance_count=instance_count, current_load=current_load, requests=requests
                )
            )
        return endpoint_loads

    def _select_next_eligible_request(
        self, endpoint_loads: List[EndpointUtilizationSummary]
    ) -> Optional[ImageRequestStatusRecord]:
        """
        Find the next eligible request to process.

        :param endpoint_loads: List of endpoint load information
        :return: The next request to process, if any
        """
        oldest_request = None
        last_load = None
        for endpoint_load in sorted(endpoint_loads, key=lambda x: x.load_factor):
            if last_load is not None and endpoint_load.load_factor > last_load:
                break
            if endpoint_load.requests:
                current_oldest_request = min(endpoint_load.requests, key=attrgetter("request_time"))
                if oldest_request is None or oldest_request.request_time > current_oldest_request.request_time:
                    oldest_request = current_oldest_request
                    last_load = endpoint_load.load_factor

        return oldest_request
