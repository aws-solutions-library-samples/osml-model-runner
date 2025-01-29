#  Copyright 2023-2025 Amazon.com, Inc. or its affiliates.

import logging

from osgeo import gdal

from aws.osml.gdal import load_gdal_dataset, set_gdal_default_configuration
from aws.osml.model_runner.api import get_image_path

from .api import ImageRequest, RegionRequest
from .app_config import ServiceConfig
from .common import EndpointUtils, ThreadingLocalContextFilter
from .database import EndpointStatisticsTable, JobItem, JobTable, RegionRequestItem, RegionRequestTable
from .exceptions import RetryableJobException, SelfThrottledRegionException
from .image_request_handler import ImageRequestHandler
from .queue import RequestQueue
from .region_request_handler import RegionRequestHandler
from .scheduler.fifo_image_scheduler import FIFOImageScheduler
from .status import ImageStatusMonitor, RegionStatusMonitor
from .tile_worker import TilingStrategy, VariableOverlapTilingStrategy

# Set up logging configuration
logger = logging.getLogger(__name__)
gdal.UseExceptions()


class ModelRunner:
    """
    Main class for operating the ModelRunner application. It monitors input queues for processing requests,
    decomposes the image into smaller regions and tiles, invokes an ML model on each tile, and aggregates
    the results into a single output, which can be sent to the configured output sinks.
    """

    def __init__(self, tiling_strategy: TilingStrategy = VariableOverlapTilingStrategy()) -> None:
        """
        Initialize a model runner with the injectable behaviors.

        :param tiling_strategy: Defines how a larger image will be broken into chunks for processing

        :return: None
        """
        self.config = ServiceConfig()
        self.tiling_strategy = tiling_strategy

        # Set up the job scheduler
        self.image_job_scheduler = FIFOImageScheduler(RequestQueue(self.config.image_queue, wait_seconds=0))

        # Set up internal queues and monitors
        self.region_request_queue = RequestQueue(self.config.region_queue, wait_seconds=10)
        self.region_requests_iter = iter(self.region_request_queue)

        # Set up tables and status monitors
        self.job_table = JobTable(self.config.job_table)
        self.region_request_table = RegionRequestTable(self.config.region_request_table)
        self.endpoint_statistics_table = EndpointStatisticsTable(self.config.endpoint_statistics_table)
        self.image_status_monitor = ImageStatusMonitor(self.config.image_status_topic)
        self.region_status_monitor = RegionStatusMonitor(self.config.region_status_topic)
        self.endpoint_utils = EndpointUtils()

        # Handlers for image and region processing
        self.region_request_handler = RegionRequestHandler(
            region_request_table=self.region_request_table,
            job_table=self.job_table,
            region_status_monitor=self.region_status_monitor,
            endpoint_statistics_table=self.endpoint_statistics_table,
            tiling_strategy=self.tiling_strategy,
            endpoint_utils=self.endpoint_utils,
            config=self.config,
        )
        self.image_request_handler = ImageRequestHandler(
            job_table=self.job_table,
            image_status_monitor=self.image_status_monitor,
            endpoint_statistics_table=self.endpoint_statistics_table,
            tiling_strategy=self.tiling_strategy,
            region_request_queue=self.region_request_queue,
            region_request_table=self.region_request_table,
            endpoint_utils=self.endpoint_utils,
            config=self.config,
            region_request_handler=self.region_request_handler,
        )

        self.running = False

    def run(self) -> None:
        """
        Start the ModelRunner to continuously monitor and process work queues.

        :return: None
        """
        logger.info("Starting ModelRunner")
        self.running = True
        self.monitor_work_queues()

    def stop(self) -> None:
        """
        Stop the ModelRunner.

        :return: None
        """
        logger.info("Stopping ModelRunner")
        self.running = False

    def monitor_work_queues(self) -> None:
        """
        Continuously monitors the SQS queues for RegionRequest and ImageRequest.
        :return: None
        """
        set_gdal_default_configuration()
        logger.info("Beginning monitoring request queues")
        while self.running:
            try:
                # If there are regions to process
                if not self._process_region_requests():
                    # Move along to the next image request if present
                    self._process_image_requests()
            except Exception as err:
                logger.error(f"Unexpected error in monitor_work_queues: {err}")
                self.running = False
        logger.info("Stopped monitoring request queues")

    def _process_region_requests(self) -> bool:
        """
        Process messages from the region request queue.

        :return: True if a region request was processed, False if not.
        """
        logger.debug("Checking work queue for regions to process...")
        try:
            receipt_handle, region_request_attributes = next(self.region_requests_iter)
        except StopIteration:
            # No region requests available in the queue
            logger.debug("No region requests available to process.")
            return False

        if region_request_attributes:
            ThreadingLocalContextFilter.set_context(region_request_attributes)
            try:
                region_request = RegionRequest(region_request_attributes)
                image_path = get_image_path(region_request.image_url, region_request.image_read_role)
                raster_dataset, sensor_model = load_gdal_dataset(image_path)
                region_request_item = self._get_or_create_region_request_item(region_request)
                image_request_item = self.region_request_handler.process_region_request(
                    region_request, region_request_item, raster_dataset, sensor_model
                )
                if self.job_table.is_image_request_complete(image_request_item):
                    self.image_request_handler.complete_image_request(
                        region_request, str(raster_dataset.GetDriver().ShortName).upper(), raster_dataset, sensor_model
                    )
                self.region_request_queue.finish_request(receipt_handle)
            except RetryableJobException as err:
                logger.warning(f"Retrying region request due to: {err}")
                self.region_request_queue.reset_request(receipt_handle, visibility_timeout=0)
            except SelfThrottledRegionException as err:
                logger.warning(f"Retrying region request due to: {err}")
                self.region_request_queue.reset_request(
                    receipt_handle, visibility_timeout=int(self.config.throttling_retry_timeout)
                )
            except Exception as err:
                logger.exception(f"Error processing region request: {err}")
                self.region_request_queue.finish_request(receipt_handle)
            finally:
                return True
        else:
            return False

    def _process_image_requests(self) -> bool:
        """
        Processes messages from the image job scheduler.

        :return: True if an image request was processed, False if not.
        """
        image_request = self.image_job_scheduler.get_next_scheduled_request()
        if image_request:
            try:
                ThreadingLocalContextFilter.set_context(image_request.__dict__)
                logger.info(f"Starting processing for image request: {image_request.job_id}")
                self.image_request_handler.process_image_request(image_request)
                self.image_job_scheduler.finish_request(image_request)
            except RetryableJobException:
                self.image_job_scheduler.finish_request(image_request, should_retry=True)
            except Exception as err:
                logger.error(f"Error processing image request: {err}")
                self._fail_image_request(image_request, err)
                self.image_job_scheduler.finish_request(image_request)
            return True
        else:
            return False

    def _fail_image_request(self, image_request: ImageRequest, error: Exception) -> None:
        """
        Handles failing an image request by updating its status and logging the failure.

        This method is called when an image request cannot be processed due to an error.
        It marks the image request as failed and updates the job status using the
        `ImageRequestHandler`.

        :param image_request: The image request that failed to process.
        :param error: The exception that caused the failure.

        :return: None
        """
        min_image_id = image_request.image_id if image_request else ""
        min_job_id = image_request.job_id if image_request else ""
        minimal_job_item = JobItem(image_id=min_image_id, job_id=min_job_id, processing_duration=0)
        self.image_request_handler.fail_image_request(minimal_job_item, error)

    def _get_or_create_region_request_item(self, region_request: RegionRequest) -> RegionRequestItem:
        """
        Retrieves or creates a `RegionRequestItem` in the region request table.

        This method checks if a region request already exists in the `RegionRequestTable`.
        If it does, it retrieves the existing request; otherwise, it creates a new
        `RegionRequestItem` from the provided `RegionRequest` and starts the region
        processing.

        :param region_request: The region request to process.

        :return: The retrieved or newly created `RegionRequestItem`.
        """
        region_request_item = self.region_request_table.get_region_request(region_request.region_id, region_request.image_id)
        if region_request_item is None:
            region_request_item = RegionRequestItem.from_region_request(region_request)
            self.region_request_table.start_region_request(region_request_item)
            logger.debug(
                f"Added region request: image id {region_request_item.image_id}, region id {region_request_item.region_id}"
            )
        return region_request_item
