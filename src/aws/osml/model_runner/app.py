#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging

from osgeo import gdal

from aws.osml.gdal import load_gdal_dataset, set_gdal_default_configuration

from .api import ImageRequest, InvalidImageRequestException, RegionRequest
from .app_config import ServiceConfig
from .common import EndpointUtils, ThreadingLocalContextFilter
from .database import EndpointStatisticsTable, JobItem, JobTable, RegionRequestItem, RegionRequestTable
from .exceptions import RetryableJobException, SelfThrottledRegionException
from .image_request_handler import ImageRequestHandler
from .queue import RequestQueue
from .region_request_handler import RegionRequestHandler
from .status import ImageStatusMonitor, RegionStatusMonitor
from .tile_worker import TilingStrategy, VariableOverlapTilingStrategy

# Set up logging configuration
logger = logging.getLogger(__name__)

# GDAL 4.0 will begin using exceptions as the default; at this point the software is written to assume
# no exceptions so we call this explicitly until the software can be updated to match.
gdal.UseExceptions()


class ModelRunner:
    """
    Main class for operating the ModelRunner application. It monitors input queues for processing requests,
    decomposes the image into a set of smaller regions and tiles, invokes an ML model endpoint with each tile, and
    finally aggregates all the results into a single output which can be deposited into the desired output sinks.
    """

    def __init__(self, tiling_strategy: TilingStrategy = VariableOverlapTilingStrategy()) -> None:
        """
        Initialize a model runner with the injectable behaviors.

        :param tiling_strategy: class defining how a larger image will be broken into chunks for processing
        """
        self.config = ServiceConfig()
        self.tiling_strategy = tiling_strategy
        self.image_request_queue = RequestQueue(self.config.image_queue, wait_seconds=0)
        self.image_requests_iter = iter(self.image_request_queue)
        self.job_table = JobTable(self.config.job_table)
        self.region_request_table = RegionRequestTable(self.config.region_request_table)
        self.endpoint_statistics_table = EndpointStatisticsTable(self.config.endpoint_statistics_table)
        self.region_request_queue = RequestQueue(self.config.region_queue, wait_seconds=10)
        self.region_requests_iter = iter(self.region_request_queue)
        self.image_status_monitor = ImageStatusMonitor(self.config.image_status_topic)
        self.region_status_monitor = RegionStatusMonitor(self.config.region_status_topic)
        self.endpoint_utils = EndpointUtils()
        # Pass dependencies into RegionRequestHandler
        self.region_request_handler = RegionRequestHandler(
            region_request_table=self.region_request_table,
            job_table=self.job_table,
            region_status_monitor=self.region_status_monitor,
            endpoint_statistics_table=self.endpoint_statistics_table,
            tiling_strategy=self.tiling_strategy,
            endpoint_utils=self.endpoint_utils,
            config=self.config,
        )
        # Pass dependencies into ImageRequestHandler
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
        Starts ModelRunner in a loop that continuously monitors the image work queue and region work queue.

        :return: None
        """
        self.monitor_work_queues()

    def stop(self) -> None:
        """
        Stops ModelRunner by setting the global run variable to False.

        :return: None
        """
        self.running = False

    def monitor_work_queues(self) -> None:
        """
        Monitors SQS queues for ImageRequest and RegionRequest The region work queue is checked first and will wait for
        up to 10 seconds to start work. Only if no regions need to be processed in that time will this worker check to
        see if a new image can be started. Ultimately this setup is intended to ensure that all the regions for an image
        are completed by the cluster before work begins on more images.

        :return: None
        """
        # Set the running state to True
        self.running = True

        # Set up the GDAL configuration options that should remain unchanged for the life of this execution
        set_gdal_default_configuration()
        try:
            while self.running:
                logger.debug("Checking work queue for regions to process ...")
                (receipt_handle, region_request_attributes) = next(self.region_requests_iter)
                ThreadingLocalContextFilter.set_context(region_request_attributes)

                # If we found a region request on the queue
                if region_request_attributes is not None:
                    try:
                        # Parse the message into a working RegionRequest
                        region_request = RegionRequest(region_request_attributes)

                        # If the image request has a s3 url lets augment its path for virtual hosting
                        if "s3:/" in region_request.image_url:
                            # Validate that image exists in S3
                            ImageRequest.validate_image_path(region_request.image_url, region_request.image_read_role)
                            image_path = region_request.image_url.replace("s3:/", "/vsis3", 1)
                        else:
                            image_path = region_request.image_url

                        # Load the image into a GDAL dataset
                        raster_dataset, sensor_model = load_gdal_dataset(image_path)
                        image_format = str(raster_dataset.GetDriver().ShortName).upper()

                        # Get RegionRequestItem if not create new RegionRequestItem
                        region_request_item = self.region_request_table.get_region_request(
                            region_request.region_id, region_request.image_id
                        )
                        if region_request_item is None:
                            # Create a new item from the region request
                            region_request_item = RegionRequestItem.from_region_request(region_request)

                            # Add the item to the table and start it processing
                            self.region_request_table.start_region_request(region_request_item)
                            logging.debug(
                                (
                                    f"Adding region request: image id: {region_request_item.image_id} - "
                                    f"region id: {region_request_item.region_id}"
                                )
                            )

                        # Process our region request
                        image_request_item = self.region_request_handler.process_region_request(
                            region_request, region_request_item, raster_dataset, sensor_model
                        )

                        # Check if the image is complete
                        if self.job_table.is_image_request_complete(image_request_item):
                            # If so complete the image request
                            self.image_request_handler.complete_image_request(
                                region_request, image_format, raster_dataset, sensor_model
                            )

                        # Update the queue
                        self.region_request_queue.finish_request(receipt_handle)
                    except RetryableJobException:
                        self.region_request_queue.reset_request(receipt_handle, visibility_timeout=0)
                    except SelfThrottledRegionException:
                        self.region_request_queue.reset_request(
                            receipt_handle,
                            visibility_timeout=int(self.config.throttling_retry_timeout),
                        )
                    except Exception as err:
                        logger.error(f"There was a problem processing the region request: {err}")
                        self.region_request_queue.finish_request(receipt_handle)
                else:
                    logger.debug("Checking work queue for images to process ...")
                    (receipt_handle, image_request_message) = next(self.image_requests_iter)

                    # If we found a request on the queue
                    if image_request_message is not None:
                        image_request = None
                        try:
                            # Parse the message into a working ImageRequest
                            image_request = ImageRequest.from_external_message(image_request_message)
                            ThreadingLocalContextFilter.set_context(image_request.__dict__)

                            # Check that our image request looks good
                            if not image_request.is_valid():
                                error = f"Invalid image request: {image_request_message}"
                                logger.exception(error)
                                raise InvalidImageRequestException(error)

                            # Process the request
                            self.image_request_handler.process_image_request(image_request)

                            # Update the queue
                            self.image_request_queue.finish_request(receipt_handle)
                        except RetryableJobException:
                            self.image_request_queue.reset_request(receipt_handle, visibility_timeout=0)
                        except Exception as err:
                            logger.error(f"There was a problem processing the image request: {err}")
                            min_image_id = image_request.image_id if image_request and image_request.image_id else ""
                            min_job_id = image_request.job_id if image_request and image_request.job_id else ""
                            minimal_job_item = JobItem(
                                image_id=min_image_id,
                                job_id=min_job_id,
                                processing_duration=0,
                            )
                            self.image_request_handler.fail_image_request(minimal_job_item, err)
                            self.image_request_queue.finish_request(receipt_handle)
        finally:
            # If we stop monitoring the queue set run state to false
            self.running = False
