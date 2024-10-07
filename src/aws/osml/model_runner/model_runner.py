#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging

from osgeo import gdal

from aws.osml.gdal import load_gdal_dataset, set_gdal_default_configuration

from .api import ImageRequest, InvalidImageRequestException, RegionRequest
from .common import EndpointUtils, ThreadingLocalContextFilter
from .config import ServiceConfig
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
        self.running = False

        # Pass dependencies into RegionRequestHandler
        self.region_request_handler = RegionRequestHandler(
            region_request_table=self.region_request_table,
            job_table=self.job_table,
            region_status_monitor=self.region_status_monitor,
            endpoint_statistics_table=self.endpoint_statistics_table,
            tiling_strategy=self.tiling_strategy,
            region_request_queue=self.region_request_queue,
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

    def run(self) -> None:
        """
        Starts ModelRunner in a loop that continuously monitors the image work queue and region work queue.
        """
        self.monitor_work_queues()

    def stop(self) -> None:
        """
        Stops ModelRunner by setting the global run variable to False.
        """
        self.running = False

    def monitor_work_queues(self) -> None:
        """
        Monitors SQS queues for ImageRequest and RegionRequest.
        """
        self.running = True
        set_gdal_default_configuration()

        try:
            while self.running:
                logger.debug("Checking work queue for regions to process ...")
                (receipt_handle, region_request_attributes) = next(self.region_requests_iter)
                ThreadingLocalContextFilter.set_context(region_request_attributes)

                if region_request_attributes is not None:
                    try:
                        region_request = RegionRequest(region_request_attributes)

                        if "s3:/" in region_request.image_url:
                            ImageRequest.validate_image_path(region_request.image_url, region_request.image_read_role)
                            image_path = region_request.image_url.replace("s3:/", "/vsis3", 1)
                        else:
                            image_path = region_request.image_url

                        raster_dataset, sensor_model = load_gdal_dataset(image_path)
                        image_format = str(raster_dataset.GetDriver().ShortName).upper()

                        region_request_item = self.region_request_table.get_region_request(
                            region_request.region_id, region_request.image_id
                        )
                        if region_request_item is None:
                            region_request_item = RegionRequestItem.from_region_request(region_request)

                        image_request_item = self.region_request_handler.process_region_request(
                            region_request, region_request_item, raster_dataset, sensor_model
                        )

                        if self.job_table.is_image_request_complete(image_request_item):
                            self.image_request_handler.complete_image_request(
                                region_request, image_format, raster_dataset, sensor_model
                            )

                        self.region_request_queue.finish_request(receipt_handle)
                    except RetryableJobException:
                        self.region_request_queue.reset_request(receipt_handle, visibility_timeout=0)
                    except SelfThrottledRegionException:
                        self.region_request_queue.reset_request(
                            receipt_handle, visibility_timeout=int(self.config.throttling_retry_timeout)
                        )
                    except Exception as err:
                        logger.error(f"There was a problem processing the region request: {err}")
                        self.region_request_queue.finish_request(receipt_handle)
                else:
                    logger.debug("Checking work queue for images to process ...")
                    (receipt_handle, image_request_message) = next(self.image_requests_iter)

                    if image_request_message is not None:
                        image_request = None
                        try:
                            image_request = ImageRequest.from_external_message(image_request_message)
                            ThreadingLocalContextFilter.set_context(image_request.__dict__)

                            if not image_request.is_valid():
                                error = f"Invalid image request: {image_request_message}"
                                logger.exception(error)
                                raise InvalidImageRequestException(error)

                            self.image_request_handler.process_image_request(image_request)
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
            self.running = False
