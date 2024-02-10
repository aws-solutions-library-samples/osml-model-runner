#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import ast
import functools
import json
import logging
import math
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from json import dumps
from math import degrees
from secrets import token_hex
from typing import Any, Dict, List, Optional, Tuple

import shapely.geometry.base
from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from geojson import Feature
from osgeo import gdal
from osgeo.gdal import Dataset

from aws.osml.gdal import (
    GDALConfigEnv,
    GDALDigitalElevationModelTileFactory,
    get_image_extension,
    load_gdal_dataset,
    set_gdal_default_configuration,
)
from aws.osml.photogrammetry import DigitalElevationModel, ElevationModel, ImageCoordinate, SensorModel, SRTMTileSet

from .api import VALID_MODEL_HOSTING_OPTIONS, ImageRequest, InvalidImageRequestException, RegionRequest, SinkMode
from .app_config import MetricLabels, ServiceConfig
from .common import (
    EndpointUtils,
    FeatureDistillationDeserializer,
    GeojsonDetectionField,
    ImageDimensions,
    ImageRegion,
    ImageRequestStatus,
    RegionRequestStatus,
    ThreadingLocalContextFilter,
    Timer,
    build_embedded_metrics_config,
    get_credentials_for_assumed_role,
    mr_post_processing_options_factory,
)
from .database import EndpointStatisticsTable, FeatureTable, JobItem, JobTable, RegionRequestItem, RegionRequestTable
from .exceptions import (
    AggregateFeaturesException,
    AggregateOutputFeaturesException,
    InvalidFeaturePropertiesException,
    InvalidImageURLException,
    LoadImageException,
    ProcessImageException,
    ProcessRegionException,
    RetryableJobException,
    SelfThrottledRegionException,
    UnsupportedModelException,
)
from .inference import FeatureSelector, calculate_processing_bounds, get_source_property
from .queue import RequestQueue
from .sink import SinkFactory
from .status import StatusMonitor
from .tile_worker import generate_crops, process_tiles, setup_tile_workers

# Set up metrics configuration
build_embedded_metrics_config()

# Set up logging configuration
logger = logging.getLogger(__name__)


class ModelRunner:
    """
    Main class for operating the ModelRunner application. It monitors input queues for processing requests,
    decomposes the image into a set of smaller regions and tiles, invokes an ML model endpoint with each tile, and
    finally aggregates all the results into a single output which can be deposited into the desired output sinks.
    """

    def __init__(self) -> None:
        self.image_request_queue = RequestQueue(ServiceConfig.image_queue, wait_seconds=0)
        self.image_requests_iter = iter(self.image_request_queue)
        self.job_table = JobTable(ServiceConfig.job_table)
        self.region_request_table = RegionRequestTable(ServiceConfig.region_request_table)
        self.endpoint_statistics_table = EndpointStatisticsTable(ServiceConfig.endpoint_statistics_table)
        self.region_request_queue = RequestQueue(ServiceConfig.region_queue, wait_seconds=10)
        self.region_requests_iter = iter(self.region_request_queue)
        self.status_monitor = StatusMonitor()
        self.elevation_model = ModelRunner.create_elevation_model()
        self.endpoint_utils = EndpointUtils()
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

    @staticmethod
    def create_elevation_model() -> Optional[ElevationModel]:
        """
        Create an elevation model if the relevant options are set in the service configuration.

        :return: Optional[ElevationModel] = the elevation model or None if not configured
        """
        if ServiceConfig.elevation_data_location:
            return DigitalElevationModel(
                SRTMTileSet(
                    version=ServiceConfig.elevation_data_version,
                    format_extension=ServiceConfig.elevation_data_extension,
                ),
                GDALDigitalElevationModelTileFactory(ServiceConfig.elevation_data_location),
            )

        return None

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
                            region_pixel_bounds = f"{region_request.region_bounds[0]}{region_request.region_bounds[1]}"
                            region_request_item = RegionRequestItem(
                                image_id=region_request.image_id,
                                job_id=region_request.job_id,
                                region_pixel_bounds=region_pixel_bounds,
                                region_id=region_request.region_id,
                            )
                            self.region_request_table.start_region_request(region_request_item)
                            logging.info(
                                "Adding region request: imageid: {0} - regionid: {1}".format(
                                    region_request_item.image_id, region_request_item.region_id
                                )
                            )

                        # Process our region request
                        image_request_item = self.process_region_request(
                            region_request, region_request_item, raster_dataset, sensor_model
                        )

                        # Check if the image is complete
                        if self.job_table.is_image_request_complete(image_request_item):
                            # If so complete the image request
                            self.complete_image_request(region_request, image_format)

                        # Update the queue
                        self.region_request_queue.finish_request(receipt_handle)
                    except RetryableJobException:
                        self.region_request_queue.reset_request(receipt_handle, visibility_timeout=0)
                    except SelfThrottledRegionException:
                        self.region_request_queue.reset_request(
                            receipt_handle,
                            visibility_timeout=int(ServiceConfig.throttling_retry_timeout),
                        )
                    except Exception as err:
                        logger.error("There was a problem processing the region request: {}".format(err))
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
                                error = "Invalid image request: {}".format(image_request_message)
                                logger.exception(error)
                                raise InvalidImageRequestException(error)

                            # Process the request
                            self.process_image_request(image_request)

                            # Update the queue
                            self.image_request_queue.finish_request(receipt_handle)
                        except RetryableJobException:
                            self.image_request_queue.reset_request(receipt_handle, visibility_timeout=0)
                        except Exception as err:
                            logger.error("There was a problem processing the image request: {}".format(err))
                            min_image_id = image_request.image_id if image_request and image_request.image_id else ""
                            min_job_id = image_request.job_id if image_request and image_request.job_id else ""
                            min_job_arn = image_request.job_arn if image_request and image_request.job_arn else ""
                            minimal_job_item = JobItem(
                                image_id=min_image_id,
                                job_id=min_job_id,
                                job_arn=min_job_arn,
                                processing_time=Decimal(0),
                            )
                            self.fail_image_request_send_messages(minimal_job_item, err)
                            self.image_request_queue.finish_request(receipt_handle)
        finally:
            # If we stop monitoring the queue set run state to false
            self.running = False

    def process_image_request(self, image_request: ImageRequest) -> None:
        """
        Processes ImageRequest objects that are picked up from  queue. Loads the specified image into memory to be
        chipped apart into regions and sent downstream for processing via RegionRequest. This will also process the
        first region chipped from the image. # This worker will process the first region of this image since it has
        already loaded the dataset from S3 and is ready to go. Any additional regions will be queued for processing by
        other workers in this cluster.

        :param image_request: ImageRequest = the image request derived from the ImageRequest SQS message

        :return: None
        """
        image_request_item = None
        try:
            if ServiceConfig.self_throttling:
                max_regions = self.endpoint_utils.calculate_max_regions(
                    image_request.model_name, image_request.model_invocation_role
                )
                # Add entry to the endpoint statistics table
                self.endpoint_statistics_table.upsert_endpoint(image_request.model_name, max_regions)

            # Update the image status to started and include relevant image meta-data
            logger.info("Starting processing of {}".format(image_request.image_url))
            image_request_item = JobItem(
                image_id=image_request.image_id,
                job_id=image_request.job_id,
                job_arn=image_request.job_arn,
                tile_size=str(image_request.tile_size),
                tile_overlap=str(image_request.tile_overlap),
                model_name=image_request.model_name,
                model_invoke_mode=image_request.model_invoke_mode,
                outputs=dumps(image_request.outputs),
                image_url=image_request.image_url,
                image_read_role=image_request.image_read_role,
                feature_properties=dumps(image_request.feature_properties),
            )
            feature_distillation_option_list = image_request.get_feature_distillation_option()
            if feature_distillation_option_list:
                image_request_item.feature_distillation_option = dumps(
                    asdict(feature_distillation_option_list[0], dict_factory=mr_post_processing_options_factory)
                )

            # Start the image processing
            self.job_table.start_image_request(image_request_item)
            self.status_monitor.process_event(image_request_item, ImageRequestStatus.STARTED, "Started image request")

            # Check we have a valid image request, throws if not
            self.validate_model_hosting(image_request_item)

            # Load the relevant image meta data into memory
            image_extension, raster_dataset, sensor_model, all_regions = self.load_image_request(
                image_request_item, image_request.roi
            )

            if sensor_model is None:
                logging.warning(
                    "Dataset {} did not have a geo transform. Results are not geo-referenced.".format(
                        image_request_item.image_id
                    )
                )

            # If we got valid outputs
            if raster_dataset and all_regions and image_extension:
                image_request_item.region_count = Decimal(len(all_regions))
                image_request_item.width = Decimal(raster_dataset.RasterXSize)
                image_request_item.height = Decimal(raster_dataset.RasterYSize)
                try:
                    image_request_item.extents = json.dumps(ModelRunner.get_extents(raster_dataset, sensor_model))
                except Exception as e:
                    logger.warning(f"Could not get extents for image: {image_request_item.image_id}")
                    logger.exception(e)

                feature_properties: List[dict] = json.loads(image_request_item.feature_properties)

                # If we can get a valid source metadata from the source image - attach it to features
                # else, just pass in whatever custom features if they were provided
                source_metadata = get_source_property(image_extension, raster_dataset)
                if isinstance(source_metadata, dict):
                    feature_properties.append(source_metadata)

                # Update the feature properties
                image_request_item.feature_properties = json.dumps(feature_properties)

                # Update the image request job to have new derived image data
                image_request_item = self.job_table.update_image_request(image_request_item)

                self.status_monitor.process_event(image_request_item, ImageRequestStatus.IN_PROGRESS, "Processing regions")

                # Place the resulting region requests on the appropriate work queue
                self.queue_region_request(all_regions, image_request, raster_dataset, sensor_model, image_extension)

        except Exception as err:
            # We failed try and gracefully update our image request
            if image_request_item:
                self.fail_image_request(image_request_item, err)
            else:
                minimal_job_item = JobItem(
                    image_id=image_request.image_id,
                    job_id=image_request.job_id,
                    job_arn=image_request.job_arn,
                    processing_time=Decimal(0),
                )
                self.fail_image_request(minimal_job_item, err)

            # Let the application know that we failed to process image
            raise ProcessImageException("Failed to process image region!") from err

    def queue_region_request(
        self,
        all_regions: List[ImageRegion],
        image_request: ImageRequest,
        raster_dataset: Dataset,
        sensor_model: Optional[SensorModel],
        image_extension: Optional[str],
    ) -> None:
        """
        Loads the list of regions into the queue. First it will create a RequestRequestItem and creates
        an entry into the RegionRequestTable for traceability. Then process the region request. Once it's completed,
        it will update an entry in the RegionRequestTable.

        :param image_extension: = the GDAL derived image extension
        :param all_regions: List[ImageRegion] = the list of image regions
        :param image_request: ImageRequest = the image request
        :param raster_dataset: Dataset = the raster dataset containing the region
        :param sensor_model: Optional[SensorModel] = the sensor model for this raster dataset

        :return None
        """
        # Set aside the first region
        first_region = all_regions.pop(0)
        for region in all_regions:
            logger.info("Queueing region: {}".format(region))

            region_pixel_bounds = f"{region[0]}{region[1]}"
            region_id = f"{region_pixel_bounds}-{token_hex(16)}"
            region_request = RegionRequest(
                image_request.get_shared_values(), region_bounds=region, region_id=region_id, image_extension=image_extension
            )

            # Create a new entry to the region request being started
            region_request_item = RegionRequestItem(
                image_id=image_request.image_id,
                job_id=image_request.job_id,
                region_pixel_bounds=region_pixel_bounds,
                region_id=region_id,
            )
            self.region_request_table.start_region_request(region_request_item)
            logging.info(
                "Adding region request: imageid: {0} - regionid: {1}".format(
                    region_request_item.image_id, region_request_item.region_id
                )
            )

            # Send the attributes of the region request as the message.
            self.region_request_queue.send_request(region_request.__dict__)

        # Go ahead and process the first region
        logger.info("Processing first region {}: {}".format(0, first_region))

        region_pixel_bounds = f"{first_region[0]}{first_region[1]}"
        region_id = f"{region_pixel_bounds}-{token_hex(16)}"
        first_region_request = RegionRequest(
            image_request.get_shared_values(),
            region_bounds=first_region,
            region_id=region_id,
            image_extension=image_extension,
        )

        # Add item to RegionRequestTable
        region_request_item = RegionRequestItem(
            image_id=image_request.image_id,
            job_id=image_request.job_id,
            region_pixel_bounds=region_pixel_bounds,
            region_id=region_id,
        )
        self.region_request_table.start_region_request(region_request_item)
        logging.info(
            "Adding region request: imageid: {0} - regionid: {1}".format(
                region_request_item.image_id, region_request_item.region_id
            )
        )

        # Processes our region request and return the updated item
        image_request_item = self.process_region_request(
            first_region_request, region_request_item, raster_dataset, sensor_model
        )

        # If the image is finished then complete it
        if self.job_table.is_image_request_complete(image_request_item):
            image_format = str(raster_dataset.GetDriver().ShortName).upper()
            self.complete_image_request(first_region_request, image_format)

    @metric_scope
    def process_region_request(
        self,
        region_request: RegionRequest,
        region_request_item: RegionRequestItem,
        raster_dataset: gdal.Dataset,
        sensor_model: Optional[SensorModel] = None,
        metrics: MetricsLogger = None,
    ) -> JobItem:
        """
        Processes RegionRequest objects that are delegated for processing. Loads the specified region of an image into
        memory to be processed by tile-workers. If a raster_dataset is not provided directly it will poll the image
        from the region request.

        :param region_request: RegionRequest = the region request
        :param region_request_item: RegionRequestItem = the region request to update
        :param raster_dataset: gdal.Dataset = the raster dataset containing the region
        :param sensor_model: Optional[SensorModel] = the sensor model for this raster dataset
        :param metrics: MetricsLogger = the metrics logger to use to report metrics.

        :return: None
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()

        if not region_request.is_valid():
            logger.error("Invalid Region Request! {}".format(region_request.__dict__))
            raise ValueError("Invalid Region Request")

        if isinstance(metrics, MetricsLogger):
            image_format = str(raster_dataset.GetDriver().ShortName).upper()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.REGION_PROCESSING_OPERATION,
                    MetricLabels.MODEL_NAME_DIMENSION: region_request.model_name,
                    MetricLabels.INPUT_FORMAT_DIMENSION: image_format,
                }
            )

        if ServiceConfig.self_throttling:
            max_regions = self.endpoint_utils.calculate_max_regions(
                region_request.model_name, region_request.model_invocation_role
            )
            # Add entry to the endpoint statistics table
            self.endpoint_statistics_table.upsert_endpoint(region_request.model_name, max_regions)
            in_progress = self.endpoint_statistics_table.current_in_progress_regions(region_request.model_name)

            if in_progress >= max_regions:
                if isinstance(metrics, MetricsLogger):
                    metrics.put_metric(MetricLabels.THROTTLES, 1, str(Unit.COUNT.value))
                logger.info("Throttling region request. (Max: {} In-progress: {}".format(max_regions, in_progress))
                raise SelfThrottledRegionException

            # Increment the endpoint region counter
            self.endpoint_statistics_table.increment_region_count(region_request.model_name)

        try:
            with Timer(
                task_str="Processing region {} {}".format(region_request.image_url, region_request.region_bounds),
                metric_name=MetricLabels.DURATION,
                logger=logger,
                metrics_logger=metrics,
            ):
                # Set up our threaded tile worker pool
                tile_queue, tile_workers = setup_tile_workers(region_request, sensor_model, self.elevation_model)

                # Process all our tiles
                total_tile_count, tile_error_count = process_tiles(
                    region_request, tile_queue, tile_workers, raster_dataset, sensor_model
                )

                # Update table w/ total tile counts
                region_request_item.total_tiles = Decimal(total_tile_count)
                region_request_item = self.region_request_table.update_region_request(region_request_item)

            # Update the image request to complete this region
            image_request_item = self.job_table.complete_region_request(region_request.image_id, bool(tile_error_count))

            # Update region request table if that region succeeded
            region_request_item = self.region_request_table.complete_region_request(
                region_request_item, self.calculate_region_status(total_tile_count, tile_error_count)
            )

            # Write CloudWatch Metrics to the Logs
            if isinstance(metrics, MetricsLogger):
                # TODO: Consider adding the +1 invocation to timer
                metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

            # Return the updated item
            return image_request_item

        except Exception as err:
            logger.error("Failed to process image region: {}".format(err))
            # update the table to take in that exception
            region_request_item.message = "Failed to process image region: {0}".format(err)
            return self.fail_region_request(region_request_item, metrics)

        finally:
            # Decrement the endpoint region counter
            if ServiceConfig.self_throttling:
                self.endpoint_statistics_table.decrement_region_count(region_request.model_name)

    @staticmethod
    def load_image_request(
        image_request_item: JobItem,
        roi: shapely.geometry.base.BaseGeometry,
    ) -> Tuple[str, Dataset, Optional[SensorModel], List[ImageRegion]]:
        """
        Loads the required image file metadata into memory to be chipped apart into regions and
        distributed for region processing.

        :param image_request_item: JobItem = the region request to update.
        :param roi: BaseGeometry = the region of interest shape

        :return: Tuple[Queue, List[TileWorker]: A list of tile workers and the queue that manages them
        """
        # If this request contains an execution role retrieve credentials that will be used to
        # access data
        assumed_credentials = None
        if image_request_item.image_read_role:
            assumed_credentials = get_credentials_for_assumed_role(image_request_item.image_read_role)

        # This will update the GDAL configuration options to use the security credentials for this
        # request. Any GDAL managed AWS calls (i.e. incrementally fetching pixels from a dataset
        # stored in S3) within this "with" statement will be made using customer credentials. At
        # the end of the "with" scope the credentials will be removed.
        with GDALConfigEnv().with_aws_credentials(assumed_credentials):
            # Use GDAL to access the dataset and geo positioning metadata
            if not image_request_item.image_url:
                raise InvalidImageURLException("No image URL specified. Image URL is required.")

            # If the image request has a valid s3 image url, otherwise this is a local file
            if "s3:/" in image_request_item.image_url:
                # Validate that image exists in S3
                ImageRequest.validate_image_path(image_request_item.image_url, image_request_item.image_read_role)

                image_path = image_request_item.image_url.replace("s3:/", "/vsis3", 1)
            else:
                image_path = image_request_item.image_url

            # Use gdal to load the image url we were given
            raster_dataset, sensor_model = load_gdal_dataset(image_path)
            image_extension = get_image_extension(image_path)

            # Determine how much of this image should be processed.
            # Bounds are: UL corner (row, column) , dimensions (w, h)
            processing_bounds = calculate_processing_bounds(raster_dataset, roi, sensor_model)
            if not processing_bounds:
                logger.info("Requested ROI does not intersect image. Nothing to do")
                raise LoadImageException("Failed to create processing bounds for image!")
            else:
                # Calculate a set of ML engine-sized regions that we need to process for this image
                # Region size chosen to break large images into pieces that can be handled by a
                # single tile worker
                region_size: ImageDimensions = ast.literal_eval(ServiceConfig.region_size)
                if not image_request_item.tile_overlap:
                    region_overlap = (0, 0)
                else:
                    region_overlap = ast.literal_eval(image_request_item.tile_overlap)

                all_regions = generate_crops(processing_bounds, region_size, region_overlap)

        return image_extension, raster_dataset, sensor_model, all_regions

    def fail_image_request(self, image_request_item: JobItem, err: Exception) -> None:
        """
        Handles failure events/exceptions for image requests and tries to update the status monitor accordingly

        :param image_request_item: JobItem = the image request that failed.
        :param err: Exception = the exception that caused the failure

        :return: None
        """
        self.fail_image_request_send_messages(image_request_item, err)
        self.job_table.end_image_request(image_request_item.image_id)

    def fail_image_request_send_messages(self, image_request_item: JobItem, err: Exception) -> None:
        """
        Updates failed metrics and update the status monitor accordingly

        :param image_request_item: JobItem = the image request that failed.
        :param err: Exception = the exception that caused the failure

        :return: None
        """
        logger.exception("Failed to start image processing!: {}".format(err))
        self.status_monitor.process_event(image_request_item, ImageRequestStatus.FAILED, str(err))

    def complete_image_request(self, region_request: RegionRequest, image_format: str) -> None:
        """
        Runs after every region has completed processing to check if that was the last region and run required
        completion logic for the associated ImageRequest.

        :param region_request: RegionRequest = the region request to update.
        :param image_format: the format of the image
        :return: None
        """
        try:
            # Grab the full image request item from the table
            image_request_item = self.job_table.get_image_request(region_request.image_id)

            # Check if the image request is finished
            logger.info("Last region of image request was completed, aggregating features for image!")
            # Set up our feature table to work with the region quest
            feature_table = FeatureTable(ServiceConfig.feature_table, region_request.tile_size, region_request.tile_overlap)
            # Aggregate all the features from our job
            features = self.aggregate_features(image_request_item, feature_table)
            features = self.select_features(image_request_item, features)
            features = self.add_properties_to_features(image_request_item, features)

            # Sink the features into the right outputs
            is_write_succeeded = self.sync_features(image_request_item, features)
            if not is_write_succeeded:
                raise AggregateOutputFeaturesException(
                    "Failed to write features to S3 or Kinesis! Please check the " "log..."
                )

            # Put our end time on our image_request_item
            completed_image_request_item = self.job_table.end_image_request(image_request_item.image_id)

            # Ensure we have a valid start time for our record
            # TODO: Figure out why we wouldn't have a valid start time?!?!
            if completed_image_request_item.processing_time is not None:
                image_request_status = self.status_monitor.get_image_request_status(completed_image_request_item)
                self.status_monitor.process_event(
                    completed_image_request_item, image_request_status, "Completed image processing"
                )
                self.generate_image_processing_metrics(completed_image_request_item, image_format)
            else:
                raise InvalidImageRequestException("ImageRequest has no start time")

        except Exception as err:
            raise AggregateFeaturesException("Failed to aggregate features for region!") from err

    @metric_scope
    def generate_image_processing_metrics(
        self, image_request_item: JobItem, image_format: str, metrics: MetricsLogger = None
    ) -> None:
        """
        Output the metrics for the full image processing timeline.

        :param image_request_item: the completed image request item that tracks the duration and error counts
        :param image_format: the input image format
        :param metrics: the current metric scope
        """
        if not metrics:
            logger.warning("Unable to generate image processing metrics. Metrics logger is None!")
            return

        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.IMAGE_PROCESSING_OPERATION,
                    MetricLabels.MODEL_NAME_DIMENSION: image_request_item.model_name,
                    MetricLabels.INPUT_FORMAT_DIMENSION: image_format,
                }
            )

            metrics.put_metric(MetricLabels.DURATION, float(image_request_item.processing_time), str(Unit.SECONDS.value))
            metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))
            if image_request_item.region_error > 0:
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))

    def fail_region_request(
        self,
        region_request_item: RegionRequestItem,
        metrics: MetricsLogger = None,
    ) -> JobItem:
        """
        Fails a region if it failed to process successfully and updates the table accordingly before
        raising an exception

        :param region_request_item: RegionRequestItem = the region request to update
        :param metrics: MetricsLogger = the metrics logger to use to report metrics.

        :return: None
        """
        if isinstance(metrics, MetricsLogger):
            metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
        try:
            region_request_item = self.region_request_table.complete_region_request(
                region_request_item, RegionRequestStatus.FAILED
            )
            return self.job_table.complete_region_request(region_request_item.image_id, error=True)
        except Exception as status_error:
            logger.error("Unable to update region status in job table")
            logger.exception(status_error)
        raise ProcessRegionException("Failed to process image region!")

    def validate_model_hosting(self, image_request: JobItem):
        """
        Validates that the image request is valid. If not, raises an exception.

        :param image_request: JobItem = the image request

        :return: None
        """
        if not image_request.model_invoke_mode or image_request.model_invoke_mode not in VALID_MODEL_HOSTING_OPTIONS:
            error = f"Application only supports ${VALID_MODEL_HOSTING_OPTIONS} Endpoints"
            self.status_monitor.process_event(
                image_request,
                ImageRequestStatus.FAILED,
                error,
            )
            raise UnsupportedModelException(error)

    @staticmethod
    @metric_scope
    def aggregate_features(
        image_request_item: JobItem, feature_table: FeatureTable, metrics: MetricsLogger = None
    ) -> List[Feature]:
        """
        For a given image processing job - aggregate all the features that were collected for it and
        put them in the correct output sync locations.

        :param image_request_item: JobItem = the image request
        :param feature_table: FeatureTable = the table storing features from all completed regions
        :param metrics: the current metrics scope

        :return: List[geojson.Feature] = the list of features
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.FEATURE_AGG_OPERATION,
                }
            )

        with Timer(
            task_str="Aggregating Features", metric_name=MetricLabels.DURATION, logger=logger, metrics_logger=metrics
        ):
            features = feature_table.get_features(image_request_item.image_id)
            logger.info(f"Total features aggregated: {len(features)}")

        return features

    @staticmethod
    def identify_overlap(
        feature: Feature, shape: Tuple[int, int], overlap: Tuple[int, int], origin: Tuple[int, int] = (0, 0)
    ) -> Tuple[int, int, int, int]:
        """
        Generate a tuple that contains the min and max indexes of adjacent tiles or regions for a given feature. If
        the min and max values for both x and y are the same then this feature does not touch an overlap region.

        :param feature: the geojson Feature that must contain properties to identify its location in an image
        :param shape: the width, height of the area in pixels
        :param overlap: the x, y overlap between areas in pixels
        :param origin: the x, y coordinate of the area in relation to the full image

        :return: a tuple: minx, maxx, miny, maxy that identifies any overlap.
        """
        bbox = feature["properties"][GeojsonDetectionField.BOUNDS]

        # If an offset origin was supplied adjust the bbox so the key is relative to the origin.
        bbox = (bbox[0] - origin[0], bbox[1] - origin[1], bbox[2] - origin[0], bbox[3] - origin[1])

        stride_x = shape[0] - overlap[0]
        stride_y = shape[0] - overlap[1]

        max_x_index = int(bbox[2] / stride_x)
        max_y_index = int(bbox[3] / stride_y)

        min_x_index = int(bbox[0] / stride_x)
        min_y_index = int(bbox[1] / stride_y)
        min_x_offset = int(bbox[0]) % stride_x
        min_y_offset = int(bbox[1]) % stride_y

        if min_x_offset < overlap[0] and min_x_index > 0:
            min_x_index -= 1
        if min_y_offset < overlap[1] and min_y_index > 0:
            min_y_index -= 1

        return min_x_index, max_x_index, min_y_index, max_y_index

    @staticmethod
    def group_features_by_overlap(
        features: List[Feature], shape: Tuple[int, int], overlap: Tuple[int, int], origin: Tuple[int, int] = (0, 0)
    ) -> Dict[Tuple[int, int, int, int], List[Feature]]:
        """
        Group all the feature items by tile id

        :param features: List[FeatureItem] = the list of feature items
        :param shape: the width, height of the area in pixels
        :param overlap: the x, y overlap between areas in pixels
        :param origin: the x, y coordinate of the area in relation to the full image

        :return: a mapping of overlap id to a list of features that intersect that overlap region
        """
        grouped_features: Dict[Tuple[int, int, int, int], List[Feature]] = {}
        for feature in features:
            overlap_key = ModelRunner.identify_overlap(feature, shape, overlap, origin)
            grouped_features.setdefault(overlap_key, []).append(feature)
        return grouped_features

    @staticmethod
    @metric_scope
    def select_features(
        image_request_item: JobItem, features: List[Feature], metrics: MetricsLogger = None
    ) -> List[Feature]:
        """
        Selects the desired features using the options in the JobItem (NMS, SOFT_NMS, etc.).
        This code applies a feature selector only to the features that came from regions of the image
        that were processed multiple times. First features are grouped based on the region they were
        processed in. Any features found in the overlap area between regions are run through the
        FeatureSelector. If they were not part of an overlap area between regions, they will be grouped
        based on tile boundaries. Any features that fall into the overlap of adjacent tiles are filtered
        by the FeatureSelector. All other features should not be duplicates; they are added to the result
        without additional filtering.

        Computationally, this implements two critical factors that lower the overall processing time for the
        O(N^2) selection algorithms. First, it will filter out the majority of features that couldn't possibly
        have duplicates generated by our tiled image processing; Second, it runs the selection algorithms
        incrementally on much smaller groups of features.

        :param image_request_item: JobItem = the image request
        :param features: List[Feature] = the list of geojson features to process
        :param metrics: MetricsLogger = the metrics logger to use to report metrics.
        :return: List[Feature] = the list of geojson features after processing
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.FEATURE_SELECTION_OPERATION,
                }
            )
            metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

        with Timer(
            task_str="Select (deduplicate) image features",
            metric_name=MetricLabels.DURATION,
            logger=logger,
            metrics_logger=metrics,
        ):
            feature_distillation_option_dict = json.loads(image_request_item.feature_distillation_option)
            feature_distillation_option = FeatureDistillationDeserializer().deserialize(feature_distillation_option_dict)
            feature_selector = FeatureSelector(feature_distillation_option)

            region_size = ast.literal_eval(ServiceConfig.region_size)
            tile_size = ast.literal_eval(image_request_item.tile_size)
            overlap = ast.literal_eval(image_request_item.tile_overlap)

            logger.debug("FeatureSelection: Starting overlap-aware deduplication of features.")
            total_skipped = 0
            deduped_features = []
            features_grouped_by_region = ModelRunner.group_features_by_overlap(features, region_size, overlap)
            for region_key, region_features in features_grouped_by_region.items():
                region_stride = (region_size[0] - overlap[0], region_size[1] - overlap[1])
                region_origin = (region_stride[0] * region_key[0], region_stride[1] * region_key[1])

                if region_key[0] != region_key[1] or region_key[2] != region_key[3]:
                    # The Group contains contributions from multiple regions, run selection on the entire group
                    deduped_features.extend(feature_selector.select_features(region_features))
                else:
                    # Not an overlap between regions group these features using tile size to identify overlaps
                    features_grouped_by_tile = ModelRunner.group_features_by_overlap(
                        region_features, tile_size, overlap, region_origin
                    )

                    for tile_key, tile_features in features_grouped_by_tile.items():
                        if tile_key[0] != tile_key[1] or tile_key[2] != tile_key[3]:
                            # Group contains contributions from multiple tiles, run selection
                            deduped_features.extend(feature_selector.select_features(tile_features))
                        else:
                            # No overlap between tiles, features can be added directly to the result
                            total_skipped += len(tile_features)
                            deduped_features.extend(tile_features)

            logger.debug(
                f"FeatureSelection: Skipped processing of {total_skipped} of {len(features)} features. "
                "They were not inside an overlap region."
            )

            return deduped_features

    @staticmethod
    @metric_scope
    def sync_features(image_request_item: JobItem, features: List[Feature], metrics: MetricsLogger = None) -> bool:
        """
        Writing the features output to S3 and/or Kinesis Stream

        :param image_request_item: JobItem = the job table item for an image request
        :param features: List[Features] = the list of features to update
        :param metrics: the current metrics scope

        :return: bool = if it has successfully written to an output sync
        """

        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.FEATURE_DISSEMINATE_OPERATION,
                }
            )
            metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

        with Timer(
            task_str="Sink image features",
            metric_name=MetricLabels.DURATION,
            logger=logger,
            metrics_logger=metrics,
        ):
            tracking_output_sinks = {
                "S3": False,
                "Kinesis": False,
            }  # format: job_id = {"s3": true, "kinesis": true}

            # Ensure we have outputs defined for where to dump our features
            if image_request_item.outputs:
                logging.info("Writing aggregate feature for job '{}'".format(image_request_item.job_id))
                for sink in SinkFactory.outputs_to_sinks(json.loads(image_request_item.outputs)):
                    if sink.mode == SinkMode.AGGREGATE and image_request_item.job_id:
                        is_write_output_succeeded = sink.write(image_request_item.job_id, features)
                        tracking_output_sinks[sink.name()] = is_write_output_succeeded

                # Log them let them know if both written to both outputs (S3 and Kinesis) or one in another
                # If both couldn't write to either stream because both were down, return False. Otherwise True
                if tracking_output_sinks["S3"] and not tracking_output_sinks["Kinesis"]:
                    logging.info("OSMLModelRunner was able to write the features to S3 but not Kinesis. Continuing...")
                    return True
                elif not tracking_output_sinks["S3"] and tracking_output_sinks["Kinesis"]:
                    logging.info("OSMLModelRunner was able to write the features to Kinesis but not S3. Continuing...")
                    return True
                elif tracking_output_sinks["S3"] and tracking_output_sinks["Kinesis"]:
                    logging.info("OSMLModelRunner was able to write the features to both S3 and Kinesis. Continuing...")
                    return True
                else:
                    logging.error("OSMLModelRunner was not able to write the features to either S3 or Kinesis. Failing...")
                    return False
            else:
                raise InvalidImageRequestException("No output destinations were defined for this image request!")

    def add_properties_to_features(self, image_request_item: JobItem, features: List[Feature]) -> List[Feature]:
        """
        Add arbitrary and controlled property dictionaries to geojson feature properties

        :param image_request_item: JobItem = the job table item for an image request
        :param features: List[geojson.Feature] = the list of features to update

        :return: List[geojson.Feature] = updated list of features
        """
        try:
            feature_properties: List[dict] = json.loads(image_request_item.feature_properties)
            for feature in features:
                # Update the features with their inference metadata
                feature["properties"].update(
                    self.get_inference_metadata_property(image_request_item, feature["properties"]["inferenceTime"])
                )

                # For the custom provided feature properties, update
                for feature_property in feature_properties:
                    feature["properties"].update(feature_property)

                # Remove unneeded feature properties if they are present
                if feature.get("properties", {}).get("inferenceTime"):
                    del feature["properties"]["inferenceTime"]
                if feature.get("properties", {}).get(GeojsonDetectionField.BOUNDS):
                    del feature["properties"][GeojsonDetectionField.BOUNDS]
                if feature.get("properties", {}).get("detection_score"):
                    del feature["properties"]["detection_score"]
                if feature.get("properties", {}).get("feature_types"):
                    del feature["properties"]["feature_types"]
                if feature.get("properties", {}).get("image_id"):
                    del feature["properties"]["image_id"]
                if feature.get("properties", {}).get("adjusted_feature_types"):
                    del feature["properties"]["adjusted_feature_types"]

        except Exception as err:
            logging.exception(err)
            raise InvalidFeaturePropertiesException("Could not apply custom properties to features!")
        return features

    @staticmethod
    def get_inference_metadata_property(image_request_item: JobItem, inference_time: str) -> Dict[str, Any]:
        """
        Create an inference dictionary property to append to geojson features

        :param image_request_item: JobItem = the job table item for an image request
        :param inference_time: str = the time the inference was made in epoch millisec

        :return: Dict[str, Any] = an inference metadata dictionary property to attach to features
        """
        seconds = float(image_request_item.start_time) / 1000.0
        receive_time = datetime.fromtimestamp(seconds).isoformat()
        inference_metadata_property = {
            "inferenceMetadata": {
                "jobId": image_request_item.job_id,
                "filePath": image_request_item.image_url,
                "receiveTime": receive_time,
                "inferenceTime": inference_time,
                "tileOverlapFeatureSelection": image_request_item.feature_distillation_option,
            }
        }
        return inference_metadata_property

    @staticmethod
    def get_extents(ds: gdal.Dataset, sm: SensorModel) -> Dict[str, Any]:
        """
        Returns the geographic extents of the given GDAL dataset.

        :param ds: GDAL dataset.
        :param sm: OSML Sensor Model imputed for dataset
        :return: Dictionary with keys 'north', 'south', 'east', 'west' representing the extents.
        """
        try:
            # Compute WGS-84 world coordinates for each image corners to impute the extents for visualizations
            image_corners = [[0, 0], [ds.RasterXSize, 0], [ds.RasterXSize, ds.RasterYSize], [0, ds.RasterYSize]]
            geo_image_corners = [sm.image_to_world(ImageCoordinate(corner)) for corner in image_corners]
            locations = [(degrees(p.latitude), degrees(p.longitude)) for p in geo_image_corners]
            feature_bounds = functools.reduce(
                lambda prev, f: [
                    min(f[0], prev[0]),
                    min(f[1], prev[1]),
                    max(f[0], prev[2]),
                    max(f[1], prev[3]),
                ],
                locations,
                [math.inf, math.inf, -math.inf, -math.inf],
            )

            return {
                "north": feature_bounds[2],
                "south": feature_bounds[0],
                "east": feature_bounds[3],
                "west": feature_bounds[1],
            }
        except Exception as e:
            logger.error(f"Error in getting extents: {e}")

    @staticmethod
    def calculate_region_status(total_tile_count: int, tile_error_count: int) -> RegionRequestStatus:
        """
        Calculate the processing status of a region upon completion
        :param total_tile_count: number of tiles that were processed
        :param tile_error_count: number of tiles with errors
        :return: RegionRequestStatus
        """
        region_status = RegionRequestStatus.SUCCESS
        if total_tile_count == tile_error_count:
            region_status = RegionRequestStatus.FAILED
        if 0 < tile_error_count < total_tile_count:
            region_status = RegionRequestStatus.PARTIAL
        return region_status
