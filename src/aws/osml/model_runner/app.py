#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import ast
import json
import logging
from dataclasses import asdict
from json import dumps
from typing import List, Optional, Tuple

import shapely.geometry.base
from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from osgeo import gdal
from osgeo.gdal import Dataset

from aws.osml.gdal import GDALConfigEnv, get_image_extension, load_gdal_dataset, set_gdal_default_configuration
from aws.osml.photogrammetry import SensorModel

from .api import VALID_MODEL_HOSTING_OPTIONS, ImageRequest, InvalidImageRequestException, RegionRequest
from .app_config import MetricLabels, ServiceConfig
from .common import (
    EndpointUtils,
    ImageDimensions,
    ImageRegion,
    RequestStatus,
    ThreadingLocalContextFilter,
    Timer,
    get_credentials_for_assumed_role,
    mr_post_processing_options_factory,
)
from .database import EndpointStatisticsTable, FeatureTable, JobItem, JobTable, RegionRequestItem, RegionRequestTable
from .exceptions import (
    AggregateFeaturesException,
    AggregateOutputFeaturesException,
    InvalidImageURLException,
    LoadImageException,
    ProcessImageException,
    RetryableJobException,
    SelfThrottledRegionException,
    UnsupportedModelException,
)
from .inference import calculate_processing_bounds, get_extents, get_source_property
from .inference.feature_utils import add_properties_to_features
from .queue import RequestQueue
from .region_request_handler import RegionRequestHandler
from .sink import SinkFactory
from .status import ImageStatusMonitor, RegionStatusMonitor
from .tile_worker import TilingStrategy, VariableOverlapTilingStrategy, select_features

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
                            self.complete_image_request(region_request, image_format, raster_dataset, sensor_model)

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
                            self.process_image_request(image_request)

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
            if self.config.self_throttling:
                max_regions = self.endpoint_utils.calculate_max_regions(
                    image_request.model_name, image_request.model_invocation_role
                )
                # Add entry to the endpoint statistics table
                self.endpoint_statistics_table.upsert_endpoint(image_request.model_name, max_regions)

            # Update the image status to started and include relevant image meta-data
            logger.debug(f"Starting processing of {image_request.image_url}")
            image_request_item = JobItem(
                image_id=image_request.image_id,
                job_id=image_request.job_id,
                tile_size=str(image_request.tile_size),
                tile_overlap=str(image_request.tile_overlap),
                model_name=image_request.model_name,
                model_invoke_mode=image_request.model_invoke_mode,
                outputs=dumps(image_request.outputs),
                image_url=image_request.image_url,
                image_read_role=image_request.image_read_role,
                feature_properties=dumps(image_request.feature_properties),
                roi_wkt=image_request.roi.wkt if image_request.roi else None,
            )
            feature_distillation_option_list = image_request.get_feature_distillation_option()
            if feature_distillation_option_list:
                image_request_item.feature_distillation_option = dumps(
                    asdict(feature_distillation_option_list[0], dict_factory=mr_post_processing_options_factory)
                )

            # Start the image processing
            self.job_table.start_image_request(image_request_item)
            self.image_status_monitor.process_event(image_request_item, RequestStatus.STARTED, "Started image request")

            # Check we have a valid image request, throws if not
            self.validate_model_hosting(image_request_item)

            # Load the relevant image meta data into memory
            image_extension, raster_dataset, sensor_model, all_regions = self.load_image_request(
                image_request_item, image_request.roi
            )

            if sensor_model is None:
                logging.warning(
                    f"Dataset {image_request_item.image_id} has no geo transform. Results are not geo-referenced."
                )

            # If we got valid outputs
            if raster_dataset and all_regions and image_extension:
                image_request_item.region_count = len(all_regions)
                image_request_item.width = int(raster_dataset.RasterXSize)
                image_request_item.height = int(raster_dataset.RasterYSize)
                try:
                    image_request_item.extents = json.dumps(get_extents(raster_dataset, sensor_model))
                except Exception as e:
                    logger.warning(f"Could not get extents for image: {image_request_item.image_id}")
                    logger.exception(e)

                feature_properties: List[dict] = json.loads(image_request_item.feature_properties)

                # If we can get a valid source metadata from the source image - attach it to features
                # else, just pass in whatever custom features if they were provided
                source_metadata = get_source_property(image_request_item.image_url, image_extension, raster_dataset)
                if isinstance(source_metadata, dict):
                    feature_properties.append(source_metadata)

                # Update the feature properties
                image_request_item.feature_properties = json.dumps(feature_properties)

                # Update the image request job to have new derived image data
                self.job_table.update_image_request(image_request_item)

                self.image_status_monitor.process_event(image_request_item, RequestStatus.IN_PROGRESS, "Processing regions")

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
                    processing_duration=0,
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

        :return: None
        """
        # Set aside the first region
        first_region = all_regions.pop(0)
        for region in all_regions:
            logger.debug(f"Queueing region: {region}")

            region_request = RegionRequest(
                image_request.get_shared_values(),
                region_bounds=region,
                region_id=f"{region[0]}{region[1]}-{image_request.job_id}",
                image_extension=image_extension,
            )

            # Create a new entry to the region request being started
            region_request_item = RegionRequestItem.from_region_request(region_request)
            self.region_request_table.start_region_request(region_request_item)
            logging.debug(
                (
                    f"Adding region request: image id: {region_request_item.image_id} - "
                    f"region id: {region_request_item.region_id}"
                )
            )

            # Send the attributes of the region request as the message.
            self.region_request_queue.send_request(region_request.__dict__)

        # Go ahead and process the first region
        logger.debug(f"Processing first region {0}: {first_region}")

        first_region_request = RegionRequest(
            image_request.get_shared_values(),
            region_bounds=first_region,
            region_id=f"{first_region[0]}{first_region[1]}-{image_request.job_id}",
            image_extension=image_extension,
        )

        # Add item to RegionRequestTable
        first_region_request_item = RegionRequestItem.from_region_request(first_region_request)
        self.region_request_table.start_region_request(first_region_request_item)
        logging.debug(f"Adding region_id: {first_region_request_item.region_id}")

        # Processes our region request and return the updated item
        image_request_item = self.region_request_handler.process_region_request(
            first_region_request, first_region_request_item, raster_dataset, sensor_model
        )

        # If the image is finished then complete it
        if self.job_table.is_image_request_complete(image_request_item):
            image_format = str(raster_dataset.GetDriver().ShortName).upper()
            self.complete_image_request(first_region_request, image_format, raster_dataset, sensor_model)

    def load_image_request(
        self,
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
                logger.warning("Requested ROI does not intersect image. Nothing to do")
                raise LoadImageException("Failed to create processing bounds for image!")
            else:
                # Calculate a set of ML engine-sized regions that we need to process for this image
                # Region size chosen to break large images into pieces that can be handled by a
                # single tile worker
                region_size: ImageDimensions = ast.literal_eval(self.config.region_size)
                tile_size: ImageDimensions = ast.literal_eval(image_request_item.tile_size)
                if not image_request_item.tile_overlap:
                    minimum_overlap = (0, 0)
                else:
                    minimum_overlap = ast.literal_eval(image_request_item.tile_overlap)

                all_regions = self.tiling_strategy.compute_regions(
                    processing_bounds, region_size, tile_size, minimum_overlap
                )

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
        logger.exception(f"Failed to start image processing!: {err}")
        self.image_status_monitor.process_event(image_request_item, RequestStatus.FAILED, str(err))

    @metric_scope
    def complete_image_request(
        self,
        region_request: RegionRequest,
        image_format: str,
        raster_dataset: gdal.Dataset,
        sensor_model: SensorModel,
        metrics: MetricsLogger = None,
    ) -> None:
        """
        Runs after every region has completed processing to check if that was the last region and run required
        completion logic for the associated ImageRequest.

        :param region_request: RegionRequest = the region request to update.
        :param image_format: Format of the image data
        :param raster_dataset: the image data rater
        :param sensor_model: the image sensor model
        :param metrics: the current metric scope

        :return: None
        """
        try:
            # Grab the full image request item from the table
            image_request_item = self.job_table.get_image_request(region_request.image_id)

            logger.debug("Last region of image request was completed, aggregating features for image!")

            roi = None
            if image_request_item.roi_wkt:
                logger.debug(f"Using ROI from request to set processing boundary: {image_request_item.roi_wkt}")
                roi = shapely.wkt.loads(image_request_item.roi_wkt)
            processing_bounds = calculate_processing_bounds(raster_dataset, roi, sensor_model)
            logger.debug(f"Processing boundary from {roi} is {processing_bounds}")

            # Set up our feature table to work with the region quest
            feature_table = FeatureTable(self.config.feature_table, region_request.tile_size, region_request.tile_overlap)
            # Aggregate all the features from our job
            features = feature_table.aggregate_features(image_request_item)
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
                features = select_features(
                    image_request_item.feature_distillation_option,
                    features,
                    processing_bounds,
                    self.config.region_size,
                    image_request_item.tile_size,
                    image_request_item.tile_overlap,
                    self.tiling_strategy,
                )
                features = add_properties_to_features(
                    image_request_item.job_id, image_request_item.feature_properties, features
                )

                # Sink the features into the right outputs
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
                    is_write_succeeded = SinkFactory.sink_features(
                        image_request_item.job_id, image_request_item.outputs, features
                    )
                    if not is_write_succeeded:
                        raise AggregateOutputFeaturesException(
                            "Failed to write features to S3 or Kinesis! Please check the " "log..."
                        )

            # Put our end time on our image_request_item
            completed_image_request_item = self.job_table.end_image_request(image_request_item.image_id)

            # Ensure we have a valid start time for our record
            # TODO: Figure out why we wouldn't have a valid start time?!?!
            if completed_image_request_item.processing_duration is not None:
                image_request_status = self.image_status_monitor.get_status(completed_image_request_item)
                self.image_status_monitor.process_event(
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

            metrics.put_metric(MetricLabels.DURATION, float(image_request_item.processing_duration), str(Unit.SECONDS.value))
            metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))
            if image_request_item.region_error > 0:
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))

    def validate_model_hosting(self, image_request: JobItem):
        """
        Validates that the image request is valid. If not, raises an exception.

        :param image_request: JobItem = the image request

        :return: None
        """
        if not image_request.model_invoke_mode or image_request.model_invoke_mode not in VALID_MODEL_HOSTING_OPTIONS:
            error = f"Application only supports ${VALID_MODEL_HOSTING_OPTIONS} Endpoints"
            self.image_status_monitor.process_event(
                image_request,
                RequestStatus.FAILED,
                error,
            )
            raise UnsupportedModelException(error)
