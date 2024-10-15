#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import ast
import json
import logging
from dataclasses import asdict
from json import dumps
from typing import List, Optional, Tuple

import shapely.geometry.base
from aws_embedded_metrics import MetricsLogger, metric_scope
from aws_embedded_metrics.unit import Unit
from geojson import Feature
from osgeo import gdal
from osgeo.gdal import Dataset

from aws.osml.gdal import GDALConfigEnv, get_image_extension, load_gdal_dataset
from aws.osml.model_runner.api import get_image_path
from aws.osml.photogrammetry import SensorModel

from .api import VALID_MODEL_HOSTING_OPTIONS, ImageRequest, RegionRequest
from .app_config import MetricLabels, ServiceConfig
from .common import (
    EndpointUtils,
    ImageDimensions,
    ImageRegion,
    RequestStatus,
    Timer,
    get_credentials_for_assumed_role,
    mr_post_processing_options_factory,
)
from .database import EndpointStatisticsTable, FeatureTable, JobItem, JobTable, RegionRequestItem, RegionRequestTable
from .exceptions import (
    AggregateFeaturesException,
    AggregateOutputFeaturesException,
    LoadImageException,
    ProcessImageException,
    UnsupportedModelException,
)
from .inference import calculate_processing_bounds, get_source_property
from .inference.feature_utils import add_properties_to_features, get_extents
from .queue import RequestQueue
from .region_request_handler import RegionRequestHandler
from .sink import SinkFactory
from .status import ImageStatusMonitor
from .tile_worker import TilingStrategy, select_features

# Set up logging configuration
logger = logging.getLogger(__name__)

# GDAL 4.0 will begin using exceptions as the default; at this point the software is written to assume
# no exceptions so we call this explicitly until the software can be updated to match.
gdal.UseExceptions()


class ImageRequestHandler:
    """
    Class responsible for handling ImageRequest processing.
    """

    def __init__(
        self,
        job_table: JobTable,
        image_status_monitor: ImageStatusMonitor,
        endpoint_statistics_table: EndpointStatisticsTable,
        tiling_strategy: TilingStrategy,
        region_request_queue: RequestQueue,
        region_request_table: RegionRequestTable,
        endpoint_utils: EndpointUtils,
        config: ServiceConfig,
        region_request_handler: RegionRequestHandler,
    ) -> None:
        """
        Initialize the ImageRequestHandler with the necessary dependencies.

        :param job_table: The job table for managing image processing jobs.
        :param image_status_monitor: A monitor to track image request statuses.
        :param endpoint_statistics_table: Table for tracking endpoint statistics.
        :param tiling_strategy: The strategy for handling image tiling into regions.
        :param region_request_queue: Queue to send region requests for processing.
        :param region_request_table: Table to track region request progress and results.
        :param endpoint_utils: Utility class for handling endpoint-related operations.
        :param config: Configuration settings for the service.
        :param region_request_handler: Handler for processing individual region requests.
        """

        self.job_table = job_table
        self.image_status_monitor = image_status_monitor
        self.endpoint_statistics_table = endpoint_statistics_table
        self.tiling_strategy = tiling_strategy
        self.region_request_queue = region_request_queue
        self.region_request_table = region_request_table
        self.endpoint_utils = endpoint_utils
        self.config = config
        self.region_request_handler = region_request_handler

    def process_image_request(self, image_request: ImageRequest) -> None:
        """
        Processes an ImageRequest object. Loads the specified image into memory, splits it into regions,
        and sends these regions for downstream processing via RegionRequest. The first region is processed
        directly by this method, while the remaining regions are queued for other workers.

        :param image_request: The image request to process, derived from the ImageRequest SQS message.

        :raises ProcessImageException: If image processing fails.
        :return: None
        """
        job_item = None
        try:
            if self.config.self_throttling:
                max_regions = self.endpoint_utils.calculate_max_regions(
                    image_request.model_name, image_request.model_invocation_role
                )
                # Add entry to the endpoint statistics table
                self.endpoint_statistics_table.upsert_endpoint(image_request.model_name, max_regions)

            # Update the image status to started and include relevant image meta-data
            logger.debug(f"Starting processing of {image_request.image_url}")
            job_item = JobItem.from_image_request(image_request)
            feature_distillation_option_list = image_request.get_feature_distillation_option()
            if feature_distillation_option_list:
                job_item.feature_distillation_option = dumps(
                    asdict(feature_distillation_option_list[0], dict_factory=mr_post_processing_options_factory)
                )

            # Start the image processing
            self.job_table.start_image_request(job_item)
            self.image_status_monitor.process_event(job_item, RequestStatus.STARTED, "Started image request")

            # Check we have a valid image request, throws if not
            self.validate_model_hosting(job_item)

            # Load the relevant image meta data into memory
            extension, ds, sensor_model, regions = self.load_image_request(job_item, image_request.roi)

            if sensor_model is None:
                logging.warning(f"Dataset {job_item.image_id} has no geo transform. Results are not geo-referenced.")

            # If we got valid outputs
            if ds and regions and extension:
                job_item.region_count = len(regions)
                job_item.width = int(ds.RasterXSize)
                job_item.height = int(ds.RasterYSize)
                try:
                    job_item.extents = json.dumps(get_extents(ds, sensor_model))
                except Exception as e:
                    logger.warning(f"Could not get extents for image: {job_item.image_id}")
                    logger.exception(e)

                feature_properties: List[dict] = json.loads(job_item.feature_properties)

                # If we can get a valid source metadata from the source image - attach it to features
                # else, just pass in whatever custom features if they were provided
                source_metadata = get_source_property(job_item.image_url, extension, ds)
                if isinstance(source_metadata, dict):
                    feature_properties.append(source_metadata)

                # Update the feature properties
                job_item.feature_properties = json.dumps(feature_properties)

                # Update the image request job to have new derived image data
                self.job_table.update_image_request(job_item)

                self.image_status_monitor.process_event(job_item, RequestStatus.IN_PROGRESS, "Processing regions")

                # Place the resulting region requests on the appropriate work queue
                self.queue_region_request(regions, image_request, ds, sensor_model, extension)

        except Exception as err:
            # We failed try and gracefully update our image request
            if job_item:
                self.fail_image_request(job_item, err)
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
        Queue all image regions for processing. Each region is added to the queue, with traceability maintained
        in the RegionRequestTable. The first region is processed directly, while the others are sent to the queue.

        :param all_regions: List of image regions to process.
        :param image_request: The image request associated with these regions.
        :param raster_dataset: The GDAL dataset containing the image regions.
        :param sensor_model: The sensor model for this raster dataset, if available.
        :param image_extension: The file extension of the image.

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
        job_item = self.region_request_handler.process_region_request(
            first_region_request, first_region_request_item, raster_dataset, sensor_model
        )

        # If the image is finished then complete it
        if self.job_table.is_image_request_complete(job_item):
            image_format = str(raster_dataset.GetDriver().ShortName).upper()
            self.complete_image_request(first_region_request, image_format, raster_dataset, sensor_model)

    def load_image_request(
        self,
        job_item: JobItem,
        roi: shapely.geometry.base.BaseGeometry,
    ) -> Tuple[str, Dataset, Optional[SensorModel], List[ImageRegion]]:
        """
        Loads image metadata and prepares it for processing. The image is divided into regions
        for distribution across workers.

        :param job_item: The image request object containing job information.
        :param roi: Region of interest to restrict image processing, provided as a geometry.

        :raises InvalidImageURLException: If the image URL is not valid.
        :raises LoadImageException: If loading image or processing bounds fails.
        :return: Tuple containing image extension, GDAL dataset, optional sensor model, and list of regions to process.
        """
        # If this request contains an execution role retrieve credentials that will be used to
        # access data
        assumed_credentials = None
        if job_item.image_read_role:
            assumed_credentials = get_credentials_for_assumed_role(job_item.image_read_role)

        # This will update the GDAL configuration options to use the security credentials for this
        # request. Any GDAL managed AWS calls (i.e. incrementally fetching pixels from a dataset
        # stored in S3) within this "with" statement will be made using customer credentials. At
        # the end of the "with" scope the credentials will be removed.
        with GDALConfigEnv().with_aws_credentials(assumed_credentials):
            # Extract the virtual image path from the request
            image_path = get_image_path(job_item.image_url, job_item.image_read_role)

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
                tile_size: ImageDimensions = ast.literal_eval(job_item.tile_size)
                if not job_item.tile_overlap:
                    minimum_overlap = (0, 0)
                else:
                    minimum_overlap = ast.literal_eval(job_item.tile_overlap)

                all_regions = self.tiling_strategy.compute_regions(
                    processing_bounds, region_size, tile_size, minimum_overlap
                )

        return image_extension, raster_dataset, sensor_model, all_regions

    def fail_image_request(self, job_item: JobItem, err: Exception) -> None:
        """
        Handles image request failure. Updates the status to 'failed' and ends the request in the job table.

        :param job_item: The job item for the failed image request.
        :param err: The exception that caused the failure.

        :return: None
        """
        logger.exception(f"Failed to start image processing!: {err}")
        self.image_status_monitor.process_event(job_item, RequestStatus.FAILED, str(err))
        self.job_table.end_image_request(job_item.image_id)

    def complete_image_request(
        self, region_request: RegionRequest, image_format: str, raster_dataset: gdal.Dataset, sensor_model: SensorModel
    ) -> None:
        """
        Completes the image request after all regions have been processed. Aggregates and sinks the features,
        then finalizes the request.

        :param region_request: The region request that completed.
        :param image_format: The format of the image file.
        :param raster_dataset: The GDAL dataset of the processed image.
        :param sensor_model: The sensor model for the image, if available.

        :raises AggregateFeaturesException: If feature aggregation fails.
        :return: None
        """
        try:
            # Retrieve the full image request
            job_item = self.job_table.get_image_request(region_request.image_id)

            # Log the completion of the last region and proceed with aggregation
            logger.debug("Last region of image request was completed, aggregating features for image!")

            # Set up the feature table
            feature_table = FeatureTable(self.config.feature_table, region_request.tile_size, region_request.tile_overlap)

            # Aggregate features
            features = feature_table.aggregate_features(job_item)
            logger.debug(f"Aggregated {len(features)} features for region {region_request.region_id}")

            # Deduplicate features
            deduped_features = self.deduplicate(job_item, features, raster_dataset, sensor_model)

            # Add the relevant properties to our final features
            final_features = add_properties_to_features(job_item.job_id, job_item.feature_properties, deduped_features)

            # Sink features to target outputs
            self.sink_features(job_item, final_features)

            # Finalize and update the job table with the completed request
            self.end_image_request(job_item, image_format)

        except Exception as err:
            raise AggregateFeaturesException("Failed to aggregate features for region!") from err

    @metric_scope
    def deduplicate(
        self,
        job_item: JobItem,
        features: List[Feature],
        raster_dataset: gdal.Dataset,
        sensor_model: SensorModel,
        metrics: MetricsLogger = None,
    ) -> List[Feature]:
        """
        Deduplicate the features and add additional properties to them, if applicable.

        :param metrics:
        :param job_item: The image processing job item containing job-specific information.
        :param features: A list of GeoJSON features to deduplicate.
        :param raster_dataset: The GDAL dataset representing the image being processed.
        :param sensor_model: The sensor model associated with the dataset, used for georeferencing.
        :param metrics: Optional metrics logger for tracking performance metrics.

        :return: A list of deduplicated features with additional properties added.
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.FEATURE_SELECTION_OPERATION,
                }
            )
        with Timer(
            task_str="Select (deduplicate) image features",
            metric_name=MetricLabels.DURATION,
            logger=logger,
            metrics_logger=metrics,
        ):
            # Calculate processing bounds based on the region of interest (ROI) and sensor model
            processing_bounds = self.calculate_processing_bounds(raster_dataset, sensor_model, job_item.roi_wkt)

            # Select and deduplicate features based on configuration options and processing bounds
            deduplicated_features = select_features(
                job_item.feature_distillation_option,
                features,
                processing_bounds,
                self.config.region_size,
                job_item.tile_size,
                job_item.tile_overlap,
                self.tiling_strategy,
            )

        return deduplicated_features

    def validate_model_hosting(self, image_request: JobItem):
        """
        Validates that the image request's model invocation mode is supported. If not, raises an exception.

        :param image_request: The image processing job item to validate.

        :raises UnsupportedModelException: If the model invocation mode is not supported.
        :return: None
        """
        if not image_request.model_invoke_mode or image_request.model_invoke_mode not in VALID_MODEL_HOSTING_OPTIONS:
            error = f"Application only supports {VALID_MODEL_HOSTING_OPTIONS} Endpoints"
            self.image_status_monitor.process_event(
                image_request,
                RequestStatus.FAILED,
                error,
            )
            raise UnsupportedModelException(error)

    @metric_scope
    def end_image_request(self, job_item: JobItem, image_format: str, metrics: MetricsLogger = None) -> None:
        """
        Finalizes the image request, updates the job status, and logs the necessary metrics.

        :param job_item: The image processing job item to finalize.
        :param image_format: The format of the image being processed (e.g., TIFF, NITF).
        :param metrics: Optional metrics logger for tracking performance metrics.

        :return: None
        """
        completed_job_item = self.job_table.end_image_request(job_item.image_id)

        # Retrieve the image request status and send status updates
        image_request_status = self.image_status_monitor.get_status(completed_job_item)
        self.image_status_monitor.process_event(completed_job_item, image_request_status, "Completed image processing")

        # Log metrics for the image processing duration, invocation, and errors (if any)
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.IMAGE_PROCESSING_OPERATION,
                    MetricLabels.MODEL_NAME_DIMENSION: job_item.model_name,
                    MetricLabels.INPUT_FORMAT_DIMENSION: image_format,
                }
            )
            metrics.put_metric(MetricLabels.DURATION, float(job_item.processing_duration), str(Unit.SECONDS.value))
            metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))
            if job_item.region_error > 0:
                metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))

    @staticmethod
    def calculate_processing_bounds(
        raster_dataset: gdal.Dataset,
        sensor_model: SensorModel,
        roi_wkt: Optional[str] = None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Calculate the processing bounds for the image, based on the region of interest (ROI) and sensor model.

        :param raster_dataset: The GDAL dataset representing the image being processed.
        :param sensor_model: The sensor model used to georeference the dataset.
        :param roi_wkt: Optional Well-Known Text (WKT) representing the region of interest for image processing.

        :raises AggregateFeaturesException: If the processing bounds cannot be calculated.
        :return: A tuple representing the upper-left corner (row, column) and dimensions (width, height) of the bounds.
        """
        roi = None
        if roi_wkt:
            logger.debug(f"Using ROI from request to set processing boundary: {roi_wkt}")
            roi = shapely.to_wkt(roi_wkt)

        processing_bounds = calculate_processing_bounds(raster_dataset, roi, sensor_model)
        logger.debug(f"Processing boundary from {roi} is {processing_bounds}")

        if not processing_bounds:
            raise AggregateFeaturesException("Failed to calculate processing bounds!")

        return processing_bounds

    @staticmethod
    @metric_scope
    def sink_features(job_item: JobItem, features: List[Feature], metrics: MetricsLogger = None) -> None:
        """
        Sink the deduplicated features to the specified output (e.g., S3, Kinesis, etc.).

        :param job_item: The job item representing the image processing request.
        :param features: The list of deduplicated GeoJSON features to sink.
        :param metrics: Optional metrics logger to track feature sinking performance.

        :raises AggregateOutputFeaturesException: If sinking the features to the output fails.
        :return: None
        """
        if isinstance(metrics, MetricsLogger):
            metrics.set_dimensions()
            metrics.put_dimensions(
                {
                    MetricLabels.OPERATION_DIMENSION: MetricLabels.FEATURE_DISSEMINATE_OPERATION,
                }
            )
        with Timer(
            task_str="Sink image features",
            metric_name=MetricLabels.DURATION,
            logger=logger,
            metrics_logger=metrics,
        ):
            # Sink features to the desired output (S3, Kinesis, etc.)
            is_write_succeeded = SinkFactory.sink_features(job_item.job_id, job_item.outputs, features)
            if not is_write_succeeded:
                raise AggregateOutputFeaturesException("Failed to write features to S3 or Kinesis!")
