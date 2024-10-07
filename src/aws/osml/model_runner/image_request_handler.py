#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import ast
import functools
import json
import logging
import math
from dataclasses import asdict
from json import dumps
from math import degrees
from typing import Any, Dict, List, Optional, Tuple

import shapely.geometry.base
from aws_embedded_metrics import MetricsLogger, metric_scope
from aws_embedded_metrics.unit import Unit
from geojson import Feature
from osgeo import gdal
from osgeo.gdal import Dataset

from aws.osml.gdal import GDALConfigEnv, get_image_extension, load_gdal_dataset
from aws.osml.photogrammetry import ImageCoordinate, SensorModel

from .api import VALID_MODEL_HOSTING_OPTIONS, ImageRequest, InvalidImageRequestException, RegionRequest, SinkMode
from .common import (
    EndpointUtils,
    FeatureDistillationDeserializer,
    GeojsonDetectionField,
    ImageDimensions,
    ImageRegion,
    RequestStatus,
    Timer,
    get_credentials_for_assumed_role,
    mr_post_processing_options_factory,
)
from .config import MetricLabels, ServiceConfig
from .database import EndpointStatisticsTable, FeatureTable, JobItem, JobTable, RegionRequestItem, RegionRequestTable
from .exceptions import (
    AggregateFeaturesException,
    AggregateOutputFeaturesException,
    InvalidFeaturePropertiesException,
    InvalidImageURLException,
    LoadImageException,
    ProcessImageException,
    UnsupportedModelException,
)
from .inference import FeatureSelector, calculate_processing_bounds, get_source_property
from .queue import RequestQueue
from .region_request_handler import RegionRequestHandler
from .sink import SinkFactory
from .status import ImageStatusMonitor
from .tile_worker import TilingStrategy

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

        :param job_table: The job table for image processing.
        :param image_status_monitor: A monitor to track image request status.
        :param endpoint_statistics_table: Table for tracking endpoint statistics.
        :param tiling_strategy: The strategy for handling image tiling.
        :param region_request_queue: Queue to send region requests.
        :param region_request_table: Table to track region request progress.
        :param endpoint_utils: Utility class for handling endpoint-related operations.
        :param config: Configuration settings for the service.
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
        Processes ImageRequest objects picked up from the queue.
        """
        image_request_item = None
        try:
            if self.config.self_throttling:
                max_regions = self.endpoint_utils.calculate_max_regions(
                    image_request.model_name, image_request.model_invocation_role
                )
                # Add entry to the endpoint statistics table
                self.endpoint_statistics_table.upsert_endpoint(image_request.model_name, max_regions)

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

            self.job_table.start_image_request(image_request_item)
            self.image_status_monitor.process_event(image_request_item, RequestStatus.STARTED, "Started image request")

            self.validate_model_hosting(image_request_item)

            image_extension, raster_dataset, sensor_model, all_regions = self.load_image_request(
                image_request_item, image_request.roi
            )

            if sensor_model is None:
                logging.warning(
                    f"Dataset {image_request_item.image_id} has no geo transform. Results are not geo-referenced."
                )

            if raster_dataset and all_regions and image_extension:
                image_request_item.region_count = len(all_regions)
                image_request_item.width = int(raster_dataset.RasterXSize)
                image_request_item.height = int(raster_dataset.RasterYSize)
                try:
                    image_request_item.extents = json.dumps(self.get_extents(raster_dataset, sensor_model))
                except Exception as e:
                    logger.warning(f"Could not get extents for image: {image_request_item.image_id}")
                    logger.exception(e)

                feature_properties: List[dict] = json.loads(image_request_item.feature_properties)
                source_metadata = get_source_property(image_request_item.image_url, image_extension, raster_dataset)
                if isinstance(source_metadata, dict):
                    feature_properties.append(source_metadata)

                image_request_item.feature_properties = json.dumps(feature_properties)
                self.job_table.update_image_request(image_request_item)
                self.image_status_monitor.process_event(image_request_item, RequestStatus.IN_PROGRESS, "Processing regions")

                self.queue_region_request(all_regions, image_request, raster_dataset, sensor_model, image_extension)

        except Exception as err:
            if image_request_item:
                self.fail_image_request(image_request_item, err)
            else:
                minimal_job_item = JobItem(
                    image_id=image_request.image_id,
                    job_id=image_request.job_id,
                    processing_duration=0,
                )
                self.fail_image_request(minimal_job_item, err)
            raise ProcessImageException("Failed to process image region!") from err

    def complete_image_request(
        self, region_request: RegionRequest, image_format: str, raster_dataset: gdal.Dataset, sensor_model: SensorModel
    ) -> None:
        """
        Runs after every region has completed processing to check if that was the last region and run required
        completion logic for the associated ImageRequest.
        """
        try:
            image_request_item = self.job_table.get_image_request(region_request.image_id)

            logger.debug("Last region of image request was completed, aggregating features for image!")

            roi = None
            if image_request_item.roi_wkt:
                logger.debug(f"Using ROI from request to set processing boundary: {image_request_item.roi_wkt}")
                roi = shapely.wkt.loads(image_request_item.roi_wkt)
            processing_bounds = calculate_processing_bounds(raster_dataset, roi, sensor_model)
            logger.debug(f"Processing boundary from {roi} is {processing_bounds}")

            feature_table = FeatureTable(self.config.feature_table, region_request.tile_size, region_request.tile_overlap)
            features = feature_table.aggregate_features(image_request_item)
            features = self.select_features(image_request_item, features, processing_bounds)
            features = self.add_properties_to_features(image_request_item, features)

            is_write_succeeded = self.sink_features(image_request_item, features)
            if not is_write_succeeded:
                raise AggregateOutputFeaturesException("Failed to write features to S3 or Kinesis! Please check the log...")

            completed_image_request_item = self.job_table.end_image_request(image_request_item.image_id)

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
        Handles failure events/exceptions for image requests and tries to update the status monitor accordingly.

        :param image_request_item: JobItem = the image request that failed.
        :param err: Exception = the exception that caused the failure
        :return: None
        """
        self.fail_image_request(image_request_item, err)
        self.job_table.end_image_request(image_request_item.image_id)

    def queue_region_request(
        self,
        all_regions: List[ImageRegion],
        image_request: ImageRequest,
        raster_dataset: Dataset,
        sensor_model: Optional[SensorModel],
        image_extension: Optional[str],
    ) -> None:
        """
        Queues region requests and handles processing of the first region.
        """
        first_region = all_regions.pop(0)
        for region in all_regions:
            logger.debug(f"Queueing region: {region}")

            region_request = RegionRequest(
                image_request.get_shared_values(),
                region_bounds=region,
                region_id=f"{region[0]}{region[1]}-{image_request.job_id}",
                image_extension=image_extension,
            )

            region_request_item = RegionRequestItem.from_region_request(region_request)
            self.region_request_table.start_region_request(region_request_item)

            logger.debug(
                f"Adding region request: image id: {region_request_item.image_id} - "
                f"region id: {region_request_item.region_id}"
            )

            self.region_request_queue.send_request(region_request.__dict__)

        logger.debug(f"Processing first region {first_region}: {first_region}")

        first_region_request = RegionRequest(
            image_request.get_shared_values(),
            region_bounds=first_region,
            region_id=f"{first_region[0]}{first_region[1]}-{image_request.job_id}",
            image_extension=image_extension,
        )

        first_region_request_item = RegionRequestItem.from_region_request(first_region_request)

        image_request_item = self.region_request_handler.process_region_request(
            first_region_request, first_region_request_item, raster_dataset, sensor_model
        )

        if self.job_table.is_image_request_complete(image_request_item):
            image_format = str(raster_dataset.GetDriver().ShortName).upper()
            self.complete_image_request(first_region_request, image_format, raster_dataset, sensor_model)

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

    @metric_scope
    def select_features(
        self,
        image_request_item: JobItem,
        features: List[Feature],
        processing_bounds: Optional[ImageRegion],
        metrics: MetricsLogger = None,
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
        :param processing_bounds: the requested area of the image
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

            region_size = ast.literal_eval(self.config.region_size)
            tile_size = ast.literal_eval(image_request_item.tile_size)
            overlap = ast.literal_eval(image_request_item.tile_overlap)
            deduped_features = self.tiling_strategy.cleanup_duplicate_features(
                processing_bounds, region_size, tile_size, overlap, features, feature_selector
            )

            return deduped_features

    @staticmethod
    @metric_scope
    def sink_features(image_request_item: JobItem, features: List[Feature], metrics: MetricsLogger = None) -> bool:
        """
        Writing the features output to S3 and/or Kinesis Stream

        :param image_request_item: JobItem = the job table item for an image request
        :param features: List[Features] = the list of features to update
        :param metrics: the current metrics scope

        :return: bool = if it has successfully written to an output sink
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
                logging.debug(f"Writing aggregate feature for job '{image_request_item.job_id}'")
                for sink in SinkFactory.outputs_to_sinks(json.loads(image_request_item.outputs)):
                    if sink.mode == SinkMode.AGGREGATE and image_request_item.job_id:
                        is_write_output_succeeded = sink.write(image_request_item.job_id, features)
                        tracking_output_sinks[sink.name()] = is_write_output_succeeded

                # Log them let them know if both written to both outputs (S3 and Kinesis) or one in another
                # If both couldn't write to either stream because both were down, return False. Otherwise True
                if tracking_output_sinks["S3"] and not tracking_output_sinks["Kinesis"]:
                    logging.debug("LModelRunner was able to write the features to S3 but not Kinesis. Continuing...")
                    return True
                elif not tracking_output_sinks["S3"] and tracking_output_sinks["Kinesis"]:
                    logging.debug("ModelRunner was able to write the features to Kinesis but not S3. Continuing...")
                    return True
                elif tracking_output_sinks["S3"] and tracking_output_sinks["Kinesis"]:
                    logging.debug("ModelRunner was able to write the features to both S3 and Kinesis. Continuing...")
                    return True
                else:
                    logging.error("ModelRunner was not able to write the features to either S3 or Kinesis. Failing...")
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
                if feature.get("properties", {}).get(GeojsonDetectionField.GEOM):
                    del feature["properties"][GeojsonDetectionField.GEOM]
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
        inference_metadata_property = {
            "inferenceMetadata": {
                "jobId": image_request_item.job_id,
                "inferenceDT": inference_time,
            }
        }
        return inference_metadata_property

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
