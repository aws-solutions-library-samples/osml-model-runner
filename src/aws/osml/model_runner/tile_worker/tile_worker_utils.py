#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import ast
import json
import logging
import tempfile
from pathlib import Path
from queue import Queue
from secrets import token_hex
from typing import List, Optional, Tuple

from aws_embedded_metrics import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from geojson import Feature
from osgeo import gdal

from aws.osml.features import Geolocator, ImagedFeaturePropertyAccessor
from aws.osml.gdal import GDALConfigEnv
from aws.osml.image_processing.gdal_tile_factory import GDALTileFactory
from aws.osml.model_runner.api import RegionRequest
from aws.osml.model_runner.app_config import MetricLabels, ServiceConfig
from aws.osml.model_runner.common import (
    FeatureDistillationDeserializer,
    ImageRegion,
    Timer,
    get_credentials_for_assumed_role,
)
from aws.osml.model_runner.database import FeatureTable, RegionRequestItem, RegionRequestTable
from aws.osml.model_runner.inference import FeatureSelector
from aws.osml.model_runner.inference.endpoint_factory import FeatureDetectorFactory
from aws.osml.photogrammetry import ElevationModel, SensorModel

from .exceptions import ProcessTilesException, SetupTileWorkersException
from .tile_worker import TileWorker
from .tiling_strategy import TilingStrategy

logger = logging.getLogger(__name__)


def setup_tile_workers(
    region_request: RegionRequest,
    sensor_model: Optional[SensorModel] = None,
    elevation_model: Optional[ElevationModel] = None,
) -> Tuple[Queue, List[TileWorker]]:
    """
    Sets up a pool of tile-workers to process image tiles from a region request

    :param region_request: RegionRequest = the region request to update.
    :param sensor_model: Optional[SensorModel] = the sensor model for this raster dataset
    :param elevation_model: Optional[ElevationModel] = an elevation model used to fix the elevation of the image coordinate

    :return: Tuple[Queue, List[TileWorker] = a list of tile workers and the queue that manages them
    """
    try:
        model_invocation_credentials = None
        if region_request.model_invocation_role:
            model_invocation_credentials = get_credentials_for_assumed_role(region_request.model_invocation_role)

        # Set up a Queue to manage our tile workers
        tile_queue: Queue = Queue()
        tile_workers = []

        for _ in range(int(ServiceConfig.workers)):
            # Set up our feature table to work with the region quest
            feature_table = FeatureTable(
                ServiceConfig.feature_table,
                region_request.tile_size,
                region_request.tile_overlap,
            )

            # Set up our feature table to work with the region quest
            region_request_table = RegionRequestTable(ServiceConfig.region_request_table)

            # Ignoring mypy error - if model_name was None the call to validate the region
            # request at the start of this function would have failed
            feature_detector = FeatureDetectorFactory(
                endpoint=region_request.model_name,
                endpoint_mode=region_request.model_invoke_mode,
                assumed_credentials=model_invocation_credentials,
            ).build()

            geolocator = None
            if sensor_model is not None:
                geolocator = Geolocator(ImagedFeaturePropertyAccessor(), sensor_model, elevation_model=elevation_model)

            worker = TileWorker(tile_queue, feature_detector, geolocator, feature_table, region_request_table)
            worker.start()
            tile_workers.append(worker)

        logger.debug(f"Setup pool of {len(tile_workers)} tile workers")

        return tile_queue, tile_workers
    except Exception as err:
        logger.exception(f"Failed to setup tile workers!: {err}")
        raise SetupTileWorkersException("Failed to setup tile workers!") from err


def process_tiles(
    tiling_strategy: TilingStrategy,
    region_request_item: RegionRequestItem,
    tile_queue: Queue,
    tile_workers: List[TileWorker],
    raster_dataset: gdal.Dataset,
    sensor_model: Optional[SensorModel] = None,
) -> Tuple[int, int]:
    """
    Loads a GDAL dataset into memory and processes it with a pool of tile workers.

    :param tiling_strategy: the approach used to decompose the region into tiles for the ML model
    :param region_request_item: RegionRequestItem = the region request to update.
    :param tile_queue: Queue = keeps the image in the queue for processing
    :param tile_workers: List[TileWorker] = the list of tile workers
    :param raster_dataset: gdal.Dataset = the raster dataset containing the region
    :param sensor_model: Optional[SensorModel] = the sensor model for this raster dataset

    :return: Tuple[int, int, List[ImageRegion]] = number of tiles processed, number of tiles with an error
    """

    # Grab completed tiles from region item
    # Explicitly cast to Tuple[Tuple[int, int], Tuple[int, int]]
    # Ensure the bounds have exactly two integers before converting
    region_bounds: Tuple[Tuple[int, int], Tuple[int, int]] = (
        (region_request_item.region_bounds[0][0], region_request_item.region_bounds[0][1]),
        (region_request_item.region_bounds[1][0], region_request_item.region_bounds[1][1]),
    )

    # Explicitly cast tile_size to Tuple[int, int]
    tile_size: Tuple[int, int] = (region_request_item.tile_size[0], region_request_item.tile_size[1])

    # Explicitly cast tile_overlap to Tuple[int, int]
    tile_overlap: Tuple[int, int] = (region_request_item.tile_overlap[0], region_request_item.tile_overlap[1])

    tile_array = tiling_strategy.compute_tiles(region_bounds, tile_size, tile_overlap)

    if region_request_item.succeeded_tiles is not None:
        # Filter ImageRegions based on matching in succeeded_tiles
        filtered_regions = [
            region
            for region in tile_array
            if [[region[0][0], region[0][1]], [region[1][0], region[1][1]]] not in region_request_item.succeeded_tiles
        ]
        if len(tile_array) != len(tile_array):
            logger.debug(f"{len(tile_array) - len(tile_array)} tiles have already been processed!")

        tile_array = filtered_regions

    total_tile_count = len(tile_array)
    try:
        # This will update the GDAL configuration options to use the security credentials for
        # this request. Any GDAL managed AWS calls (i.e. incrementally fetching pixels from a
        # dataset stored in S3) within this "with" statement will be made using customer
        # credentials. At the end of the "with" scope the credentials will be removed.
        image_read_credentials = None
        if region_request_item.image_read_role:
            image_read_credentials = get_credentials_for_assumed_role(region_request_item.image_read_role)

        with GDALConfigEnv().with_aws_credentials(image_read_credentials):
            # Use the request and metadata from the raster dataset to create a set of keyword
            # arguments for the gdal.Translate() function. This will configure that function to
            # create image tiles using the format, compression, etc. needed by the CV container.
            gdal_tile_factory = GDALTileFactory(
                raster_dataset=raster_dataset,
                tile_format=region_request_item.tile_format,
                tile_compression=region_request_item.tile_compression,
                sensor_model=sensor_model,
            )

            # Calculate a set of ML engine sized regions that we need to process for this image
            # and set up a temporary directory to store the temporary files. The entire directory
            # will be deleted at the end of this image's processing
            with tempfile.TemporaryDirectory() as tmp:
                # Ignoring mypy error - if region_bounds was None the call to validate the
                # image region request at the start of this function would have failed
                for tile_bounds in tile_array:
                    # Create a temp file name for the encoded region
                    region_image_filename = (
                        f"{token_hex(16)}-region-{tile_bounds[0][0]}-{tile_bounds[0][1]}-"
                        f"{tile_bounds[1][0]}-{tile_bounds[1][1]}.{region_request_item.tile_format}"
                    )

                    # Set a path for the tmp image
                    tmp_image_path = Path(tmp, region_image_filename)

                    # Generate an encoded tile of the requested image region
                    absolute_tile_path = _create_tile(gdal_tile_factory, tile_bounds, tmp_image_path)
                    if not absolute_tile_path:
                        continue

                    # Put the image info on the tile worker queue allowing each tile to be
                    # processed in parallel.
                    image_info = {
                        "image_path": tmp_image_path,
                        "region": tile_bounds,
                        "image_id": region_request_item.image_id,
                        "job_id": region_request_item.job_id,
                        "region_id": region_request_item.region_id,
                    }

                    # Place the image info onto our processing queue
                    tile_queue.put(image_info)

                # Put enough empty messages on the queue to shut down the workers
                for i in range(len(tile_workers)):
                    tile_queue.put(None)

                # Ensure the wait for tile workers happens within the context where we create
                # the temp directory. If the context is exited before all workers return then
                # the directory will be deleted, and we will potentially lose tiles.
                # Wait for all the workers to finish gracefully before we clean up the temp directory
                tile_error_count = 0
                for worker in tile_workers:
                    worker.join()
                    tile_error_count += worker.failed_tile_count

        logger.debug(
            (
                f"Model Runner Stats Processed {total_tile_count} image tiles for "
                f"region {region_request_item.region_bounds}. {tile_error_count} tiles failed to process."
            )
        )
    except Exception as err:
        logger.exception(f"File processing tiles: {err}")
        raise ProcessTilesException("Failed to process tiles!") from err

    return total_tile_count, tile_error_count


@metric_scope
def _create_tile(gdal_tile_factory, tile_bounds, tmp_image_path, metrics: MetricsLogger = None) -> Optional[str]:
    """
    Create an encoded tile of the requested image region.

    :param gdal_tile_factory: the factory used to create the tile
    :param tile_bounds: the requested tile boundary
    :param tmp_image_path: the output location of the tile
    :param metrics: the current metrics scope
    :return: the resulting tile path or None if the tile could not be created
    """
    if isinstance(metrics, MetricsLogger):
        metrics.set_dimensions()
        metrics.put_dimensions(
            {
                MetricLabels.OPERATION_DIMENSION: MetricLabels.TILE_GENERATION_OPERATION,
                MetricLabels.INPUT_FORMAT_DIMENSION: str(gdal_tile_factory.raster_dataset.GetDriver().ShortName).upper(),
            }
        )

    # Use GDAL to create an encoded tile of the image region
    absolute_tile_path = tmp_image_path.absolute()
    with Timer(
        task_str=f"Creating image tile: {absolute_tile_path}",
        metric_name=MetricLabels.DURATION,
        logger=logger,
        metrics_logger=metrics,
    ):
        if isinstance(metrics, MetricsLogger):
            metrics.put_metric(MetricLabels.INVOCATIONS, 1, str(Unit.COUNT.value))

        encoded_tile_data = gdal_tile_factory.create_encoded_tile(
            [tile_bounds[0][1], tile_bounds[0][0], tile_bounds[1][0], tile_bounds[1][1]]
        )

        with open(absolute_tile_path, "wb") as binary_file:
            binary_file.write(encoded_tile_data)

    # GDAL doesn't always generate errors, so we need to make sure the NITF
    # encoded region was actually created.
    if not tmp_image_path.is_file():
        logger.error(
            "GDAL unable to create tile %s. Does not exist!",
            absolute_tile_path,
        )
        if isinstance(metrics, MetricsLogger):
            metrics.put_metric(MetricLabels.ERRORS, 1, str(Unit.COUNT.value))
        return None
    else:
        logger.debug(
            "Created %s size %s",
            absolute_tile_path,
            sizeof_fmt(tmp_image_path.stat().st_size),
        )

    return absolute_tile_path


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def select_features(
    feature_distillation_option: str,
    features: List[Feature],
    processing_bounds: ImageRegion,
    region_size: str,
    tile_size: str,
    tile_overlap: str,
    tiling_strategy: TilingStrategy,
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

    :param region_size:
    :param feature_distillation_option: str = the options used in selecting features (e.g., NMS/SOFT_NMS, thresholds)
    :param features: List[Feature] = the list of geojson features to process
    :param processing_bounds: the requested area of the image
    :param region_size: str = region size to use for feature dedup
    :param tile_size: str = size of the tiles used during processing
    :param tile_overlap: str = overlap between tiles during processing
    :param tiling_strategy: the tiling strategy to use for feature dedup
    :return: List[Feature] = the list of geojson features after processing
    """
    feature_distillation_option_dict = json.loads(feature_distillation_option)
    feature_distillation_option = FeatureDistillationDeserializer().deserialize(feature_distillation_option_dict)
    feature_selector = FeatureSelector(feature_distillation_option)

    region_size = ast.literal_eval(region_size)
    tile_size = ast.literal_eval(tile_size)
    overlap = ast.literal_eval(tile_overlap)
    deduped_features = tiling_strategy.cleanup_duplicate_features(
        processing_bounds, region_size, tile_size, overlap, features, feature_selector
    )

    return deduped_features
