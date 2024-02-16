#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import tempfile
from pathlib import Path
from queue import Queue
from secrets import token_hex
from typing import List, Optional, Tuple

from aws_embedded_metrics import MetricsLogger
from aws_embedded_metrics.metric_scope import metric_scope
from aws_embedded_metrics.unit import Unit
from osgeo import gdal

from aws.osml.gdal import GDALConfigEnv
from aws.osml.image_processing.gdal_tile_factory import GDALTileFactory
from aws.osml.model_runner.api import RegionRequest
from aws.osml.model_runner.app_config import MetricLabels, ServiceConfig
from aws.osml.model_runner.common import ImageDimensions, ImageRegion, Timer, get_credentials_for_assumed_role
from aws.osml.model_runner.database import FeatureTable
from aws.osml.model_runner.tile_worker import FeatureRefinery, TileWorker
from aws.osml.photogrammetry import ElevationModel, SensorModel

from ..inference.endpoint_factory import FeatureDetectorFactory
from .exceptions import ProcessTilesException, SetupTileWorkersException

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

            # Ignoring mypy error - if model_name was None the call to validate the region
            # request at the start of this function would have failed
            feature_detector = FeatureDetectorFactory(
                endpoint=region_request.model_name,
                endpoint_mode=region_request.model_invoke_mode,
                assumed_credentials=model_invocation_credentials,
            ).build()

            feature_refinery = None
            if sensor_model is not None:
                feature_refinery = FeatureRefinery(sensor_model, elevation_model=elevation_model)

            worker = TileWorker(tile_queue, feature_detector, feature_refinery, feature_table)
            worker.start()
            tile_workers.append(worker)

        logger.info("Setup pool of {} tile workers".format(len(tile_workers)))

        return tile_queue, tile_workers
    except Exception as err:
        logger.exception("Failed to setup tile workers!: {}".format(err))
        raise SetupTileWorkersException("Failed to setup tile workers!") from err


def process_tiles(
    region_request: RegionRequest,
    tile_queue: Queue,
    tile_workers: List[TileWorker],
    raster_dataset: gdal.Dataset,
    sensor_model: Optional[SensorModel] = None,
) -> Tuple[int, int]:
    """
    Loads a GDAL dataset into memory and processes it with a pool of tile workers.

    :param region_request: RegionRequest = the region request to update.
    :param tile_queue: Queue = keeps the image in the queue for processing
    :param tile_workers: List[Tileworker] = the list of tile workers
    :param raster_dataset: gdal.Dataset = the raster dataset containing the region
    :param sensor_model: Optional[SensorModel] = the sensor model for this raster dataset

    :return: Tuple[int, int] = number of tiles processed, number of tiles with an error
    """
    try:
        # This will update the GDAL configuration options to use the security credentials for
        # this request. Any GDAL managed AWS calls (i.e. incrementally fetching pixels from a
        # dataset stored in S3) within this "with" statement will be made using customer
        # credentials. At the end of the "with" scope the credentials will be removed.
        image_read_credentials = None
        if region_request.image_read_role:
            image_read_credentials = get_credentials_for_assumed_role(region_request.image_read_role)

        with GDALConfigEnv().with_aws_credentials(image_read_credentials):
            # Use the request and metadata from the raster dataset to create a set of keyword
            # arguments for the gdal.Translate() function. This will configure that function to
            # create image tiles using the format, compression, etc. needed by the CV container.
            gdal_tile_factory = GDALTileFactory(
                raster_dataset=raster_dataset,
                tile_format=region_request.tile_format,
                tile_compression=region_request.tile_compression,
                sensor_model=sensor_model,
            )

            # Calculate a set of ML engine sized regions that we need to process for this image
            # and set up a temporary directory to store the temporary files. The entire directory
            # will be deleted at the end of this image's processing
            total_tile_count = 0
            with tempfile.TemporaryDirectory() as tmp:
                # Ignoring mypy error - if region_bounds was None the call to validate the
                # image region request at the start of this function would have failed
                for tile_bounds in generate_crops(
                    region_request.region_bounds,  # type: ignore[arg-type]
                    region_request.tile_size,
                    region_request.tile_overlap,
                ):
                    # Create a temp file name for the encoded region
                    region_image_filename = "{}-region-{}-{}-{}-{}.{}".format(
                        token_hex(16),
                        tile_bounds[0][0],
                        tile_bounds[0][1],
                        tile_bounds[1][0],
                        tile_bounds[1][1],
                        region_request.tile_format,
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
                        "image_id": region_request.image_id,
                        "job_id": region_request.job_id,
                    }
                    # Increment our tile count tracking
                    total_tile_count += 1

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
                    tile_error_count += worker.feature_detector.error_count

        logger.info(
            "Model Runner Stats Processed {} image tiles for region {}. {} tile errors.".format(
                total_tile_count, region_request.region_bounds, tile_error_count
            )
        )
    except Exception as err:
        logger.exception("File processing tiles: {}", err)
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
        task_str="Creating image tile: {}".format(absolute_tile_path),
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
        logger.info(
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


def generate_crops(region: ImageRegion, chip_size: ImageDimensions, overlap: ImageDimensions) -> List[ImageRegion]:
    """
    Yields a list of overlapping chip bounding boxes for the given region. Chips will start
    in the upper left corner of the region (i.e. region[0][0], region[0][1]) and will be spaced
    such that they have the specified horizontal and vertical overlap.

    :param region: ImageDimensions = a tuple for the bounding box of the region ((ul_r, ul_c), (width, height))
    :param chip_size: ImageDimensions = a tuple for the chip dimensions (width, height)
    :param overlap: ImageDimensions = a tuple for the overlap (width, height)

    :return: List[ImageRegion] = an iterable list of tuples for the chip bounding boxes [((ul_r, ul_c), (w, h)), ...]
    """
    if overlap[0] >= chip_size[0] or overlap[1] >= chip_size[1]:
        raise ValueError("Overlap must be less than chip size! chip_size = " + str(chip_size) + " overlap = " + str(overlap))

    # Calculate the spacing for the chips taking into account the horizontal and vertical overlap
    # and how many are needed to cover the region
    stride_x = chip_size[0] - overlap[0]
    stride_y = chip_size[1] - overlap[1]
    num_x = ceildiv(region[1][0], stride_x)
    num_y = ceildiv(region[1][1], stride_y)

    crops = []
    for r in range(0, num_y):
        for c in range(0, num_x):
            # Calculate the bounds of the chip ensuring that the chip does not extend
            # beyond the edge of the requested region
            ul_x = region[0][1] + c * stride_x
            ul_y = region[0][0] + r * stride_y
            w = min(chip_size[0], (region[0][1] + region[1][0]) - ul_x)
            h = min(chip_size[1], (region[0][0] + region[1][1]) - ul_y)
            if w > overlap[0] and h > overlap[1]:
                crops.append(((ul_y, ul_x), (w, h)))

    return crops


def ceildiv(a: int, b: int) -> int:
    """
    Integer ceiling division

    :param a: int = numerator
    :param b: int = denominator

    :return: ceil(a/b)
    """
    return -(-a // b)
