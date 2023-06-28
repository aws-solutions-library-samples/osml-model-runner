#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import tempfile
from pathlib import Path
from queue import Queue
from secrets import token_hex
from typing import Any, Dict, List, Optional, Tuple

from aws_embedded_metrics import MetricsLogger
from aws_embedded_metrics.unit import Unit
from osgeo import gdal

from aws.osml.gdal import GDALConfigEnv, get_type_and_scales
from aws.osml.model_runner.api import RegionRequest
from aws.osml.model_runner.app_config import MetricLabels, ServiceConfig
from aws.osml.model_runner.common import (
    ImageCompression,
    ImageDimensions,
    ImageFormats,
    ImageRegion,
    Timer,
    get_credentials_for_assumed_role,
)
from aws.osml.model_runner.database import FeatureTable
from aws.osml.model_runner.inference import SMDetector
from aws.osml.model_runner.tile_worker import FeatureRefinery, TileWorker
from aws.osml.photogrammetry import ElevationModel, SensorModel

from .exceptions import ProcessTilesException, SetupTileWorkersException

logger = logging.getLogger(__name__)


def setup_tile_workers(
    region_request: RegionRequest,
    sensor_model: Optional[SensorModel] = None,
    elevation_model: Optional[ElevationModel] = None,
    metrics: MetricsLogger = None,
) -> Tuple[Queue, List[TileWorker]]:
    """
    Sets up a pool of tile-workers to process image tiles from a region request

    :param region_request: RegionRequest = the region request to update.
    :param sensor_model: Optional[SensorModel] = the sensor model for this raster dataset
    :param elevation_model: Optional[ElevationModel] = an elevation model used to fix the elevation of the image coordinate
    :param metrics: MetricsLogger = the metrics logger to use to report metrics.

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
            feature_detector = SMDetector(region_request.model_name, model_invocation_credentials)  # type: ignore[arg-type]

            feature_refinery = None
            if sensor_model is not None:
                feature_refinery = FeatureRefinery(sensor_model, elevation_model=elevation_model)

            worker = TileWorker(tile_queue, feature_detector, feature_refinery, feature_table, metrics)
            worker.start()
            tile_workers.append(worker)

        logger.info("Setup pool of {} tile workers".format(len(tile_workers)))

        return tile_queue, tile_workers
    except Exception as err:
        logger.exception("Failed to setup tile workers!: {}", err)
        raise SetupTileWorkersException("Failed to setup tile workers!") from err


def process_tiles(
    region_request: RegionRequest,
    tile_queue: Queue,
    tile_workers: List[TileWorker],
    raster_dataset: gdal.Dataset,
    metrics: MetricsLogger = None,
) -> int:
    """
    Loads a GDAL dataset into memory and processes it with a pool of tile workers.

    :param region_request: RegionRequest = the region request to update.
    :param tile_queue: Queue = keeps the image in the queue for processing
    :param tile_workers: List[Tileworker] = the list of tile workers
    :param raster_dataset: gdal.Dataset = the raster dataset containing the region
    :param metrics: MetricsLogger = the metrics logger to use to report metrics.

    :return: int = number of tiles processed
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
            gdal_translate_kwargs = create_gdal_translate_kwargs(
                region_request.tile_format, region_request.tile_compression, raster_dataset
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
                    # Create a temp file name for the NITF encoded region
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

                    # Use GDAL to create an encoded tile of the image region
                    # From GDAL documentation:
                    #   srcWin --- subwindow in pixels to extract:
                    #               [left_x, top_y, width, height]
                    absolute_tile_path = tmp_image_path.absolute()
                    with Timer(
                        task_str="Creating image tile: {}".format(absolute_tile_path),
                        metric_name=MetricLabels.TILING_LATENCY,
                        logger=logger,
                        metrics_logger=metrics,
                    ):
                        # Use GDAL to transform the source image
                        gdal.Translate(
                            str(absolute_tile_path),
                            raster_dataset,
                            srcWin=[
                                tile_bounds[0][1],
                                tile_bounds[0][0],
                                tile_bounds[1][0],
                                tile_bounds[1][1],
                            ],
                            **gdal_translate_kwargs,
                        )

                    # GDAL doesn't always generate errors, so we need to make sure the NITF
                    # encoded region was actually created.
                    if not tmp_image_path.is_file():
                        logger.error(
                            "GDAL unable to create tile %s. Does not exist!",
                            absolute_tile_path,
                        )
                        if isinstance(metrics, MetricsLogger):
                            metrics.put_metric(MetricLabels.TILE_CREATION_FAILURE, 1, str(Unit.COUNT.value))
                            metrics.put_metric(MetricLabels.REGION_PROCESSING_ERROR, 1, str(Unit.COUNT.value))
                        continue
                    else:
                        logger.info(
                            "Created %s size %s",
                            absolute_tile_path,
                            sizeof_fmt(tmp_image_path.stat().st_size),
                        )

                    # Put the image info on the tile worker queue allowing each tile to be
                    # processed in parallel.
                    image_info = {
                        "image_path": tmp_image_path,
                        "region": tile_bounds,
                        "image_id": region_request.image_id,
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
                for worker in tile_workers:
                    worker.join()

        logger.info(
            "Model Runner Stats Processed {} image tiles for region {}.".format(
                total_tile_count, region_request.region_bounds
            )
        )
    except Exception as err:
        logger.exception("File processing tiles: {}", err)
        raise ProcessTilesException("Failed to process tiles!") from err

    return total_tile_count


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


def create_gdal_translate_kwargs(
    image_format: ImageFormats, image_compression: ImageCompression, raster_dataset: gdal.Dataset
) -> Dict[str, Any]:
    """
    This function creates a set of keyword arguments suitable for passing to the gdal.Translate
    function. The values for these options are derived from the region processing request and
    the raster dataset itself.

    See: https://gdal.org/python/osgeo.gdal-module.html#Translate
    See: https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions

    :param image_format: ImageFormats = the format of the input image
    :param image_compression: ImageCompression = the compression used on the input image
    :param raster_dataset: gdal.Dataset = the raster dataset to translate

    :return: Dict[str, any] = the dictionary of translate keyword arguments
    """
    # Figure out what type of image this is and calculate a scale that does not force any range
    # remapping
    # TODO: Consider adding an option to have this driver perform the DRA. That option would change
    #       the scale_params output by this calculation
    output_type, scale_params = get_type_and_scales(raster_dataset)

    gdal_translate_kwargs = {
        "scaleParams": scale_params,
        "outputType": output_type,
        "format": image_format,
    }

    creation_options = ""
    if image_format == ImageFormats.NITF:
        # Creation options specific to the NITF raster driver.
        # See: https://gdal.org/drivers/raster/nitf.html
        if image_compression is None:
            # Default NITF tiles to JPEG2000 compression if not specified
            creation_options += "IC=C8"
        elif image_compression == ImageCompression.J2K:
            creation_options += "IC=C8"
        elif image_compression == ImageCompression.JPEG:
            creation_options += "IC=C3"
        elif image_compression == ImageCompression.NONE:
            creation_options += "IC=NC"
        else:
            logging.warning("Invalid compress specified for NITF image defaulting to JPEG2000!")
            creation_options += "IC=C8"

    if image_format == ImageFormats.GTIFF:
        # Creation options specific to the GeoTIFF raster driver.
        # See: https://gdal.org/drivers/raster/nitf.html
        if image_compression is None:
            # Default GeoTiff tiles to LZQ compression if not specified
            creation_options += "COMPRESS=LZW"
        elif image_compression == ImageCompression.LZW:
            creation_options += "COMPRESS=LZW"
        elif image_compression == ImageCompression.JPEG:
            creation_options += "COMPRESS=JPEG"
        elif image_compression == ImageCompression.NONE:
            creation_options += "COMPRESS=NONE"
        else:
            logging.warning("Invalid compress specified for GTIFF image defaulting to LZW!")
            creation_options += "COMPRESS=LZW"

    # TODO: Expand this to offer support for compression using other file formats

    if creation_options != "":
        gdal_translate_kwargs["creationOptions"] = creation_options

    return gdal_translate_kwargs


def ceildiv(a: int, b: int) -> int:
    """
    Integer ceiling division

    :param a: int = numerator
    :param b: int = denominator

    :return: ceil(a/b)
    """
    return -(-a // b)
