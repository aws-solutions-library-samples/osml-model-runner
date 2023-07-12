#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import argparse
import time
import uuid
from typing import List, Tuple

from osgeo import gdal, gdalconst


def get_type_and_scales(raster_dataset: gdal.Dataset) -> Tuple[int, List[List[int]]]:
    scale_params = []
    num_bands = raster_dataset.RasterCount
    output_type = gdalconst.GDT_Byte
    min = 0
    max = 255
    for band_num in range(1, num_bands + 1):
        band = raster_dataset.GetRasterBand(band_num)
        output_type = band.DataType
        if output_type == gdalconst.GDT_Byte:
            min = 0
            max = 255
        elif output_type == gdalconst.GDT_UInt16:
            min = 0
            max = 65535
        elif output_type == gdalconst.GDT_Int16:
            min = -32768
            max = 32767
        elif output_type == gdalconst.GDT_UInt32:
            min = 0
            max = 4294967295
        elif output_type == gdalconst.GDT_Int32:
            min = -2147483648
            max = 2147483647
        else:
            print("Image uses unsupported GDAL datatype {}. Defaulting to [0,255] range".format(output_type))

        scale_params.append([min, max, min, max])

    return output_type, scale_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="s3://...")
    parser.add_argument("-ts", "--tile-size", default=1024)
    parser.add_argument("-tf", "--tile-format", default="NITF")
    args = parser.parse_args()

    max_curl_chunk_size = 10 * 1024 * 1024
    # For information on these options and their usage please see:
    # https://gdal.org/user/configoptions.html
    gdal_default_environment_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_CACHEMAX": "75%",
        "GDAL_NUM_THREADS": "1",
        "GDAL_GCPS_TO_GEOTRANSFORM_APPROX_OK": "YES",
        "GDAL_MAX_DATASET_POOL_SIZE": "1000",
        "BIGTIFF_OVERVIEW": "YES",
        "USE_TILE_AS_BLOCK": "YES",
        # This flag will setup verbose output for GDAL. In particular it will show you each range
        # read for the file if using the /vsis3 virtual file system.
        "CPL_DEBUG": "ON",
        "CPL_VSIL_CURL_CHUNK_SIZE": str(max_curl_chunk_size),
        "CPL_VSIL_CURL_CACHE_SIZE": str(max_curl_chunk_size * 100),
        "VSI_CACHE": "YES",
        "VSI_CACHE_SIZE": "10000000000",
    }

    for key, val in gdal_default_environment_options.items():
        gdal.SetConfigOption(key, str(val))

    gdalvfs = args.image.replace("s3:/", "/vsis3", 1)

    print("******************************************************************")
    print(f"Opening Image: {gdalvfs}")
    print("******************************************************************")
    start = time.time()
    ds = gdal.Open(gdalvfs)
    width = ds.RasterXSize
    height = ds.RasterYSize
    num_bands = ds.RasterCount
    driver_name = ds.GetDriver().ShortName
    image_structure_metadata = ds.GetMetadata("IMAGE_STRUCTURE")
    compression = image_structure_metadata.get("COMPRESSION")
    nbits = image_structure_metadata.get("NBITS")
    xml_tres = ds.GetMetadata("xml:TRE")
    metadata_elapsed = time.time() - start

    output_type, scale_params = get_type_and_scales(ds)
    gdal_translate_kwargs = {
        "scaleParams": scale_params,
        "outputType": output_type,
        "format": args.tile_format,
    }
    creation_options = ""
    if args.tile_format == "NITF":
        creation_options += "IC=C8"
    if len(creation_options) > 0:
        gdal_translate_kwargs["creationOptions"] = creation_options

    tiles_elapsed = []
    #   srcWin --- subwindow in pixels to extract:
    #               [left_x, top_y, width, height]
    for src_win in [
        [0, 0, args.tile_size, args.tile_size],
        [
            (width - 1) - args.tile_size,
            (height - 1) - args.tile_size,
            args.tile_size,
            args.tile_size,
        ],
    ]:
        print("TILE: ***********************************")
        temp_ds_name = "/vsimem/" + str(uuid.uuid4())
        start = time.time()
        gdal.Translate(temp_ds_name, ds, srcWin=src_win, **gdal_translate_kwargs)
        tiles_elapsed.append(time.time() - start)

    print("******************************************************************")
    time_string = ""
    for value in tiles_elapsed:
        time_string += " {:>8.2f}".format(value)
    size_string = f"{width}x{height}x{num_bands}"
    print(f"METRICS: {driver_name:^10s} {size_string:>20s} {compression:^8s} {nbits} {metadata_elapsed:>8.2f}{time_string}")
    print("******************************************************************")
