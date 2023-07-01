#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import pathlib
import re
import sys
from typing import Iterator, TextIO
from urllib.parse import urlparse
from xml.etree import ElementTree

import boto3
from osgeo import gdal

SUBDATASET_NAME_PATTERN = re.compile("SUBDATASET_(\d+)_NAME")
IMAGE_SUFFIXES = [".ntf", ".NTF", ".nitf", ".NITF", ".tif", ".TIF", ".tiff", ".TIFF"]


def write_datasset_summary(image_name: str, ds: gdal.Dataset, text_output: TextIO) -> None:
    driver_name = ds.GetDriver().ShortName
    num_bands = ds.RasterCount
    image_structure_metadata = ds.GetMetadata("IMAGE_STRUCTURE")

    if ds.GetGeoTransform(can_return_null=True) is not None:
        geo_value = "GEO"
    elif ds.GetGCPs() is not None:
        geo_value = "GEO-GCP"
    else:
        geo_value = "None"

    tre_names = []
    xml_tres = ds.GetMetadata("xml:TRE")
    if xml_tres is not None and len(xml_tres) > 0:
        parsed_tres = ElementTree.fromstring(xml_tres[0])
        tre_names = [tre.get("name") for tre in parsed_tres.findall("./tre")]

    rpc_value = None
    for tre_name in tre_names:
        if tre_name.startswith("RPC"):
            rpc_value = tre_name
            break
    if not rpc_value:
        rpc_value = "None"

    num_polys = tre_names.count("RSMPCA")
    num_grids = tre_names.count("RSMGGA")
    num_adjustments = tre_names.count("RSMAPA")

    rsm_value = "None"
    if num_polys > 0:
        rsm_value = "POLY"
        if num_polys > 1:
            rsm_value += "-MSEG"
    if num_grids > 0:
        rsm_value = "GRID"
        if num_grids > 1:
            rsm_value += "-MSEG"
    if num_adjustments > 0:
        rsm_value += "-ADJ"

    text_output.write(
        f"{image_name}, "
        f"{driver_name}, "
        f"{ds.RasterXSize}, "
        f"{ds.RasterYSize}, "
        f"{num_bands}, "
        f"{image_structure_metadata.get('COMPRESSION')}, "
        f"{image_structure_metadata.get('NBITS')}, "
        f"{geo_value}, "
        f"{rpc_value}, "
        f"{rsm_value}"
        "\n"
    )


def create_summary(image_path: str, text_output: TextIO) -> None:
    ds = gdal.Open(image_path)
    if ds:
        print(f"Processing: {image_path}")
        subdatasets_metadata = ds.GetMetadata("SUBDATASETS")
        if len(subdatasets_metadata.keys()) > 0:
            for key, value in subdatasets_metadata.items():
                match = SUBDATASET_NAME_PATTERN.match(key)
                if match:
                    write_datasset_summary(value, gdal.Open(value), text_output)
        else:
            write_datasset_summary(image_path, ds, text_output)


def s3_image_listing(root_path: str) -> Iterator[str]:
    parsed_url = urlparse(root_path)
    bucket_name = parsed_url.netloc
    key_prefix = parsed_url.path[1:]

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    kwargs = {"Bucket": bucket_name, "Prefix": key_prefix}
    for page in paginator.paginate(**kwargs):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]

                # This is a hack because some of the development test data buckets have 1000s of tiles from the
                # SpaceNet buildings challenges. They're not really relevant for AIP so they're excluded here.
                if "SN2_buildings" in key:
                    continue
                for suffix in IMAGE_SUFFIXES:
                    if key.endswith(suffix):
                        yield str(f"/vsis3/{bucket_name}/{key}")
                        break


def local_image_listing(root_path: str) -> Iterator[str]:
    p = pathlib.Path(root_path)

    for extension in IMAGE_SUFFIXES:
        for name in p.glob(f"*{extension}"):
            yield str(pathlib.Path(p, name))


if __name__ == "__main__":
    if sys.argv[1].startswith("s3:"):
        image_list_generator = s3_image_listing
    else:
        image_list_generator = local_image_listing

    with open(sys.argv[2], "w") as csv_output_file:
        csv_output_file.write("image,format,width,height,bands,compression,bits,geo,rpc,rsm\n")
        for image_path in image_list_generator(sys.argv[1]):
            create_summary(image_path, csv_output_file)
