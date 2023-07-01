#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import re
import sys
from xml.etree import ElementTree

from osgeo import gdal

SUBDATASET_NAME_PATTERN = re.compile("SUBDATASET_(\d+)_NAME")


def print_datasset_summary(ds: gdal.Dataset) -> None:
    print("****************************************************************")
    print(f"Size:         {ds.RasterXSize} x {ds.RasterYSize}")
    print(f"GeoTransform: {ds.GetGeoTransform(can_return_null=True) is not None}")
    print(f"GCPs:         {ds.GetGCPs() is not None}")

    image_structure_metadata = ds.GetMetadata("IMAGE_STRUCTURE")
    print(f"Compression:  {image_structure_metadata.get('COMPRESSION')}")
    print(f"NBITS:        {image_structure_metadata.get('NBITS')}")

    num_bands = ds.RasterCount
    print(f"Num Bands:    {num_bands}")
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        print(
            f"  Band {i}: Extent {band.XSize} x {band.YSize} "
            f"Blocking {band.GetBlockSize()} "
            f"Type {gdal.GetDataTypeName(band.DataType)}"
        )

    tre_names = []
    xml_tres = ds.GetMetadata("xml:TRE")
    if xml_tres is not None and len(xml_tres) > 0:
        parsed_tres = ElementTree.fromstring(xml_tres[0])
        tre_names = [tre.get("name") for tre in parsed_tres.findall("./tre")]
    print(f"TREs: {tre_names}")
    print("****************************************************************")


if __name__ == "__main__":
    image_path = sys.argv[1]
    ds = gdal.Open(image_path)

    driver_name = ds.GetDriver().ShortName
    print(f"Driver:       {driver_name}")

    subdatasets_metadata = ds.GetMetadata("SUBDATASETS")
    if len(subdatasets_metadata.keys()) > 0:
        print("MULTI IMAGE DATASET")
        for key, value in subdatasets_metadata.items():
            match = SUBDATASET_NAME_PATTERN.match(key)
            if match:
                print("****************************************************************")
                print(f"IMAGE {match.group(1)}")
                print_datasset_summary(gdal.Open(value))
    else:
        print("SINGLE IMAGE DATASET")
        print_datasset_summary(ds)
