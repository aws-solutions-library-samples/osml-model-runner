#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import unittest
from math import radians
from unittest import TestCase

import pytest


class TestGDALDemTileFactory(TestCase):
    def test_load_geotiff_tile(self):
        from aws.osml.model_runner.gdal.gdal_dem_tile_factory import GDALDigitalElevationModelTileFactory
        from aws.osml.model_runner.photogrammetry import GeodeticWorldCoordinate

        tile_factory = GDALDigitalElevationModelTileFactory("./test/data")
        elevation_array, sensor_model = tile_factory.get_tile("n47_e034_3arc_v2.tif")

        assert elevation_array is not None
        assert elevation_array.shape == (1201, 1201)
        assert sensor_model is not None

        center_image = sensor_model.world_to_image(GeodeticWorldCoordinate([radians(34.5), radians(47.5), 0.0]))

        assert center_image.x == pytest.approx(600.5, abs=1.0)
        assert center_image.y == pytest.approx(600.5, abs=1.0)


if __name__ == "__main__":
    unittest.main()
