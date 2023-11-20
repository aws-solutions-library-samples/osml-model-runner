#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import unittest
from math import degrees
from typing import List

import geojson
import numpy as np
import pytest
import shapely


class TestFeatureUtils(unittest.TestCase):
    def test_features_conversion_none(self):
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        shapes = features_to_image_shapes(self.build_gdal_sensor_model(), None)
        assert len(shapes) == 0

    def test_features_conversion_no_geometry(self):
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        malformed_feature = {"id": "test_feature"}
        with pytest.raises(ValueError) as e_info:
            features_to_image_shapes(self.build_gdal_sensor_model(), [malformed_feature])
        assert str(e_info.value) == "Feature does not contain a valid geometry"

    def test_features_conversion_unsupported_type(self):
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        malformed_feature = {
            "id": "test_feature",
            "geometry": {"type": "NewType", "coordinates": [-77.0364, 38.8976, 0.0]},
        }
        with pytest.raises(ValueError) as e_info:
            features_to_image_shapes(self.build_gdal_sensor_model(), [malformed_feature])
        assert str(e_info.value) == "Unable to convert feature due to unrecognized or invalid geometry"

    def test_features_conversion(self):
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        with open("./test/data/feature_examples.geojson", "r") as geojson_file:
            features: List[geojson.Feature] = geojson.load(geojson_file)["features"]

        # We should have 1 feature for each of the 6 geojson types
        assert len(features) == 6

        shapes = features_to_image_shapes(self.build_gdal_sensor_model(), features)
        assert len(shapes) == len(features)
        assert isinstance(shapes[0], shapely.geometry.Point)
        assert isinstance(shapes[1], shapely.geometry.MultiPoint)
        assert isinstance(shapes[2], shapely.geometry.LineString)
        assert isinstance(shapes[3], shapely.geometry.MultiLineString)
        assert isinstance(shapes[4], shapely.geometry.Polygon)
        assert isinstance(shapes[5], shapely.geometry.MultiPolygon)

    def test_polygon_feature_conversion(self):
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        sample_image_bounds = [(0, 0), (19584, 0), (19584, 19584), (0, 19584)]
        polygon_feature: geojson.Feature = geojson.Feature(
            geometry=geojson.geometry.Polygon(
                [
                    (
                        [-43.681640625, -22.939453125, 0.0],
                        [-43.59375, -22.939453125, 0.0],
                        [-43.59375, -23.02734375, 0.0],
                        [-43.681640625, -23.02734375, 0.0],
                        [-43.681640625, -22.939453125, 0.0],
                    )
                ]
            )
        )

        shape = features_to_image_shapes(self.build_gdal_sensor_model(), [polygon_feature])[0]

        assert isinstance(shape, shapely.geometry.Polygon)
        for i in range(0, len(sample_image_bounds)):
            print("TEST: " + str(i))
            print("SIB: " + str(sample_image_bounds[i]))
            print("SEC: " + str(shape.exterior.coords[i]))
            assert pytest.approx(sample_image_bounds[i], rel=0.49, abs=0.49) == shape.exterior.coords[i]

    def test_calculate_processing_bounds_no_roi(self):
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds

        ds, sensor_model = self.get_dataset_and_camera()

        processing_bounds = calculate_processing_bounds(ds, None, sensor_model)

        assert processing_bounds == ((0, 0), (101, 101))

    def test_calculate_processing_bounds_full_image(self):
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds
        from aws.osml.photogrammetry import ImageCoordinate

        ds, sensor_model = self.get_dataset_and_camera()

        chip_ul = sensor_model.image_to_world(ImageCoordinate([0, 0]))
        chip_lr = sensor_model.image_to_world(ImageCoordinate([101, 101]))
        min_vals = np.minimum(chip_ul.coordinate, chip_lr.coordinate)
        max_vals = np.maximum(chip_ul.coordinate, chip_lr.coordinate)
        polygon_coords = []
        polygon_coords.append([degrees(min_vals[0]), degrees(min_vals[1])])
        polygon_coords.append([degrees(min_vals[0]), degrees(max_vals[1])])
        polygon_coords.append([degrees(max_vals[0]), degrees(max_vals[1])])
        polygon_coords.append([degrees(max_vals[0]), degrees(min_vals[1])])
        polygon_coords.append([degrees(min_vals[0]), degrees(min_vals[1])])
        roi = shapely.geometry.Polygon(polygon_coords)

        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)

        assert processing_bounds == ((0, 0), (101, 101))

    def test_calculate_processing_bounds_intersect(self):
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds
        from aws.osml.photogrammetry import ImageCoordinate

        ds, sensor_model = self.get_dataset_and_camera()

        chip_ul = sensor_model.image_to_world(ImageCoordinate([-10, -10]))
        chip_lr = sensor_model.image_to_world(ImageCoordinate([50, 50]))
        min_vals = np.minimum(chip_ul.coordinate, chip_lr.coordinate)
        max_vals = np.maximum(chip_ul.coordinate, chip_lr.coordinate)
        polygon_coords = []
        polygon_coords.append([degrees(min_vals[0]), degrees(min_vals[1])])
        polygon_coords.append([degrees(min_vals[0]), degrees(max_vals[1])])
        polygon_coords.append([degrees(max_vals[0]), degrees(max_vals[1])])
        polygon_coords.append([degrees(max_vals[0]), degrees(min_vals[1])])
        polygon_coords.append([degrees(min_vals[0]), degrees(min_vals[1])])
        roi = shapely.geometry.Polygon(polygon_coords)

        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)

        # Processing bounds is in ((r, c), (w, h))
        assert processing_bounds == ((0, 0), (50, 50))

    def test_calculate_processing_bounds_chip(self):
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds
        from aws.osml.photogrammetry import ImageCoordinate

        ds, sensor_model = self.get_dataset_and_camera()

        chip_ul = sensor_model.image_to_world(ImageCoordinate([10, 15]))
        chip_lr = sensor_model.image_to_world(ImageCoordinate([70, 90]))
        min_vals = np.minimum(chip_ul.coordinate, chip_lr.coordinate)
        max_vals = np.maximum(chip_ul.coordinate, chip_lr.coordinate)
        polygon_coords = []
        polygon_coords.append([degrees(min_vals[0]), degrees(min_vals[1])])
        polygon_coords.append([degrees(min_vals[0]), degrees(max_vals[1])])
        polygon_coords.append([degrees(max_vals[0]), degrees(max_vals[1])])
        polygon_coords.append([degrees(max_vals[0]), degrees(min_vals[1])])
        polygon_coords.append([degrees(min_vals[0]), degrees(min_vals[1])])
        roi = shapely.geometry.Polygon(polygon_coords)

        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)

        # Processing bounds is in ((r, c), (w, h))
        assert processing_bounds == ((15, 10), (60, 75))

    def test_get_source_property_not_available(self):
        from aws.osml.model_runner.inference.feature_utils import get_source_property

        ds, sensor_model = self.get_dataset_and_camera()
        source_property = get_source_property("UNSUPPORTED", ds)
        assert source_property is None

    def test_get_source_property_exception(self):
        from aws.osml.model_runner.inference.feature_utils import get_source_property

        source_property = get_source_property("NITF", dataset=None)
        assert source_property is None

    @staticmethod
    def build_gdal_sensor_model():
        from aws.osml.photogrammetry import GDALAffineSensorModel

        # Test coordinate calculations using geotransform matrix from sample SpaceNet RIO image
        transform = [
            -43.681640625,
            4.487879136029412e-06,
            0.0,
            -22.939453125,
            0.0,
            -4.487879136029412e-06,
        ]
        return GDALAffineSensorModel(transform)

    @staticmethod
    def get_dataset_and_camera():
        from aws.osml.gdal.gdal_utils import load_gdal_dataset

        ds, sensor_model = load_gdal_dataset("./test/data/GeogToWGS84GeoKey5.tif")
        return ds, sensor_model


if __name__ == "__main__":
    unittest.main()
