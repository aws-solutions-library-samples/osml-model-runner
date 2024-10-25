#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from math import degrees
from typing import List

import geojson
import numpy as np
import pytest
import shapely
from osgeo import gdal

gdal.DontUseExceptions()


class TestFeatureUtils(unittest.TestCase):
    def test_features_conversion_none(self):
        """
        Test that converting features with None input returns an empty list.
        Ensure that an exception is raised if skipping invalid inputs is disabled.
        """
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        shapes = features_to_image_shapes(self.build_gdal_sensor_model(), None)
        assert len(shapes) == 0

        with self.assertRaises(ValueError):
            features_to_image_shapes(self.build_gdal_sensor_model(), None, False)

    def test_features_conversion_no_geometry(self):
        """
        Test that features without 'geometry' are skipped.
        Ensure that an exception is raised if skipping invalid inputs is disabled.
        """
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        malformed_feature = {"id": "test_feature"}
        shapes = features_to_image_shapes(self.build_gdal_sensor_model(), [malformed_feature])
        assert len(shapes) == 0

        with self.assertRaises(ValueError):
            features_to_image_shapes(self.build_gdal_sensor_model(), [malformed_feature], False)

    def test_features_conversion_unsupported_type(self):
        """
        Test that features with unsupported geometry types are skipped.
        Ensure that an exception is raised if skipping invalid inputs is disabled.
        """
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        malformed_feature = {
            "id": "test_feature",
            "geometry": {"type": "NewType", "coordinates": [-77.0364, 38.8976, 0.0]},
        }
        shapes = features_to_image_shapes(self.build_gdal_sensor_model(), [malformed_feature])
        assert len(shapes) == 0

        with self.assertRaises(ValueError):
            features_to_image_shapes(self.build_gdal_sensor_model(), [malformed_feature], False)

    def test_features_conversion(self):
        """
        Test converting valid GeoJSON features to Shapely shapes.
        Ensure that the conversion matches the expected geometry types.
        """
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        with open("./test/data/feature_examples.geojson", "r") as geojson_file:
            features: List[geojson.Feature] = geojson.load(geojson_file)["features"]

        assert len(features) == 6
        shapes = features_to_image_shapes(self.build_gdal_sensor_model(), features)
        assert len(shapes) == len(features)
        assert isinstance(shapes[0], shapely.geometry.Point)
        assert isinstance(shapes[1], shapely.geometry.MultiPoint)
        assert isinstance(shapes[2], shapely.geometry.LineString)
        assert isinstance(shapes[3], shapely.geometry.MultiLineString)
        assert isinstance(shapes[4], shapely.geometry.Polygon)
        assert isinstance(shapes[5], shapely.geometry.MultiPolygon)

    def test_features_conversion_mixed_skip(self):
        """
        Test processing a mix of valid and invalid features with skipping enabled.
        Ensure that only valid features are converted.
        """
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        with open("./test/data/feature_examples.geojson", "r") as geojson_file:
            features: List[geojson.Feature] = geojson.load(geojson_file)["features"]

        features.append({"id": "test_feature"})  # Add invalid feature
        shapes = features_to_image_shapes(self.build_gdal_sensor_model(), features)
        assert len(shapes) == 6

    def test_features_conversion_mixed_no_skip(self):
        """
        Test processing a mix of valid and invalid features with skipping disabled.
        Ensure that an exception is raised when encountering invalid features.
        """
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        with open("./test/data/feature_examples.geojson", "r") as geojson_file:
            features: List[geojson.Feature] = geojson.load(geojson_file)["features"]

        features.append({"id": "test_feature"})  # Add invalid feature

        with self.assertRaises(ValueError):
            features_to_image_shapes(self.build_gdal_sensor_model(), features, False)

    def test_polygon_feature_conversion(self):
        """
        Test converting a GeoJSON polygon to a Shapely polygon.
        Ensure that the resulting shape matches expected coordinates.
        """
        from aws.osml.model_runner.inference.feature_utils import features_to_image_shapes

        sample_image_bounds = [(0, 0), (19584, 0), (19584, 19584), (0, 19584)]
        polygon_feature = geojson.Feature(
            geometry=geojson.geometry.Polygon(
                [
                    [
                        [-43.681640625, -22.939453125, 0.0],
                        [-43.59375, -22.939453125, 0.0],
                        [-43.59375, -23.02734375, 0.0],
                        [-43.681640625, -23.02734375, 0.0],
                        [-43.681640625, -22.939453125, 0.0],
                    ]
                ]
            )
        )
        shape = features_to_image_shapes(self.build_gdal_sensor_model(), [polygon_feature])[0]
        assert isinstance(shape, shapely.geometry.Polygon)
        for i in range(len(sample_image_bounds)):
            assert pytest.approx(sample_image_bounds[i], rel=0.49, abs=0.49) == shape.exterior.coords[i]

    def test_convert_nested_coordinate_lists_single_vs_nested(self):
        """
        Test the conversion of single versus nested coordinates using a mock conversion function.
        """
        from aws.osml.model_runner.inference.feature_utils import convert_nested_coordinate_lists

        single_coord = [-77.0364, 38.8976]
        nested_coords = [[-77.0364, 38.8976], [-77.0365, 38.8977]]

        converted_single = convert_nested_coordinate_lists(single_coord, lambda x: x)
        converted_nested = convert_nested_coordinate_lists(nested_coords, lambda x: x)

        assert isinstance(converted_single, tuple)
        assert isinstance(converted_nested, list)
        assert len(converted_nested) == 2

    def test_calculate_processing_bounds_no_roi(self):
        """
        Test calculating processing bounds without an ROI; should return the full image dimensions.
        """
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds

        ds, sensor_model = self.get_dataset_and_camera()
        processing_bounds = calculate_processing_bounds(ds, None, sensor_model)
        assert processing_bounds == ((0, 0), (101, 101))

    def test_calculate_processing_bounds_full_image(self):
        """
        Test calculating processing bounds with an ROI covering the full image.
        """
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds
        from aws.osml.photogrammetry import ImageCoordinate

        ds, sensor_model = self.get_dataset_and_camera()
        chip_ul = sensor_model.image_to_world(ImageCoordinate([0, 0]))
        chip_lr = sensor_model.image_to_world(ImageCoordinate([101, 101]))
        min_vals = np.minimum(chip_ul.coordinate, chip_lr.coordinate)
        max_vals = np.maximum(chip_ul.coordinate, chip_lr.coordinate)
        polygon_coords = [
            [degrees(min_vals[0]), degrees(min_vals[1])],
            [degrees(min_vals[0]), degrees(max_vals[1])],
            [degrees(max_vals[0]), degrees(max_vals[1])],
            [degrees(max_vals[0]), degrees(min_vals[1])],
            [degrees(min_vals[0]), degrees(min_vals[1])],
        ]
        roi = shapely.geometry.Polygon(polygon_coords)

        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)
        assert processing_bounds == ((0, 0), (101, 101))

    def test_calculate_processing_bounds_intersect(self):
        """
        Test calculating processing bounds with an ROI partially intersecting the image.
        """
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds
        from aws.osml.photogrammetry import ImageCoordinate

        ds, sensor_model = self.get_dataset_and_camera()
        chip_ul = sensor_model.image_to_world(ImageCoordinate([-10, -10]))
        chip_lr = sensor_model.image_to_world(ImageCoordinate([50, 50]))
        min_vals = np.minimum(chip_ul.coordinate, chip_lr.coordinate)
        max_vals = np.maximum(chip_ul.coordinate, chip_lr.coordinate)
        polygon_coords = [
            [degrees(min_vals[0]), degrees(min_vals[1])],
            [degrees(min_vals[0]), degrees(max_vals[1])],
            [degrees(max_vals[0]), degrees(max_vals[1])],
            [degrees(max_vals[0]), degrees(min_vals[1])],
            [degrees(min_vals[0]), degrees(min_vals[1])],
        ]
        roi = shapely.geometry.Polygon(polygon_coords)

        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)
        assert processing_bounds == ((0, 0), (50, 50))

    def test_calculate_processing_bounds_chip(self):
        """
        Test calculating processing bounds for a specific chip within the image.
        """
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds
        from aws.osml.photogrammetry import ImageCoordinate

        ds, sensor_model = self.get_dataset_and_camera()
        chip_ul = sensor_model.image_to_world(ImageCoordinate([10, 15]))
        chip_lr = sensor_model.image_to_world(ImageCoordinate([70, 90]))
        min_vals = np.minimum(chip_ul.coordinate, chip_lr.coordinate)
        max_vals = np.maximum(chip_ul.coordinate, chip_lr.coordinate)
        polygon_coords = [
            [degrees(min_vals[0]), degrees(min_vals[1])],
            [degrees(min_vals[0]), degrees(max_vals[1])],
            [degrees(max_vals[0]), degrees(max_vals[1])],
            [degrees(max_vals[0]), degrees(min_vals[1])],
            [degrees(min_vals[0]), degrees(min_vals[1])],
        ]
        roi = shapely.geometry.Polygon(polygon_coords)

        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)
        assert processing_bounds == ((15, 10), (60, 75))

    def test_get_source_property_not_available(self):
        """
        Test retrieving a source property for an unsupported image type; should return None.
        """
        from aws.osml.model_runner.inference.feature_utils import get_source_property

        ds, sensor_model = self.get_dataset_and_camera()
        source_property = get_source_property("./test/data/GeogToWGS84GeoKey5.tif", "UNSUPPORTED", ds)
        assert source_property is None

    def test_get_source_property_exception(self):
        """
        Test that getting a source property handles exceptions gracefully and returns None.
        """
        from aws.osml.model_runner.inference.feature_utils import get_source_property

        source_property = get_source_property("./test/data/GeogToWGS84GeoKey5.tif", "NITF", dataset=None)
        assert source_property is None

    @staticmethod
    def build_gdal_sensor_model():
        from aws.osml.photogrammetry import GDALAffineSensorModel

        # Create a mock GDAL sensor model for testing transformations
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
