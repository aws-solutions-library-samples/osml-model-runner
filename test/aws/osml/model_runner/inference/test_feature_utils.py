#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import unittest
from typing import List

import geojson
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

        ds, sensor_model = self.get_dataset_and_camera()

        roi = shapely.wkt.loads("POLYGON ((8 50, 10 50, 10 60, 8 60, 8 50))")

        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)

        assert processing_bounds == ((0, 0), (101, 101))

    def test_calculate_processing_bounds_intersect(self):
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds

        ds, sensor_model = self.get_dataset_and_camera()

        roi = shapely.wkt.loads("POLYGON ((8 52, 9.001043490711101 52.0013898967889, 9 54, 8 54, 8 52))")

        # Manually verify the lon/lat coordinates of the image positions used in this test with these
        # print statements
        # print(sensor_model.image_to_world((0, 0)))
        # print(sensor_model.image_to_world((50, 50)))
        # print(sensor_model.image_to_world((101, 101)))
        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)

        # Processing bounds is in ((r, c), (w, h))
        assert processing_bounds == ((0, 0), (50, 50))

    def test_calculate_processing_bounds_chip(self):
        from aws.osml.model_runner.inference.feature_utils import calculate_processing_bounds

        ds, sensor_model = self.get_dataset_and_camera()
        roi = shapely.wkt.loads(
            "POLYGON (("
            "8.999932379599102 52.0023621190119, 8.999932379599102 52.0002787856769, "
            "9.001599046267101 52.0002787856769, 9.001599046267101 52.0023621190119, "
            "8.999932379599102 52.0023621190119"
            "))"
        )

        # Manually verify the lon/lat coordinates of the image positions used in this test with these
        # print statements
        # print(sensor_model.image_to_world((10, 15)))
        # print(sensor_model.image_to_world((70, 90)))
        processing_bounds = calculate_processing_bounds(ds, roi, sensor_model)

        # Processing bounds is in ((r, c), (w, h))
        assert processing_bounds == ((15, 10), (60, 75))

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
