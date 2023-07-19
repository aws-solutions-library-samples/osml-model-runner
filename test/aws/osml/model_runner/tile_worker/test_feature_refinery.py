#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import unittest
from secrets import token_hex

import geojson
from defusedxml import ElementTree


class TestFeatureRefinery(unittest.TestCase):
    def setUp(self) -> None:
        from aws.osml.model_runner.tile_worker.feature_refinery import FeatureRefinery

        self.sensor_model = self.build_sensor_model()
        self.sample_geojson_detections = self.build_geojson_detections()
        self.synthetic_detections = self.build_synthetic_detections()
        self.feature_refinery = FeatureRefinery(self.sensor_model)

    def test_build_features_for_sparse_tile(self):
        self.feature_refinery.refine_features_for_tile(self.sample_geojson_detections)
        for feature in self.sample_geojson_detections:
            assert feature["geometry"]["type"] == "Polygon"
            assert "bbox" in feature
            assert "center_latitude" in feature["properties"]
            assert "center_longitude" in feature["properties"]

    def test_build_features_for_dense_tile(self):
        self.feature_refinery.refine_features_for_tile(self.synthetic_detections)
        for feature in self.synthetic_detections:
            assert feature["geometry"]["type"] == "Polygon"
            assert "bbox" in feature
            assert "center_latitude" in feature["properties"]
            assert "center_longitude" in feature["properties"]

    @staticmethod
    def build_geojson_detections():
        with open("./test/data/detections.geojson", "r") as geojson_file:
            return geojson.load(geojson_file)["features"]

    @staticmethod
    def build_synthetic_detections():
        features_per_row = 40
        feature_stride = 15
        feature_size = 5
        selected_features = []
        for y in range(0, features_per_row):
            for x in range(0, features_per_row):
                selected_features.append(
                    geojson.Feature(
                        id=token_hex(16),
                        geometry=None,
                        properties={
                            "bounds_imcoords": [
                                x * feature_stride,
                                y * feature_stride,
                                x * feature_stride + feature_size,
                                y * feature_stride + feature_size,
                            ]
                        },
                    )
                )
        return selected_features

    @staticmethod
    def build_sensor_model():
        from aws.osml.gdal.sensor_model_factory import SensorModelFactory, SensorModelTypes

        with open("test/data/sample-metadata-ms-rpc00b.xml", "rb") as xml_file:
            xml_tres = ElementTree.parse(xml_file)
            sensor_model_builder = SensorModelFactory(
                2048,
                2048,
                xml_tres=xml_tres,
                selected_sensor_model_types=[SensorModelTypes.PROJECTIVE, SensorModelTypes.RPC],
            )
            return sensor_model_builder.build()


if __name__ == "__main__":
    unittest.main()
