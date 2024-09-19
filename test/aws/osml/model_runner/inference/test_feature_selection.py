#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase

import geojson
from geojson import Feature, Point


class TestFeatureSelection(TestCase):
    def test_feature_selection_empty_list(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_option = FeatureDistillationNMS()
        feature_selector = FeatureSelector(options=feature_selection_option)

        features = feature_selector.select_features([])
        assert len(features) == 0

    def test_feature_selection_none_list(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_option = FeatureDistillationNMS()
        feature_selector = FeatureSelector(options=feature_selection_option)

        features = feature_selector.select_features(None)
        assert len(features) == 0

    def test_feature_selection_nms_no_overlap(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_option = FeatureDistillationNMS()
        feature_selector = FeatureSelector(options=feature_selection_option)

        with open("./test/data/detections.geojson", "r") as geojson_file:
            sample_features = geojson.load(geojson_file)["features"]
        assert len(sample_features) > 0

        processed_features = feature_selector.select_features(sample_features)
        assert len(sample_features) == len(processed_features)

    def test_feature_selection_nms_overlaps(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_option = FeatureDistillationNMS()
        feature_selector = FeatureSelector(options=feature_selection_option)

        # The actual geojson isn't used by nms. The only thing we care about for determining
        # duplicates is the bounds_imcoords property so these shapes are all just simple
        # points
        # feature_a and feature_b have IoU ~0.80
        original_features = [
            Feature(
                id="feature_a",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [50, 50, 100, 100]},
            ),
            Feature(
                id="feature_b",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [45, 45, 101, 101]},
            ),
            Feature(
                id="feature_c",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [250, 250, 300, 275]},
            ),
        ]
        processed_features = feature_selector.select_features(original_features)
        assert len(processed_features) == 2

    def test_feature_selection_no_selection(self):
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selector = FeatureSelector()

        # feature_a and feature_b have IoU ~0.80
        original_features = [
            Feature(
                id="feature_a",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [50, 50, 100, 100]},
            ),
            Feature(
                id="feature_b",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [45, 45, 101, 101]},
            ),
            Feature(
                id="feature_c",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [250, 250, 300, 275]},
            ),
        ]
        processed_features = feature_selector.select_features(original_features)
        assert processed_features == original_features

    def test_feature_selection_unknown_algorithm(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS, MRPostProcessingAlgorithmType
        from aws.osml.model_runner.inference import FeatureSelector
        from aws.osml.model_runner.inference.exceptions import FeatureDistillationException

        class FakeFeatureSelectionAlgorithm(MRPostProcessingAlgorithmType):
            MAGIC = "MAGIC"

        feature_selection_option = FeatureDistillationNMS(algorithm_type=FakeFeatureSelectionAlgorithm.MAGIC)
        feature_selector = FeatureSelector(options=feature_selection_option)

        # feature_a and feature_b have IoU ~0.80
        original_features = [
            Feature(
                id="feature_a",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [50, 50, 100, 100]},
            ),
            Feature(
                id="feature_b",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [45, 45, 101, 101]},
            ),
            Feature(
                id="feature_c",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [250, 250, 300, 275]},
            ),
        ]
        with self.assertRaises(FeatureDistillationException):
            feature_selector.select_features(original_features)

    def test_feature_selection_nms_overlaps_custom_threshold(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_option_1 = FeatureDistillationNMS(iou_threshold=0.4)
        feature_selector_1 = FeatureSelector(options=feature_selection_option_1)
        feature_selection_option_2 = FeatureDistillationNMS(iou_threshold=0.7)
        feature_selector_2 = FeatureSelector(options=feature_selection_option_2)

        # These features have an IoU of ~0.47
        original_features = [
            Feature(
                id="feature_a",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [40, 40, 90, 90]},
            ),
            Feature(
                id="feature_b",
                geometry=Point((0, 0)),
                properties={"bounds_imcoords": [50, 50, 100, 100]},
            ),
        ]
        assert len(feature_selector_1.select_features(original_features)) == 1
        assert len(feature_selector_2.select_features(original_features)) == 2

    def test_feature_selection_nms_overlaps_multiple_categories(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_options = FeatureDistillationNMS()
        feature_selector = FeatureSelector(options=feature_selection_options)

        # feature_a and feature_b have IoU ~0.80 but different labels
        original_features = [
            Feature(
                id="feature_a",
                geometry=Point((0, 0)),
                properties={
                    "bounds_imcoords": [50, 50, 100, 100],
                    "featureClasses": [{"iri": "ground_motor_passenger_vehicle", "score": 0.45}],
                },
            ),
            Feature(
                id="feature_b",
                geometry=Point((0, 0)),
                properties={
                    "bounds_imcoords": [45, 45, 101, 101],
                    "featureClasses": [{"iri": "boat", "score": 0.57}],
                },
            ),
            Feature(
                id="feature_c",
                geometry=Point((0, 0)),
                properties={
                    "bounds_imcoords": [250, 250, 300, 275],
                    "featureClasses": [{"iri": "ground_motor_passenger_vehicle", "score": 0.80}],
                },
            ),
        ]
        processed_features = feature_selector.select_features(original_features)
        assert len(processed_features) == 3

    def test_feature_selection_nms_point_feature(self):
        from aws.osml.model_runner.common import FeatureDistillationNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_option = FeatureDistillationNMS()
        feature_selector = FeatureSelector(options=feature_selection_option)

        test_feature = [
            Feature(
                geometry=Point((85.000111, 32.983222, 0.0)),
                id="point-feature",
                properties={
                    "bounds_imcoords": [409.6, 409.6, 409.6, 409.6],
                    "featureClasses": [{"iri": "boat", "score": 0.85}],
                },
            )
        ]
        processed_features = feature_selector.select_features(test_feature)
        assert len(processed_features) == 1

    def test_feature_selection_soft_nms_overlaps(self):
        from aws.osml.model_runner.common import FeatureDistillationSoftNMS
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selection_option = FeatureDistillationSoftNMS()
        feature_selector = FeatureSelector(options=feature_selection_option)

        # feature_a and feature_b have IoU ~0.80
        original_features = [
            Feature(
                id="feature_a",
                geometry=Point((0, 0)),
                properties={
                    "bounds_imcoords": [50, 50, 100, 100],
                    "featureClasses": [{"iri": "boat", "score": 0.85}],
                },
            ),
            Feature(
                id="feature_b",
                geometry=Point((0, 0)),
                properties={
                    "bounds_imcoords": [45, 45, 101, 101],
                    "featureClasses": [{"iri": "boat", "score": 0.93}],
                },
            ),
            Feature(
                id="feature_c",
                geometry=Point((0, 0)),
                properties={
                    "bounds_imcoords": [250, 255, 300, 275],
                    "featureClasses": [{"iri": "boat", "score": 0.80}],
                },
            ),
        ]
        processed_features = feature_selector.select_features(original_features)
        assert len(processed_features) == 3
        processed_properties = [feature["properties"] for feature in processed_features]
        expected_properties = [
            {
                "bounds_imcoords": [45, 45, 101, 101],
                "featureClasses": [{"iri": "boat", "score": 0.93, "rawScore": 0.93}],
            },
            {
                "bounds_imcoords": [250, 255, 300, 275],
                "featureClasses": [{"iri": "boat", "score": 0.80, "rawScore": 0.80}],
            },
            {
                "bounds_imcoords": [50, 50, 100, 100],
                "featureClasses": [
                    {
                        "iri": "boat",
                        "score": 0.85,
                        "rawScore": 0.85,
                    }
                ],
            },
        ]
        assert processed_properties == expected_properties

    def test_feature_selection_soft_nms_single_feature(self):
        from aws.osml.model_runner.inference import FeatureSelector

        feature_selector = FeatureSelector()

        test_feature = [
            Feature(
                bbox=[85.00011108395785, 32.98316663951341, 85.00016669381992, 32.98322224937548],
                geometry={
                    "coordinates": [
                        [
                            [85.000111, 32.983222, 0.0],
                            [85.000111, 32.983167, 0.0],
                            [85.000167, 32.983167, 0.0],
                            [85.000167, 32.983222, 0.0],
                            [85.000111, 32.983222, 0.0],
                        ]
                    ],
                    "type": "Polygon",
                },
                id="5d91b723be67911e407b30f5645e61d9",
                properties={
                    "bounds_imcoords": [409.6, 409.6, 614.4, 614.4],
                    "center_latitude": 32.98319444444444,
                    "center_longitude": 85.00013888888888,
                    "detection": {
                        "coordinates": [],
                        "ontology": [{"detectionScore": 1.0, "iri": "sample_object"}],
                        "pixelCoordinates": [[409.6, 409.6], [409.6, 614.4], [614.4, 614.4], [614.4, 409.6]],
                        "type": "Polygon",
                    },
                    "detection_score": 1.0,
                    "feature_types": {"sample_object": 1.0},
                    "image_id": "test-image-id:./test/data/small.ntf",
                    "inferenceTime": "2023-02-21T05:02:00.917276",
                },
                type="Feature",
            )
        ]
        processed_features = feature_selector.select_features(test_feature)
        assert len(processed_features) == 1
