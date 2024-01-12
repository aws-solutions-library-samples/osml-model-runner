#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

SAMPLE_REGION_REQUEST_DATA = {
    "tile_size": (10, 10),
    "tile_overlap": (1, 1),
    "tile_format": "NITF",
    "image_id": "test-image-id",
    "image_url": "test-image-url",
    "region_bounds": ((0, 0), (50, 50)),
    "model_name": "test-model-name",
    "model_invoke_mode": "SM_ENDPOINT",
    "output_bucket": "unit-test",
    "output_prefix": "region-request",
    "execution_role": "arn:aws:iam::012345678910:role/OversightMLBetaInvokeRole",
}


class TestRequestUtils(unittest.TestCase):
    def setUp(self):
        self.sample_request_data = self.build_request_data()

    def tearDown(self):
        self.sample_request_data = None

    def test_invalid_request_image_id(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.image_id = ""
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_request_image_url(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.image_url = ""
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_request_model_name(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.model_name = ""
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_request_model_invoke_mode(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.model_invoke_mode = None
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_request_tile_size(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.tile_size = None
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_size = 0
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_size = (-1, 0)
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_size = (-1, -1)
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_tile_overlap(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.tile_overlap = None
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_overlap = 0
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_overlap = (-1, 0)
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_overlap = (0, -1)
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_overlap = (-1, -1)
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_overlap = (10, 10)
        self.sample_request_data.tile_size = (5, 12)
        assert not shared_properties_are_valid(self.sample_request_data)

        self.sample_request_data.tile_overlap = (10, 10)
        self.sample_request_data.tile_size = (12, 5)
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_request_tile_format(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.tile_format = None
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_request_image_read_role(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.image_read_role = "012345678910:role/OversightMLS3ReadOnly"
        assert not shared_properties_are_valid(self.sample_request_data)

    def test_invalid_request_model_invocation_role(self):
        from aws.osml.model_runner.api.request_utils import shared_properties_are_valid

        self.sample_request_data.model_invocation_role = "012345678910:role/OversightMLModelInvoker"
        assert not shared_properties_are_valid(self.sample_request_data)

    @staticmethod
    def build_request_data():
        from aws.osml.model_runner.api.region_request import RegionRequest

        return RegionRequest(SAMPLE_REGION_REQUEST_DATA)


if __name__ == "__main__":
    unittest.main()
