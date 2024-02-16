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


class TestRegionRequest(unittest.TestCase):
    def test_invalid_data(self):
        from aws.osml.model_runner.api.region_request import RegionRequest

        rr = RegionRequest(
            SAMPLE_REGION_REQUEST_DATA,
            image_id="",
        )

        assert not rr.is_valid()

    def test_invalid_region_bounds(self):
        from aws.osml.model_runner.api.region_request import RegionRequest

        rr = RegionRequest(
            SAMPLE_REGION_REQUEST_DATA,
            region_bounds=None,
        )

        assert not rr.is_valid()


if __name__ == "__main__":
    unittest.main()
