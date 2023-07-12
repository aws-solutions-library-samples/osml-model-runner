#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import unittest
from test.data.sample_request_data import SAMPLE_REGION_REQUEST_DATA


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
