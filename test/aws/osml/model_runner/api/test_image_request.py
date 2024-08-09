#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

legacy_execution_role = "arn:aws:iam::012345678910:role/OversightMLBetaInvokeRole"


class TestImageRequest(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual DDB resources/tables for each test to use
        """
        self.sample_request_data = self.build_request_data()

    def tearDown(self):
        self.sample_request_data = None

    def test_invalid_data(self):
        from aws.osml.model_runner.api.image_request import ImageRequest

        ir = ImageRequest(
            self.sample_request_data,
            image_id="",
        )

        assert not ir.is_valid()

    def test_invalid_job_id(self):
        from aws.osml.model_runner.api.image_request import ImageRequest

        ir = ImageRequest(
            self.sample_request_data,
            job_id=None,
        )

        assert not ir.is_valid()

    @staticmethod
    def build_request_data():
        return {
            "job_id": "test-job",
            "image_id": "test-image-id",
            "image_url": "test-image-url",
            "image_read_role": "arn:aws:iam::012345678910:role/OversightMLS3ReadOnly",
            "output_bucket": "unit-test",
            "output_prefix": "region-request",
            "tile_size": (10, 10),
            "tile_overlap": (1, 1),
            "tile_format": "NITF",
            "model_name": "test-model-name",
            "model_invoke_mode": "SM_ENDPOINT",
            "model_invocation_role": "arn:aws:iam::012345678910:role/OversightMLModelInvoker",
        }


if __name__ == "__main__":
    unittest.main()
