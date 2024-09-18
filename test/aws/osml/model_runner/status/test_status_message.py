#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest

from aws.osml.model_runner.common import RequestStatus
from aws.osml.model_runner.status.status_message import StatusMessage


class TestStatusMessage(unittest.TestCase):
    def setUp(self):
        """Set up test case with sample StatusMessage data."""
        self.status_message = StatusMessage(
            status=RequestStatus.SUCCESS,
            job_id="1234",
            image_id="image-5678",
            region_id="region-9999",
            processing_duration=1234,
            failed_tiles=[
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
        )

    def test_asdict(self):
        """Test the asdict method to ensure correct conversion."""
        expected_dict = {
            "status": RequestStatus.SUCCESS,
            "job_id": "1234",
            "image_id": "image-5678",
            "region_id": "region-9999",
            "processing_duration": 1234,
            "failed_tiles": [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
            ],
        }
        self.assertEqual(self.status_message.asdict(), expected_dict)

    def test_asdict_str_values(self):
        """Test the asdict_str_values method for proper string conversion."""
        expected_dict_str = {
            "status": "SUCCESS",
            "job_id": "1234",
            "image_id": "image-5678",
            "region_id": "region-9999",
            "processing_duration": "1234",
            "failed_tiles": "[{'1': [[1, 2], [3, 4]]}, {'2': [[5, 6], [7, 8]]}]",
        }
        self.assertEqual(self.status_message.asdict_str_values(), expected_dict_str)

    def test_asdict_with_missing_fields(self):
        """Test asdict and asdict_str_values methods when optional fields are None."""
        status_message = StatusMessage(status=RequestStatus.FAILED, job_id="5678")
        expected_dict = {"status": RequestStatus.FAILED, "job_id": "5678"}
        self.assertEqual(status_message.asdict(), expected_dict)

        expected_dict_str = {"status": "FAILED", "job_id": "5678"}
        self.assertEqual(status_message.asdict_str_values(), expected_dict_str)

    def test_asdict_str_values_processing_duration(self):
        """Test that processing_duration is correctly formatted in seconds."""
        status_message = StatusMessage(status=RequestStatus.SUCCESS, job_id="5678", processing_duration=int("3000"))
        dict_str_values = status_message.asdict_str_values()
        self.assertEqual(dict_str_values["processing_duration"], "3000")

    def test_asdict_str_values_failed_tiles(self):
        """Test that failed_tiles are formatted correctly as string values."""
        status_message = StatusMessage(
            status=RequestStatus.FAILED,
            job_id="5678",
            failed_tiles=[[[1, 2], [3, 4]]],
        )
        dict_str_values = status_message.asdict_str_values()
        self.assertEqual(dict_str_values["failed_tiles"], "[{'1': [[1, 2], [3, 4]]}]")


if __name__ == "__main__":
    unittest.main()
