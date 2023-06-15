#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from decimal import Decimal
from unittest import TestCase

from shapely.geometry import Point


class TestImageStatusHelper(TestCase):
    def setUp(self):
        from aws.osml.model_runner.api.inference import ModelInvokeMode
        from aws.osml.model_runner.common.typing import ImageCompression, ImageFormats, ImageRequestStatus
        from aws.osml.model_runner.sink.kinesis_sink import KinesisSink
        from aws.osml.model_runner.sink.s3_sink import S3Sink

        self.mock_image_status = ImageRequestStatus.IN_PROGRESS
        self.mock_job_id = "job id"
        self.mock_job_arn = "job arn"
        self.mock_image_id = "image id"
        self.mock_image_url = "https://image-link.ntf"
        self.mock_image_read_role = "ImageReader"
        self.mock_outputs = [S3Sink("test_bucket", "folder"), KinesisSink("test_stream")]
        self.mock_model_name = "NOOP_MODEL_NAME"
        self.mock_processing_duration = Decimal(42.5)
        self.mock_model_invoke_mode = ModelInvokeMode.NONE
        self.mock_model_invocation_role = "ModelInvocationRole"
        self.mock_tile_size = (5000, 5000)
        self.mock_tile_overlap = (50, 50)
        self.mock_tile_format = ImageFormats.NITF
        self.mock_tile_compression = ImageCompression.JPEG
        self.mock_roi = Point(25, 25)
        self.mock_region_bounds = ((0, 0), (100, 100))

    def tearDown(self):
        pass

    def test_image_status_item_to_dict_of_strings(self):
        from aws.osml.model_runner.status.image_request_status import ImageRequestStatusMessage

        mock_item = ImageRequestStatusMessage(
            image_status=self.mock_image_status,
            job_id=self.mock_job_id,
            job_arn=self.mock_job_arn,
            image_id=self.mock_image_id,
            image_url=self.mock_image_url,
            image_read_role=self.mock_image_read_role,
            outputs=self.mock_outputs,
            model_name=self.mock_model_name,
            processing_duration=self.mock_processing_duration,
            model_invoke_mode=self.mock_model_invoke_mode,
            model_invocation_role=self.mock_model_invocation_role,
            tile_size=self.mock_tile_size,
            tile_overlap=self.mock_tile_overlap,
            tile_format=self.mock_tile_format,
            tile_compression=self.mock_tile_compression,
            roi=self.mock_roi,
            region_bounds=self.mock_region_bounds,
        )
        expected_dict_of_strings = {
            "image_status": str(self.mock_image_status.value),
            "job_id": self.mock_job_id,
            "job_arn": self.mock_job_arn,
            "image_id": self.mock_image_id,
            "image_url": self.mock_image_url,
            "image_read_role": self.mock_image_read_role,
            "outputs": str([str(item) for item in self.mock_outputs]),
            "model_name": self.mock_model_name,
            "processing_duration": str(self.mock_processing_duration),
            "model_invoke_mode": str(self.mock_model_invoke_mode.value),
            "model_invocation_role": self.mock_model_invocation_role,
            "tile_size": str(self.mock_tile_size),
            "tile_overlap": str(self.mock_tile_overlap),
            "tile_format": str(self.mock_tile_format.value),
            "tile_compression": str(self.mock_tile_compression.value),
            "roi": str(self.mock_roi),
            "region_bounds": str(self.mock_region_bounds),
        }
        dict_of_strings = mock_item.asdict_str_values()
        assert dict_of_strings == expected_dict_of_strings

    def test_image_status_item_to_dict_of_strings_remove_nones(self):
        from aws.osml.model_runner.status.image_request_status import ImageRequestStatusMessage

        mock_item = ImageRequestStatusMessage(
            image_status=self.mock_image_status,
            job_id=self.mock_job_id,
        )
        expected_dict_of_strings = {
            "image_status": str(self.mock_image_status.value),
            "job_id": self.mock_job_id,
        }
        dict_of_strings = mock_item.asdict_str_values()
        assert dict_of_strings == expected_dict_of_strings
