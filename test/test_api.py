#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from typing import Any, Dict
from unittest import TestCase, main
from unittest.mock import patch

import boto3
import pytest
import shapely.geometry
from botocore.stub import Stubber

TEST_S3_FULL_BUCKET_PATH = "s3://test-results-bucket/test/data/small.ntf"

base_request = {
    "jobArn": "arn:aws:oversightml:us-east-1:012345678910:ipj/test-job",
    "jobName": "test-job",
    "jobId": "5f4e8a55-95cf-4d96-95cd-9b037f767eff",
    "imageUrls": ["s3://fake-bucket/images/test-image-id"],
    "imageProcessor": {"name": "test-model-name", "type": "SM_ENDPOINT"},
}


class TestModelRunnerAPI(TestCase):
    def test_region_request_constructor(self):
        from aws.osml.model_runner.api.image_request import ModelInvokeMode
        from aws.osml.model_runner.api.region_request import RegionRequest
        from aws.osml.model_runner.common.typing import ImageCompression, ImageFormats

        region_request_template = {
            "model_name": "test-model-name",
            "model_invoke_mode": "SM_ENDPOINT",
            "model_invocation_role": "arn:aws:iam::012345678910:role/OversightMLBetaModelInvokerRole",
        }

        rr = RegionRequest(
            region_request_template,
            image_id="test-image-id",
            image_url="s3://fake-bucket/images/test-image-id",
            image_read_role="arn:aws:iam::012345678910:role/OversightMLBetaS3ReadOnly",
            region_bounds=[0, 1, 2, 3],
        )

        # Check to ensure we've created a valid request
        assert rr.is_valid()

        # Checks to ensure the dictionary provided values are set
        assert rr.model_name == "test-model-name"
        assert rr.model_invoke_mode == ModelInvokeMode.SM_ENDPOINT
        assert rr.model_invocation_role == "arn:aws:iam::012345678910:role/OversightMLBetaModelInvokerRole"

        # Checks to ensure the keyword arguments are set
        assert rr.image_id == "test-image-id"
        assert rr.image_url == "s3://fake-bucket/images/test-image-id"
        assert rr.image_read_role == "arn:aws:iam::012345678910:role/OversightMLBetaS3ReadOnly"
        assert rr.region_bounds == [0, 1, 2, 3]

        # Checks to ensure the defaults are set
        assert rr.tile_size == (1024, 1024)
        assert rr.tile_overlap == (50, 50)
        assert rr.tile_format == ImageFormats.NITF
        assert rr.tile_compression == ImageCompression.NONE

    def test_image_request_constructor(self):
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.common.typing import ImageCompression
        from aws.osml.model_runner.sink.sink import Sink
        from aws.osml.model_runner.sink.sink_factory import SinkFactory

        image_request_template = {
            "model_name": "test-model-name",
            "model_invoke_mode": "SM_ENDPOINT",
            "image_read_role": "arn:aws:iam::012345678910:role/OversightMLBetaS3ReadOnly",
        }
        fake_s3_sink = {
            "type": "S3",
            "bucket": "fake-bucket",
            "prefix": "images/outputs",
            "mode": "Aggregate",
        }
        ir = ImageRequest(
            image_request_template,
            job_arn="arn:aws:oversightml:us-east-1:012345678910:ipj/test-job",
            job_name="test-job",
            job_id="5f4e8a55-95cf-4d96-95cd-9b037f767eff",
            image_id="5f4e8a55-95cf-4d96-95cd-9b037f767eff:s3://fake-bucket/images/test-image-id",
            image_url="s3://fake-bucket/images/test-image-id",
            outputs=[fake_s3_sink],
        )

        assert ir.is_valid()
        assert ir.image_url == "s3://fake-bucket/images/test-image-id"
        assert ir.image_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff:s3://fake-bucket/images/test-image-id"
        assert ir.image_read_role == "arn:aws:iam::012345678910:role/OversightMLBetaS3ReadOnly"
        assert ir.tile_size == (1024, 1024)
        assert ir.tile_overlap == (50, 50)
        assert ir.model_name == "test-model-name"
        assert ir.model_invoke_mode == "SM_ENDPOINT"
        assert ir.model_invocation_role == ""
        assert ir.tile_format == "NITF"
        assert ir.tile_compression == ImageCompression.NONE
        assert ir.job_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff"
        assert ir.job_arn == "arn:aws:oversightml:us-east-1:012345678910:ipj/test-job"
        assert len(ir.outputs) == 1
        sinks = SinkFactory.outputs_to_sinks(ir.outputs)
        s3_sink: Sink = sinks[0]
        assert s3_sink.name() == "S3"
        assert s3_sink.__getattribute__("bucket") == "fake-bucket"
        assert s3_sink.__getattribute__("prefix") == "images/outputs"
        assert ir.roi is None

    @patch("aws.osml.model_runner.common.credentials_utils.sts_client")
    def test_image_request_from_message(self, mock_sts):
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.common.typing import ImageCompression
        from aws.osml.model_runner.sink.sink import Sink
        from aws.osml.model_runner.sink.sink_factory import SinkFactory

        test_access_key_id = "123456789"
        test_secret_access_key = "987654321"
        test_secret_token = "SecretToken123"
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": test_access_key_id,
                "SecretAccessKey": test_secret_access_key,
                "SessionToken": test_secret_token,
            }
        }
        updates: Dict[str, Any] = {
            "jobStatus": "SUBMITTED",
            "processingSubmitted": "2021-09-14T00:18:32.130000+00:00",
            "imageReadRole": "arn:aws:iam::012345678910:role/OversightMLS3ReadOnly",
            "outputs": [
                {
                    "type": "S3",
                    "bucket": "fake-bucket",
                    "prefix": "images/outputs",
                    "assumedRole": "arn:aws:iam::012345678910:role/OversightMLBetaS3ReadOnlyRole",
                }
            ],
            "imageProcessorTileSize": 2048,
            "imageProcessorTileOverlap": 100,
            "imageProcessorTileFormat": "PNG",
            "imageProcessorTileCompression": ImageCompression.NONE,
            "regionOfInterest": "POLYGON((0.5 0.5,5 0,5 5,0 5,0.5 0.5), (1.5 1,4 3,4 1,1.5 1))",
        }
        message_body = base_request.copy()
        message_body.update(updates)

        ir = ImageRequest.from_external_message(message_body)

        assert ir.is_valid()
        assert ir.image_url == "s3://fake-bucket/images/test-image-id"
        assert ir.image_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff:s3://fake-bucket/images/test-image-id"
        assert ir.image_read_role == "arn:aws:iam::012345678910:role/OversightMLS3ReadOnly"
        assert ir.tile_size == (2048, 2048)
        assert ir.tile_overlap == (100, 100)
        assert ir.model_name == "test-model-name"
        assert ir.model_invoke_mode == "SM_ENDPOINT"
        assert ir.tile_format == "PNG"
        assert ir.tile_compression == ImageCompression.NONE
        assert ir.job_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff"
        assert ir.job_arn == "arn:aws:oversightml:us-east-1:012345678910:ipj/test-job"
        assert len(ir.outputs) == 1
        sinks = SinkFactory.outputs_to_sinks(ir.outputs)
        s3_sink: Sink = sinks[0]
        assert s3_sink.name() == "S3"
        assert s3_sink.__getattribute__("bucket") == "fake-bucket"
        assert s3_sink.__getattribute__("prefix") == "images/outputs"
        assert isinstance(ir.roi, shapely.geometry.Polygon)

    def test_image_request_from_minimal_message_legacy_output(self):
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.common.typing import ImageCompression
        from aws.osml.model_runner.sink.sink import Sink
        from aws.osml.model_runner.sink.sink_factory import SinkFactory

        updates: Dict[str, Any] = {"outputBucket": "fake-bucket", "outputPrefix": "images/outputs"}
        message_body = base_request.copy()
        message_body.update(updates)

        ir = ImageRequest.from_external_message(message_body)

        assert ir.is_valid()
        assert ir.image_url == "s3://fake-bucket/images/test-image-id"
        assert ir.image_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff:s3://fake-bucket/images/test-image-id"
        assert ir.image_read_role == ""
        assert ir.tile_size == (1024, 1024)
        assert ir.tile_overlap == (50, 50)
        assert ir.model_name == "test-model-name"
        assert ir.model_invoke_mode == "SM_ENDPOINT"
        assert ir.model_invocation_role == ""
        assert ir.tile_format == "NITF"
        assert ir.tile_compression == ImageCompression.NONE
        assert ir.job_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff"
        assert ir.job_arn == "arn:aws:oversightml:us-east-1:012345678910:ipj/test-job"
        assert len(ir.outputs) == 1
        sinks = SinkFactory.outputs_to_sinks(ir.outputs)
        s3_sink: Sink = sinks[0]
        assert s3_sink.name() == "S3"
        assert s3_sink.__getattribute__("bucket") == "fake-bucket"
        assert s3_sink.__getattribute__("prefix") == "images/outputs"
        assert ir.roi is None

    def test_image_request_multiple_sinks(self):
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.common.typing import ImageCompression
        from aws.osml.model_runner.sink.sink import Sink
        from aws.osml.model_runner.sink.sink_factory import SinkFactory

        updates: Dict[str, Any] = {
            "outputs": [
                {
                    "type": "S3",
                    "bucket": "fake-bucket",
                    "prefix": "images/outputs",
                    "mode": "Aggregate",
                },
                {"type": "Kinesis", "stream": "FakeStream", "batchSize": 500},
            ]
        }
        message_body = base_request.copy()
        message_body.update(updates)

        ir = ImageRequest.from_external_message(message_body)

        assert ir.is_valid()
        assert ir.image_url == "s3://fake-bucket/images/test-image-id"
        assert ir.image_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff:s3://fake-bucket/images/test-image-id"
        assert ir.image_read_role == ""
        assert ir.tile_size == (1024, 1024)
        assert ir.tile_overlap == (50, 50)
        assert ir.model_name == "test-model-name"
        assert ir.model_invoke_mode == "SM_ENDPOINT"
        assert ir.model_invocation_role == ""
        assert ir.tile_format == "NITF"
        assert ir.tile_compression == ImageCompression.NONE
        assert ir.job_id == "5f4e8a55-95cf-4d96-95cd-9b037f767eff"
        assert ir.job_arn == "arn:aws:oversightml:us-east-1:012345678910:ipj/test-job"
        assert len(ir.outputs) == 2
        sinks = SinkFactory.outputs_to_sinks(ir.outputs)
        s3_sink: Sink = sinks[0]
        assert s3_sink.name() == "S3"
        assert s3_sink.__getattribute__("bucket") == "fake-bucket"
        assert s3_sink.__getattribute__("prefix") == "images/outputs"
        kinesis_sink: Sink = sinks[1]
        assert kinesis_sink.name() == "Kinesis"
        assert kinesis_sink.__getattribute__("stream") == "FakeStream"
        assert kinesis_sink.__getattribute__("batch_size") == 500
        assert ir.roi is None

    def test_image_request_invalid_sink(self):
        from aws.osml.model_runner.api.exceptions import InvalidImageRequestException
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.sink.sink_factory import SinkFactory

        updates: Dict[str, Any] = {"outputs": [{"type": "SQS", "queue": "FakeQueue"}]}
        message_body = base_request.copy()
        message_body.update(updates)

        with self.assertRaises(InvalidImageRequestException):
            ir = ImageRequest.from_external_message(message_body)
            SinkFactory.outputs_to_sinks(ir.outputs)

    def test_image_request_invalid_image_path(self):
        from aws.osml.model_runner.api.exceptions import InvalidS3ObjectException
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.app_config import BotoConfig

        s3_client = boto3.client("s3", config=BotoConfig.default)
        s3_client_stub = Stubber(s3_client)
        s3_client_stub.activate()

        s3_client_stub.add_client_error(
            "head_object",
            service_error_code="404",
            service_message="Not Found",
            expected_params={"Bucket": TEST_S3_FULL_BUCKET_PATH},
        )

        with pytest.raises(InvalidS3ObjectException):
            ImageRequest.validate_image_path(TEST_S3_FULL_BUCKET_PATH, None)


if __name__ == "__main__":
    main()
