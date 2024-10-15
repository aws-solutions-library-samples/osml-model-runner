#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import os
from importlib import reload
from test.configuration import MOCK_MODEL_RESPONSE, TEST_CONFIG
from unittest import TestCase, main
from unittest.mock import Mock, patch

import boto3
import geojson
from botocore.stub import ANY, Stubber
from moto import mock_aws
from osgeo import gdal


@mock_aws
class TestModelRunnerEndToEnd(TestCase):
    """
    Unit tests for the ModelRunner application.

    This test suite covers different functionalities of the ModelRunner,
    such as processing image and region requests, AWS resource interactions,
    and handling exceptions during tile processing.
    """

    def setUp(self) -> None:
        """
        Set up virtual AWS resources for use in unit tests.
        Creates DynamoDB tables, S3 buckets, SNS topics, SQS queues, and
        mock SageMaker endpoints required for the tests.
        """
        from aws.osml.model_runner.api import RegionRequest
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.endpoint_statistics_table import EndpointStatisticsTable
        from aws.osml.model_runner.database.feature_table import FeatureTable
        from aws.osml.model_runner.database.job_table import JobTable
        from aws.osml.model_runner.database.region_request_table import RegionRequestTable
        from aws.osml.model_runner.model_runner import ModelRunner
        from aws.osml.model_runner.status import ImageStatusMonitor, RegionStatusMonitor

        # Required to avoid warnings from GDAL
        gdal.DontUseExceptions()

        # Set default custom feature properties that should exist on the output features=
        self.test_custom_feature_properties = {
            "modelMetadata": {
                "modelName": "test-model-name",
                "ontologyName": "test-ontology--name",
                "ontologyVersion": "test-ontology-version",
                "classification": "test-classification",
            }
        }

        # Set default feature properties that should exist on the output features
        self.test_feature_source_property = [
            {
                "location": TEST_CONFIG["IMAGE_FILE"],
                "format": "NITF",
                "category": "VIS",
                "sourceId": "Checks an uncompressed 1024x1024 8 bit mono image with GEOcentric data. Airfield",
                "sourceDT": "1996-12-17T10:26:30Z",
            }
        ]

        # Build a mock region request for testing
        self.region_request = RegionRequest(
            {
                "tile_size": (10, 10),
                "tile_overlap": (1, 1),
                "tile_format": "NITF",
                "image_id": TEST_CONFIG["IMAGE_ID"],
                "image_url": TEST_CONFIG["IMAGE_FILE"],
                "region_bounds": ((0, 0), (50, 50)),
                "model_name": TEST_CONFIG["MODEL_ENDPOINT"],
                "model_invoke_mode": "SM_ENDPOINT",
                "image_extension": TEST_CONFIG["IMAGE_EXTENSION"],
            }
        )

        # Build a mock image request for testing
        self.image_request = ImageRequest.from_external_message(
            {
                "jobName": TEST_CONFIG["IMAGE_ID"],
                "jobId": TEST_CONFIG["JOB_ID"],
                "imageUrls": [TEST_CONFIG["IMAGE_FILE"]],
                "outputs": [
                    {"type": "S3", "bucket": TEST_CONFIG["RESULTS_BUCKET"], "prefix": f"{TEST_CONFIG['IMAGE_ID']}"},
                    {"type": "Kinesis", "stream": TEST_CONFIG["RESULTS_STREAM"], "batchSize": 1000},
                ],
                "featureProperties": [self.test_custom_feature_properties],
                "imageProcessor": {"name": TEST_CONFIG["MODEL_ENDPOINT"], "type": "SM_ENDPOINT"},
                "imageProcessorTileSize": 2048,
                "imageProcessorTileOverlap": 50,
                "imageProcessorTileFormat": "NITF",
                "imageProcessorTileCompression": "JPEG",
                "randomKey": "random-value",
            }
        )

        # Build the required virtual DDB tables
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)

        self.image_request_ddb = self.ddb.create_table(
            TableName=os.environ["JOB_TABLE"],
            KeySchema=TEST_CONFIG["JOB_TABLE_KEY_SCHEMA"],
            AttributeDefinitions=TEST_CONFIG["JOB_TABLE_ATTRIBUTE_DEFINITIONS"],
            BillingMode="PAY_PER_REQUEST",
        )
        self.job_table = JobTable(os.environ["JOB_TABLE"])

        self.image_request_ddb = self.ddb.create_table(
            TableName=os.environ["REGION_REQUEST_TABLE"],
            KeySchema=TEST_CONFIG["REGION_REQUEST_TABLE_KEY_SCHEMA"],
            AttributeDefinitions=TEST_CONFIG["REGION_REQUEST_TABLE_ATTRIBUTE_DEFINITIONS"],
            BillingMode="PAY_PER_REQUEST",
        )
        self.region_request_table = RegionRequestTable(os.environ["REGION_REQUEST_TABLE"])

        self.endpoint_statistics_ddb = self.ddb.create_table(
            TableName=os.environ["ENDPOINT_TABLE"],
            KeySchema=TEST_CONFIG["ENDPOINT_TABLE_KEY_SCHEMA"],
            AttributeDefinitions=TEST_CONFIG["ENDPOINT_TABLE_ATTRIBUTE_DEFINITIONS"],
            BillingMode="PAY_PER_REQUEST",
        )
        self.endpoint_statistics_table = EndpointStatisticsTable(os.environ["ENDPOINT_TABLE"])

        self.feature_ddb = self.ddb.create_table(
            TableName=os.environ["FEATURE_TABLE"],
            KeySchema=TEST_CONFIG["FEATURE_TABLE_KEY_SCHEMA"],
            AttributeDefinitions=TEST_CONFIG["FEATURE_TABLE_ATTRIBUTE_DEFINITIONS"],
            BillingMode="PAY_PER_REQUEST",
        )
        self.feature_table = FeatureTable(
            os.environ["FEATURE_TABLE"],
            self.image_request.tile_size,
            self.image_request.tile_overlap,
        )

        # Build a virtual S3 and Kinesis output sink
        self.s3 = boto3.client("s3", config=BotoConfig.default)

        self.image_bucket = self.s3.create_bucket(
            Bucket=TEST_CONFIG["IMAGE_BUCKET"],
            CreateBucketConfiguration={"LocationConstraint": os.environ["AWS_DEFAULT_REGION"]},
        )

        with open(TEST_CONFIG["IMAGE_FILE"], "rb") as data:
            self.s3.upload_fileobj(data, TEST_CONFIG["IMAGE_BUCKET"], TEST_CONFIG["IMAGE_KEY"])

        self.results_bucket = self.s3.create_bucket(
            Bucket=TEST_CONFIG["RESULTS_BUCKET"],
            CreateBucketConfiguration={"LocationConstraint": os.environ["AWS_DEFAULT_REGION"]},
        )

        self.kinesis = boto3.client("kinesis", config=BotoConfig.default)
        self.results_stream = self.kinesis.create_stream(
            StreamName=TEST_CONFIG["RESULTS_STREAM"], StreamModeDetails={"StreamMode": "ON_DEMAND"}
        )

        # Build a virtual image status topic and queue
        self.sns = boto3.client("sns", config=BotoConfig.default)
        image_status_topic_arn = self.sns.create_topic(Name=os.environ["IMAGE_STATUS_TOPIC"]).get("TopicArn")

        self.sqs = boto3.client("sqs", config=BotoConfig.default)
        image_status_queue_url = self.sqs.create_queue(QueueName="mock_queue").get("QueueUrl")
        image_status_queue_attributes = self.sqs.get_queue_attributes(
            QueueUrl=image_status_queue_url, AttributeNames=["QueueArn"]
        )
        image_status_queue_arn = image_status_queue_attributes.get("Attributes").get("QueueArn")

        self.sns.subscribe(TopicArn=image_status_topic_arn, Protocol="sqs", Endpoint=image_status_queue_arn)
        self.image_status_monitor = ImageStatusMonitor(image_status_topic_arn)

        # Build a virtual region status topic and queue
        region_status_topic_arn = self.sns.create_topic(Name=os.environ["REGION_STATUS_TOPIC"]).get("TopicArn")
        self.region_status_monitor = RegionStatusMonitor(region_status_topic_arn)

        region_status_queue_url = self.sqs.create_queue(QueueName="mock_region_queue").get("QueueUrl")
        region_status_queue_attributes = self.sqs.get_queue_attributes(
            QueueUrl=region_status_queue_url, AttributeNames=["QueueArn"]
        )
        region_status_queue_arn = region_status_queue_attributes.get("Attributes").get("QueueArn")

        self.sns.subscribe(TopicArn=region_status_topic_arn, Protocol="sqs", Endpoint=region_status_queue_arn)

        # Build a virtual SageMaker endpoint
        self.sm = boto3.client("sagemaker", config=BotoConfig.default)
        self.sm.create_model(
            ModelName=TEST_CONFIG["MODEL_NAME"],
            PrimaryContainer=TEST_CONFIG["SM_MODEL_CONTAINER"],
            ExecutionRoleArn=f"arn:aws:iam::{TEST_CONFIG['ACCOUNT_ID']}:role/FakeRole",
        )

        config_name = "TestConfig"
        production_variants = TEST_CONFIG["ENDPOINT_PRODUCTION_VARIANTS"]
        self.sm.create_endpoint_config(EndpointConfigName=config_name, ProductionVariants=production_variants)
        self.sm.create_endpoint(EndpointName=TEST_CONFIG["MODEL_ENDPOINT"], EndpointConfigName=config_name)

        # Plug in the required virtual resources to our ModelRunner instance
        self.model_runner = ModelRunner()
        self.model_runner.job_table = self.job_table
        self.model_runner.region_request_table = self.region_request_table
        self.model_runner.endpoint_statistics_table = self.endpoint_statistics_table
        self.model_runner.image_status_monitor = self.image_status_monitor
        self.model_runner.region_status_monitor = self.region_status_monitor
        self.model_runner.region_request_handler.job_table = self.job_table
        self.model_runner.region_request_handler.region_request_table = self.region_request_table
        self.model_runner.region_request_handler.endpoint_statistics_table = self.endpoint_statistics_table
        self.model_runner.region_request_handler.region_status_monitor = self.region_status_monitor
        self.model_runner.region_request_handler.job_table = self.job_table
        self.model_runner.image_request_handler.region_request_table = self.region_request_table
        self.model_runner.image_request_handler.region_request_handler = self.model_runner.region_request_handler
        self.model_runner.image_request_handler.endpoint_statistics_table = self.endpoint_statistics_table
        self.model_runner.image_request_handler.job_table = self.job_table
        self.model_runner.image_request_handler.image_status_monitor = self.image_status_monitor

    def tearDown(self) -> None:
        """
        Delete virtual AWS resources after each test.
        """
        self.image_request_ddb.delete()
        self.endpoint_statistics_ddb.delete()
        self.feature_ddb.delete()
        self.feature_table = None
        self.s3 = None
        self.kinesis = None
        self.results_stream = None
        self.image_bucket = None
        self.results_bucket = None
        self.model_runner = None
        self.sns = None
        self.sqs = None
        self.image_status_monitor = None
        self.region_status_monitor = None

    def test_aws_osml_model_runner_importable(self) -> None:
        """
        Ensure that aws.osml.model_runner can be imported without errors.
        """
        import aws.osml.model_runner  # noqa: F401

    def test_run(self) -> None:
        """
        Test that the run method in ModelRunner initiates the work queue monitoring process.
        """
        self.model_runner.monitor_work_queues = Mock()
        self.model_runner.run()
        self.model_runner.monitor_work_queues.assert_called_once()

    def test_stop(self) -> None:
        """
        Test that the stop method stops the ModelRunner.
        """
        self.model_runner.running = True
        self.model_runner.stop()
        assert self.model_runner.running is False

    def test_end_to_end(self) -> None:
        """
        Test the process of handling an image request, ensuring that jobs are marked as complete,
        features are created, and the correct metadata is stored in S3. Checks that we calculated
        the max in progress regions with the test instance type is set to m5.12xl with 48 vcpus.
        """
        with patch("aws.osml.model_runner.inference.sm_detector.boto3") as mock_boto3:
            # Build stubbed model client for ModelRunner to interact with
            mock_boto3.client.return_value = self.get_stubbed_sm_client()
            self.model_runner.image_request_handler.process_image_request(self.image_request)

            # Ensure that the single region was processed successfully
            image_request_item = self.job_table.get_image_request(self.image_request.image_id)
            assert image_request_item.region_success == 1

            # Ensure that the detection outputs arrived in our DDB table
            features = self.feature_table.get_features(self.image_request.image_id)
            assert len(features) == 1
            assert features[0]["geometry"]["type"] == "Polygon"

            # Ensure that the detection outputs arrived in our output bucket
            results_key = self.s3.list_objects(Bucket=TEST_CONFIG["RESULTS_BUCKET"])["Contents"][0]["Key"]
            results_contents = self.s3.get_object(Bucket=TEST_CONFIG["RESULTS_BUCKET"], Key=results_key)["Body"].read()
            results_features = geojson.loads(results_contents.decode("utf-8"))["features"]
            assert len(results_features) > 0

            # Test that we get the correct model metadata appended to our feature outputs
            actual_model_metadata = results_features[0]["properties"]["modelMetadata"]
            expected_model_metadata = self.test_custom_feature_properties.get("modelMetadata")
            assert actual_model_metadata == expected_model_metadata

            # Test that we get the correct source metadata appended to our feature outputs
            actual_source_metadata = results_features[0]["properties"]["sourceMetadata"]
            expected_source_metadata = self.test_feature_source_property
            assert actual_source_metadata == expected_source_metadata

            # Default scale factor set to 10 and workers per cpu is 1 so: floor((10 * 1 * 48) / 1) = 480
            regions = self.model_runner.endpoint_utils.calculate_max_regions(endpoint_name=TEST_CONFIG["MODEL_ENDPOINT"])
            assert 480 == regions

    @patch.dict("os.environ", values={"ELEVATION_DATA_LOCATION": TEST_CONFIG["ELEVATION_DATA_LOCATION"]})
    def test_create_elevation_model(self) -> None:
        """
        Test that the ModelRunner correctly creates an elevation model based on the SRTM DEM tile set.
        The import and reload statements are necessary to force the ServiceConfig to update with the
        patched environment variables.
        """
        import aws.osml.model_runner.app_config

        reload(aws.osml.model_runner.app_config)
        from aws.osml.gdal.gdal_dem_tile_factory import GDALDigitalElevationModelTileFactory
        from aws.osml.model_runner.app_config import ServiceConfig
        from aws.osml.photogrammetry.digital_elevation_model import DigitalElevationModel
        from aws.osml.photogrammetry.srtm_dem_tile_set import SRTMTileSet

        assert ServiceConfig.elevation_data_location == TEST_CONFIG["ELEVATION_DATA_LOCATION"]
        config = ServiceConfig()
        elevation_model = config.create_elevation_model()
        assert elevation_model
        assert isinstance(elevation_model, DigitalElevationModel)
        assert isinstance(elevation_model.tile_set, SRTMTileSet)
        assert isinstance(elevation_model.tile_factory, GDALDigitalElevationModelTileFactory)

        assert elevation_model.tile_set.format_extension == ".tif"
        assert elevation_model.tile_set.prefix == ""
        assert elevation_model.tile_set.version == "1arc_v3"

        assert elevation_model.tile_factory.tile_directory == TEST_CONFIG["ELEVATION_DATA_LOCATION"]

    def test_create_elevation_model_disabled(self) -> None:
        """
        Test that no elevation model is created when ELEVATION_DATA_LOCATION is not set in the environment.
        The import and reload statements are necessary to force the ServiceConfig to update with the
        patched environment variables.
        """
        import aws.osml.model_runner.app_config

        reload(aws.osml.model_runner.app_config)
        from aws.osml.model_runner.app_config import ServiceConfig

        assert ServiceConfig.elevation_data_location is None
        config = ServiceConfig()
        elevation_model = config.create_elevation_model()

        assert not elevation_model

    @staticmethod
    def get_stubbed_sm_client() -> boto3.client:
        """
        Get a stubbed SageMaker client for use in testing.

        :return: A stubbed SageMaker Runtime client.
        """
        sm_client = boto3.client("sagemaker-runtime")
        sm_runtime_stub = Stubber(sm_client)
        sm_runtime_stub.add_response(
            "invoke_endpoint",
            expected_params={"EndpointName": TEST_CONFIG["MODEL_ENDPOINT"], "Body": ANY},
            service_response=MOCK_MODEL_RESPONSE,
        )
        sm_runtime_stub.activate()

        return sm_client


if __name__ == "__main__":
    main()
