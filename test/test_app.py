#  Copyright 2023 Amazon.com, Inc. or its affiliates.
import os
import unittest
from importlib import reload

import boto3
import geojson
import mock
from botocore.exceptions import ClientError
from mock import Mock
from moto import mock_dynamodb, mock_ec2, mock_kinesis, mock_s3, mock_sagemaker, mock_sns, mock_sqs
from osgeo import gdal

TEST_MOCK_PUT_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "put_item"))
TEST_MOCK_UPDATE_EXCEPTION = Mock(side_effect=ClientError({"Error": {"Code": 500, "Message": "ClientError"}}, "update_item"))

TEST_ACCOUNT_ID = "123456789123"
TEST_IMAGE_ID = "test-image-id"
TEST_IMAGE_EXTENSION = "NITF"
TEST_JOB_ID = "test-job-id"
TEST_ELEVATION_DATA_LOCATION = "s3://TEST-BUCKET/ELEVATION-DATA-LOCATION"
TEST_MODEL_ENDPOINT = "NOOP_BOUNDS_MODEL_NAME"
TEST_MODEL_NAME = "FakeCVModel"
TEST_RESULTS_BUCKET = "test-results-bucket"
TEST_IMAGE_FILE = "./test/data/small.ntf"
TEST_IMAGE_BUCKET = "test-image-bucket"
TEST_IMAGE_KEY = "small.ntf"
TEST_S3_FULL_BUCKET_PATH = "s3://test-results-bucket/test/data/small.ntf"
TEST_RESULTS_STREAM = "test-results-stream"
TEST_SM_MODEL_CONTAINER = {
    "Image": "123456789123.dkr.ecr.us-east-1.amazonaws.com/test:1",
    "ModelDataUrl": "s3://MyBucket/model.tar.gz",
}
TEST_ENDPOINT_PRODUCTION_VARIANTS = [
    {
        "VariantName": "Primary",
        "ModelName": TEST_MODEL_NAME,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.12xlarge",
    },
]

# DDB Configurations
TEST_JOB_TABLE_KEY_SCHEMA = [{"AttributeName": "image_id", "KeyType": "HASH"}]
TEST_JOB_TABLE_ATTRIBUTE_DEFINITIONS = [{"AttributeName": "image_id", "AttributeType": "S"}]

TEST_ENDPOINT_TABLE_KEY_SCHEMA = [{"AttributeName": "endpoint", "KeyType": "HASH"}]
TEST_ENDPOINT_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "endpoint", "AttributeType": "S"},
]

TEST_FEATURE_TABLE_KEY_SCHEMA = [
    {"AttributeName": "hash_key", "KeyType": "HASH"},
    {"AttributeName": "range_key", "KeyType": "RANGE"},
]
TEST_FEATURE_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "hash_key", "AttributeType": "S"},
    {"AttributeName": "range_key", "AttributeType": "S"},
]

TEST_REGION_REQUEST_TABLE_KEY_SCHEMA = [
    {"AttributeName": "region_id", "KeyType": "HASH"},
    {"AttributeName": "image_id", "KeyType": "RANGE"},
]
TEST_REGION_REQUEST_TABLE_ATTRIBUTE_DEFINITIONS = [
    {"AttributeName": "region_id", "AttributeType": "S"},
    {"AttributeName": "image_id", "AttributeType": "S"},
]


class RegionRequestMatcher:
    def __init__(self, region_request):
        self.region_request = region_request

    def __eq__(self, other):
        if other is None:
            return self.region_request is None
        else:
            return other["region"] == self.region_request["region"] and other["image_id"] == self.region_request["image_id"]


@mock_dynamodb
@mock_ec2
@mock_s3
@mock_sagemaker
@mock_sqs
@mock_sns
@mock_kinesis
class TestModelRunner(unittest.TestCase):
    def setUp(self):
        """
        Set up virtual AWS resources for use by our unit tests
        """
        from aws.osml.model_runner.api import RegionRequest
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.app import ModelRunner
        from aws.osml.model_runner.app_config import BotoConfig
        from aws.osml.model_runner.database.endpoint_statistics_table import EndpointStatisticsTable
        from aws.osml.model_runner.database.feature_table import FeatureTable
        from aws.osml.model_runner.database.job_table import JobTable
        from aws.osml.model_runner.database.region_request_table import RegionRequestTable
        from aws.osml.model_runner.status.sns_helper import SNSHelper

        # GDAL 4.0 will begin using exceptions as the default; at this point the software is written to assume
        # no exceptions so we call this explicitly until the software can be updated to match.
        gdal.DontUseExceptions()

        # Create custom properties to be passed into the image request
        self.test_custom_feature_properties = {
            "modelMetadata": {
                "modelName": "test-model-name",
                "ontologyName": "test-ontology--name",
                "ontologyVersion": "test-ontology-version",
                "classification": "test-classification",
            }
        }

        # This is the expected results for the source property derived from the small test image
        self.test_feature_source_property = [
            {
                "fileType": "NITF",
                "info": {
                    "imageCategory": "VIS",
                    "metadata": {
                        "sourceId": "Checks an uncompressed 1024x1024 8 bit mono image with GEOcentric data. Airfield",
                        "sourceDt": "1996-12-17T10:26:30",
                        "classification": "UNCLASSIFIED",
                    },
                },
            }
        ]

        self.region_request = RegionRequest(
            {
                "tile_size": (10, 10),
                "tile_overlap": (1, 1),
                "tile_format": "NITF",
                "image_id": TEST_IMAGE_ID,
                "image_url": TEST_IMAGE_FILE,
                "region_bounds": ((0, 0), (50, 50)),
                "model_name": TEST_MODEL_ENDPOINT,
                "model_invoke_mode": "SM_ENDPOINT",
                "image_extension": TEST_IMAGE_EXTENSION,
            }
        )

        # Build fake image request to work with
        self.image_request = ImageRequest.from_external_message(
            {
                "jobArn": f"arn:aws:oversightml:{os.environ['AWS_DEFAULT_REGION']}:{TEST_ACCOUNT_ID}:job/{TEST_IMAGE_ID}",
                "jobName": TEST_IMAGE_ID,
                "jobId": TEST_IMAGE_ID,
                "imageUrls": [TEST_IMAGE_FILE],
                "outputs": [
                    {"type": "S3", "bucket": TEST_RESULTS_BUCKET, "prefix": f"{TEST_IMAGE_ID}/"},
                    {"type": "Kinesis", "stream": TEST_RESULTS_STREAM, "batchSize": 1000},
                ],
                "featureProperties": [self.test_custom_feature_properties],
                "imageProcessor": {"name": TEST_MODEL_ENDPOINT, "type": "SM_ENDPOINT"},
                "imageProcessorTileSize": 2048,
                "imageProcessorTileOverlap": 50,
                "imageProcessorTileFormat": "NITF",
                "imageProcessorTileCompression": "JPEG",
            }
        )

        # Prepare something ahead of all tests
        # Create virtual DDB tables to write test data into
        self.ddb = boto3.resource("dynamodb", config=BotoConfig.default)

        # Job tracking table
        self.image_request_ddb = self.ddb.create_table(
            TableName=os.environ["JOB_TABLE"],
            KeySchema=TEST_JOB_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_JOB_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )
        self.job_table = JobTable(os.environ["JOB_TABLE"])

        # Region Request tracking table
        self.image_request_ddb = self.ddb.create_table(
            TableName=os.environ["REGION_REQUEST_TABLE"],
            KeySchema=TEST_REGION_REQUEST_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_REGION_REQUEST_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )
        self.region_request_table = RegionRequestTable(os.environ["REGION_REQUEST_TABLE"])

        # Endpoint statistics table
        self.endpoint_statistics_ddb = self.ddb.create_table(
            TableName=os.environ["ENDPOINT_TABLE"],
            KeySchema=TEST_ENDPOINT_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_ENDPOINT_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )
        self.endpoint_statistics_table = EndpointStatisticsTable(os.environ["ENDPOINT_TABLE"])

        # Feature tracking table
        self.feature_ddb = self.ddb.create_table(
            TableName=os.environ["FEATURE_TABLE"],
            KeySchema=TEST_FEATURE_TABLE_KEY_SCHEMA,
            AttributeDefinitions=TEST_FEATURE_TABLE_ATTRIBUTE_DEFINITIONS,
            BillingMode="PAY_PER_REQUEST",
        )
        self.feature_table = FeatureTable(
            os.environ["FEATURE_TABLE"],
            self.image_request.tile_size,
            self.image_request.tile_overlap,
        )

        # Create fake buckets for images and results
        self.s3 = boto3.client("s3", config=BotoConfig.default)

        # Create a fake bucket to store images
        self.image_bucket = self.s3.create_bucket(
            Bucket=TEST_IMAGE_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": os.environ["AWS_DEFAULT_REGION"]},
        )
        # Load our test image into our bucket
        with open(TEST_IMAGE_FILE, "rb") as data:
            self.s3.upload_fileobj(data, TEST_IMAGE_BUCKET, TEST_IMAGE_KEY)

        # Create a fake bucket to store results in
        self.results_bucket = self.s3.create_bucket(
            Bucket=TEST_RESULTS_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": os.environ["AWS_DEFAULT_REGION"]},
        )

        # Create a fake stream to store results in
        self.kinesis = boto3.client("kinesis", region_name=os.environ["AWS_DEFAULT_REGION"])
        self.results_stream = self.kinesis.create_stream(
            StreamName=TEST_RESULTS_STREAM, StreamModeDetails={"StreamMode": "ON_DEMAND"}
        )

        # Create a fake sns topic for reporting job status
        self.sns = boto3.client("sns", config=BotoConfig.default)
        sns_response = self.sns.create_topic(Name=os.environ["IMAGE_STATUS_TOPIC"])
        self.mock_topic_arn = sns_response.get("TopicArn")

        # Create a fake sqs queue to consume the sns topic events
        self.sqs = boto3.client("sqs", config=BotoConfig.default)
        sqs_response = self.sqs.create_queue(QueueName="mock_queue")
        self.mock_queue_url = sqs_response.get("QueueUrl")
        queue_attributes = self.sqs.get_queue_attributes(QueueUrl=self.mock_queue_url, AttributeNames=["QueueArn"])
        queue_arn = queue_attributes.get("Attributes").get("QueueArn")

        # Subscribe our sns topic to the queue
        self.sns.subscribe(TopicArn=self.mock_topic_arn, Protocol="sqs", Endpoint=queue_arn)

        # Set up our status monitor for the queue
        self.image_status_sns = SNSHelper(self.mock_topic_arn)

        # Create a fake bounds model
        self.sm = boto3.client("sagemaker", config=BotoConfig.default)
        self.sm.create_model(
            ModelName=TEST_MODEL_NAME,
            PrimaryContainer=TEST_SM_MODEL_CONTAINER,
            ExecutionRoleArn=f"arn:aws:iam::{TEST_ACCOUNT_ID}:role/FakeRole",
        )
        # Create a fake endpoint config
        config_name = "UnitTestConfig"
        production_variants = TEST_ENDPOINT_PRODUCTION_VARIANTS
        self.sm.create_endpoint_config(EndpointConfigName=config_name, ProductionVariants=production_variants)
        # Create a fake endpoint
        self.sm.create_endpoint(EndpointName=TEST_MODEL_ENDPOINT, EndpointConfigName=config_name)
        # Create a fake geom model endpoint
        self.sm.create_endpoint(EndpointName="NOOP_GEOM_MODEL_NAME", EndpointConfigName=config_name)

        # Build our model runner and plug in fake resources
        self.model_runner = ModelRunner()
        self.model_runner.job_table = self.job_table
        self.model_runner.region_request_table = self.region_request_table
        self.model_runner.endpoint_statistics_table = self.endpoint_statistics_table
        self.model_runner.status_monitor.image_status_sns = self.image_status_sns

    def tearDown(self):
        """
        Delete virtual resources after each test
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
        self.mock_topic_arn = None
        self.sqs = None
        self.mock_queue_url = None
        self.image_status_sns = None

    def test_aws_osml_model_runner_importable(self):
        import aws.osml.model_runner  # noqa: F401

    def test_process_bounds_image_request(self):
        from aws.osml.model_runner.database.region_request_table import RegionRequestTable

        self.model_runner.region_request_table = Mock(RegionRequestTable, autospec=True)

        self.model_runner.process_image_request(self.image_request)

        # Check to make sure the job was marked as complete
        image_request_item = self.job_table.get_image_request(self.image_request.image_id)
        assert image_request_item.region_success == 1

        # Check that we created the right amount of features
        features = self.feature_table.get_features(self.image_request.image_id)
        assert len(features) == 1

        # Check to make sure the feature was assigned a real geo coordinate
        assert features[0]["geometry"]["type"] == "Polygon"

        # Grab the feature results from virtual S3 bucket
        results_key = self.s3.list_objects(Bucket=TEST_RESULTS_BUCKET)["Contents"][0]["Key"]

        results_contents = self.s3.get_object(
            Bucket=TEST_RESULTS_BUCKET,
            Key=results_key,
        )["Body"].read()

        # Load them into memory as geojson
        results_features = geojson.loads(results_contents.decode("utf-8"))["features"]
        assert len(results_features) > 0

        # Check that the provided custom feature property was added
        assert results_features[0]["properties"]["modelMetadata"] == self.test_custom_feature_properties.get("modelMetadata")

        # Check we got the correct source data for the small.ntf file
        assert results_features[0]["properties"]["source"] == self.test_feature_source_property

        # Check that we calculated the max in progress regions
        # Test instance type is set to m5.12xl with 48 vcpus. Default
        # scale factor is set to 10 and workers per cpu is 1 so:
        # floor((10 * 1 * 48) / 1) = 480
        assert 480 == self.model_runner.endpoint_utils.calculate_max_regions(endpoint_name=TEST_MODEL_ENDPOINT)

    def test_process_geom_image_request(self):
        from aws.osml.model_runner.api.image_request import ImageRequest
        from aws.osml.model_runner.database.region_request_table import RegionRequestTable

        self.model_runner.region_request_table = Mock(RegionRequestTable, autospec=True)
        self.image_request = ImageRequest.from_external_message(
            {
                "jobArn": f"arn:aws:oversightml:{os.environ['AWS_DEFAULT_REGION']}:{TEST_ACCOUNT_ID}:job/{TEST_IMAGE_ID}",
                "jobName": TEST_IMAGE_ID,
                "jobId": TEST_IMAGE_ID,
                "imageUrls": [TEST_IMAGE_FILE],
                "outputs": [
                    {"type": "S3", "bucket": TEST_RESULTS_BUCKET, "prefix": f"{TEST_IMAGE_ID}/"},
                    {"type": "Kinesis", "stream": TEST_RESULTS_STREAM, "batchSize": 1000},
                ],
                "featureProperties": [self.test_custom_feature_properties],
                "imageProcessor": {"name": "NOOP_GEOM_MODEL_NAME", "type": "SM_ENDPOINT"},
                "imageProcessorTileSize": 2048,
                "imageProcessorTileOverlap": 50,
                "imageProcessorTileFormat": "NITF",
                "imageProcessorTileCompression": "JPEG",
            }
        )
        self.model_runner.process_image_request(self.image_request)

        # Check to make sure the job was marked as complete
        image_request_item = self.job_table.get_image_request(self.image_request.image_id)
        assert image_request_item.region_success == 1

        # Check that we created the right amount of features
        features = self.feature_table.get_features(self.image_request.image_id)
        assert len(features) == 1

        # Check to make sure the feature was assigned a real geo coordinate
        assert features[0]["geometry"]["type"] == "Polygon"

        # Grab the feature results from virtual S3 bucket
        results_key = self.s3.list_objects(Bucket=TEST_RESULTS_BUCKET)["Contents"][0]["Key"]

        results_contents = self.s3.get_object(
            Bucket=TEST_RESULTS_BUCKET,
            Key=results_key,
        )["Body"].read()

        # Load them into memory as geojson
        results_features = geojson.loads(results_contents.decode("utf-8"))["features"]
        assert len(results_features) > 0

        # Check that the provided custom feature property was added
        assert results_features[0]["properties"]["modelMetadata"] == self.test_custom_feature_properties.get("modelMetadata")

        # Check we got the correct source data for the small.ntf file
        assert results_features[0]["properties"]["source"] == self.test_feature_source_property

        # Check that we calculated the max in progress regions
        # Test instance type is set to m5.12xl with 48 vcpus. Default
        # scale factor is set to 10 and workers per cpu is 1 so:
        # floor((10 * 1 * 48) / 1) = 480
        assert 480 == self.model_runner.endpoint_utils.calculate_max_regions(endpoint_name=TEST_MODEL_ENDPOINT)

    # Remember that with multiple patch decorators the order of the mocks in the parameter list is
    # reversed (i.e. the first mock parameter is the last decorator defined). Also note that the
    # pytest fixtures must come at the end.
    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.FeatureDetectorFactory", autospec=True)
    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.FeatureTable", autospec=True)
    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.TileWorker", autospec=True)
    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.Queue", autospec=True)
    def test_process_region_request(
        self,
        mock_queue,
        mock_tile_worker,
        mock_feature_table,
        mock_feature_detector,
    ):
        from aws.osml.gdal.gdal_utils import load_gdal_dataset
        from aws.osml.model_runner.database.endpoint_statistics_table import EndpointStatisticsTable
        from aws.osml.model_runner.database.job_table import JobTable
        from aws.osml.model_runner.database.region_request_table import RegionRequestItem, RegionRequestTable

        region_request_item = RegionRequestItem(
            image_id=TEST_IMAGE_ID, region_id="test-region-id", region_pixel_bounds="(0, 0)(50, 50)"
        )

        region_queue_put_calls = [
            mock.call(RegionRequestMatcher({"region": ((0, 0), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((0, 9), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((0, 18), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((0, 27), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((0, 36), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((0, 45), (5, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((9, 0), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((9, 9), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((9, 18), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((9, 27), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((9, 36), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((9, 45), (5, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((18, 0), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((18, 9), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((18, 18), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((18, 27), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((18, 36), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((18, 45), (5, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((27, 0), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((27, 9), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((27, 18), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((27, 27), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((27, 36), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((27, 45), (5, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((36, 0), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((36, 9), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((36, 18), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((36, 27), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((36, 36), (10, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((36, 45), (5, 10)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((45, 0), (10, 5)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((45, 9), (10, 5)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((45, 18), (10, 5)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((45, 27), (10, 5)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((45, 36), (10, 5)), "image_id": "test-image-id"})),
            mock.call(RegionRequestMatcher({"region": ((45, 45), (5, 5)), "image_id": "test-image-id"})),
        ]

        # Load up our test image
        raster_dataset, sensor_model = load_gdal_dataset(self.region_request.image_url)

        self.model_runner.job_table = Mock(JobTable, autospec=True)
        self.model_runner.region_request_table = Mock(RegionRequestTable, autospec=True)
        self.model_runner.endpoint_statistics_table = Mock(EndpointStatisticsTable, autospec=True)
        self.model_runner.endpoint_statistics_table.current_in_progress_regions.return_value = 0
        self.model_runner.process_region_request(self.region_request, region_request_item, raster_dataset, sensor_model)

        # Create tile worker threads to process tiles
        num_workers = int(os.environ["WORKERS"])
        for i in range(num_workers):
            region_queue_put_calls.append(mock.call(RegionRequestMatcher(None)))

        # Check to make sure the correct number of workers were created and setup with detectors and
        # feature tables
        assert mock_tile_worker.call_count == num_workers
        assert mock_feature_detector.call_count == num_workers
        assert mock_feature_table.call_count == num_workers

        # We're testing a single region here so expecting a single call to both increment and
        # decrement for the model associated with the region
        self.model_runner.endpoint_statistics_table.increment_region_count.assert_called_once_with(TEST_MODEL_ENDPOINT)
        self.model_runner.endpoint_statistics_table.decrement_region_count.assert_called_once_with(TEST_MODEL_ENDPOINT)

        # Check to make sure a queue was created and populated with appropriate region requests
        mock_queue.assert_called_once()
        mock_queue.return_value.put.assert_has_calls(region_queue_put_calls)

    @mock.patch.dict("os.environ", values={"ELEVATION_DATA_LOCATION": TEST_ELEVATION_DATA_LOCATION})
    def test_create_elevation_model(self):
        # These imports/reloads are necessary to force the ServiceConfig instance used by model runner
        # to have the patched environment variables
        import aws.osml.model_runner.app_config

        reload(aws.osml.model_runner.app_config)
        reload(aws.osml.model_runner.app)
        from aws.osml.gdal.gdal_dem_tile_factory import GDALDigitalElevationModelTileFactory
        from aws.osml.model_runner.app import ModelRunner
        from aws.osml.model_runner.app_config import ServiceConfig
        from aws.osml.photogrammetry.digital_elevation_model import DigitalElevationModel
        from aws.osml.photogrammetry.srtm_dem_tile_set import SRTMTileSet

        assert ServiceConfig.elevation_data_location == TEST_ELEVATION_DATA_LOCATION

        elevation_model = ModelRunner.create_elevation_model()
        assert elevation_model
        assert isinstance(elevation_model, DigitalElevationModel)
        assert isinstance(elevation_model.tile_set, SRTMTileSet)
        assert isinstance(elevation_model.tile_factory, GDALDigitalElevationModelTileFactory)

        assert elevation_model.tile_set.format_extension == ".tif"
        assert elevation_model.tile_set.prefix == ""
        assert elevation_model.tile_set.version == "1arc_v3"

        assert elevation_model.tile_factory.tile_directory == TEST_ELEVATION_DATA_LOCATION

    def test_create_elevation_model_disabled(self):
        # These imports/reloads are necessary to force the ServiceConfig instance used by model runner
        # to have the patched environment variables
        import aws.osml.model_runner.app_config

        reload(aws.osml.model_runner.app_config)
        reload(aws.osml.model_runner.app)
        from aws.osml.model_runner.app import ModelRunner
        from aws.osml.model_runner.app_config import ServiceConfig

        # Check to make sure that excluding the ELEVATION_DATA_LOCATION env variable results in no elevation model
        assert ServiceConfig.elevation_data_location is None

        elevation_model = ModelRunner.create_elevation_model()
        assert not elevation_model

    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.FeatureDetectorFactory", autospec=True)
    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.FeatureTable", autospec=True)
    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.TileWorker", autospec=True)
    @mock.patch("aws.osml.model_runner.tile_worker.tile_worker_utils.Queue", autospec=True)
    def test_process_region_request_throttled(
        self,
        mock_queue,
        mock_tile_worker,
        mock_feature_table,
        mock_feature_detector,
    ):
        from aws.osml.gdal.gdal_utils import load_gdal_dataset
        from aws.osml.model_runner.database.endpoint_statistics_table import EndpointStatisticsTable
        from aws.osml.model_runner.database.job_table import JobTable
        from aws.osml.model_runner.database.region_request_table import RegionRequestTable
        from aws.osml.model_runner.exceptions import SelfThrottledRegionException

        # Load up our test image
        raster_dataset, sensor_model = load_gdal_dataset(self.region_request.image_url)

        self.model_runner.job_table = Mock(JobTable, autospec=True)
        self.model_runner.region_request_table = Mock(RegionRequestTable, autospec=True)
        self.model_runner.endpoint_statistics_table = Mock(EndpointStatisticsTable, autospec=True)
        self.model_runner.endpoint_statistics_table.current_in_progress_regions.return_value = 10000

        with self.assertRaises(SelfThrottledRegionException):
            self.model_runner.process_region_request(self.region_request, raster_dataset, sensor_model)

        self.model_runner.endpoint_statistics_table.increment_region_count.assert_not_called()
        self.model_runner.endpoint_statistics_table.decrement_region_count.assert_not_called()

        assert mock_tile_worker.call_count == 0
        assert mock_feature_detector.call_count == 0
        assert mock_feature_table.call_count == 0

        # Check to make sure a queue was created and populated with appropriate region requests
        mock_queue.assert_not_called()

    @staticmethod
    def get_dataset_and_camera():
        from aws.osml.gdal.gdal_utils import load_gdal_dataset

        ds, sensor_model = load_gdal_dataset("./test/data/GeogToWGS84GeoKey5.tif")
        return ds, sensor_model


if __name__ == "__main__":
    unittest.main()
