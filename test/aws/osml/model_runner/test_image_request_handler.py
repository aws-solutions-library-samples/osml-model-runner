#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from datetime import datetime, timezone
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from aws.osml.model_runner.api import ImageRequest
from aws.osml.model_runner.app_config import ServiceConfig
from aws.osml.model_runner.common import EndpointUtils, RequestStatus
from aws.osml.model_runner.database import EndpointStatisticsTable, JobItem, JobTable, RegionRequestTable
from aws.osml.model_runner.exceptions import ProcessImageException
from aws.osml.model_runner.image_request_handler import ImageRequestHandler
from aws.osml.model_runner.queue import RequestQueue
from aws.osml.model_runner.status import ImageStatusMonitor
from aws.osml.model_runner.tile_worker import TilingStrategy


class TestImageRequestHandler(TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_job_table = MagicMock(spec=JobTable)
        self.mock_image_status_monitor = MagicMock(spec=ImageStatusMonitor)
        self.mock_endpoint_statistics_table = MagicMock(spec=EndpointStatisticsTable)
        self.mock_tiling_strategy = MagicMock(spec=TilingStrategy)
        self.mock_region_request_queue = MagicMock(spec=RequestQueue)
        self.mock_region_request_table = MagicMock(spec=RegionRequestTable)
        self.mock_endpoint_utils = MagicMock(spec=EndpointUtils)
        self.mock_config = MagicMock(spec=ServiceConfig)

        # Set up config properties
        self.mock_config.self_throttling = False

        # Instantiate the handler with mocked dependencies
        self.handler = ImageRequestHandler(
            job_table=self.mock_job_table,
            image_status_monitor=self.mock_image_status_monitor,
            endpoint_statistics_table=self.mock_endpoint_statistics_table,
            tiling_strategy=self.mock_tiling_strategy,
            region_request_queue=self.mock_region_request_queue,
            region_request_table=self.mock_region_request_table,
            endpoint_utils=self.mock_endpoint_utils,
            config=self.mock_config,
            region_request_handler=MagicMock(),
        )

        # Mock request and items
        self.mock_image_request = self.image_request = ImageRequest.from_external_message(
            {
                "jobName": "test-job-name",
                "jobId": "test-job-id",
                "imageUrls": ["./test/data/small.ntf"],
                "outputs": [
                    {"type": "S3", "bucket": "test-results-bucket", "prefix": "test-image-id"},
                    {"type": "Kinesis", "stream": ":test-results-stream", "batchSize": 1000},
                ],
                "imageProcessor": {"name": "test-model", "type": "SM_ENDPOINT"},
                "imageProcessorTileSize": 2048,
                "imageProcessorTileOverlap": 50,
                "imageProcessorTileFormat": "NITF",
                "imageProcessorTileCompression": "JPEG",
                "randomKey": "random-value",
            }
        )

        self.mock_job_item = JobItem.from_image_request(self.mock_image_request)

    def test_process_image_request_success(self):
        """
        Test successful image request processing.
        """
        # Mock internal methods
        self.handler.load_image_request = MagicMock(return_value=("tif", MagicMock(), MagicMock(), [MagicMock()]))
        self.handler.queue_region_request = MagicMock()

        # Call process_image_request
        self.handler.process_image_request(self.mock_image_request)

        # Assert that the STARTED status was called first
        self.mock_job_table.start_image_request.assert_called_once()

        # Ensure the regions were queued
        self.handler.queue_region_request.assert_called_once()

        # Ensure processing events were emitted
        self.assertEqual(self.mock_image_status_monitor.process_event.call_count, 2)

    def test_process_image_request_throttling(self):
        """
        Test image request processing when throttling is enabled.
        """
        # Enable throttling in config
        self.mock_config.self_throttling = True

        # Mock internal methods
        self.mock_endpoint_utils.calculate_max_regions.return_value = 5
        self.mock_endpoint_statistics_table.current_in_progress_regions.return_value = 5

        # Call process_image_request with throttling enabled
        with self.assertRaises(ProcessImageException):
            self.handler.process_image_request(self.mock_image_request)

        # Ensure processing continued after throttling
        self.assertEqual(self.mock_image_status_monitor.process_event.call_count, 2)

    def test_process_image_request_failure(self):
        """
        Test failure during image request processing.
        """
        # Simulate an exception in load_image_request
        self.handler.load_image_request = MagicMock(side_effect=Exception("Test error"))

        # Call process_image_request and assert the exception is raised
        with self.assertRaises(ProcessImageException):
            self.handler.process_image_request(self.mock_image_request)

        # Ensure failure handling methods were called
        self.mock_image_status_monitor.process_event.assert_called()

    @patch("aws.osml.model_runner.image_request_handler.SinkFactory.sink_features")
    @patch("aws.osml.model_runner.image_request_handler.ImageRequestHandler.deduplicate")
    @patch("aws.osml.model_runner.image_request_handler.FeatureTable.aggregate_features")
    def test_complete_image_request(self, mock_aggregate_features, mock_deduplicate, mock_sink_features):
        """
        Test successful completion of image request.
        """
        # Set up mock return values for our JobItem to complete
        self.mock_job_table.get_image_request.return_value = self.mock_job_item
        self.mock_job_item.processing_duration = 1000
        self.mock_job_item.region_error = 0

        # Set up mock return values for RegionRequest to complete
        mock_region_request = MagicMock()
        mock_raster_dataset = MagicMock()
        mock_sensor_model = MagicMock()
        mock_features = [
            {
                "type": "Feature",
                "properties": {
                    "inferenceTime": datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
                },
                "geometry": {"type": "Point", "coordinates": [-77.0364761352539, 38.89761287129639]},
            }
        ]
        mock_deduplicate.return_value = mock_features
        mock_aggregate_features.return_value = mock_features
        mock_sink_features.return_value = True

        # Call complete_image_request
        self.handler.complete_image_request(mock_region_request, "tif", mock_raster_dataset, mock_sensor_model)

        # Ensure sink_features was called correctly
        mock_sink_features.assert_called_once()

        # Ensure failure handling methods were called
        self.mock_image_status_monitor.process_event.assert_called()

    def test_fail_image_request(self):
        """
        Test fail_image_request method behavior.
        """
        # Call fail_image_request
        self.handler.fail_image_request(self.mock_job_item, Exception("Test failure"))

        # Ensure status monitor was updated and job table was called
        self.mock_image_status_monitor.process_event.assert_called_once_with(
            self.mock_job_item, RequestStatus.FAILED, "Test failure"
        )
        self.mock_job_table.end_image_request.assert_called_once_with(self.mock_job_item.image_id)


if __name__ == "__main__":
    main()
