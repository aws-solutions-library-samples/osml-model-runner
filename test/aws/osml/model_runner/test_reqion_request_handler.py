#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase, main
from unittest.mock import MagicMock

from osgeo import gdal

from aws.osml.model_runner.api import RegionRequest
from aws.osml.model_runner.app_config import ServiceConfig
from aws.osml.model_runner.common import EndpointUtils, RequestStatus
from aws.osml.model_runner.database import EndpointStatisticsTable, JobItem, JobTable, RegionRequestItem, RegionRequestTable
from aws.osml.model_runner.exceptions import SelfThrottledRegionException
from aws.osml.model_runner.region_request_handler import RegionRequestHandler
from aws.osml.model_runner.status import RegionStatusMonitor
from aws.osml.model_runner.tile_worker import TilingStrategy
from aws.osml.photogrammetry import SensorModel


class TestRegionRequestHandler(TestCase):
    def setUp(self):
        # Set up mock dependencies
        self.mock_region_request_table = MagicMock(spec=RegionRequestTable)
        self.mock_job_table = MagicMock(spec=JobTable)
        self.mock_region_status_monitor = MagicMock(spec=RegionStatusMonitor)
        self.mock_endpoint_statistics_table = MagicMock(spec=EndpointStatisticsTable)
        self.mock_tiling_strategy = MagicMock(spec=TilingStrategy)
        self.mock_endpoint_utils = MagicMock(spec=EndpointUtils)
        self.mock_config = MagicMock(spec=ServiceConfig)

        # Example config properties
        self.mock_config.self_throttling = False

        # Instantiate the handler with mocked dependencies
        self.handler = RegionRequestHandler(
            region_request_table=self.mock_region_request_table,
            job_table=self.mock_job_table,
            region_status_monitor=self.mock_region_status_monitor,
            endpoint_statistics_table=self.mock_endpoint_statistics_table,
            tiling_strategy=self.mock_tiling_strategy,
            endpoint_utils=self.mock_endpoint_utils,
            config=self.mock_config,
        )

        # Mock the region request and dataset
        self.mock_raster_dataset = MagicMock(spec=gdal.Dataset)
        self.mock_sensor_model = MagicMock(spec=SensorModel)

        # Mock the region request and dataset
        self.mock_raster_dataset = MagicMock(spec=gdal.Dataset)
        self.mock_sensor_model = MagicMock(spec=SensorModel)

        # Add necessary attributes to mock region request
        self.mock_region_request = RegionRequest(
            {
                "tile_size": (10, 10),
                "tile_overlap": (1, 1),
                "tile_format": "NITF",
                "image_id": "test-image-d",
                "image_url": "./test/data/small.ntf",
                "region_bounds": ((0, 0), (50, 50)),
                "model_name": "test-model",
                "model_invoke_mode": "SM_ENDPOINT",
                "image_extension": "NITF",
            }
        )

        # Create a mock item from the request
        self.mock_region_request_item = RegionRequestItem.from_region_request(self.mock_region_request)

        # Mock the is_valid function and set to true, so we can reverse for failure testing
        self.mock_region_request.is_valid = MagicMock(return_value=True)

    def test_process_region_request_success(self):
        """
        Test successful region processing.
        """
        # Mock tile processing behavior
        self.mock_tiling_strategy.return_value = MagicMock()
        self.mock_region_request_table.start_region_request.return_value = self.mock_region_request_item
        self.mock_region_request_table.update_region_request.return_value = self.mock_region_request_item
        self.mock_job_table.complete_region_request.return_value = MagicMock(spec=JobItem)

        # Call process_region_request
        result = self.handler.process_region_request(
            region_request=self.mock_region_request,
            region_request_item=self.mock_region_request_item,
            raster_dataset=self.mock_raster_dataset,
            sensor_model=self.mock_sensor_model,
        )

        # Assert that the region request was started and updated correctly
        self.mock_region_request_table.start_region_request.assert_called_once_with(self.mock_region_request_item)
        self.mock_region_request_table.update_region_request.assert_called_once()
        self.mock_job_table.complete_region_request.assert_called_once()
        self.mock_region_status_monitor.process_event.assert_called()
        assert isinstance(result, JobItem)

    def test_process_region_request_throttling(self):
        """
        Test region request processing when throttling is enabled.
        """
        self.mock_config.self_throttling = True

        # Mock endpoint statistics behavior
        self.mock_endpoint_utils.calculate_max_regions.return_value = 5
        self.mock_endpoint_statistics_table.current_in_progress_regions.return_value = 5

        # Assert that throttling is raised
        with self.assertRaises(SelfThrottledRegionException):
            self.handler.process_region_request(
                region_request=self.mock_region_request,
                region_request_item=self.mock_region_request_item,
                raster_dataset=self.mock_raster_dataset,
                sensor_model=self.mock_sensor_model,
            )

        self.mock_endpoint_statistics_table.increment_region_count.assert_not_called()
        self.mock_endpoint_statistics_table.decrement_region_count.assert_not_called()

    def test_process_region_request_invalid_request(self):
        """
        Test processing with an invalid RegionRequest.
        """
        # Simulate an invalid region request
        self.mock_region_request.is_valid.return_value = False

        # Assert that ValueError is raised for invalid region request
        with self.assertRaises(ValueError):
            self.handler.process_region_request(
                region_request=self.mock_region_request,
                region_request_item=self.mock_region_request_item,
                raster_dataset=self.mock_raster_dataset,
                sensor_model=self.mock_sensor_model,
            )
        self.mock_endpoint_statistics_table.increment_region_count.assert_not_called()
        self.mock_endpoint_statistics_table.decrement_region_count.assert_not_called()

    def test_process_region_request_exception(self):
        """
        Test region processing failure scenario.
        """
        self.mock_tiling_strategy.return_value = MagicMock()
        self.mock_job_table.complete_region_request.return_value = MagicMock(spec=JobItem)

        # Simulate tile processing throwing an error
        self.mock_region_request_table.update_region_request.side_effect = Exception("Tile processing failed")

        # Call process_region_request and expect failure
        result = self.handler.process_region_request(
            region_request=self.mock_region_request,
            region_request_item=self.mock_region_request_item,
            raster_dataset=self.mock_raster_dataset,
            sensor_model=self.mock_sensor_model,
        )

        # Assert that fail_region_request was called due to failure
        self.mock_region_request_table.start_region_request.assert_called_once()
        self.mock_region_status_monitor.process_event.assert_called()
        assert self.mock_region_request_item.message == "Failed to process image region: Tile processing failed"
        assert isinstance(result, JobItem)

    def test_fail_region_request(self):
        """
        Test fail_region_request method behavior.
        """
        self.mock_job_table.complete_region_request.return_value = MagicMock(spec=JobItem)
        result = self.handler.fail_region_request(self.mock_region_request_item)

        # Assert that the region request was updated with FAILED status
        self.mock_region_request_table.complete_region_request.assert_called_once_with(
            self.mock_region_request_item, RequestStatus.FAILED
        )
        self.mock_region_status_monitor.process_event.assert_called_once()
        self.mock_job_table.complete_region_request.assert_called_once()
        assert isinstance(result, JobItem)


if __name__ == "__main__":
    main()
