#  Copyright 2023-2025 Amazon.com, Inc. or its affiliates.

import unittest
from unittest.mock import MagicMock, patch

from aws.osml.model_runner.model_runner import ModelRunner, RetryableJobException


class TestModelRunner(unittest.TestCase):
    def setUp(self):
        # Create the instance of ModelRunner
        self.runner = ModelRunner()
        # Mock the process methods
        self.runner.image_request_handler = MagicMock()
        self.runner.region_request_handler = MagicMock()

    def test_run_starts_monitoring(self):
        """Test that the `run` method sets up and starts the monitoring loop."""
        # Mock method calls
        self.runner.monitor_work_queues = MagicMock()

        # Call the method
        self.runner.run()

        # Ensure the run method calls monitor_work_queues and sets `self.running`
        self.assertTrue(self.runner.running)
        self.runner.monitor_work_queues.assert_called_once()

    def test_stop_stops_running(self):
        """Test that the `stop` method correctly stops the runner."""
        # Call stop
        self.runner.stop()

        # Check if `self.running` is set to False
        self.assertFalse(self.runner.running)

    @patch("aws.osml.model_runner.model_runner.RegionRequestHandler.process_region_request")
    @patch("aws.osml.model_runner.model_runner.RequestQueue.finish_request")
    @patch("aws.osml.model_runner.model_runner.load_gdal_dataset")
    def test_process_region_requests_success(self, mock_load_gdal, mock_finish_request, mock_process_region):
        """Test processing of region requests successfully."""
        mock_region_request_item = MagicMock()
        mock_image_request_item = MagicMock()
        self.runner._get_or_create_region_request_item = MagicMock(return_value=mock_region_request_item)
        mock_load_gdal.return_value = (MagicMock(), MagicMock())
        mock_process_region.return_value = mock_image_request_item
        self.runner.job_table.is_image_request_complete = MagicMock(return_value=True)

        # Simulate queue data
        self.runner.region_requests_iter = iter([("receipt_handle", {"region_id": "region_123"})])

        # Call method
        self.runner._process_region_requests()

        # Ensure region request was processed correctly
        self.runner.image_request_handler.complete_image_request.assert_called_once()
        mock_finish_request.assert_called_once_with("receipt_handle")

    def test_process_image_request_noimage(self):
        """Test path where the scheduler does not return an ImageRequest to process"""
        with patch.object(self.runner, "image_job_scheduler", new_callable=MagicMock) as mock_scheduler:
            mock_scheduler.get_next_scheduled_request.return_value = None
            result = self.runner._process_image_requests()
            self.assertFalse(result)

    def test_process_image_requests_retryable(self):
        """Test that a RetryableJobException resets the request."""

        with patch.object(self.runner, "image_job_scheduler", new_callable=MagicMock) as mock_scheduler, patch.object(
            self.runner, "image_request_handler", new_callable=MagicMock
        ) as mock_handler:
            mock_image_request = MagicMock()
            mock_scheduler.get_next_scheduled_request.return_value = mock_image_request
            mock_handler.process_image_request.side_effect = RetryableJobException()

            result = self.runner._process_image_requests()
            self.assertTrue(result)

            mock_scheduler.finish_request.assert_called_once_with(mock_image_request, should_retry=True)

    @patch("aws.osml.model_runner.model_runner.ImageRequest")
    def test_process_image_requests_general_error(self, mock_image_request):
        """Test that general exceptions mark the image request as failed."""

        with patch.object(self.runner, "image_job_scheduler", new_callable=MagicMock) as mock_scheduler, patch.object(
            self.runner, "image_request_handler", new_callable=MagicMock
        ) as mock_handler:
            mock_image_request = MagicMock()
            mock_scheduler.get_next_scheduled_request.return_value = mock_image_request
            mock_handler.process_image_request.side_effect = Exception("Some error")

            result = self.runner._process_image_requests()
            self.assertTrue(result)

            mock_scheduler.finish_request.assert_called_once_with(mock_image_request)

    @patch("aws.osml.model_runner.model_runner.RegionRequestHandler.process_region_request")
    @patch("aws.osml.model_runner.model_runner.RequestQueue.finish_request")
    @patch("aws.osml.model_runner.model_runner.RequestQueue.reset_request")
    def test_process_region_requests_general_error(self, mock_reset_request, mock_finish_request, mock_process_region):
        """Test that general exceptions log an error and complete the request."""
        # Mock exception
        mock_process_region.side_effect = Exception("Some region processing error")

        # Simulate queue data
        self.runner.region_requests_iter = iter([("receipt_handle", {"region_id": "region_123"})])

        # Call method
        self.runner._process_region_requests()

        # Ensure the request was completed and logged
        mock_finish_request.assert_called_once_with("receipt_handle")
        mock_reset_request.assert_not_called()  # Ensure no reset on general errors


if __name__ == "__main__":
    unittest.main()
