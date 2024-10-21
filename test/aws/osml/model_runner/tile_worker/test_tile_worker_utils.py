#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase, main
from unittest.mock import Mock, patch


class TestTileWorkerUtils(TestCase):
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.TileWorker", autospec=True)
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.ServiceConfig", autospec=True)
    def test_setup_tile_workers(self, mock_service_config, mock_tile_worker):
        """
        Test the setup of tile workers, ensuring the correct number of workers is initialized
        based on the configuration and that workers are started correctly.
        """
        from aws.osml.model_runner.api import RegionRequest
        from aws.osml.model_runner.tile_worker.tile_worker_utils import setup_tile_workers

        mock_tile_worker.start = Mock()
        mock_num_tile_workers = 4
        mock_service_config.workers = mock_num_tile_workers
        mock_region_request = RegionRequest(
            {
                "tile_size": (10, 10),
                "tile_overlap": (1, 1),
                "tile_format": "NITF",
                "image_id": "1",
                "image_url": "/mock/path",
                "region_bounds": ((0, 0), (50, 50)),
                "model_invoke_mode": "SM_ENDPOINT",
                "image_extension": "fake",
            }
        )
        mock_sensor_model = None
        mock_elevation_model = None
        work_queue, tile_worker_list = setup_tile_workers(mock_region_request, mock_sensor_model, mock_elevation_model)

        # Assert that the correct number of tile workers are created
        assert len(tile_worker_list) == mock_num_tile_workers
        # Verify that each worker's start method is called
        for worker in tile_worker_list:
            # Assert that the mock function was called exactly 4 times
            self.assertEqual(worker.start.call_count, 4)

    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.FeatureTable", autospec=True)
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.TileWorker", autospec=True)
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.ServiceConfig", autospec=True)
    def test_setup_tile_workers_exception(self, mock_service_config, mock_tile_worker, mock_feature_table):
        """
        Test that an exception during tile worker setup raises a SetupTileWorkersException.
        """
        from aws.osml.model_runner.api import RegionRequest
        from aws.osml.model_runner.tile_worker.exceptions import SetupTileWorkersException
        from aws.osml.model_runner.tile_worker.tile_worker_utils import setup_tile_workers

        mock_tile_worker.start = Mock()
        mock_num_tile_workers = 4
        mock_service_config.workers = mock_num_tile_workers
        mock_feature_table.side_effect = Exception("Mock processing exception")
        mock_region_request = RegionRequest(
            {
                "tile_size": (10, 10),
                "tile_overlap": (1, 1),
                "tile_format": "NITF",
                "image_id": "1",
                "image_url": "/mock/path",
                "region_bounds": ((0, 0), (50, 50)),
                "model_invoke_mode": "SM_ENDPOINT",
                "image_extension": "fake",
            }
        )
        mock_sensor_model = None
        mock_elevation_model = None
        with self.assertRaises(SetupTileWorkersException):
            # Attempt to set up workers should fail and raise the specified exception
            setup_tile_workers(mock_region_request, mock_sensor_model, mock_elevation_model)

    def test_process_tiles(self):
        """
        Test processing of image tiles using a tiling strategy, ensuring all expected tiles are processed
        without errors. The test also validates successful integration with GDAL datasets.
        """
        from aws.osml.model_runner.api import RegionRequest
        from aws.osml.model_runner.database import RegionRequestItem
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy
        from aws.osml.model_runner.tile_worker.tile_worker_utils import process_tiles, setup_tile_workers

        # Mock the RegionRequest and RegionRequestItem
        mock_region_request = RegionRequest(
            {
                "tile_size": (10, 10),
                "tile_overlap": (0, 0),
                "tile_format": "NITF",
                "image_id": "1",
                "image_url": "/mock/path",
                "region_bounds": ((0, 0), (50, 50)),
                "model_invoke_mode": "SM_ENDPOINT",
                "image_extension": "fake",
                "failed_tiles": [],
            }
        )
        region_request_item = RegionRequestItem.from_region_request(mock_region_request)

        # Load the testing Dataset and SensorModel
        ds, sensor_model = self.get_dataset_and_camera()

        # Setup tile workers
        work_queue, tile_worker_list = setup_tile_workers(mock_region_request, sensor_model, None)

        # Execute process_tiles
        total_tile_count, tile_error_count = process_tiles(
            tiling_strategy=VariableTileTilingStrategy(),
            region_request_item=region_request_item,
            tile_queue=work_queue,
            tile_workers=tile_worker_list,
            raster_dataset=ds,
            sensor_model=sensor_model,
        )

        # Verify expected results
        assert total_tile_count == 25
        assert tile_error_count == 0

    def test_next_greater_multiple(self):
        """
        Test finding the next greater multiple of a number.
        """
        assert 16 == self.next_greater_multiple(1, 16)
        assert 16 == self.next_greater_multiple(15, 16)
        assert 16 == self.next_greater_multiple(16, 16)
        assert 32 == self.next_greater_multiple(17, 16)
        assert 48 == self.next_greater_multiple(42, 16)
        assert 64 == self.next_greater_multiple(50, 16)
        assert 528 == self.next_greater_multiple(513, 16)

    def test_next_greater_power_of_two(self):
        """
        Test finding the next greater power of two for a given number.
        """
        assert 1 == self.next_greater_power_of_two(1)
        assert 2 == self.next_greater_power_of_two(2)
        assert 4 == self.next_greater_power_of_two(3)
        assert 8 == self.next_greater_power_of_two(8)
        assert 64 == self.next_greater_power_of_two(42)
        assert 128 == self.next_greater_power_of_two(100)
        assert 256 == self.next_greater_power_of_two(255)
        assert 512 == self.next_greater_power_of_two(400)

    def test_sizeof_fmt(self):
        """
        Test the human-readable size formatting function.
        """
        from aws.osml.model_runner.tile_worker.tile_worker_utils import sizeof_fmt

        assert sizeof_fmt(250) == "250.0B"
        assert sizeof_fmt(1024**3) == "1.0GiB"
        assert sizeof_fmt(1024**8) == "1.0YiB"

    @staticmethod
    def get_dataset_and_camera():
        """
        Utility method to load a dataset and associated sensor model from a test file.
        """
        from aws.osml.gdal.gdal_utils import load_gdal_dataset

        return load_gdal_dataset("./test/data/GeogToWGS84GeoKey5.tif")

    @staticmethod
    def next_greater_multiple(n: int, m: int) -> int:
        """
        Return the minimum value that is greater than or equal to n that is evenly divisible by m.
        """
        if n % m == 0:
            return n
        return n + (m - n % m)

    @staticmethod
    def next_greater_power_of_two(n: int) -> int:
        """
        Returns the smallest power of 2 that is greater than or equal to the input parameter.
        """
        count = 0
        if n and not (n & (n - 1)):
            return n
        while n != 0:
            n >>= 1
            count += 1
        return 1 << count


if __name__ == "__main__":
    main()
