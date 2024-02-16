#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from unittest import TestCase, main
from unittest.mock import Mock, patch

import pytest


class TestTileWorkerUtils(TestCase):
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.TileWorker", autospec=True)
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.ServiceConfig", autospec=True)
    def test_setup_tile_workers(self, mock_service_config, mock_tile_worker):
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
        assert len(tile_worker_list) == mock_num_tile_workers

    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.FeatureTable", autospec=True)
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.TileWorker", autospec=True)
    @patch("aws.osml.model_runner.tile_worker.tile_worker_utils.ServiceConfig", autospec=True)
    def test_setup_tile_workers_exception(self, mock_service_config, mock_tile_worker, mock_feature_table):
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
            # with self.assertRaises(ValueError):
            setup_tile_workers(mock_region_request, mock_sensor_model, mock_elevation_model)

    def test_chip_generator(self):
        from aws.osml.model_runner.tile_worker.tile_worker_utils import generate_crops

        chip_list = []
        for chip in generate_crops(((5, 10), (1024, 1024)), (300, 300), (44, 44)):
            chip_list.append(chip)

        assert len(chip_list) == 16
        assert chip_list[0] == ((5, 10), (300, 300))
        assert chip_list[1] == ((5, 266), (300, 300))
        assert chip_list[3] == ((5, 778), (256, 300))
        assert chip_list[12] == ((773, 10), (300, 256))
        assert chip_list[15] == ((773, 778), (256, 256))

        chip_list = []
        for chip in generate_crops(((0, 0), (5000, 2500)), (2048, 2048), (0, 0)):
            chip_list.append(chip)

        assert len(chip_list) == 6
        assert chip_list[0] == ((0, 0), (2048, 2048))
        assert chip_list[1] == ((0, 2048), (2048, 2048))
        assert chip_list[2] == ((0, 4096), (904, 2048))
        assert chip_list[3] == ((2048, 0), (2048, 452))
        assert chip_list[4] == ((2048, 2048), (2048, 452))
        assert chip_list[5] == ((2048, 4096), (904, 452))

        chip_list = []
        for chip in generate_crops(((150, 150), (5000, 5000)), (2048, 2048), (1024, 1024)):
            chip_list.append(chip)

    def test_invalid_chip_generator(self):
        from aws.osml.model_runner.tile_worker.tile_worker_utils import generate_crops

        with pytest.raises(ValueError):
            chip_list = []
            for chip in generate_crops(((5, 10), (1024, 1024)), (300, 300), (301, 0)):
                chip_list.append(chip)

        with pytest.raises(ValueError):
            chip_list = []
            for chip in generate_crops(((5, 10), (1024, 1024)), (300, 300), (0, 301)):
                chip_list.append(chip)

    def test_next_greater_multiple(self):
        assert 16 == self.next_greater_multiple(1, 16)
        assert 16 == self.next_greater_multiple(15, 16)
        assert 16 == self.next_greater_multiple(16, 16)
        assert 32 == self.next_greater_multiple(17, 16)
        assert 48 == self.next_greater_multiple(42, 16)
        assert 64 == self.next_greater_multiple(50, 16)
        assert 528 == self.next_greater_multiple(513, 16)

    def test_next_greater_power_of_two(self):
        assert 1 == self.next_greater_power_of_two(1)
        assert 2 == self.next_greater_power_of_two(2)
        assert 4 == self.next_greater_power_of_two(3)
        assert 8 == self.next_greater_power_of_two(8)
        assert 64 == self.next_greater_power_of_two(42)
        assert 128 == self.next_greater_power_of_two(100)
        assert 256 == self.next_greater_power_of_two(255)
        assert 512 == self.next_greater_power_of_two(400)

    def test_sizeof_fmt(self):
        from aws.osml.model_runner.tile_worker.tile_worker_utils import sizeof_fmt

        dummy_250_b = sizeof_fmt(250)
        # Black formatter doesn't play well with the **'s wrapped in brackets
        # fmt: off
        dummy_1_gb = sizeof_fmt(1024 ** 3)
        dummy_1_yib = sizeof_fmt(1024 ** 8)
        # fmt: on
        assert dummy_250_b == "250.0B"
        assert dummy_1_gb == "1.0GiB"
        assert dummy_1_yib == "1.0YiB"

    @staticmethod
    def get_dataset_and_camera():
        from aws.osml.gdal.gdal_utils import load_gdal_dataset

        ds, sensor_model = load_gdal_dataset("./test/data/GeogToWGS84GeoKey5.tif")
        return ds, sensor_model

    @staticmethod
    def next_greater_multiple(n: int, m: int) -> int:
        """
        Return the minimum value that is greater than or equal to n that is evenly divisible by m.

        :param n: the input value
        :param m: the multiple
        :return: the minimum multiple of m greater than n
        """
        if n % m == 0:
            return n

        return n + (m - n % m)

    @staticmethod
    def next_greater_power_of_two(n: int) -> int:
        """
        Returns the number that is both a power of 2 and greater than or equal to the input parameter.
        For example input 100 returns 128.

        :param n: the input integer
        :return: power of 2 greater than or equal to input
        """

        count = 0

        # First n in the below condition is for the case where n is 0
        # Second condition is only true if n is already a power of 2
        if n and not (n & (n - 1)):
            return n

        while n != 0:
            n >>= 1
            count += 1

        return 1 << count


if __name__ == "__main__":
    main()
