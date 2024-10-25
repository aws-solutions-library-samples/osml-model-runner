#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase, main

import pytest

from aws.osml.model_runner.tile_worker.tiling_strategy import generate_crops


class TestTilingStrategy(TestCase):
    def test_chip_generator_partial_overlap(self):
        """
        Test that the chip generator correctly produces crops based on the given image size,
        crop size, and overlap for a partial overlap scenario.
        """

        # Set up chips with partial overlap
        chip_list = []
        for chip in generate_crops(((5, 10), (1024, 1024)), (300, 300), (44, 44)):
            chip_list.append(chip)

        # Verify the total number of chips and their positions/sizes
        assert len(chip_list) == 16
        assert chip_list[0] == ((5, 10), (300, 300))
        assert chip_list[1] == ((5, 266), (300, 300))
        assert chip_list[3] == ((5, 778), (256, 300))
        assert chip_list[12] == ((773, 10), (300, 256))
        assert chip_list[15] == ((773, 778), (256, 256))

    def test_chip_generator_no_overlap(self):
        """
        Test that the chip generator correctly produces crops based on the given image size,
        crop size, and overlap for a scenario with no overlap.
        """

        # Set up chips with no overlap, producing complete tiles
        chip_list = []
        for chip in generate_crops(((0, 0), (5000, 2500)), (2048, 2048), (0, 0)):
            chip_list.append(chip)

        # Verify the total number of chips and their positions/sizes
        assert len(chip_list) == 6
        assert chip_list[0] == ((0, 0), (2048, 2048))
        assert chip_list[1] == ((0, 2048), (2048, 2048))
        assert chip_list[2] == ((0, 4096), (904, 2048))
        assert chip_list[3] == ((2048, 0), (2048, 452))
        assert chip_list[4] == ((2048, 2048), (2048, 452))
        assert chip_list[5] == ((2048, 4096), (904, 452))

    def test_chip_generator_full_overlap(self):
        """
        Test that the chip generator correctly produces crops based on the given image size,
        crop size, and overlap for a full overlap scenario.
        """

        # Set up chips with full overlap, large crop sizes
        chip_list = []
        for chip in generate_crops(((150, 150), (5000, 5000)), (2048, 2048), (1024, 1024)):
            chip_list.append(chip)

        # Verify the output is logically handled even without assert checks (e.g., no exceptions)
        assert len(chip_list) == 16

    def test_invalid_chip_generator(self):
        """
        Test that the chip generator raises an error for invalid overlap configurations.
        Verifies scenarios where the overlap exceeds crop size, which is not allowed.
        """

        # Overlap values larger than crop dimensions should raise a ValueError
        with pytest.raises(ValueError):
            chip_list = []
            for chip in generate_crops(((5, 10), (1024, 1024)), (300, 300), (301, 0)):
                chip_list.append(chip)

        with pytest.raises(ValueError):
            chip_list = []
            for chip in generate_crops(((5, 10), (1024, 1024)), (300, 300), (0, 301)):
                chip_list.append(chip)


if __name__ == "__main__":
    main()
