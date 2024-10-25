#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase, main
from unittest.mock import Mock


class TestVariableTileTilingStrategy(TestCase):
    def test_compute_regions(self):
        """
        Test that regions are correctly computed for a full-sized image based on
        the specified nominal region size, tile size, and overlap.
        """
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        # Define image and tiling parameters
        full_image_size = (25000, 12000)
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Compute regions
        regions = tiling_strategy.compute_regions(((0, 0), full_image_size), nominal_region_size, tile_size, overlap)

        # Verify the number of computed regions
        assert len(regions) == 6

        # Verify that all computed regions match expected results
        for r in regions:
            assert r in [
                ((0, 0), (10000, 10000)),
                ((0, 9900), (10000, 10000)),
                ((0, 19800), (5200, 10000)),
                ((9900, 0), (10000, 2100)),
                ((9900, 9900), (10000, 2100)),
                ((9900, 19800), (5200, 2100)),
            ]

    def test_compute_regions_roi(self):
        """
        Test that regions are correctly computed within a specific region of interest (ROI)
        based on the specified nominal region size, tile size, and overlap.
        """
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        # Define tiling parameters
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Compute regions within the ROI
        regions = tiling_strategy.compute_regions(((200, 8000), (17000, 11800)), nominal_region_size, tile_size, overlap)

        # Verify the number of computed regions
        assert len(regions) == 4

        # Verify that all computed regions match expected results
        for r in regions:
            assert r in [
                ((200, 8000), (10000, 10000)),
                ((200, 17900), (7100, 10000)),
                ((10100, 8000), (10000, 1900)),
                ((10100, 17900), (7100, 1900)),
            ]

    def test_compute_regions_tiny_image(self):
        """
        Test that regions are correctly computed for a tiny image where the
        nominal region size is larger than the image itself.
        """
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        # Define tiling parameters
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Compute regions for a tiny image
        regions = tiling_strategy.compute_regions(((0, 0), (12000, 2000)), nominal_region_size, tile_size, overlap)

        # Verify the number of computed regions
        assert len(regions) == 2

        # Verify that all computed regions match expected results
        for r in regions:
            assert r in [((0, 0), (10000, 2000)), ((0, 9900), (2100, 2000))]

    def test_compute_tiles(self):
        """
        Test that tiles are correctly computed within specified regions based on tile size and overlap.
        """
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Test full region tiling
        tiles = tiling_strategy.compute_tiles(((0, 0), (10000, 10000)), tile_size, overlap)
        assert len(tiles) == 9
        for t in tiles:
            assert t in [
                ((0, 0), (4096, 4096)),
                ((0, 3996), (4096, 4096)),
                ((0, 7992), (2008, 4096)),
                ((3996, 0), (4096, 4096)),
                ((3996, 3996), (4096, 4096)),
                ((3996, 7992), (2008, 4096)),
                ((7992, 0), (4096, 2008)),
                ((7992, 3996), (4096, 2008)),
                ((7992, 7992), (2008, 2008)),
            ]

        # Test region on the right edge of the image
        tiles = tiling_strategy.compute_tiles(((0, 19800), (5200, 10000)), tile_size, overlap)
        assert len(tiles) == 6
        for t in tiles:
            assert t in [
                ((0, 19800), (4096, 4096)),
                ((0, 23796), (1204, 4096)),
                ((3996, 19800), (4096, 4096)),
                ((3996, 23796), (1204, 4096)),
                ((7992, 19800), (4096, 2008)),
                ((7992, 23796), (1204, 2008)),
            ]

        # Test region on the bottom edge of the image
        tiles = tiling_strategy.compute_tiles(((9900, 9900), (10000, 2100)), tile_size, overlap)
        assert len(tiles) == 3
        for t in tiles:
            assert t in [((9900, 9900), (4096, 2100)), ((9900, 13896), (4096, 2100)), ((9900, 17892), (2008, 2100))]

        # Test bottom right corner region
        tiles = tiling_strategy.compute_tiles(((9900, 19800), (5200, 2100)), tile_size, overlap)
        assert len(tiles) == 2
        for t in tiles:
            assert t in [((9900, 19800), (4096, 2100)), ((9900, 23796), (1204, 2100))]

    def test_compute_tiles_tiny_region(self):
        """
        Test that tiles are correctly computed for a small region that is smaller than the tile size.
        """
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Compute tiles for a small region
        tiles = tiling_strategy.compute_tiles(((10, 50), (1024, 2048)), tile_size, overlap)

        # Verify that only a single tile is produced for the tiny region
        assert len(tiles) == 1
        assert tiles[0] == ((10, 50), (1024, 2048))

    def test_deconflict_features(self):
        """
        Test that duplicate features are properly deconflicted based on specified rules.
        """
        from geojson import Feature

        from aws.osml.model_runner.inference import FeatureSelector
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        # Define image and tiling parameters
        full_image_size = (25000, 12000)
        full_image_region = ((0, 0), full_image_size)
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Sample feature inputs, including duplicates
        features = [
            Feature(properties={"imageBBox": [19804, 9904, 19824, 9924]}),
            Feature(properties={"imageBBox": [19805, 9905, 19825, 9925]}),
            Feature(properties={"imageBBox": [13900, 11000, 17510, 13910]}),
            Feature(properties={"imageBBox": [13900, 11000, 17510, 13910]}),
            Feature(properties={"imageBBox": [4000, 4000, 4010, 4010]}),
            Feature(properties={"imageBBox": [4000, 4000, 4010, 4010]}),
            Feature(properties={"imageBBox": [16000, 4000, 16010, 4010]}),
            Feature(properties={"imageBBox": [16000, 4000, 16010, 4010]}),
            Feature(properties={"imageBBox": [10, 10, 10, 10]}),
            Feature(properties={"imageBBox": [10, 10, 10, 10]}),
        ]

        # Mock feature selector to deconflict overlapping features
        class DummyFeatureSelector(FeatureSelector):
            def select_features(self, features):
                if len(features) > 0:
                    return [features[0]]
                return []

        mock_feature_selector = Mock(wraps=DummyFeatureSelector())

        # Deconflict features using the tiling strategy
        deduped_features = tiling_strategy.cleanup_duplicate_features(
            full_image_region, nominal_region_size, tile_size, overlap, features, mock_feature_selector
        )

        # Verify the correct number of deconflicted features
        assert len(deduped_features) == 6

        # Verify that the feature selector was called for each overlapping group
        assert len(mock_feature_selector.method_calls) == 4


if __name__ == "__main__":
    main()
