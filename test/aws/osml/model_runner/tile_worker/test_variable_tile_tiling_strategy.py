#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase, main
from unittest.mock import Mock


class TestVariableTileTilingStrategy(TestCase):
    def test_compute_regions(self):
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        full_image_size = (25000, 12000)
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        regions = tiling_strategy.compute_regions(((0, 0), full_image_size), nominal_region_size, tile_size, overlap)
        assert len(regions) == 6
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
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        regions = tiling_strategy.compute_regions(((200, 8000), (17000, 11800)), nominal_region_size, tile_size, overlap)
        assert len(regions) == 4
        for r in regions:
            assert r in [
                ((200, 8000), (10000, 10000)),
                ((200, 17900), (7100, 10000)),
                ((10100, 8000), (10000, 1900)),
                ((10100, 17900), (7100, 1900)),
            ]

    def test_compute_regions_tiny_image(self):
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        regions = tiling_strategy.compute_regions(((0, 0), (12000, 2000)), nominal_region_size, tile_size, overlap)
        assert len(regions) == 2
        for r in regions:
            assert r in [((0, 0), (10000, 2000)), ((0, 9900), (2100, 2000))]

    def test_compute_tiles(self):
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # First full region
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

        # A region on the right edge of the image
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

        # A region on the bottom edge of the image
        tiles = tiling_strategy.compute_tiles(((9900, 9900), (10000, 2100)), tile_size, overlap)
        assert len(tiles) == 3
        for t in tiles:
            assert t in [((9900, 9900), (4096, 2100)), ((9900, 13896), (4096, 2100)), ((9900, 17892), (2008, 2100))]

        # The bottom right corner region
        tiles = tiling_strategy.compute_tiles(((9900, 19800), (5200, 2100)), tile_size, overlap)
        assert len(tiles) == 2
        for t in tiles:
            assert t in [((9900, 19800), (4096, 2100)), ((9900, 23796), (1204, 2100))]

    def test_compute_tiles_tiny_region(self):
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # First full region
        tiles = tiling_strategy.compute_tiles(((10, 50), (1024, 2048)), tile_size, overlap)
        assert len(tiles) == 1
        assert tiles[0] == ((10, 50), (1024, 2048))

    def test_deconflict_features(self):
        from geojson import Feature

        from aws.osml.model_runner.inference import FeatureSelector
        from aws.osml.model_runner.tile_worker import VariableTileTilingStrategy

        tiling_strategy = VariableTileTilingStrategy()

        full_image_size = (25000, 12000)
        full_image_region = ((0, 0), full_image_size)
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        features = [
            # Duplicate features in overlap of two regions, keep 1
            Feature(properties={"imageBBox": [19804, 9904, 19824, 9924]}),
            Feature(properties={"imageBBox": [19805, 9905, 19825, 9925]}),
            # Duplicate features in overlap of two tiles in lower edge region, keep 1
            Feature(properties={"imageBBox": [13900, 11000, 17510, 13910]}),
            Feature(properties={"imageBBox": [13900, 11000, 17510, 13910]}),
            # Duplicate features in overlap of tiles in center of first region, keep 1
            Feature(properties={"imageBBox": [4000, 4000, 4010, 4010]}),
            Feature(properties={"imageBBox": [4000, 4000, 4010, 4010]}),
            # Duplicate features in overlap of tiles in center of second region, keep 1
            Feature(properties={"imageBBox": [16000, 4000, 16010, 4010]}),
            Feature(properties={"imageBBox": [16000, 4000, 16010, 4010]}),
            # Features duplicate but do not touch overlap regions, keep 2
            Feature(properties={"imageBBox": [10, 10, 10, 10]}),
            Feature(properties={"imageBBox": [10, 10, 10, 10]}),
        ]

        class DummyFeatureSelector(FeatureSelector):
            def select_features(self, features):
                if len(features) > 0:
                    return [features[0]]
                return []

        mock_feature_selector = Mock(wraps=DummyFeatureSelector())
        deduped_features = tiling_strategy.cleanup_duplicate_features(
            full_image_region, nominal_region_size, tile_size, overlap, features, mock_feature_selector
        )

        print(deduped_features)

        # Check to ensure we have the correct number of features returned and that the feature selector
        # was called once for each overlapping group
        assert len(deduped_features) == 6
        assert len(mock_feature_selector.method_calls) == 4


if __name__ == "__main__":
    main()
