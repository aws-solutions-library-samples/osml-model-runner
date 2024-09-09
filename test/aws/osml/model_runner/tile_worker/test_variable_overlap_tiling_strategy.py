#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from unittest import TestCase, main
from unittest.mock import Mock


class TestVariableOverlapTilingStrategy(TestCase):
    def test_compute_regions_full_image(self):
        from aws.osml.model_runner.tile_worker import VariableOverlapTilingStrategy

        tiling_strategy = VariableOverlapTilingStrategy()

        full_image_size = (25000, 12000)
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Check a full image
        regions = tiling_strategy.compute_regions(((0, 0), full_image_size), nominal_region_size, tile_size, overlap)
        assert len(regions) == 8
        for r in regions:
            assert r in [
                ((0, 0), (7580, 8048)),
                ((0, 6968), (7580, 8048)),
                ((0, 13936), (7580, 8048)),
                ((0, 20904), (4096, 8048)),
                ((7904, 0), (7580, 4096)),
                ((7904, 6968), (7580, 4096)),
                ((7904, 13936), (7580, 4096)),
                ((7904, 20904), (4096, 4096)),
            ]

    def test_compute_regions_roi(self):
        from aws.osml.model_runner.tile_worker import VariableOverlapTilingStrategy

        tiling_strategy = VariableOverlapTilingStrategy()

        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Check regions generated from a subset of an image
        regions = tiling_strategy.compute_regions(((200, 8000), (17000, 11800)), nominal_region_size, tile_size, overlap)
        assert len(regions) == 6
        for r in regions:
            assert r in [
                ((200, 8000), (7322, 7948)),
                ((200, 14452), (7322, 7948)),
                ((200, 20904), (4096, 7948)),
                ((7904, 8000), (7322, 4096)),
                ((7904, 14452), (7322, 4096)),
                ((7904, 20904), (4096, 4096)),
            ]

    def test_compute_regions_tiny_image(self):
        from aws.osml.model_runner.tile_worker import VariableOverlapTilingStrategy

        tiling_strategy = VariableOverlapTilingStrategy()

        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # Check regions generated from an image that has a dimension smaller than the fixed tile size
        regions = tiling_strategy.compute_regions(((0, 0), (12000, 2000)), nominal_region_size, tile_size, overlap)
        assert len(regions) == 2
        for r in regions:
            assert r in [((0, 0), (8048, 2000)), ((0, 7904), (4096, 2000))]

    def test_compute_tiles(self):
        from aws.osml.model_runner.tile_worker import VariableOverlapTilingStrategy

        tiling_strategy = VariableOverlapTilingStrategy()
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # First full region
        tiles = tiling_strategy.compute_tiles(((0, 0), (7580, 8048)), tile_size, overlap)
        assert len(tiles) == 4
        for t in tiles:
            assert t in [
                ((0, 0), (4096, 4096)),
                ((0, 3484), (4096, 4096)),
                ((3952, 0), (4096, 4096)),
                ((3952, 3484), (4096, 4096)),
            ]

        # A region on the right edge of the image
        tiles = tiling_strategy.compute_tiles(((0, 20904), (4096, 8048)), tile_size, overlap)
        assert len(tiles) == 2
        for t in tiles:
            assert t in [((0, 20904), (4096, 4096)), ((3952, 20904), (4096, 4096))]

        # A region on the bottom edge of the image
        tiles = tiling_strategy.compute_tiles(((7904, 13936), (7580, 4096)), tile_size, overlap)
        assert len(tiles) == 2
        for t in tiles:
            assert t in [((7904, 13936), (4096, 4096)), ((7904, 17420), (4096, 4096))]

        # The bottom right corner region
        tiles = tiling_strategy.compute_tiles(((7904, 20904), (4096, 4096)), tile_size, overlap)
        assert len(tiles) == 1
        assert tiles[0] == ((7904, 20904), (4096, 4096))

    def test_compute_tiles_tiny_region(self):
        from aws.osml.model_runner.tile_worker import VariableOverlapTilingStrategy

        tiling_strategy = VariableOverlapTilingStrategy()
        overlap = (100, 100)
        tile_size = (4096, 4096)

        # If the requested region is smaller than the actual tile size then the full region
        # will be turned into a single tile. This is an odd edge case that we don't expect
        # to see much but we don't want to error out and will instead just fall back to
        # a partial tile strategy.
        tiles = tiling_strategy.compute_tiles(((10, 50), (1024, 2048)), tile_size, overlap)
        assert len(tiles) == 1
        assert tiles[0] == ((10, 50), (1024, 2048))

    def test_deconflict_features(self):
        from geojson import Feature

        from aws.osml.model_runner.inference import FeatureSelector
        from aws.osml.model_runner.tile_worker import VariableOverlapTilingStrategy

        tiling_strategy = VariableOverlapTilingStrategy()

        full_image_size = (25000, 12000)
        full_image_region = ((0, 0), full_image_size)
        nominal_region_size = (10000, 10000)
        overlap = (100, 100)
        tile_size = (4096, 4096)

        features = [
            # Duplicate features in overlap of two regions, keep 1
            Feature(properties={"imageBBox": [20904, 7904, 20924, 7924]}),
            Feature(properties={"imageBBox": [20905, 7905, 20925, 7925]}),
            # Duplicate features in overlap of two tiles in lower edge region, keep 1
            Feature(properties={"imageBBox": [17500, 10000, 17510, 10010]}),
            Feature(properties={"imageBBox": [17500, 10000, 17510, 10010]}),
            # Duplicate features in overlap of tiles in center of first region, keep 1
            Feature(properties={"imageBBox": [4000, 4000, 4010, 4010]}),
            Feature(properties={"imageBBox": [4000, 4000, 4010, 4010]}),
            # Duplicate features in overlap of tiles in center of second region, keep 1
            Feature(properties={"imageBBox": [10000, 4000, 10010, 4010]}),
            Feature(properties={"imageBBox": [10000, 4000, 10010, 4010]}),
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

        # Check to ensure we have the correct number of features returned and that the feature selector
        # was called once for each overlapping group
        assert len(deduped_features) == 6
        assert len(mock_feature_selector.method_calls) == 4


if __name__ == "__main__":
    main()
