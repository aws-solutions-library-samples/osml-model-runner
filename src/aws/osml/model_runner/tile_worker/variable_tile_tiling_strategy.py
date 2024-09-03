#  Copyright 2024 Amazon.com, Inc. or its affiliates.

import logging
from typing import Dict, List, Tuple

from geojson import Feature

from ..common import ImageDimensions, ImageRegion, get_feature_image_bounds
from ..inference import FeatureSelector
from .tiling_strategy import TilingStrategy, generate_crops

logger = logging.getLogger(__name__)


class VariableTileTilingStrategy(TilingStrategy):
    def compute_regions(
        self,
        processing_bounds: ImageRegion,
        region_size: ImageDimensions,
        tile_size: ImageDimensions,
        overlap: ImageDimensions,
    ) -> List[ImageRegion]:
        """
        Identify the regions that should be created from this image.

        :param processing_bounds: the bounds of the full image or area of interest in pixels ((r, c), (w, h))
        :param region_size: the size of the regions in pixels (w, h)
        :param tile_size: the size of the tiles in pixels (w, y)
        :param overlap: the amount of overlap (w, h)

        :return: a collection of region boundaries
        """
        return generate_crops(processing_bounds, region_size, overlap, only_full_tiles=False)

    def compute_tiles(self, region: ImageRegion, tile_size: ImageDimensions, overlap: ImageDimensions) -> List[ImageRegion]:
        """
        Identify the tiles that should be created from this region.

        :param region: the bounds of the region in pixels ((r, c), (w, h))
        :param tile_size: the size of the tiles in pixels (w, h)
        :param overlap: the amount of overlap (w, h)

        :return: a collection of tile boundaries
        """
        return generate_crops(region, tile_size, overlap, only_full_tiles=False)

    def cleanup_duplicate_features(
        self,
        processing_bounds: ImageRegion,
        region_size: ImageDimensions,
        tile_size: ImageDimensions,
        overlap: ImageDimensions,
        features: List[Feature],
        feature_selector: FeatureSelector,
    ) -> List[Feature]:
        """
        This method handles cleaning up duplicates caused by tiling by applying the feature selector to any features
        that come from overlap regions.

        :param processing_bounds: the bounds of the full image or area of interest in pixels ((r, c), (w, h))
        :param region_size: the size of the regions in pixels (w, h)
        :param tile_size: the size of the tiles in pixels (w, y)
        :param overlap: the amount of overlap (w, h)
        :param features: the collection of features to deduplicate
        :param feature_selector: the algorithm that will be used to resolve duplicates

        :return: the collection of features with duplicates removed
        """
        logger.debug("FeatureSelection: Starting overlap-aware deduplication of features.")

        total_skipped = 0
        deduped_features = []
        features_grouped_by_region = self._group_features_by_overlap(features, region_size, overlap)
        for region_key, region_features in features_grouped_by_region.items():
            region_stride = (region_size[0] - overlap[0], region_size[1] - overlap[1])
            region_origin = (region_stride[0] * region_key[0], region_stride[1] * region_key[2])

            if region_key[0] != region_key[1] or region_key[2] != region_key[3]:
                # The Group contains contributions from multiple regions, run selection on the entire group
                deduped_features.extend(feature_selector.select_features(region_features))
            else:
                # Not an overlap between regions group these features using tile size to identify overlaps
                features_grouped_by_tile = self._group_features_by_overlap(
                    region_features, tile_size, overlap, region_origin
                )

                for tile_key, tile_features in features_grouped_by_tile.items():
                    if tile_key[0] != tile_key[1] or tile_key[2] != tile_key[3]:
                        # Group contains contributions from multiple tiles, run selection
                        deduped_features.extend(feature_selector.select_features(tile_features))
                    else:
                        # No overlap between tiles, features can be added directly to the result
                        total_skipped += len(tile_features)
                        deduped_features.extend(tile_features)

        logger.debug(
            f"FeatureSelection: Skipped processing of {total_skipped} of {len(features)} features. "
            "They were not inside an overlap region."
        )

        return deduped_features

    @staticmethod
    def _identify_overlap(
        feature: Feature, shape: Tuple[int, int], overlap: Tuple[int, int], origin: Tuple[int, int] = (0, 0)
    ) -> Tuple[int, int, int, int]:
        """
        Generate a tuple that contains the min and max indexes of adjacent tiles or regions for a given feature. If
        the min and max values for both x and y are the same then this feature does not touch an overlap region.

        :param feature: the geojson Feature that must contain properties to identify its location in an image
        :param shape: the width, height of the area in pixels
        :param overlap: the x, y overlap between areas in pixels
        :param origin: the x, y coordinate of the area in relation to the full image

        :return: a tuple: minx, maxx, miny, maxy that identifies any overlap.
        """
        bbox = get_feature_image_bounds(feature)

        # If an offset origin was supplied adjust the bbox so the key is relative to the origin.
        bbox = (bbox[0] - origin[0], bbox[1] - origin[1], bbox[2] - origin[0], bbox[3] - origin[1])

        stride_x = shape[0] - overlap[0]
        stride_y = shape[1] - overlap[1]

        max_x_index = int(bbox[2] / stride_x)
        max_y_index = int(bbox[3] / stride_y)

        min_x_index = int(bbox[0] / stride_x)
        min_y_index = int(bbox[1] / stride_y)
        min_x_offset = int(bbox[0]) % stride_x
        min_y_offset = int(bbox[1]) % stride_y

        if min_x_offset < overlap[0] and min_x_index > 0:
            min_x_index -= 1
        if min_y_offset < overlap[1] and min_y_index > 0:
            min_y_index -= 1

        return min_x_index, max_x_index, min_y_index, max_y_index

    @staticmethod
    def _group_features_by_overlap(
        features: List[Feature], shape: Tuple[int, int], overlap: Tuple[int, int], origin: Tuple[int, int] = (0, 0)
    ) -> Dict[Tuple[int, int, int, int], List[Feature]]:
        """
        Group all the feature items by tile id

        :param features: List[FeatureItem] = the list of feature items
        :param shape: the width, height of the area in pixels
        :param overlap: the x, y overlap between areas in pixels
        :param origin: the x, y coordinate of the area in relation to the full image

        :return: a mapping of overlap id to a list of features that intersect that overlap region
        """
        grouped_features: Dict[Tuple[int, int, int, int], List[Feature]] = {}
        for feature in features:
            overlap_key = VariableTileTilingStrategy._identify_overlap(feature, shape, overlap, origin)
            grouped_features.setdefault(overlap_key, []).append(feature)
        return grouped_features
