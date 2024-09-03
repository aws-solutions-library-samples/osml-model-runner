#  Copyright 2024 Amazon.com, Inc. or its affiliates.

from abc import ABC, abstractmethod
from typing import List

from geojson import Feature

from ..common import ImageDimensions, ImageRegion
from ..inference import FeatureSelector


class TilingStrategy(ABC):
    """
    TilingStrategy defines an abstract interface for the code that determines how a large image is broken down into
    regions and tiles.
    """

    @abstractmethod
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

    @abstractmethod
    def compute_tiles(self, region: ImageRegion, tile_size: ImageDimensions, overlap: ImageDimensions) -> List[ImageRegion]:
        """
        Identify the tiles that should be created from this region.

        :param region: the bounds of the region in pixels ((r, c), (w, h))
        :param tile_size: the size of the tiles in pixels (w, h)
        :param overlap: the amount of overlap (w, h)

        :return: a collection of tile boundaries
        """

    @abstractmethod
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


def generate_crops(
    region: ImageRegion, chip_size: ImageDimensions, overlap: ImageDimensions, only_full_tiles: bool = False
) -> List[ImageRegion]:
    """
    Yields a list of overlapping chip bounding boxes for the given region. Chips will start
    in the upper left corner of the region (i.e. region[0][0], region[0][1]) and will be spaced
    such that they have the specified horizontal and vertical overlap.

    :param region: a tuple for the bounding box of the region ((ul_r, ul_c), (width, height))
    :param chip_size: a tuple for the chip dimensions (width, height)
    :param overlap: a tuple for the overlap (width, height)
    :param only_full_tiles: true if we only want to generate tiles that match the chip_size

    :return: an iterable list of tuples for the chip bounding boxes [((ul_r, ul_c), (w, h)), ...]
    """
    if overlap[0] >= chip_size[0] or overlap[1] >= chip_size[1]:
        raise ValueError("Overlap must be less than chip size! chip_size = " + str(chip_size) + " overlap = " + str(overlap))

    # Calculate the spacing for the chips taking into account the horizontal and vertical overlap
    # and how many are needed to cover the region
    stride_x = chip_size[0] - overlap[0]
    stride_y = chip_size[1] - overlap[1]
    num_x = ceildiv(region[1][0], stride_x)
    num_y = ceildiv(region[1][1], stride_y)

    crops = []
    for r in range(0, num_y):
        for c in range(0, num_x):
            # Calculate the bounds of the chip ensuring that the chip does not extend
            # beyond the edge of the requested region
            ul_x = region[0][1] + c * stride_x
            ul_y = region[0][0] + r * stride_y
            w = min(chip_size[0], (region[0][1] + region[1][0]) - ul_x)
            h = min(chip_size[1], (region[0][0] + region[1][1]) - ul_y)
            if only_full_tiles:
                if w == chip_size[0] and h == chip_size[1]:
                    crops.append(((ul_y, ul_x), (w, h)))
            elif w > overlap[0] and h > overlap[1]:
                crops.append(((ul_y, ul_x), (w, h)))

    return crops


def ceildiv(a: int, b: int) -> int:
    """
    Integer ceiling division

    :param a: numerator
    :param b: denominator

    :return: ceil(a/b)
    """
    return -(-a // b)
