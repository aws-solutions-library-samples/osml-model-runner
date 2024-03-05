#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

from typing import Optional, Tuple

import geojson

from aws.osml.features import ImagedFeaturePropertyAccessor

property_accessor = ImagedFeaturePropertyAccessor()


def get_feature_image_bounds(feature: geojson.Feature) -> Optional[Tuple[float, float, float, float]]:
    """
    This is a utility function that creates a [minx, miny, maxx, maxy] tuple for the boundary of the
    image geometry of a feature. If no image geometry property can be found None is returned.

    :param feature: the feature to calculate bounds for
    :return: the bounds or None
    """
    image_geometry = property_accessor.find_image_geometry(feature)
    if not image_geometry:
        return None
    return image_geometry.bounds
