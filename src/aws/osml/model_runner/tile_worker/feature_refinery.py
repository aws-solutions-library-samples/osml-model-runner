#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import functools
import math
from typing import List, Optional, Tuple

import geojson
import numpy as np
from scipy.interpolate import RectBivariateSpline
from shapely.geometry import Polygon

from aws.osml.model_runner.common import GeojsonDetectionField
from aws.osml.photogrammetry import (
    CompositeSensorModel,
    ElevationModel,
    GeodeticWorldCoordinate,
    ImageCoordinate,
    SensorModel,
)


class LocationGridInterpolator:
    """
    This class can be used to approximate geodetic world coordinates from a grid of correspondences that is
    computed over a given area.
    """

    def __init__(
        self,
        sensor_model: SensorModel,
        elevation_model: Optional[ElevationModel],
        grid_area_ulx: float,
        grid_area_uly: float,
        grid_area_width: float,
        grid_area_height: float,
        grid_resolution: int,
    ) -> None:
        """
        Construct the grid of correspondences of the requested size/resolution from using the sensor model provided.

        :param sensor_model: SensorModel = the sensor model for the image
        :param elevation_model: Optional[Elevationmodel] = an optional external elevation model
        :param grid_area_ulx: float = the x component of the upper left corner of the grid in pixel space
        :param grid_area_uly: float = the y component of the upper left corner of the grid in pixel space
        :param grid_area_width: float = the width of the grid in pixels
        :param grid_area_height: float = the height of the grid in pixels
        :param grid_resolution: int = the number of points to calculate across the grid. Total points will be resolution^^2

        :return: None
        """
        xs = np.linspace(grid_area_ulx, grid_area_ulx + grid_area_width, grid_resolution)
        ys = np.linspace(grid_area_uly, grid_area_uly + grid_area_height, grid_resolution)
        longitude_values = np.empty(len(xs) * len(ys))
        latitude_values = np.empty(len(longitude_values))
        i = 0
        for y in ys:
            for x in xs:
                world_coordinate = sensor_model.image_to_world(ImageCoordinate([x, y]), elevation_model=elevation_model)
                longitude_values[i] = world_coordinate.longitude
                latitude_values[i] = world_coordinate.latitude
                i += 1
        xgrid, ygrid = np.meshgrid(xs, ys)
        longitude_values.shape = xgrid.shape
        latitude_values.shape = xgrid.shape

        self.longitude_interpolator = RectBivariateSpline(xs, ys, longitude_values, kx=1, ky=1)
        self.latitude_interpolator = RectBivariateSpline(xs, ys, latitude_values, kx=1, ky=1)
        self.elevation_model = elevation_model

    def __call__(self, *args, **kwargs):
        """
        Call this interpolation function given an image coordinate array.

        :param args: a single argument for the coordinate array
        :param kwargs: not used
        :return: a GeodeticWorldCoordinate for that image location
        """
        image_coord = args[0]
        world_coord = [
            self.longitude_interpolator(image_coord[0], image_coord[1])[0][0],
            self.latitude_interpolator(image_coord[0], image_coord[1])[0][0],
            0.0,
        ]
        world_coordinate = GeodeticWorldCoordinate(world_coord)
        if self.elevation_model is not None:
            self.elevation_model.set_elevation(world_coordinate)
        return world_coordinate


class FeatureRefinery:
    """
    A FeatureRefinery is a class that improves and updates the features returned by object detectors. Its primary
    purpose is to assign geographic coordinates for the features, but it can be expanded over time to make other
    adjustments (e.g. assign classification metadata).
    """

    def __init__(
        self,
        sensor_model: SensorModel,
        elevation_model: Optional[ElevationModel] = None,
        approximation_grid_size: int = 11,
    ) -> None:
        """
        Construct a refinery given the context objects necessary for geolocating features.

        :param sensor_model: SensorModel = sensor model for the image
        :param elevation_model: Optional[ElevationModel] = external elevation model
        :param approximation_grid_size: int = resolution of the approximation grid to use

        :return: None
        """
        self.sensor_model = sensor_model
        self.elevation_model = elevation_model
        self.approximation_grid_size = approximation_grid_size

    def refine_features_for_tile(self, features: List[geojson.Feature]) -> None:
        """
        Update the features to contain additional information from the context provided.

        :param features: List[geojson.Feature] = the input features to refine
        :return: None, the features are updated in place
        """
        if len(features) < self.approximation_grid_size * self.approximation_grid_size:
            # This is a sparsely populated tile, the cost of computing the grid is likely larger than
            # calculating locations for the detections directly. In this case we geolocate the individual
            # features one by one
            self._geolocate_individual_features(features)
        else:
            # Create an approximation grid and then compute geolocation
            self._geolocate_features_using_approximation_grid(features)

    def _geolocate_individual_features(self, features: List[geojson.Feature]) -> None:
        """
        This method computes geolocations directly for each feature. It is useful for sparse feature sets where the
        cost of computing the precise locations for an approximation grid is higher than geolocating the individual
        features.

        :param features: List[geojson.Feature] = the input features
        :return: None, but the individual features have their geometry property updated
        """
        for feature in features:
            # Calculate the center of the detection and convert it to a geodetic coordinate
            initial_sensor_model = self.sensor_model
            if isinstance(self.sensor_model, CompositeSensorModel):
                initial_sensor_model = self.sensor_model.approximate_sensor_model

            if GeojsonDetectionField.GEOM in feature["properties"]:
                mask = Polygon(feature["properties"][GeojsonDetectionField.GEOM])
                center_xy = (mask.centroid.x, mask.centroid.y)
                image_coords = mask.exterior.coords
            elif GeojsonDetectionField.BOUNDS in feature["properties"]:
                bbox = feature["properties"][GeojsonDetectionField.BOUNDS]
                center_xy = [
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0,
                ]
                # Calculate image coordinates and update feature
                image_coords = self.imcoords_bbox_to_polygon(bbox)

            approximate_center_location = initial_sensor_model.image_to_world(
                ImageCoordinate(center_xy), elevation_model=self.elevation_model
            )

            # Calculate the geodetic coordinates of the bounding box
            polygon_image_coords = [
                initial_sensor_model.image_to_world(ImageCoordinate(image_coord), elevation_model=self.elevation_model)
                for image_coord in image_coords
            ]

            if not isinstance(self.sensor_model, CompositeSensorModel):
                final_center_location = approximate_center_location
            else:
                # Convert the detection center to a geodetic coordinate using the precision sensor model and then
                # calculate a delta between the approximate result and the precision result.
                final_center_location = self.sensor_model.image_to_world(
                    ImageCoordinate(center_xy),
                    elevation_model=self.elevation_model,
                    options={
                        "initial_guess": [
                            approximate_center_location.longitude,
                            approximate_center_location.latitude,
                        ],
                        "initial_search_distance": math.radians(0.005),
                    },
                )
                delta_longitude = final_center_location.longitude - approximate_center_location.longitude
                delta_latitude = final_center_location.latitude - approximate_center_location.latitude

                # Move all the boundary coordinates by the delta. This assumes that the boundary coordinates are
                # relatively close to the center and that they will exhibit the same shift that we saw between the
                # approximate center and the precision center
                for image_coord in polygon_image_coords:
                    image_coord.longitude += delta_longitude
                    image_coord.latitude += delta_latitude

            # Convert the geodetic coordinates to the [degrees, degrees, meters] array that is expected by GeoJSON.
            # The first coordinate is added to the end of the list to close the bounding box as required by some
            # visualization tools.
            polygon_coords = [
                FeatureRefinery.radians_coordinate_to_degrees(image_coord) for image_coord in polygon_image_coords
            ]
            polygon_coords.append(polygon_coords[0])

            # Note that for geojson polygons the "coordinates" member must be an array of
            # LinearRing coordinate arrays. For Polygons with multiple rings, the first must be
            # the exterior ring and any others must be interior rings or holes. We only have an
            # exterior ring hence creating an array of the coordinates is appropriate here.
            feature["geometry"] = geojson.Polygon([tuple(polygon_coords)])

            self.compute_center_lat_long(feature, polygon_coords, final_center_location)

    def _geolocate_features_using_approximation_grid(self, features: List[geojson.Feature]) -> None:
        """
        This method computes geolocations for features using an approximation grid. It is useful for dense feature
        sets where the cost of computing precise locations for a set of close features is expensive and unnecessary.
        Here we first compute a grid of locations covering the features in question that is then used to efficiently
        assign geolocations for all the features.

        :param features: List[geojson.Feature] = the input features
        :return: None, but the individual features have their geometry property updated
        """

        # Compute the boundary of these features. Normally this will be similar to the tile boundary but the
        # interpolation needs to be setup to cover the entire extent, so we calculate it explicitly here. If the
        # features happen to be very tightly packed and only occupy a small portion of the tile we will gain some
        # benefit by creating the same resolution of approximation grid over the smaller area.
        feature_bounds = functools.reduce(
            lambda prev, f: [
                min(f["properties"][GeojsonDetectionField.BOUNDS][0], prev[0]),
                min(f["properties"][GeojsonDetectionField.BOUNDS][1], prev[1]),
                max(f["properties"][GeojsonDetectionField.BOUNDS][2], prev[2]),
                max(f["properties"][GeojsonDetectionField.BOUNDS][3], prev[3]),
            ],
            features,
            [math.inf, math.inf, -math.inf, -math.inf],
        )

        # Use the feature boundary to set up an approximation grid for the region"bounds_imcoords"
        grid_area_ulx = feature_bounds[0]
        grid_area_uly = feature_bounds[1]
        grid_area_width = feature_bounds[2] - feature_bounds[0]
        grid_area_height = feature_bounds[3] - feature_bounds[1]
        tile_interpolation_grid = LocationGridInterpolator(
            self.sensor_model,
            self.elevation_model,
            grid_area_ulx,
            grid_area_uly,
            grid_area_width,
            grid_area_height,
            self.approximation_grid_size,
        )

        for feature in features:
            # Calculate the center of the detection and convert it to a geodetic coordinate using the
            # approximation grid.
            if GeojsonDetectionField.GEOM in feature["properties"]:
                mask = Polygon(feature["properties"][GeojsonDetectionField.GEOM])
                center_xy = (mask.centroid.x, mask.centroid.y)
                image_coords = mask.exterior.coords
            elif GeojsonDetectionField.BOUNDS in feature["properties"]:
                bbox = feature["properties"][GeojsonDetectionField.BOUNDS]
                center_xy = [
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0,
                ]
                # Calculate image coordinates and update feature
                image_coords = self.imcoords_bbox_to_polygon(bbox)

            center_location = tile_interpolation_grid(center_xy)

            # Calculate the geodetic coordinates of the bounding box using the approximation grid. This also
            # converts the resulting GeodeticWorldCoordinate to he [degrees, degrees, meters] array that is
            # expected by GeoJSON. The first coordinate is added to the end of the list to close the bounding
            # box as required by some visualization tools.
            polygon_coords = [
                FeatureRefinery.radians_coordinate_to_degrees(tile_interpolation_grid(image_coord))
                for image_coord in image_coords
            ]
            polygon_coords.append(polygon_coords[0])

            # Note that for geojson polygons the "coordinates" member must be an array of
            # LinearRing coordinate arrays. For Polygons with multiple rings, the first must be
            # the exterior ring and any others must be interior rings or holes. We only have an
            # exterior ring hence creating an array of the coordinates is appropriate here.
            feature["geometry"] = geojson.Polygon([tuple(polygon_coords)])

            self.compute_center_lat_long(feature, polygon_coords, center_location)

    @staticmethod
    def imcoords_bbox_to_polygon(bbox: List[float]) -> List[List[float]]:
        """
        Converts a bbox of image coordinates into a four point polygon of coordinates.

        :param bbox: List[float] = list of pixel coordinates bounding box: [a, b, c ,d]
        :return: List[List[float]] = list of pixel coordinates as a 4 point polygon: [[a,b],[a,d],[c,d],[c,b]]
        """
        return [
            [bbox[0], bbox[1]],
            [bbox[0], bbox[3]],
            [bbox[2], bbox[3]],
            [bbox[2], bbox[1]],
        ]

    @staticmethod
    def feature_property_transformation(feature: geojson.Feature) -> None:
        """
        Mutate a feature list property blob inplace into the bellow structure:

        "properties": {
            "detection": {
                "type": "Polygon",
                "pixelCoordinates": [
                    [0, 0],
                    [0, 10],
                    [10, 10],
                    [10, 0]
                ],
                "ontology": [
                    {
                    "iri": "https://myOntology.com/object/001",
                    "detectionScore": 0.42
                    }
                ]
            }
        }

        :param feature: geojson.Feature = the feature that needs it's "properties" blob conformed.

        :return: None
        """

        # Create an ontology based on the models returned feature_types
        ontology = []

        for feature_type in feature["properties"]["feature_types"]:
            ontology.append(
                {
                    "iri": feature_type,
                    "detectionScore": feature["properties"]["feature_types"][feature_type],
                }
            )

        if GeojsonDetectionField.GEOM in feature["properties"]:
            pixel_coordinates = feature["properties"][GeojsonDetectionField.GEOM]

        elif GeojsonDetectionField.BOUNDS in feature["properties"]:
            # Grab the input coordinate boundaries from the model output
            bbox = feature["properties"][GeojsonDetectionField.BOUNDS]
            pixel_coordinates = FeatureRefinery.imcoords_bbox_to_polygon(bbox)

        # Create detection property for feature output
        feature["properties"]["detection"] = {
            "type": "Polygon",
            "pixelCoordinates": pixel_coordinates,
            "ontology": ontology,
        }

    @staticmethod
    def radians_coordinate_to_degrees(
        coordinate: GeodeticWorldCoordinate,
    ) -> Tuple[float, float, float]:
        """
        GeoJSON coordinate order is a decimal longitude, latitude with an optional height as a 3rd value
        (i.e. [lon, lat, ht]). The WorldCoordinate uses the same ordering but the longitude and latitude are expressed
        in radians rather than degrees.

        :param coordinate: GeodeticWorldCoordinate = the geodetic world coordinate (longitude, latitude, elevation)

        :return: Tuple[float, float, float] = degrees(longitude), degrees(latitude), elevation
        """
        return (
            math.degrees(coordinate.longitude),
            math.degrees(coordinate.latitude),
            coordinate.elevation,
        )

    @staticmethod
    def compute_center_lat_long(
        feature: geojson.Feature,
        polygon_coords: List[tuple],
        final_center_location: GeodeticWorldCoordinate,
    ) -> None:
        """
        Geojson features can optionally have a bounding box that contains [min lon, min lat,
        max lon, max lat]. This code computes that bounding box from the polygon boundary
        coordinates

        :param feature: geojson.Feature = the feature that needs it's "properties" blob conformed.
        :param polygon_coords: List[tuple] = list of polygon coordinates
        :param final_center_location: GeodeticWorldCoordinate = Geodetic coordinate using the precision sensor model
                                    and then calculated using a delta between approximate result and the precision result

        :return None
        """
        feature["bbox"] = functools.reduce(
            lambda prev, coord: [
                min(coord[0], prev[0]),
                min(coord[1], prev[1]),
                max(coord[0], prev[2]),
                max(coord[1], prev[3]),
            ],
            polygon_coords,
            [math.inf, math.inf, -math.inf, -math.inf],
        )

        # Adding these because some visualization tools (e.g. kepler.gl) can perform more
        # advanced rendering (e.g. cluster layers) if the data points have single coordinates.
        feature["properties"]["center_longitude"] = math.degrees(final_center_location.longitude)
        feature["properties"]["center_latitude"] = math.degrees(final_center_location.latitude)
