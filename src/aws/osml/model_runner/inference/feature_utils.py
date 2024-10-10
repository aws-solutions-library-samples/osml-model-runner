#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import functools
import json
import logging
import math
from datetime import datetime
from io import BufferedReader
from json import dumps
from math import degrees, radians
from secrets import token_hex
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import shapely
from geojson import Feature, FeatureCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, loads
from osgeo import gdal
from shapely.geometry.base import BaseGeometry

from aws.osml.model_runner.common import GeojsonDetectionField, ImageDimensions
from aws.osml.photogrammetry import GeodeticWorldCoordinate, ImageCoordinate, SensorModel

from .exceptions import InvalidFeaturePropertiesException

logger = logging.getLogger(__name__)


def features_to_image_shapes(sensor_model: SensorModel, features: List[Feature]) -> List[BaseGeometry]:
    """
    Convert geojson objects/shapes to shapely shapes

    :param sensor_model: SensorModel = the model to use for the transform
    :param features: List[geojson.Features] = the features to convert

    :return: List[BaseGeometry] = a list of shapely shapes
    """
    shapes: List[BaseGeometry] = []
    if not features:
        return shapes
    for feature in features:
        if "geometry" not in feature:
            raise ValueError("Feature does not contain a valid geometry")

        feature_geometry = feature["geometry"]

        image_coords = convert_nested_coordinate_lists(feature_geometry["coordinates"], sensor_model.world_to_image)

        feature_geometry["coordinates"] = image_coords

        if isinstance(feature_geometry, Point):
            shapes.append(shapely.geometry.Point(image_coords))
        elif isinstance(feature_geometry, LineString):
            shapes.append(shapely.geometry.LineString(image_coords))
        elif isinstance(feature_geometry, Polygon):
            shapes.append(shapely.geometry.shape(feature_geometry))
        elif isinstance(feature_geometry, MultiPoint):
            shapes.append(shapely.geometry.MultiPoint(image_coords))
        elif isinstance(feature_geometry, MultiLineString):
            shapes.append(shapely.geometry.MultiLineString(image_coords))
        elif isinstance(feature_geometry, MultiPolygon):
            shapes.append(shapely.geometry.shape(feature_geometry))
        else:
            # Unlikely to get here as we're handling all valid geojson types but if the spec
            # ever changes or if a consumer passes in a custom dictionary that isn't valid
            # we want to handle it gracefully
            raise ValueError("Unable to convert feature due to unrecognized or invalid geometry")

    return shapes


def convert_nested_coordinate_lists(coordinates_or_lists: List, conversion_function: Callable) -> Union[Tuple, List]:
    """
    Convert a nested list of coordinates to 3D world GIS coordinates

    :param coordinates_or_lists: List = a coordinate or list of coordinates to transform
    :param conversion_function: Callable = the function to use for the GIS transform

    :return: Union[Tuple, List] = the transformed list of coordinates
    """
    if not isinstance(coordinates_or_lists[0], List):
        # This appears to be a single coordinate so run it through the supplied conversion
        # function (i.e. world_to_image). Ensure that the coordinate has an elevation and convert
        # the longitude, latitude to radians to meet the expectations of the sensor model.
        world_coordinate_3d = [radians(coordinates_or_lists[0]), radians(coordinates_or_lists[1])]
        if len(coordinates_or_lists) == 2:
            world_coordinate_3d.append(0.0)
        else:
            world_coordinate_3d.append(coordinates_or_lists[2])
        image_coordinate = conversion_function(GeodeticWorldCoordinate(world_coordinate_3d))
        return tuple(list(image_coordinate.coordinate))
    else:
        # This appears to be a list of lists (i.e. a LineString, Polygon, etc.) so invoke this
        # conversion routine recursively to preserve the nesting structure of the input
        output_list = []
        for coordinate_list in coordinates_or_lists:
            output_list.append(convert_nested_coordinate_lists(coordinate_list, conversion_function))
        return output_list


def create_mock_feature_collection(payload: BufferedReader, geom=False) -> FeatureCollection:
    """
    This function allows us to emulate what we would expect a model to return to MR, a geojson formatted
    FeatureCollection. This allows us to bypass using a real model if the NOOP_MODEL_NAME is given as the
    model name in the image request. This is the same logic used by our current default dummy model to select
    detection points in our pipeline.

    :param payload: BufferedReader = object that holds the data that will be  sent to the feature generator
    :param geom: Bool = whether or not to return the geom_imcoords field in the geojson
    :return: FeatureCollection = feature collection containing the center point of a tile given as a detection point
    """
    logging.debug("Creating a fake feature collection to use for testing ModelRunner!")

    # Use GDAL to open the image. The binary payload from the HTTP request is used to create an in-memory
    # virtual file system for GDAL which is then opened to decode the image into a dataset which will give us
    # access to a NumPy array for the pixels.
    temp_ds_name = "/vsimem/" + token_hex(16)
    gdal.FileFromMemBuffer(temp_ds_name, payload.read())
    ds = gdal.Open(temp_ds_name)
    height, width = ds.RasterYSize, ds.RasterXSize
    logging.debug(f"Processing image of size: {width}x{height}")

    # Create a single detection bbox that is at the center of and sized proportionally to the image

    center_xy = width / 2, height / 2
    fixed_object_size_xy = width * 0.1, height * 0.1
    fixed_object_bbox = [
        center_xy[0] - fixed_object_size_xy[0],
        center_xy[1] - fixed_object_size_xy[1],
        center_xy[0] + fixed_object_size_xy[0],
        center_xy[1] + fixed_object_size_xy[1],
    ]

    fixed_object_polygon = [
        (center_xy[0] - fixed_object_size_xy[0], center_xy[1] - fixed_object_size_xy[1]),
        (center_xy[0] - fixed_object_size_xy[0], center_xy[1] + fixed_object_size_xy[1]),
        (center_xy[0] + fixed_object_size_xy[0], center_xy[1] + fixed_object_size_xy[1]),
        (center_xy[0] + fixed_object_size_xy[0], center_xy[1] - fixed_object_size_xy[1]),
    ]

    # Convert that bbox detection into a sample GeoJSON formatted detection. Note that the world coordinates
    # are not normally provided by the model container, so they're defaulted to 0,0 here since GeoJSON features
    # require a geometry.
    json_results = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"coordinates": [0.0, 0.0], "type": "Point"},
                "id": token_hex(16),
                "properties": {
                    "detection_score": 1.0,
                    "feature_types": {"sample_object": 1.0},
                    "image_id": token_hex(16),
                },
            }
        ],
    }
    if geom is True:
        json_results["features"][0]["properties"]["geom_imcoords"] = fixed_object_polygon
    else:
        json_results["features"][0]["properties"]["bounds_imcoords"] = fixed_object_bbox
    return loads(dumps(json_results))


def calculate_processing_bounds(
    ds: gdal.Dataset, roi: Optional[BaseGeometry], sensor_model: Optional[SensorModel]
) -> Optional[Tuple[ImageDimensions, ImageDimensions]]:
    """
    An area of interest converter

    :param ds: gdal.Dataset = GDAL dataset
    :param roi: Optional[BaseGeometry] = ROI shape
    :param sensor_model: Optional[SensorModel] = Sensor model to use for transformations

    :return: Optional[Tuple[ImageDimensions, ImageDimensions]] = Image dimensions associated with the ROI request
    """
    processing_bounds: Optional[Tuple[ImageDimensions, ImageDimensions]] = (
        (0, 0),
        (ds.RasterXSize, ds.RasterYSize),
    )
    if roi is not None and sensor_model is not None:
        full_image_area = shapely.geometry.Polygon(
            [(0, 0), (0, ds.RasterYSize), (ds.RasterXSize, ds.RasterYSize), (ds.RasterXSize, 0)]
        )

        # This is making the assumption that the ROI is a shapely Polygon, and it only considers
        # the exterior boundary (i.e. we don't handle cases where the WKT for the ROI has holes).
        # It also assumes that the coordinates of the WKT string are in longitude latitude order
        # to match GeoJSON
        world_coordinates_3d = []
        list_coordinates = shapely.geometry.mapping(roi)["coordinates"][0]
        for coord in list_coordinates:
            if len(coord) == 3:
                world_coordinates_3d.append(coord)
            else:
                world_coordinates_3d.append(coord + (0.0,))
        roi_area = features_to_image_shapes(
            sensor_model,
            [Feature(geometry=Polygon([tuple(world_coordinates_3d)]))],
        )[0]

        if roi_area.intersects(full_image_area):
            area_to_process = roi_area.intersection(full_image_area)

            # Shapely bounds are (minx, miny, maxx, maxy); convert this to the ((r, c), (w, h))
            # expected by the tiler
            processing_bounds = (
                (round(area_to_process.bounds[1]), round(area_to_process.bounds[0])),
                (
                    round(area_to_process.bounds[2] - area_to_process.bounds[0]),
                    round(area_to_process.bounds[3] - area_to_process.bounds[1]),
                ),
            )
        else:
            processing_bounds = None

    return processing_bounds


def get_source_property(image_location: str, image_extension: str, dataset: gdal.Dataset) -> Optional[Dict]:
    """
    Get the source property from NITF image

    :param image_location: the location of the source image
    :param image_extension: the file extension type of the source image
    :param dataset: the GDAL dataset to probe for source data

    :return: the source dictionary property to attach to features
    """
    # Currently we only support deriving source metadata from NITF images
    if image_extension == "NITF":
        try:
            metadata = dataset.GetMetadata()
            # Extract metadata headers from NITF
            data_type = metadata.get("NITF_ICAT", None)
            source_id = metadata.get("NITF_FTITLE", None)
            # Format of datetime string follows 14 digit spec in MIL-STD-2500C for NITFs
            source_dt = (
                datetime.strptime(metadata.get("NITF_IDATIM"), "%Y%m%d%H%M%S").isoformat(timespec="seconds") + "Z"
                if metadata.get("NITF_IDATIM")
                else None
            )

            # Build a source property for features
            source_property = {
                "sourceMetadata": [
                    {
                        "location": image_location,
                        "format": "NITF",
                        "category": data_type,
                        "sourceId": source_id,
                        "sourceDT": source_dt,
                    }
                ]
            }

            return source_property
        except Exception as err:
            logging.warning(f"Source metadata not available for {image_extension} image extension! {err}")
            return None
    else:
        logging.warning(f"Source metadata not available for {image_extension} image extension!")
        return None


def get_extents(ds: gdal.Dataset, sm: SensorModel) -> Dict[str, Any]:
    """
    Returns the geographic extents of the given GDAL dataset.

    :param ds: GDAL dataset.
    :param sm: OSML Sensor Model imputed for dataset
    :return: Dictionary with keys 'north', 'south', 'east', 'west' representing the extents.
    """
    # Compute WGS-84 world coordinates for each image corners to impute the extents for visualizations
    image_corners = [[0, 0], [ds.RasterXSize, 0], [ds.RasterXSize, ds.RasterYSize], [0, ds.RasterYSize]]
    geo_image_corners = [sm.image_to_world(ImageCoordinate(corner)) for corner in image_corners]
    locations = [(degrees(p.latitude), degrees(p.longitude)) for p in geo_image_corners]
    feature_bounds = functools.reduce(
        lambda prev, f: [
            min(f[0], prev[0]),
            min(f[1], prev[1]),
            max(f[0], prev[2]),
            max(f[1], prev[3]),
        ],
        locations,
        [math.inf, math.inf, -math.inf, -math.inf],
    )

    return {
        "north": feature_bounds[2],
        "south": feature_bounds[0],
        "east": feature_bounds[3],
        "west": feature_bounds[1],
    }


def add_properties_to_features(job_id: str, feature_properties: str, features: List[Feature]) -> List[Feature]:
    """
    Add arbitrary and controlled property dictionaries to geojson feature properties
    :param job_id: str = unique identifier for the job
    :param feature_properties: str = additional feature properties or metadata from the image processing
    :param features: List[geojson.Feature] = the list of features to update

    :return: List[geojson.Feature] = updated list of features
    """
    try:
        feature_properties: List[dict] = json.loads(feature_properties)
        for feature in features:
            # Update the features with their inference metadata
            feature["properties"].update(get_inference_metadata_property(job_id, feature["properties"]["inferenceTime"]))

            # For the custom provided feature properties, update
            for feature_property in feature_properties:
                feature["properties"].update(feature_property)

            # Remove unneeded feature properties if they are present
            if feature.get("properties", {}).get("inferenceTime"):
                del feature["properties"]["inferenceTime"]
            if feature.get("properties", {}).get(GeojsonDetectionField.BOUNDS):
                del feature["properties"][GeojsonDetectionField.BOUNDS]
            if feature.get("properties", {}).get(GeojsonDetectionField.GEOM):
                del feature["properties"][GeojsonDetectionField.GEOM]
            if feature.get("properties", {}).get("detection_score"):
                del feature["properties"]["detection_score"]
            if feature.get("properties", {}).get("feature_types"):
                del feature["properties"]["feature_types"]
            if feature.get("properties", {}).get("image_id"):
                del feature["properties"]["image_id"]
            if feature.get("properties", {}).get("adjusted_feature_types"):
                del feature["properties"]["adjusted_feature_types"]

    except Exception as err:
        logging.exception(err)
        raise InvalidFeaturePropertiesException("Could not apply custom properties to features!")
    return features


def get_inference_metadata_property(job_id: str, inference_time: str) -> Dict[str, Any]:
    """
    Create an inference dictionary property to append to geojson features

    :param job_id: str = unique identifier for the job
    :param inference_time: str = the time the inference was made in epoch millisec

    :return: Dict[str, Any] = an inference metadata dictionary property to attach to features
    """
    inference_metadata_property = {
        "inferenceMetadata": {
            "jobId": job_id,
            "inferenceDT": inference_time,
        }
    }
    return inference_metadata_property
