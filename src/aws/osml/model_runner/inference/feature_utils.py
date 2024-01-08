#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
from datetime import datetime
from io import BufferedReader
from json import dumps
from math import radians
from secrets import token_hex
from typing import Callable, Dict, List, Optional, Tuple, Union

import shapely
from geojson import Feature, FeatureCollection, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, loads
from osgeo import gdal
from shapely.geometry.base import BaseGeometry

from aws.osml.model_runner.common import Classification, ImageDimensions, get_image_classification
from aws.osml.photogrammetry import GeodeticWorldCoordinate, SensorModel


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


def get_source_property(image_extension: str, dataset: gdal.Dataset) -> Optional[Dict]:
    """
    Get the source property from NITF image

    :param image_extension: str = the file extension type of the source image
    :param dataset: gdal.Dataset = the GDAL dataset to probe for source data

    :return: Optional[Dict] = the source dictionary property to attach to features
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
                datetime.strptime(metadata.get("NITF_IDATIM"), "%Y%m%d%H%M%S").isoformat()
                if metadata.get("NITF_IDATIM")
                else None
            )
            # Determine the image classification from the metadata
            source_classification = get_image_classification(dataset)
            source_classification_str = (
                source_classification.classification if isinstance(source_classification, Classification) else None
            )

            # Build a source property for features
            source_property = {
                "source": [
                    {
                        "fileType": "NITF",
                        "info": {
                            "imageCategory": data_type,
                            "metadata": {
                                "sourceId": source_id,
                                "sourceDt": source_dt,
                                "classification": source_classification_str,
                            },
                        },
                    }
                ]
            }

            return source_property
        except Exception as err:
            logging.warning("Source metadata not available for {} image extension! {}".format(image_extension, err))
            return None
    else:
        logging.warning("Source metadata not available for {} image extension!".format(image_extension))
        return None
