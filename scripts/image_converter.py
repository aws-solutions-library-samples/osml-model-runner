#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import os
from argparse import ArgumentParser
from glob import glob
from sys import exit
from typing import List

from osgeo.gdal import Dataset, Open, Translate, TranslateOptions


def convert_format(ds: Dataset, output_image_path: str, output_format: str) -> int:
    options = TranslateOptions(format=output_format)
    try:
        ds = Translate(output_image_path, ds, options=options)
        if ds:
            logging.info("Successfully converted image to {}.".format(output_image_path))
            return 1
        else:
            logging.error("Failed to convert to {}.".format(output_image_path))
            return 0
    except Exception as err:
        logging.error("Failed to convert to {}. {}: {}".format(output_image_path, type(err).__name__, err))
        return 0


def main(input_directory: str, output_formats: List[str], output_directory: str = None, input_extension: str = None) -> int:
    search_dir = input_directory if input_extension is None else f"{input_directory}/*.{input_extension}"
    success_count = 0
    input_files = glob(search_dir)
    for path in input_files:
        dir_file = os.path.split(path)
        directory = dir_file[0]
        file = dir_file[1]
        name_ext = os.path.splitext(file)
        filename = name_ext[0]
        ds = Open(path)
        for image_format in output_formats:
            output_dir = os.path.join(directory, image_format.lower()) if output_directory is None else output_directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{filename}.{image_format.lower()}")
            success = convert_format(ds, output_path, image_format)
            success_count += success
    if success_count / len(output_formats) == len(input_files):
        logging.info("Successfully converted all images.")
        return 0
    else:
        logging.error(
            "Failed to convert some images. (Converted {}/{}).".format(
                int(success_count / len(output_formats)), len(input_files)
            )
        )
        return 1


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ", level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("-i", "--input_directory", help="<Required> Input file directory", required=True)
    parser.add_argument("-o", "--output_directory", help="<Optional> Output file directory")
    parser.add_argument(
        "-of", "--output_formats", nargs="+", help="<Required> Space delimited list of output formats", required=True
    )
    parser.add_argument("-if", "--input_format", help="<Optional> Input file extension filter")
    args = parser.parse_args()

    logging.info("Running image conversion with options: {}.".format(vars(args)))

    exit(
        main(
            input_directory=args.input_directory,
            output_formats=args.output_formats,
            output_directory=args.output_directory,
            input_extension=args.input_format,
        )
    )
