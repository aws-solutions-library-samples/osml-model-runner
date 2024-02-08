#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import argparse
import logging
import os
import signal
import sys
from types import FrameType
from typing import Optional

from codeguru_profiler_agent import Profiler
from pythonjsonlogger import jsonlogger

from aws.osml.model_runner.app import ModelRunner
from aws.osml.model_runner.common import ThreadingLocalContextFilter


def handler_stop_signals(signal_num: int, frame: Optional[FrameType], model_runner: ModelRunner) -> None:
    model_runner.stop()


def configure_logging(verbose: bool) -> None:
    """
    This function configures the Python logging module to use a JSON formatter with and thread local context
    variables.

    :param verbose: if true the logging level will be set to DEBUG, otherwise it will be set to INFO.
    """
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG

    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    ch.addFilter(ThreadingLocalContextFilter(["job_id", "image_id"]))
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(job_id)s %(image_id)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    ch.setFormatter(formatter)

    root_logger.addHandler(ch)


def map_signals(model_runner: ModelRunner) -> None:
    signal.signal(signal.SIGINT, lambda signum, frame: handler_stop_signals(signum, frame, model_runner))
    signal.signal(signal.SIGTERM, lambda signum, frame: handler_stop_signals(signum, frame, model_runner))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def setup_code_profiling() -> None:
    codeguru_profiling_group = os.environ.get("CODEGURU_PROFILING_GROUP")
    if codeguru_profiling_group:
        Profiler(profiling_group_name=codeguru_profiling_group).start()


def main() -> int:
    model_runner = ModelRunner()

    map_signals(model_runner)
    args = parse_args()
    configure_logging(args.verbose)
    setup_code_profiling()

    model_runner.run()
    return 1


if __name__ == "__main__":
    sys.exit(main())
