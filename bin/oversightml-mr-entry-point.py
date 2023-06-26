#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import argparse
import logging
import os
import signal
import sys
from types import FrameType
from typing import Optional

from codeguru_profiler_agent import Profiler

from aws.osml.model_runner.app import ModelRunner


def handler_stop_signals(signal_num: int, frame: Optional[FrameType], model_runner: ModelRunner) -> None:
    model_runner.stop()


def configure_logging(verbose: bool) -> None:
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG

    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    formatter = logging.Formatter("%(levelname)-8s %(message)s")
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
