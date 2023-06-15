#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import time
from typing import Optional

from aws_embedded_metrics.logger.metrics_logger import MetricsLogger
from aws_embedded_metrics.unit import Unit


class Timer(object):
    def __init__(
        self,
        task_str: str,
        metric_name: str,
        logger: logging.Logger,
        metrics_logger: MetricsLogger = None,
    ) -> None:
        self.str = task_str
        self.metric_name = metric_name
        self.logger = logger
        self.metrics_logger: MetricsLogger = metrics_logger
        self.normalizing_factor: Optional[float] = None
        self.normalizing_unit: Optional[str] = None
        self.use_milliseconds = False

    def set_use_milliseconds(self, use_milliseconds: bool = True) -> None:
        """
        Use milliseconds format

        :param use_milliseconds: bool = should the timer be in milliseconds or not

        :return: None
        """
        self.use_milliseconds = use_milliseconds

    def set_normalizing_factor(self, normalizing_factor: float, normalizing_unit: str) -> None:
        """
        Set normalizing factor

        :param normalizing_factor: float = formatted factor (in time)
        :param normalizing_unit: str = formatted unit

        :return: None
        """
        self.normalizing_factor = normalizing_factor
        self.normalizing_unit = normalizing_unit

    def __enter__(self):
        self.logger.info(f"Starting: {self.str}")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: str, exc_val: str, exec_traceback: str) -> None:
        end_time = time.time()
        total_time = (end_time - self.start_time) * (1000 if self.use_milliseconds else 1)
        time_unit = Unit.MILLISECONDS.value if self.use_milliseconds else Unit.SECONDS.value
        try:
            if self.metrics_logger:
                self.metrics_logger.put_metric(self.metric_name, total_time, time_unit)
                self.logger.info(f"{self.str} took {str(total_time)} {time_unit.lower()}.")
                if self.normalizing_factor and self.normalizing_unit:
                    time_per_unit = total_time / self.normalizing_factor
                    self.metrics_logger.put_metric(f"{self.metric_name}Per{self.normalizing_unit}", time_per_unit, time_unit)
                    self.logger.info(
                        f"{self.str} took {str(time_per_unit)} {time_unit.lower()} " f"per {self.normalizing_unit}."
                    )
        except Exception as e:
            self.logger.warning("Unable to log metrics: " + str(e))
