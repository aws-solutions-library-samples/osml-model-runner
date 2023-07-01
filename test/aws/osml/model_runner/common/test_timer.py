#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import time
import unittest

from mock import Mock


class TestTimer(unittest.TestCase):
    def test_timer_with_normalized_unit(self):
        from aws.osml.model_runner.common import Timer

        mock_logger = Mock()
        mock_metrics_logger = Mock()
        mock_metrics_logger.put_metric = Mock()
        mock_metric_name = "Testing"
        mock_metrics_per_unit = "TestUnit"
        with Timer(
            task_str="Running a test",
            metric_name=mock_metric_name,
            logger=mock_logger,
            metrics_logger=mock_metrics_logger,
        ) as t:
            time.sleep(1.0)
            t.set_normalizing_factor(2.0, mock_metrics_per_unit)
        call_args = mock_metrics_logger.put_metric.call_args_list
        assert len(call_args) == 2
        assert call_args[0][0][0] == mock_metric_name
        time_result = call_args[0][0][1]
        assert time_result >= 1.0
        assert call_args[0][0][2] == "Seconds"
        assert call_args[1][0][0] == f"{mock_metric_name}Per{mock_metrics_per_unit}"
        assert call_args[1][0][1] == time_result / 2.0
        assert call_args[1][0][2] == "Seconds"

    def test_timer_without_normalized_unit(self):
        from aws.osml.model_runner.common import Timer

        mock_logger = Mock()
        mock_metrics_logger = Mock()
        mock_metrics_logger.put_metric = Mock()
        mock_metric_name = "Testing"
        with Timer(
            task_str="Running a test",
            metric_name=mock_metric_name,
            logger=mock_logger,
            metrics_logger=mock_metrics_logger,
        ):
            time.sleep(1.0)
        call_args = mock_metrics_logger.put_metric.call_args_list
        assert len(call_args) == 1
        assert call_args[0][0][0] == mock_metric_name
        time_result = call_args[0][0][1]
        assert time_result >= 1.0
        assert call_args[0][0][2] == "Seconds"

    def test_timer_use_milliseconds(self):
        from aws.osml.model_runner.common import Timer

        mock_logger = Mock()
        mock_metrics_logger = Mock()
        mock_metrics_logger.put_metric = Mock()
        mock_metric_name = "Testing"
        with Timer(
            task_str="Running a test",
            metric_name=mock_metric_name,
            logger=mock_logger,
            metrics_logger=mock_metrics_logger,
        ) as t:
            time.sleep(1.0)
            t.set_use_milliseconds(True)
        call_args = mock_metrics_logger.put_metric.call_args_list
        assert len(call_args) == 1
        assert call_args[0][0][0] == mock_metric_name
        time_result = call_args[0][0][1]
        assert time_result >= 1.0
        assert call_args[0][0][2] == "Milliseconds"

    def test_timer_metrics_logger_none(self):
        from aws.osml.model_runner.common import Timer

        mock_logger = Mock()
        mock_metrics_logger = Mock()
        mock_metrics_logger.put_metric = None
        mock_metric_name = "testing"
        with Timer(
            task_str="Running a test",
            metric_name=mock_metric_name,
            logger=mock_logger,
            metrics_logger=None,
        ):
            time.sleep(1.0)

        call_args = mock_metrics_logger.put_metric
        assert call_args is None

    def test_timer_throw_exception(self):
        from aws.osml.model_runner.common import Timer

        mock_logger = Mock()
        mock_metrics_logger = Mock()
        mock_metrics_logger.put_metric = None
        mock_metric_name = "testing"
        with Timer(
            task_str="Running a test",
            metric_name=mock_metric_name,
            logger=mock_logger,
            metrics_logger=mock_metrics_logger,
        ):
            time.sleep(1.0)

        call_args = mock_metrics_logger.put_metric
        assert call_args is None


if __name__ == "__main__":
    unittest.main()
