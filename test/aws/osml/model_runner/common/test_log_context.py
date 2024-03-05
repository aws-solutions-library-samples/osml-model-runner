#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
from threading import Thread
from unittest import TestCase, main


class TestThread(TestCase):
    def test_filter_adds_thread_local_context(self):
        from aws.osml.model_runner.common import ThreadingLocalContextFilter

        context_filter = ThreadingLocalContextFilter(attribute_names=["context_value"])
        context_filter.set_context({"context_value": "A"})
        test_log_record = logging.LogRecord("test-name", logging.DEBUG, "some_module.py", 1, "test message", None, None)
        assert context_filter.filter(test_log_record) is True
        assert test_log_record.context_value == "A"

        thread_log_record = logging.LogRecord("test-name", logging.DEBUG, "some_module.py", 1, "test message", None, None)

        def sample_task():
            context_filter.set_context({"context_value": "B"})
            context_filter.filter(thread_log_record) is True

        thread = Thread(target=sample_task)
        thread.start()
        thread.join()
        assert thread_log_record.context_value == "B"

        test_log_record = logging.LogRecord("test-name", logging.DEBUG, "some_module.py", 1, "test message", None, None)
        assert context_filter.filter(test_log_record) is True
        assert test_log_record.context_value == "A"


if __name__ == "__main__":
    main()
