#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import unittest
from typing import List

from geojson import Feature

from aws.osml.model_runner.api import SinkMode
from aws.osml.model_runner.sink.sink import Sink


class MockSink(Sink):
    """
    A mock implementation of the Sink abstract class to test the interface.
    """

    @staticmethod
    def name() -> str:
        return "MockSink"

    @property
    def mode(self) -> SinkMode:
        return SinkMode.AGGREGATE

    def write(self, image_id: str, features: List[Feature]) -> bool:
        return True


class TestSink(unittest.TestCase):
    def setUp(self):
        """
        Create an instance of the MockSink class for testing.
        """
        self.sink = MockSink()

    def tearDown(self):
        """
        Clean up any resources used by the tests.
        """
        self.sink = None

    def test_str_representation(self):
        """
        Test the `__str__` method.
        Verifies that the string representation combines name and mode.
        """
        expected_str = "MockSink AGGREGATE"
        self.assertEqual(str(self.sink), expected_str)


if __name__ == "__main__":
    unittest.main()
