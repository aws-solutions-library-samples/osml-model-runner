#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import unittest
from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional

from aws.osml.model_runner.database.dataclass_ddb_mixin import DataclassDDBMixin, decimal_to_numeric, numeric_to_decimal


class TestDataclassDDBMixin(unittest.TestCase):
    """Test cases for DynamoDB utility functions."""

    def test_mixin(self):
        @dataclass
        class DummyDataclass(DataclassDDBMixin):
            one: int = 1
            one_five: float = 1.5
            optional: Optional[str] = None

        test_dataclass = DummyDataclass()
        test_dataclass_item = test_dataclass.to_ddb_item()
        self.assertIsInstance(test_dataclass_item["one"], Decimal)
        self.assertIsInstance(test_dataclass_item["one_five"], Decimal)
        self.assertEqual(len(test_dataclass_item.keys()), 2)

        new_test_dataclass = DummyDataclass.from_ddb_item(test_dataclass_item)
        self.assertIsInstance(new_test_dataclass, DummyDataclass)
        self.assertIsInstance(new_test_dataclass.one, int)
        self.assertIsInstance(new_test_dataclass.one_five, float)
        self.assertEqual(new_test_dataclass, test_dataclass)

    def test_nested_structures(self):
        original = {"number": 14.5, "text": "hello", "nested": {"values": [1, "text", None], "bool": True}}

        # Convert to decimals
        decimal_version = numeric_to_decimal(original)
        self.assertIsInstance(decimal_version["number"], Decimal)
        self.assertIsInstance(decimal_version["nested"]["values"][0], Decimal)
        self.assertEqual(decimal_version["nested"]["values"][1], "text")
        self.assertEqual(decimal_version["nested"]["values"][2], None)
        self.assertEqual(decimal_version["nested"]["bool"], True)

        # Convert back to numeric
        numeric_version = decimal_to_numeric(decimal_version)
        self.assertEqual(numeric_version, original)

    def test_nested_dataclasses(self):
        @dataclass
        class Point(DataclassDDBMixin):
            x: float
            y: float

        @dataclass
        class Line(DataclassDDBMixin):
            a: Point
            b: Point

        @dataclass
        class MultiPoint(DataclassDDBMixin):
            count: int
            points: Optional[List[Point]] = None

        test_line_as_dict = {"a": {"x": Decimal(1.0), "y": Decimal(2.0)}, "b": {"x": Decimal(3), "y": Decimal(4)}}
        test_line = Line.from_ddb_item(test_line_as_dict)
        self.assertIsInstance(test_line, Line)
        self.assertIsInstance(test_line.a, Point)
        self.assertIsInstance(test_line.b, Point)
        self.assertEqual(test_line.a.x, 1.0)
        self.assertEqual(test_line.a.y, 2.0)
        self.assertEqual(test_line.b.x, 3.0)
        self.assertEqual(test_line.b.y, 4.0)

        test_multipoint_as_dict = {
            "count": Decimal(2),
            "points": [{"x": Decimal(1.0), "y": Decimal(2.0)}, {"x": Decimal(3), "y": Decimal(4)}],
        }
        test_multipoint = MultiPoint.from_ddb_item(test_multipoint_as_dict)
        self.assertIsInstance(test_multipoint, MultiPoint)
        self.assertIsInstance(test_multipoint.points, List)
        self.assertEqual(test_multipoint.count, 2)
        self.assertEqual(len(test_multipoint.points), 2)
        self.assertIsInstance(test_multipoint.points[0], Point)
        self.assertIsInstance(test_multipoint.points[1], Point)

        test_multipoint_as_dict = {"count": Decimal(0)}
        test_multipoint = MultiPoint.from_ddb_item(test_multipoint_as_dict)
        self.assertIsInstance(test_multipoint, MultiPoint)
        self.assertEqual(test_multipoint.count, 0)
        self.assertIsNone(test_multipoint.points)


if __name__ == "__main__":
    unittest.main()
