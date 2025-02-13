#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import unittest
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from aws.osml.model_runner.database.dataclass_ddb_mixin import DataclassDDBMixin, decimal_to_numeric, numeric_to_decimal


class TestDDBUtils(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
