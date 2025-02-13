#  Copyright 2025 Amazon.com, Inc. or its affiliates.

from dataclasses import asdict, fields
from decimal import Decimal
from typing import Any, Dict


def numeric_to_decimal(value: Any) -> Any:
    """
    Convert numeric values (int, float) to Decimal, handling nested structures.

    Recursively converts numeric values to Decimal type while preserving the structure
    of nested containers (lists and dictionaries). Non-numeric values are returned unchanged.

    :param value: The value to convert, which can be a single numeric value (int, float),
                 a list containing numeric values, a dictionary containing numeric values,
                 or any non-numeric value
    :return: The converted value with the same structure as input but with numeric values
            converted to Decimal. For numeric inputs returns Decimal, for lists returns List,
            for dicts returns Dict, and for non-numeric values returns the original type
    """
    if isinstance(value, bool):
        # Need to check this first since True/False are both bool and int in python
        return value
    elif isinstance(value, (int, float)):
        return Decimal(str(value))
    elif isinstance(value, list):
        return [numeric_to_decimal(item) for item in value]
    elif isinstance(value, dict):
        return {k: numeric_to_decimal(v) for k, v in value.items()}
    return value


def decimal_to_numeric(value: Any) -> Any:
    """
    Convert Decimal values to appropriate numeric types, handling nested structures.

    Recursively converts Decimal values to their appropriate Python numeric type (int or float)
    while preserving the structure of nested containers. Non-Decimal values are returned unchanged.
    Decimal values without fractional parts are converted to int, while those with fractional
    parts are converted to float.

    :param value: The value to convert, which can be a single Decimal value,
                 a list containing Decimal values, a dictionary containing Decimal values,
                 or any non-Decimal value
    :return: The converted value with the same structure as input but with Decimal values
            converted to int or float. For whole number Decimals returns int, for fractional
            Decimals returns float, for lists returns List, for dicts returns Dict,
            and for non-Decimal values returns the original type
    """
    if isinstance(value, Decimal):
        # Convert to int if the Decimal has no decimal places
        if value % 1 == 0:
            return int(value)
        return float(value)
    elif isinstance(value, list):
        return [decimal_to_numeric(item) for item in value]
    elif isinstance(value, dict):
        return {k: decimal_to_numeric(v) for k, v in value.items()}
    return value


class DataclassDDBMixin:
    """
    This is a mixin that adds the ability to convert a dataclass to/from a dictionary of values suitable
    for use as a DynamoDB Item.
    """

    def to_ddb_item(self) -> Dict[str, Any]:
        """
        This function converts the dataclass to a dictionary of values suitable for use as a DynamoDB Item.

        - Any numeric (int, float) values are converted to an equivalent Decimal value since there are problems
          with how Python/DynamoDB handle floats.
        - Any fields that are None are excluded.

        :return: the dictionary
        """
        return numeric_to_decimal(asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}))

    @classmethod
    def from_ddb_item(cls, dictionary: Dict):
        """
        This function creates an instance of a dataclass from a DynamoDB Item dictionary.

        - Any Decimal values are converted to their numeric (int, float) equivalent values.
        - Any keys in the dictionary that do not

        :param dictionary: the DynamoDB item dictionary
        :return: the dataclass instance
        """
        # Note that this works because when using classmethod within a mixin, the cls argument refers to the class
        # that is inheriting from the mixin, not the mixin class itself. This allows the classmethod to operate on
        # the inheriting class's attributes and methods.
        field_names = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in dictionary.items() if k in field_names}
        typed_dict = decimal_to_numeric(filtered_dict)
        return cls(**typed_dict)
