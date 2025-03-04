#  Copyright 2025 Amazon.com, Inc. or its affiliates.

import inspect
from dataclasses import MISSING, asdict, fields, is_dataclass
from decimal import Decimal
from typing import Any, Dict, Type, TypeVar, Union, get_args, get_origin, get_type_hints

T = TypeVar("T")


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


def create_dataclass_from_dict(cls: Type[T], data: dict) -> T:
    """
    Creates a dataclass instance from a dictionary, handling nested dataclasses, Optional fields, Lists, and Dicts.

    :param cls: The dataclass type to create
    :param data: Dictionary containing the data

    :return: An instance of the dataclass populated with the dictionary data
    """
    if not is_dataclass(cls):
        raise ValueError(f"Class {cls.__name__} is not a dataclass")

    if data is None:
        return None

    # Get type hints for all fields
    type_hints = get_type_hints(cls)

    # Prepare kwargs for the dataclass constructor
    kwargs = {}

    for field in fields(cls):
        field_name = field.name
        field_type = type_hints[field_name]
        field_value = data.get(field_name, field.default if field.default is not MISSING else None)

        # Skip if field is not in data and is optional
        if field_value is None and _is_optional(field_type):
            continue

        # Process the field value based on its type
        kwargs[field_name] = _process_field_value(field_type, field_value)

    return cls(**kwargs)


def _is_optional(field_type: Type) -> bool:
    """
    Determines if a field type is Optional.
    """
    origin = get_origin(field_type)
    return origin is Union and type(None) in get_args(field_type)


def _process_field_value(field_type: Type, value: Any) -> Any:
    """
    Processes a field value based on its type annotation.
    Handles nested dataclasses, Lists, Dicts, and Optional types.
    """
    if value is None:
        return None

    # Get the origin type (List, Dict, etc.) and type arguments
    origin = get_origin(field_type)
    type_args = get_args(field_type)

    # Handle Optional types
    if _is_optional(field_type):
        # Get the actual type from Optional
        actual_type = next(t for t in type_args if not isinstance(t, type(None)))
        return _process_field_value(actual_type, value)

    # Handle Lists
    if origin is list:
        if not type_args:
            return value
        element_type = type_args[0]
        return [_process_field_value(element_type, item) for item in value]

    # Handle Dictionaries
    if origin is dict:
        if not type_args:
            return value
        key_type, value_type = type_args
        return {_process_field_value(key_type, k): _process_field_value(value_type, v) for k, v in value.items()}

    # Handle nested dataclasses
    if inspect.isclass(field_type) and is_dataclass(field_type):
        return create_dataclass_from_dict(field_type, value)

    # Handle basic types
    return value


class DataclassDDBMixin:
    """
    This is a mixin that adds the ability to convert a dataclass to/from a dictionary of values suitable
    for use as a DynamoDB Item.

    Note that this mixin supports nested dataclasses but it does not support abstract types or unions. The issue is
    that the information about the concrete instance type is not fully serialized to the dictionary, only attribute
    names and values are converted. The information about what types to construct is taken from the type hints and
    field annotations of the dataclass. A dataclass definition that contains an abstract type will not have enough
    information to fully reconstruct the object instance.
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
        numeric_dict = decimal_to_numeric(filtered_dict)
        return create_dataclass_from_dict(cls, numeric_dict)
