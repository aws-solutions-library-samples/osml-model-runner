#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from enum import Enum


class AutoStringEnum(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values) -> str:
        return name
