#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
from dataclasses import dataclass
from decimal import Decimal

from boto3.dynamodb.conditions import Attr
from botocore.exceptions import ClientError
from dacite import from_dict

from .ddb_helper import DDBHelper, DDBItem, DDBKey

logger = logging.getLogger(__name__)


@dataclass
class EndpointStatisticsItem(DDBItem):
    """
    EndpointStatistics is a dataclass meant to represent a single item in the
    EndpointStatisticsTable

    The data schema is defined as follows:
    endpoint: str = the Sagemaker endpoint to which the statistics pertain
    regions_in_progress: Decimal = the number of regions currently being processed
        for the associated endpoint
    max_regions: Decimal = the maximum number of regions that an endpoint can concurrently
        process before region requests should be throttled
    """

    endpoint: str
    regions_in_progress: Decimal = Decimal(0)
    max_regions: Decimal = Decimal(0)

    def __post_init__(self):
        self.ddb_key = DDBKey(hash_key="endpoint", hash_value=self.endpoint)


class EndpointStatisticsTable(DDBHelper):
    """
    EndpointStatisticsTable is a class meant to help OSML with accessing and interacting with
    the per endpoints processing statistics we track as part of the endpoint statistics table.
    It extends the DDBHelper class and provides its own item data class for use when
    working with items from the table. It also sets the key for which we index on this table
    in the constructor.

    :param table_name: str = the name of the table to interact with

    :return: None
    """

    def __init__(self, table_name: str) -> None:
        super().__init__(table_name)

    def upsert_endpoint(self, endpoint: str, max_regions: int) -> None:
        """
        Upserts an endpoint statistics entry. If the endpoint is already being
        tracked then we update the max_region count for the existing entry. If an
        endpoint is scaled up or down or the associated instance type changes the
        maximum number of concurrent regions may change.

        :param endpoint: str = Sagemaker endpoint name
        :param max_regions: int = current max concurrent regions the endpoint can process

        :return: None
        """
        logger.debug(f"Setting max region count for endpoint {endpoint} to {max_regions}")
        try:
            self.put_ddb_item(
                EndpointStatisticsItem(endpoint=endpoint, max_regions=Decimal(max_regions)),
                Attr("endpoint").not_exists(),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                self.update_ddb_item(
                    EndpointStatisticsItem(endpoint=endpoint),
                    "SET max_regions = :max",
                    {":max": max_regions},
                )
            else:
                raise

    def increment_region_count(self, endpoint: str) -> None:
        """
        Increases the in progress regions for the specified endpoint by 1.

        :param endpoint: str = Sagemaker endpoint name

        :return: None
        """
        logger.debug(f"Incremented in-progress region count for endpoint: '{endpoint}'")
        self.update_ddb_item(
            EndpointStatisticsItem(endpoint=endpoint),
            "SET regions_in_progress = regions_in_progress + :change",
            {":change": 1},
        )

    def decrement_region_count(self, endpoint: str) -> None:
        """
        Decreases the in progress regions for the specified endpoint by 1.

        :param endpoint: str = Sagemaker endpoint name

        :return: None
        """
        logger.debug(f"Decremented in-progress region count for endpoint: '{endpoint}'")
        self.update_ddb_item(
            EndpointStatisticsItem(endpoint=endpoint),
            "SET regions_in_progress = regions_in_progress - :change",
            {":change": 1},
        )

    def current_in_progress_regions(self, endpoint: str) -> int:
        """
        Retrieve the current number of in progress regions for the specified endpoint

        :param endpoint: str = Sagemaker endpoint name

        :return: int = current number of in progress regions
        """
        return int(
            from_dict(
                EndpointStatisticsItem, self.get_ddb_item(EndpointStatisticsItem(endpoint=endpoint))
            ).regions_in_progress
        )
