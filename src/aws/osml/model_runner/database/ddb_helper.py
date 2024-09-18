#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

import logging
import random
import time
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import boto3
from boto3.dynamodb.conditions import Key

from aws.osml.model_runner.app_config import BotoConfig

from .exceptions import DDBBatchWriteException, DDBUpdateException

logger = logging.getLogger(__name__)


@dataclass
class DDBKey:
    """
    DDBKey is a dataclass meant to represent a single item in a DynamoDB table via a key-value pair.

    Attributes:
        hash_key (str): The name of the hash key (partition key) used to identify the item in the DynamoDB table.
        hash_value (str): The value of the hash key (partition key) that uniquely identifies the item in the table.
        range_key (Optional[str]): The name of the range key (sort key) used to further refine the item's location.
        range_value (Optional[str]): The value of the range key (sort key) that, together with the hash key, identifies.
    """

    hash_key: str
    hash_value: str
    range_key: Optional[str] = None
    range_value: Optional[str] = None


@dataclass
class DDBItem:
    """
    DDBItem is a dataclass meant to represent a single item in a DynamoDB table via a key-value pair.

    Attributes:
        ddb_key (DDBKey): The key object representing the hash and range key pair for this DynamoDB item.
    """

    ddb_key: DDBKey = field(init=False)

    def to_put(self) -> Dict[str, str]:
        return {k: v for k, v in asdict(self).items() if v is not None and k not in self.__get_fields()}

    def to_update(self) -> Dict[str, str]:
        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None and k is not self.ddb_key.hash_key and k not in self.__get_fields()
        }

    @staticmethod
    def __get_fields():
        return [my_field.name for my_field in fields(DDBItem)]


class DDBHelper:
    """
    DDBHelper is a class meant to help OSML with accessing and interacting with DynamoDB tables.

    Attributes:
        table_name (str): The name of the DynamoDB table to interact with.
        client (boto3.resources.factory.dynamodb.ServiceResource): A DynamoDB service resource instance used.
        table (boto3.resources.factory.dynamodb.Table): A reference to the DynamoDB table for performing operations.
    """

    def __init__(self, table_name: str) -> None:
        # build a table resource to use for accessing data
        self.table_name = table_name
        self.client = boto3.resource("dynamodb", config=BotoConfig.ddb)
        self.table = self.client.Table(table_name)

    def get_ddb_item(self, ddb_item: DDBItem) -> Dict[str, Any]:
        """
        Get a DynamoDB item from table and convert Decimal values to native types

        :param ddb_item: DDBItem = item that we want to get (required)

        :return: Dict[str, Any] = response from the get_item request
        """
        response = self.table.get_item(Key=self.get_keys(ddb_item=ddb_item))
        item = response.get("Item", {})
        return self.convert_decimal(item)

    def put_ddb_item(self, ddb_item: DDBItem, condition_expression: str = None) -> Dict[str, Any]:
        """
        Put a DynamoDB item into the table with a jitter-delayed retry logic for unprocessed items.

        :param ddb_item: DDBItem = item that we want to put (required)
        :param condition_expression: str = Condition that must be satisfied in order for a PutItem operation to succeed.

        :return: Dict[str, Any] = item from the put_item response
        """
        if condition_expression:
            response = self.table.put_item(Item=ddb_item.to_put(), ConditionExpression=condition_expression)
        else:
            response = self.table.put_item(Item=ddb_item.to_put())

        return response

    def batch_write_items(self, ddb_items: List[DDBItem], max_retries: int = 500, max_delay: float = 8) -> None:
        """
        Write multiple DynamoDB items in a batch with jitter-delayed retry logic for unprocessed items.

        This method splits the provided list of `ddb_items` into batches of up to 25 items (the maximum batch size
        supported by DynamoDB). Each batch is written to the table, and if unprocessed items are returned,
        the method retries the operation with an exponential backoff with jitter. The number of retries and the
        maximum delay between retries are configurable.

        :param ddb_items: List[DDBItem] = List of items that we want to write in batch mode to the DynamoDB table.
        :param max_retries: int = Maximum number of retries for unprocessed items. Defaults to 500.
        :param max_delay: float = Maximum delay in seconds between retries, applied with jitter. Defaults to 8 seconds.

        :return: None
        """

        def _batch_write(items: Dict[str, Any], retries: int = 0, initial_delay: float = 0.125):
            """
            Execute a batch write operation to DynamoDB with jitter-delayed retry logic for unprocessed items.

            This method performs a batch write operation using the provided `items`. If any items remain unprocessed,
            it retries the operation using exponential backoff with jitter. The method limits the number of retries
            and applies a maximum delay between retries.

            :param items: Dict[str, Any] = A dictionary containing the items to write in the batch.
            :param retries: int = The number of retries attempted for unprocessed items. Defaults to 0.
            :param initial_delay: float = The initial delay before the first retry, with jitter applied on subsequent
            retries. Defaults to 0.125 seconds.

            :raises DDBBatchWriteException: Raised if items remain unprocessed after the retry limit is exceeded.
            :raises Exception: Raised if an error occurs that is not related to unprocessed items.
            """
            # Calculate smart jitter backoff for retries
            delay = random.uniform(0, min(max_delay, initial_delay * 2**retries))
            try:
                response = self.client.batch_write_item(RequestItems=items)
                unprocessed_items = response.get("UnprocessedItems", {})
                # If we failed to process some items, try again
                if unprocessed_items and retries < max_retries:
                    time.sleep(delay)
                    _batch_write(unprocessed_items, retries + 1)
                # If we still have unprocessed items after the retry limit
                elif unprocessed_items:
                    raise DDBBatchWriteException(f"Failed to process items: {unprocessed_items}.")

                logger.debug("Successfully batch wrote items to table.")
            except Exception as err:
                # If we failed to call the write, try again
                if retries < max_retries:
                    time.sleep(delay)
                    _batch_write(items, retries + 1)
                else:
                    logger.error(err)
                    raise

        # Max batch size DDB (DynamoDB) supports is 25
        batch_size = 25

        # Iterate over ddb_items in chunks of batch_size
        for i in range(0, len(ddb_items), batch_size):
            batch = ddb_items[i : i + batch_size]
            request_items = {self.table_name: []}

            # Prepare the batch request for DynamoDB
            for item in batch:
                # Ensure that the item has a method `to_put()` that formats it for DynamoDB
                put_request = {"PutRequest": {"Item": item.to_put()}}
                request_items[self.table_name].append(put_request)

            # Send the batch write request
            _batch_write(request_items)

        return

    def delete_ddb_item(self, ddb_item: DDBItem) -> Dict[str, Any]:
        """
        Delete a DynamoDB item from the table

        :param ddb_item: DDBItem = item that we want to delete (required)

        :return: Dict[str, Any] = response from the delete_item request
        """
        return self.table.delete_item(Key=self.get_keys(ddb_item=ddb_item))

    def update_ddb_item(self, ddb_item: DDBItem, update_exp: str = None, update_attr: Dict = None) -> Dict[str, Any]:
        """
        Update the DynamoDB item based on the contents of an input dictionary. If the user doesn't
        provide an update expression and attributes, one will be generated from the body.

        :param ddb_item: DDBItem = item that we want to update (required)
        :param update_exp: Optional[str] = the update expression to use for the update
        :param update_attr: Optional[list] = attribute string to use when updating DDB item

        :return: Dict[str, Any] = the new ddb item as a dict
        """
        # if we weren't provided an explicit update expression/attributes
        # then we'll build them from the body
        if not update_exp and not update_attr:
            update_item = ddb_item.to_update()
            update_exp, update_attr = self.get_update_params(update_item, ddb_item)

        # if we still don't have an update expression, then we'll just
        if update_exp and update_attr:
            response = self.table.update_item(
                Key=self.get_keys(ddb_item=ddb_item),
                UpdateExpression=update_exp,
                ExpressionAttributeValues=update_attr,
                ReturnValues="ALL_NEW",
            )

            # Convert any decimal values in the response
            return self.convert_decimal(response["Attributes"])
        else:
            raise DDBUpdateException("Failed to produce update expression or attributes for DDB update!")

    def query_items(self, ddb_item: DDBItem) -> List[Dict[str, Any]]:
        """
        Query the table for all items of a given hash_key.

        :param ddb_item: DDBItem = the hash key we want to query the table for

        :return: List[Dict[str, Any]] = the list of dictionary responses corresponding to the items returned
        """
        all_items_retrieved = False
        response = self.table.query(
            ConsistentRead=True,
            KeyConditionExpression=Key(ddb_item.ddb_key.hash_key).eq(ddb_item.ddb_key.hash_value),
        )

        # Grab all the items from the table
        items: List[dict] = []
        while not all_items_retrieved:
            items.extend(self.convert_decimal(response["Items"]))

            if "LastEvaluatedKey" in response:
                response = self.table.query(
                    ConsistentRead=True,
                    KeyConditionExpression=Key(ddb_item.ddb_key.hash_key).eq(ddb_item.ddb_key.hash_value),
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
            else:
                all_items_retrieved = True

        return items

    @staticmethod
    def get_update_params(body: Dict, ddb_item: DDBItem) -> Tuple[str, Dict[str, Any]]:
        """
        Generate an update expression and a dict of values to update a dynamodb table.

        :param body: Dict = the body of the request that contains the updated data
        :param ddb_item: DDBItem = the hash key we want to query the table for

        :return: Tuple[str, Dict[str, Any]] = the generated update expression and attributes
        """
        update_expr = ["SET "]
        update_attr = dict()

        for key, val in body.items():
            # need to omit range_key
            if key != ddb_item.ddb_key.range_key:
                update_expr.append(f" {key} = :{key},")
                update_attr[f":{key}"] = val

        return "".join(update_expr)[:-1], update_attr

    @staticmethod
    def get_keys(ddb_item: DDBItem) -> Dict[str, Any]:
        """
        Determine to see if we need to use both keys to search an item in DDB

        :param ddb_item: DDBItem = the hash key we want to query the table for

        return Dict[str, Any] = Holding either Hash Key or both Keys (Hash and Range)
        """
        if ddb_item.ddb_key.range_key is None:
            return {
                ddb_item.ddb_key.hash_key: ddb_item.ddb_key.hash_value,
            }
        else:
            return {
                ddb_item.ddb_key.hash_key: ddb_item.ddb_key.hash_value,
                ddb_item.ddb_key.range_key: ddb_item.ddb_key.range_value,
            }

    @staticmethod
    def convert_decimal(data: Any) -> Any:
        """
        Convert any Decimal values in the data to int or float, depending on the value.

        :param data: Any = the data to convert
        :returns: Any = the converted data
        """
        if isinstance(data, list):
            return [DDBHelper.convert_decimal(item) for item in data]
        elif isinstance(data, dict):
            return {k: DDBHelper.convert_decimal(v) for k, v in data.items()}
        elif isinstance(data, Decimal):
            return int(data) if data % 1 == 0 else float(data)
        else:
            return data
