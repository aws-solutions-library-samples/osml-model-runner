#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import logging
import random
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import boto3
from boto3 import dynamodb

from aws.osml.model_runner.app_config import BotoConfig

from .exceptions import DDBUpdateException

logger = logging.getLogger(__name__)


@dataclass
class DDBKey:
    """
    DDBItem is a dataclass meant to represent a single item in a DynamoDB table via a key-value pair

    The data schema is defined as follows:
    key: str = the table key to index on
    value: str = the value of the item (given a key) to use
    range_key: str = the sort key to index on
    range_value: str = the value of the item (given a key + range key) to use
    """

    hash_key: str
    hash_value: str
    range_key: Optional[str] = None
    range_value: Optional[str] = None


@dataclass
class DDBItem:
    """
    DDBItem is a dataclass meant to represent a single item in a DynamoDB table via a key-value pair

    The data schema is defined as follows:
    key: str = the table key to index on
    value: str = the value of the item (given a key) to use
    range_key: str = the sort key to index on
    range_value: str = the value of the item (given a key + range key) to use
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
    DDBHelper is a class meant to help OSML with accessing and interacting with DynamoDB tables. Generally this class
    should be inherited by downstream specific table classes to build on top of such as the FeatureTable and JobTable
    classes.

    :param table_name: str = the name of the table to interact with

    :return: None
    """

    def __init__(self, table_name: str) -> None:
        # build a table resource to use for accessing data
        self.table = boto3.resource("dynamodb", config=BotoConfig.ddb).Table(table_name)
        self.table_name = table_name

    def get_ddb_item(self, ddb_item: DDBItem) -> Dict[str, Any]:
        """
        Get a DynamoDB item from table

        :param ddb_item: DDBItem = item that we want to get (required)

        :return: Dict[str, Any] = response from the get_item request
        """
        return self.table.get_item(Key=self.get_keys(ddb_item=ddb_item))["Item"]

    def put_ddb_item(self, ddb_item: DDBItem, condition_expression: str = None) -> Dict[str, Any]:
        """
        Put a DynamoDB item into the table

        :param ddb_item: DDBItem = item that we want to put (required)
        :param condition_expression: str = condition that must be satisfied in order for a
                                            conditional PutItem operation to succeed.

        :return: Dict[str, Any] = item from the put_item response
        """
        if condition_expression:
            response = self.table.put_item(Item=ddb_item.to_put(), ConditionExpression=condition_expression)
        else:
            response = self.table.put_item(Item=ddb_item.to_put())

        return response

    def batch_write_items(self, ddb_items: List[DDBItem], max_retries: int = 5, initial_retry_delay: float = 0.5,
                          max_retry_delay: float = 16) -> Dict[str, Any]:
        """
            Write multiple DynamoDB items in a batch with exponential back-off retry logic for unprocessed items.

            :param ddb_items: List[DDBItem] = List of items that we want to write.
            :param max_retries: int = Maximum number of retries for unprocessed items.
            :param initial_retry_delay: float = Initial delay in seconds for the first retry.
            :param max_retry_delay: float = Maximum delay in seconds between retries.

            :return: None
            """

        def _batch_write(request_items, retries=0) -> Dict[str, Any]:
            response = self.table.batch_write_item(RequestItems=request_items)
            unprocessed_items = response.get('UnprocessedItems', {})

            if unprocessed_items and retries < max_retries:
                # Calculate delay with exponential back-off
                delay = min(max_retry_delay, initial_retry_delay * (2 ** retries))
                # Add random jitter to avoid the thundering herd problem
                delay += random.uniform(0, delay * 0.1)
                time.sleep(delay)

                return _batch_write(unprocessed_items, retries + 1)
            elif unprocessed_items:
                logger.error(f"Failed to process items after {max_retries} retries: {unprocessed_items}")

        batch_size = 25
        for i in range(0, len(ddb_items), batch_size):
            batch = ddb_items[i:i + batch_size]
            request_items = {self.table_name: []}

            for item in batch:
                request_items[self.table_name].append({'PutRequest': {'Item': item.to_put()}})

            return _batch_write(request_items)

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
        provide an update expression and attributes one will be generated from the body.

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

        # if we still don't have an update expression then we'll just
        if update_exp and update_attr:
            response = self.table.update_item(
                Key=self.get_keys(ddb_item=ddb_item),
                UpdateExpression=update_exp,
                ExpressionAttributeValues=update_attr,
                ReturnValues="ALL_NEW",
            )

            # Return the updated items attributes
            return response["Attributes"]
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
            KeyConditionExpression=dynamodb.conditions.Key(ddb_item.ddb_key.hash_key).eq(ddb_item.ddb_key.hash_value),
        )

        # Grab all the items from the table
        items: List[dict] = []
        while not all_items_retrieved:
            items.extend(response["Items"])

            if "LastEvaluatedKey" in response:
                response = self.table.query(
                    ConsistentRead=True,
                    KeyConditionExpression=dynamodb.conditions.Key(ddb_item.ddb_key.hash_key).eq(
                        ddb_item.ddb_key.hash_value
                    ),
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

        :return Dict[str, Any] = Holding either Hash Key or both Keys (Hash and Range)
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
