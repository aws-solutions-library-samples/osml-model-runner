#  Copyright 2023 Amazon.com, Inc. or its affiliates.

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from botocore.config import Config


@dataclass
class ServiceConfig:
    """
    ServiceConfig is a dataclass meant to house the high-level configuration settings required for Model Runner to
    operate that are provided through ENV variables. Note that required env parameters are enforced by the implied
    schema validation as os.environ[] is used to fetch the values. Optional parameters are fetched using, os.getenv(),
    which returns None.

    The data schema is defined as follows:
    region:  (str) The AWS region where the Model Runner is deployed.
    job_table: (str) The name of the job processing DDB table
    region_request_table: (str) The name of the region request processing DDB table
    feature_table: (str) The name of the feature aggregation DDB table
    image_queue: (str) The name of the image processing SQS queue
    region_queue: (str) The name of the region processing SQS queue
    workers_per_cpu: (int) The number of workers to launch per CPU
    image_timeout:  (int) The number of seconds to wait for an image to be processed
    region_timeout: (int) The number of seconds to wait for a region to be processed
    cp_api_endpoint: (str) The URL of the control plane API endpoint
    """

    # required env configuration
    aws_region: str = os.environ["AWS_DEFAULT_REGION"]
    job_table: str = os.environ["JOB_TABLE"]
    region_request_table: str = os.environ["REGION_REQUEST_TABLE"]
    endpoint_statistics_table = os.environ["ENDPOINT_TABLE"]
    feature_table: str = os.environ["FEATURE_TABLE"]
    image_queue: str = os.environ["IMAGE_QUEUE"]
    region_queue: str = os.environ["REGION_QUEUE"]
    workers_per_cpu: str = os.environ["WORKERS_PER_CPU"]
    workers: str = os.environ["WORKERS"]

    # optional elevation data
    elevation_data_location: Optional[str] = os.getenv("ELEVATION_DATA_LOCATION")
    elevation_data_extension: str = os.getenv("ELEVATION_DATA_EXTENSION", ".tif")
    elevation_data_version: str = os.getenv("ELEVATION_DATA_VERSION", "1arc_v3")

    # optional env configuration
    image_status_topic: Optional[str] = os.getenv("IMAGE_STATUS_TOPIC")
    region_status_topic: Optional[str] = os.getenv("REGION_STATUS_TOPIC")
    cp_api_endpoint: Optional[str] = os.getenv("API_ENDPOINT")
    self_throttling: bool = (
        os.getenv("SM_SELF_THROTTLING", "False") == "True" or os.getenv("SM_SELF_THROTTLING", "False") == "true"
    )

    # optional + defaulted configuration
    region_size: str = os.getenv("REGION_SIZE", "(10240, 10240)")
    throttling_vcpu_scale_factor: str = os.getenv("THROTTLING_SCALE_FACTOR", "10")
    # Time in seconds to set region request visibility timeout when a request
    # is self throttled
    throttling_retry_timeout: str = os.getenv("THROTTLING_RETRY_TIMEOUT", "10")

    # constant configuration
    kinesis_max_record_size: str = "1048576"
    ddb_max_item_size: str = "200000"
    noop_bounds_model_name: str = "NOOP_BOUNDS_MODEL_NAME"
    noop_geom_model_name: str = "NOOP_GEOM_MODEL_NAME"


@dataclass
class BotoConfig:
    """
    BotoConfig is a dataclass meant to vend our application the set of boto client configurations required for OSML

    The data schema is defined as follows:
    default:  (Config) the standard boto client configuration
    sagemaker: (Config) the sagemaker specific boto client configuration
    """

    default: Config = Config(region_name=ServiceConfig.aws_region, retries={"max_attempts": 15, "mode": "standard"})
    sagemaker: Config = Config(region_name=ServiceConfig.aws_region, retries={"max_attempts": 30, "mode": "adaptive"})
    ddb: Config = Config(
        region_name=ServiceConfig.aws_region, retries={"max_attempts": 3, "mode": "standard"}, max_pool_connections=50
    )


class MetricLabels(str, Enum):
    """
    Enumeration defining the metric labels used by OSML
    """

    # These are based on common metric names used by a variety of AWS services (e.g. Lambda)
    DURATION = "Duration"
    INVOCATIONS = "Invocations"
    ERRORS = "Errors"
    THROTTLES = "Throttles"
    RETRIES = "Retries"

    # These dimensions allow us to limit the scope of a metric value to a particular portion of the
    # ModelRunner application, a data type, or input format.
    OPERATION_DIMENSION = "Operation"
    MODEL_NAME_DIMENSION = "ModelName"
    INPUT_FORMAT_DIMENSION = "InputFormat"

    # These operation names can be used along with the Operation dimension to restrict the scope
    # of the common metrics to a specific portion of the ModelRunner application.
    IMAGE_PROCESSING_OPERATION = "ImageProcessing"
    REGION_PROCESSING_OPERATION = "RegionProcessing"
    TILE_GENERATION_OPERATION = "TileGeneration"
    TILE_PROCESSING_OPERATION = "TileProcessing"
    MODEL_INVOCATION_OPERATION = "ModelInvocation"
    FEATURE_REFINEMENT_OPERATION = "FeatureRefinement"
    FEATURE_STORAGE_OPERATION = "FeatureStorage"
    FEATURE_AGG_OPERATION = "FeatureAggregation"
    FEATURE_SELECTION_OPERATION = "FeatureSelection"
    FEATURE_DISSEMINATE_OPERATION = "FeatureDissemination"
