#  Copyright 2023 Amazon.com, Inc. or its affiliates.

from aws_embedded_metrics.config import get_config


def build_embedded_metrics_config():
    """
    Embedded metrics configuration
    """
    metrics_config = get_config()
    metrics_config.service_name = "OSML"
    metrics_config.log_group_name = "/aws/OSML/MRService"
    metrics_config.namespace = "OSML/ModelRunner"
    metrics_config.environment = "local"
