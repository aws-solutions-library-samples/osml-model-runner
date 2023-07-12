#  Copyright 2023 Amazon.com, Inc. or its affiliates.


# ModelRunner Exceptions
class RetryableJobException(Exception):
    pass


class AggregateFeaturesException(Exception):
    pass


class ProcessRegionException(Exception):
    pass


class LoadImageException(Exception):
    pass


class ProcessImageException(Exception):
    pass


class UnsupportedModelException(Exception):
    pass


class InvalidImageURLException(Exception):
    pass


class SelfThrottledRegionException(Exception):
    pass


class InvalidFeaturePropertiesException(Exception):
    pass


class AggregateOutputFeaturesException(Exception):
    pass
