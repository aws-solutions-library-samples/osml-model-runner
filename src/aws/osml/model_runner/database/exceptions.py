#  Copyright 2023 Amazon.com, Inc. or its affiliates.


# Database Exceptions
class AddFeaturesException(Exception):
    pass


class DDBUpdateException(Exception):
    pass


class DDBBatchWriteException(Exception):
    pass


class EndImageException(Exception):
    pass


class GetImageRequestItemException(Exception):
    pass


class IsImageCompleteException(Exception):
    pass


class StartImageException(Exception):
    pass


class StartRegionException(Exception):
    pass


class InvalidRegionRequestException(Exception):
    pass


class GetRegionRequestItemException(Exception):
    pass


class UpdateRegionException(Exception):
    pass


class CompleteRegionException(Exception):
    pass
