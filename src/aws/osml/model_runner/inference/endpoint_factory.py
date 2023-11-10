from typing import Dict, Optional

from aws.osml.model_runner.api import ModelInvokeMode

from .detector import Detector
from .http_detector import HTTPDetectorBuilder
from .sm_detector import SMDetectorBuilder


class FeatureDetectorFactory:
    def __init__(
        self, endpoint: str, endpoint_mode: ModelInvokeMode, assumed_credentials: Optional[Dict[str, str]] = None
    ) -> None:
        """
        :param endpoint: URL of the inference model endpoint
        :param endpoint_mode: the type of endpoint (HTTP, SageMaker)
        :param assumed_credentials: optional credentials to use with the model
        """

        self.endpoint = endpoint
        self.endpoint_mode = endpoint_mode
        self.assumed_credentials = assumed_credentials

    def build(self) -> Optional[Detector]:
        """
        :return: a feature detector based on the parameters defined during initialization
        """
        if self.endpoint_mode == ModelInvokeMode.SM_ENDPOINT:
            return SMDetectorBuilder(
                endpoint=self.endpoint,
                assumed_credentials=self.assumed_credentials,
            ).build()
        if self.endpoint_mode == ModelInvokeMode.HTTP_ENDPOINT:
            return HTTPDetectorBuilder(
                endpoint=self.endpoint,
            ).build()
