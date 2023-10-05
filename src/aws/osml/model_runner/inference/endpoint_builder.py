from abc import ABC, abstractmethod
from typing import Optional

from .detector import Detector


class FeatureEndpointBuilder(ABC):
    """
    This is an abstract base for all classes to construct Detectors for various types of endpoints.
    """

    def __init__(self) -> None:
        """
        Constructor for the builder accepting required properties or formats for detectors

        :return: None
        """
        pass

    @abstractmethod
    def build(self) -> Optional[Detector]:
        """
        Constructs the sensor model from the available information. Note that in cases where not enough information is
        available to provide any solution, this method will return None.

        :return: Optional[Detector] = the detector to generate features based on the provided build data
        """
