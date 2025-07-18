#!/usr/bin/env python3
from abc import ABC, abstractmethod

class ObjectPerception(ABC):
    """Abstract base class for object perception systems."""

    @abstractmethod
    def inference(self, image):
        """Performs inference on the input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            Any: Inference result.
        """
        pass