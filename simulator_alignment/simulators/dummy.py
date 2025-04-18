import random

from ..data_models.sample import Sample
from .base import BaseSimulator


class CopyCatAssessor(BaseSimulator):
    """Dummy assessor that simply copies the groudtruth relevance assessment"""

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        for sample in samples:
            sample.set_predicted_relevance(sample.groundtruth_relevance)

        return samples


class RandomAssessor(BaseSimulator):
    """Dummy assessor that predicts a random relevance assessment"""

    def __init__(self, min_val: int = 0, max_val: int = 3) -> None:
        """Creates a RandomAssessor instance.

        Args:
            min_val (int, optional): The minimum value that can be predicted. Defaults to 0.
            max_val (int, optional): The maximum value that can be predicted. Defaults to 3.
        """
        self.min_val = min_val
        self.max_val = max_val

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        for sample in samples:
            sample.set_predicted_relevance(random.randint(self.min_val, self.max_val))
        return samples

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.min_val}_{self.max_val}"
