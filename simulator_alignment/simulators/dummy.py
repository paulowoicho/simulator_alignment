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


class NoisyCopyCatAssessor(BaseSimulator):
    """Dummy assessor that predicts relevance such that
    for each sample, with probability self.noise_probability
    assign a random incorrect relevance in [0..max_score], else
    assign the ground‑truth relevance.
    """

    def __init__(self, noise_probability: float, max_score: int = 3) -> None:
        """Creates a NoisyCopyCatAssessor instance.

        Args:
            noise_probability (float): A probability indicating
                how often the assessor is likely to be wrong about a judgement
            max_score (int): The maximum score a sample can have as its
                relevance assessment.
        """
        self.noise_probability = noise_probability
        self.max_score = max_score
        self._other_scores: dict[int, list[int]] = {
            gt: [s for s in range(max_score + 1) if s != gt] for gt in range(max_score + 1)
        }

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        for sample in samples:
            if random.random() <= self.noise_probability:
                alt_scores = self._other_scores[sample.groundtruth_relevance]
                sample.set_predicted_relevance(random.choice(alt_scores))
            else:
                sample.set_predicted_relevance(sample.groundtruth_relevance)
        return samples

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.noise_probability}"
