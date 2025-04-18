from abc import ABC, abstractmethod
from typing import Iterator

from ..data_models.sample import Sample


class BaseDataloader(ABC):
    """A dataloader is a wrapper around a dataset. Its main utility create and yield data_models.sample.Sample objects."""

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Reurns the number of samples in the dataset."""
        ...

    @abstractmethod
    def get_sample(self) -> Iterator[Sample]:
        """Generates samples from the dataset."""
        ...

    @property
    def name(self):
        """Returns the class name."""
        return self.__class__.__name__
