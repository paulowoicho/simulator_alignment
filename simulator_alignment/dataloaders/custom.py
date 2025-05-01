from pathlib import Path
from typing import Iterator

import jsonlines

from ..data_models.sample import Sample
from .base import BaseDataloader


class CustomBenchmark(BaseDataloader):
    """Dataloader for custom benchmarks.

    See: https://github.com/paulowoicho/simulator_alignment_data
    """

    def __init__(self, dataset_path: str | Path, name: str, num_samples: int = 1500) -> None:
        """Creates an instance of CustomBenchmark

        Args:
            dataset_path (str | Path): The path to the benchmark file. This will be a jsonlines file
                where every entry is contains `query_id`, `query`, `passage_id`, `passage`, and `relevance` fields.
            name (str): The name of the benchmark.
            num_samples (int): The number of samples to use from the benchmark. Defaults to 1500.
        """
        self.dataset_path = dataset_path
        self._benchmark_name = name
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def name(self) -> str:
        return self._benchmark_name

    def get_sample(self) -> Iterator[Sample]:
        with jsonlines.open(self.dataset_path) as f:
            for idx, obj in enumerate(f, start=1):
                yield Sample(
                    query=obj["query"],
                    passage=obj["passage"],
                    groundtruth_relevance=min(obj["relevance"], 3),
                )
                if idx >= self._num_samples:
                    break
