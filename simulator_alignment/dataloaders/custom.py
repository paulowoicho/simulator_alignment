from pathlib import Path
from typing import Iterator

import jsonlines

from ..data_models.sample import Sample
from .base import BaseDataloader


class CustomBenchmark(BaseDataloader):
    """Dataloader for custom benchmarks.

    See: https://github.com/paulowoicho/simulator_alignment_data
    """

    def __init__(self, dataset_path: str | Path, name: str) -> None:
        """Creates an instance of CustomBenchmark

        Args:
            dataset_path (str | Path): The path to the benchmark file. This will be a jsonlines file
                where every entry is contains `query_id`, `query`, `passage_id`, `passage`, and `relevance` fields.
            name (str): The name of the benchmark.
        """
        self.dataset_path = dataset_path
        self._benchmark_name = name

    @property
    def num_samples(self) -> int:
        counter = 0
        with jsonlines.open(self.dataset_path) as f:
            for _ in f:
                counter += 1
        return counter

    @property
    def name(self) -> str:
        return self._benchmark_name

    def get_sample(self) -> Iterator[Sample]:
        with jsonlines.open(self.dataset_path) as f:
            for obj in f:
                yield Sample(
                    query=obj["query"],
                    passage=obj["passage"],
                    groundtruth_relevance=obj["relevance"],
                )
