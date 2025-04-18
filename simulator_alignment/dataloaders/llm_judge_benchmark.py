import csv
from pathlib import Path
from typing import Iterator

import jsonlines

from ..data_models.sample import Sample
from .base import BaseDataloader


class LLMJudgeBenchmark(BaseDataloader):
    """DataLoader for the LLMJudge Benchmark data.

    Note that relevance is defined on a scale of 0-3.
    More information available here: https://llm4eval.github.io/LLMJudge/
    """

    def __init__(
        self, qrels_path: str | Path, queries_path: str | Path, passages_path: str | Path
    ) -> None:
        """Creates an LLMJudge Benchmark dataloader.

        Note that these need to be downloaded from https://github.com/llm4eval/LLMJudge/tree/main/data
        or at least follow the format of the files there.

        Args:
            qrels_path (str | Path): Path to qrels file, which contains relevance judgements.
            queries_path (str | Path): Path to queries file, which contains a mapping of query ids to text.
            passages_path (str | Path): Path to the passages file, which contains a mapping of passage ids to text.
        """
        self.qrels_path = qrels_path

        self.query_id_to_query = {}
        with open(queries_path, encoding="utf-8") as queries_file:
            reader = csv.reader(queries_file, delimiter="\t")
            for row in reader:
                query_id, query_text = row
                self.query_id_to_query[query_id] = query_text

        self.passage_id_to_passage = {}
        with jsonlines.open(passages_path) as passages_file:
            for obj in passages_file:
                self.passage_id_to_passage[obj["docid"]] = obj["doc"]

    @property
    def num_samples(self) -> int:
        with open(self.qrels_path, encoding="utf-8") as f:
            return sum(bool(line) for line in f)

    def get_sample(self) -> Iterator[Sample]:
        with open(self.qrels_path, encoding="utf-8") as qrels_file:
            reader = csv.reader(qrels_file, delimiter=" ")
            for row in reader:
                query_id, _, passage_id, groundtruth_relevance_raw = row
                query = self.query_id_to_query[query_id]
                passage = self.passage_id_to_passage[passage_id]
                groundtruth_relevance = int(groundtruth_relevance_raw)
                yield Sample(
                    query=query, passage=passage, groundtruth_relevance=groundtruth_relevance
                )
