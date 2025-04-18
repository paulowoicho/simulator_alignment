from pathlib import Path
import tempfile
from unittest.mock import mock_open, patch

import jsonlines
import pytest

from simulator_alignment.data_models.sample import Sample
from simulator_alignment.dataloaders.llm_judge_benchmark import LLMJudgeBenchmark


class TestLLMJudgeBenchmark:
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        qrels_data = "q1 0 p1 3\nq2 0 p2 1\nq3 0 p3 0"
        queries_data = "q1\tWhat is the capital of France?\nq2\tHow tall is Mount Everest?\nq3\tWho wrote Hamlet?"
        passages_data = [
            {"docid": "p1", "doc": "Paris is the capital of France."},
            {"docid": "p2", "doc": "Mount Everest is 8,848.86 meters tall."},
            {"docid": "p3", "doc": "William Shakespeare wrote Hamlet."},
        ]

        return qrels_data, queries_data, passages_data

    @pytest.fixture
    def temp_files(self, sample_data):
        """Create temporary files with sample data"""
        qrels_data, queries_data, passages_data = sample_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create qrels file
            qrels_path = Path(tmp_dir) / "qrels.txt"
            with open(qrels_path, "w", encoding="utf-8") as f:
                f.write(qrels_data)

            # Create queries file
            queries_path = Path(tmp_dir) / "queries.tsv"
            with open(queries_path, "w", encoding="utf-8") as f:
                f.write(queries_data)

            # Create passages file
            passages_path = Path(tmp_dir) / "passages.jsonl"
            with jsonlines.open(passages_path, "w") as writer:
                for passage in passages_data:
                    writer.write(passage)

            yield qrels_path, queries_path, passages_path

    @pytest.fixture
    def llmjudge_loader(self, temp_files):
        """Create an LLMJudgeBenchmark instance"""
        qrels_path, queries_path, passages_path = temp_files
        return LLMJudgeBenchmark(qrels_path, queries_path, passages_path)

    def test_num_samples(self, llmjudge_loader):
        """Test the num_samples property"""
        assert llmjudge_loader.num_samples == 3

    def test_num_samples_empty_file(self):
        """Test num_samples with empty qrels file"""
        with tempfile.NamedTemporaryFile() as empty_file:
            with patch("builtins.open", mock_open(read_data="")):
                loader = LLMJudgeBenchmark(
                    empty_file.name, "dummy_queries.tsv", "dummy_passages.jsonl"
                )
                with patch.object(loader, "query_id_to_query", {}):
                    with patch.object(loader, "passage_id_to_passage", {}):
                        assert loader.num_samples == 0

    def test_get_sample(self, llmjudge_loader):
        """Test get_sample method returns expected samples"""
        samples = list(llmjudge_loader.get_sample())

        assert len(samples) == 3
        assert all(isinstance(sample, Sample) for sample in samples)

        # Check first sample
        assert samples[0].query == "What is the capital of France?"
        assert samples[0].passage == "Paris is the capital of France."
        assert samples[0].groundtruth_relevance == 3

        # Check second sample
        assert samples[1].query == "How tall is Mount Everest?"
        assert samples[1].passage == "Mount Everest is 8,848.86 meters tall."
        assert samples[1].groundtruth_relevance == 1

        # Check third sample
        assert samples[2].query == "Who wrote Hamlet?"
        assert samples[2].passage == "William Shakespeare wrote Hamlet."
        assert samples[2].groundtruth_relevance == 0
