from unittest.mock import MagicMock

import pytest

from simulator_alignment.data_models.sample import Sample


@pytest.fixture
def sample_list():
    """Fixture to create sample list for testing, shared across test classes"""
    return [
        Sample(query="What is Paris?", passage="Paris is in France", groundtruth_relevance=3),
        Sample(query="How tall is Everest?", passage="Everest is tall", groundtruth_relevance=2),
        Sample(query="Who is Shakespeare?", passage="A famous writer", groundtruth_relevance=1),
        Sample(query="What is Python?", passage="Unrelated text", groundtruth_relevance=0),
    ]


@pytest.fixture
def mock_openai_engine():
    """Creates a mock inference engine."""
    return MagicMock()
