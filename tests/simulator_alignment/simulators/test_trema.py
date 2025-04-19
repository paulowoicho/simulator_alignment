from unittest.mock import MagicMock

import pytest

from simulator_alignment.simulators.trema import TREMA4Prompts, TREMASumDecompose

# Fake scores returned by the model: 4 samples × 4 criteria = 16 responses
FAKE_CRITERIA_RESPONSES = [
    MagicMock(outputs=[MagicMock(text="3")]),  # Sample 1
    MagicMock(outputs=[MagicMock(text="3")]),
    MagicMock(outputs=[MagicMock(text="3")]),
    MagicMock(outputs=[MagicMock(text="3")]),
    MagicMock(outputs=[MagicMock(text="2")]),  # Sample 2
    MagicMock(outputs=[MagicMock(text="2")]),
    MagicMock(outputs=[MagicMock(text="2")]),
    MagicMock(outputs=[MagicMock(text="2")]),
    MagicMock(outputs=[MagicMock(text="1")]),  # Sample 3
    MagicMock(outputs=[MagicMock(text="1")]),
    MagicMock(outputs=[MagicMock(text="1")]),
    MagicMock(outputs=[MagicMock(text="1")]),
    MagicMock(outputs=[MagicMock(text="0")]),  # Sample 4
    MagicMock(outputs=[MagicMock(text="0")]),
    MagicMock(outputs=[MagicMock(text="0")]),
    MagicMock(outputs=[MagicMock(text="0")]),
]

# Final relevance for TREMA4Prompts (stage 2): pretend the model just echoes average rounded
FAKE_STAGE2_RESPONSES = [
    MagicMock(outputs=[MagicMock(text="3")]),
    MagicMock(outputs=[MagicMock(text="2")]),
    MagicMock(outputs=[MagicMock(text="1")]),
    MagicMock(outputs=[MagicMock(text="0")]),
]


@pytest.fixture
def mock_llm():
    """Creates a mock inference engine."""
    mock_engine = MagicMock()
    # Mock model config check
    mock_engine.llm_engine.get_model_config.return_value.model = (
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )

    mock_engine.get_tokenizer.return_value = MagicMock()
    mock_engine.get_tokenizer.return_value.apply_chat_template.return_value = [
        1,
        2,
        3,
    ]  # Fake tokens
    mock_engine.llm_engine.model_config.max_model_len = 1000

    return mock_engine


class TestSimulators:
    def test_trema4prompts_assess(self, sample_list, mock_llm):
        # First batch is 16 criteria calls, then 4 final relevance calls
        mock_llm.chat.side_effect = [FAKE_CRITERIA_RESPONSES, FAKE_STAGE2_RESPONSES]

        simulator = TREMA4Prompts(mock_llm)
        result = simulator.assess(sample_list)

        # Assert that predicted relevance is as expected
        assert [s.predicted_relevance for s in result] == [3, 2, 1, 0]
        # Assert that the model was called twice (once for criteria, once for final score)
        assert mock_llm.chat.call_count == 2

    def test_tremasumdecompose_assess(self, sample_list, mock_llm):
        mock_llm.chat.return_value = FAKE_CRITERIA_RESPONSES

        simulator = TREMASumDecompose(mock_llm)
        result = simulator.assess(sample_list)

        # Assert that predicted relevance is as expected
        assert [s.predicted_relevance for s in result] == [3, 2, 0, 0]
        # Assert that the model was called only once (for the criteria)
        assert mock_llm.chat.call_count == 1
