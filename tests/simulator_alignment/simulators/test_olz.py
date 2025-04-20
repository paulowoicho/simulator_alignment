from simulator_alignment.simulators.olz import OlzGPT4o

from .test_utils import make_response


class TestOlzGPT4o:
    def test_olz_gpt4o_assess(self, sample_list, mock_openai_engine):
        """Test that the OlzGPT4o simulator parses responses correctly."""
        outputs = ["perfectly relevant", "highly relevant", "related", "irrele"]
        mock_openai_engine.chat.completions.create.side_effect = [
            make_response(o) for o in outputs
        ]

        simulator = OlzGPT4o(mock_openai_engine)
        result = simulator.assess(sample_list)

        assert [s.predicted_relevance for s in result] == [3, 2, 1, 0]
