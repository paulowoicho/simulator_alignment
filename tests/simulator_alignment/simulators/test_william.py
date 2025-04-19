from simulator_alignment.simulators.william import WilliamUmbrela1

from .test_utils import make_response


class TestWilliamUmbrela1:
    def test_william_umbrela1_assess(self, sample_list, mock_openai_engine):
        """Test that the WilliamUmbrela1 simulator parses responses correctly."""
        outputs = [
            "##o: 3",
            "##o: 2",
            "##o: 1",
            "#4",
        ]
        mock_openai_engine.chat.completions.create.side_effect = [
            make_response(o) for o in outputs
        ]

        simulator = WilliamUmbrela1(mock_openai_engine)
        result = simulator._score_samples(sample_list)

        assert [s.predicted_relevance for s in result] == [3, 2, 1, 0]
