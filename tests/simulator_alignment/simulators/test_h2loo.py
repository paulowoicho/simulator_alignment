from simulator_alignment.simulators.h2loo import H2looFewself

from .test_utils import make_response


class TestH2looFewself:
    def test_h2loo_fewself_assess(self, sample_list, mock_openai_engine):
        outputs = [
            "##final score: 3",
            "##final score: 2",
            "##final score: 1",
            "##final score: david",
        ]
        mock_openai_engine.chat.completions.create.side_effect = [
            make_response(o) for o in outputs
        ]

        simulator = H2looFewself(mock_openai_engine)
        result = simulator.assess(sample_list)

        # Assert that predicted relevance matches the expected outcomes
        assert [s.predicted_relevance for s in result] == [3, 2, 1, 0]
