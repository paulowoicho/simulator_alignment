"""Implements simulators based on the TREMA approach in the SIGIR 2024 LLMJudge challenge.

For more details, see: https://arxiv.org/pdf/2410.14044.

Prompts are taken from https://github.com/llm4eval/LLMJudge-benchmark/blob/main/submissions/prompts/TREMA-4prompt.txt
"""

from enum import Enum
import logging
import re
from string import Template

from vllm import LLM, RequestOutput, SamplingParams

from ..data_models.sample import Sample
from .base import BaseSimulator

logger = logging.getLogger(__name__)

_STAGE_ONE_SYS_PROMPT = """
Please assess how well the provided passage meets specific criteria in relation to the query. Use the
following scoring scale (0-3) for evaluation:

0: Not relevant at all / No information provided.
1: Marginally relevant / Partially addresses the criterion.
2: Fairly relevant / Adequately addresses the criterion.
3: Highly relevant / Fully satisfies the criterion.
"""

_STAGE_ONE_USER_PROMPT = """
Please rate how well the given passage meets the $criterion criterion in relation to the query. The
output should be a single score (0-3) indicating $criterion_description.

Query: $query
Passage: $passage
$criterion:
"""

_TREM4PROMPT_STAGE_TWO_SYS_PROMPT = """
You are a search quality rater evaluating the relevance of passages. Given a query and passage, you
must provide a score on an integer scale of 0 to 3 with the following meanings:

3 = Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
2 = Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear,
or hidden amongst extraneous information.
1 = Related: The passage seems related to the query but does not answer it.
0 = Irrelevant: The passage has nothing to do with the query.

Assume that you are writing an answer to the query. If the passage seems to be related to the query
but does not include any answer to the query, mark it 1. If you would use any of the information
contained in the passage in such an answer, mark it 2. If the passage is primarily about the query, or
contains vital information about the topic, mark it 3. Otherwise, mark it 0.
"""

_TREM4PROMPT_STAGE_TWO_USER_PROMPT = """
Please rate how the given passage is relevant to the query based on the given scores. The output must
be only a score that indicates how relevant they are.

Query: $query
Passage: $passage
Exactness: $exactness_score
Topicality: $topicality_score
Coverage: $coverage_score
Contextual Fit: $contextual_fit_score
Score:
"""


class _Criteria(Enum):
    """Relevance criteria with names and descriptions."""

    EXACTNESS = ("Exactness", "How precisely does the passage answer the query.")
    COVERAGE = (
        "Coverage",
        "How much of the passage is dedicated to discussing the query and its related topics.",
    )
    TOPICALITY = (
        "Topicality",
        "Is the passage about the same subject as the whole query (not only a single word of it).",
    )
    CONTEXTUAL_FIT = ("Contextual Fit", "Does the passage provide relevant background or context.")

    @property
    def name(self):
        return self.value[0]

    @property
    def description(self):
        return self.value[1]


def _get_score(response: RequestOutput) -> int:
    """Helper method to extract the score from a model's output

    Args:
        reponse (RequestOutput): Response output from the model

    Returns:
        int: Extracted score.
    """
    m = re.search(r"\b([0-3])\b", response.outputs[0].text)
    return int(m[1]) if m else 0


def _batch_grade_criteria(inference_engine: LLM, samples: list[Sample]) -> list[dict[str, int]]:
    """Helper method to perform the first stage of Trema style assessments.

    Essentially, for every sample, it asks the model to grade the sample on 4 criteria.
    Each ask to the model is an independent request.

    Args:
        inference_engine (LLM): vLLM inference engine.
        samples (list[Sample]): Samples to grade.

    Returns:
        list[dict[str, int]]: A list of dictionaries with scores per criteria for each input sample.
    """

    criteria = list(_Criteria)
    num_criteria = len(criteria)

    batched_messages: list[list[dict[str, str]]] = []
    for sample in samples:
        batched_messages.extend(
            [
                {"role": "system", "content": _STAGE_ONE_SYS_PROMPT},
                {
                    "role": "user",
                    "content": Template(_STAGE_ONE_USER_PROMPT).substitute(
                        criterion=criterion.name,
                        criterion_description=criterion.description,
                        query=sample.query,
                        passage=sample.passage,
                    ),
                },
            ]
            for criterion in criteria
        )

    responses = inference_engine.chat(
        messages=batched_messages,  # type: ignore[arg-type]
        sampling_params=SamplingParams(temperature=0),
    )

    grades_list: list[dict[str, int]] = [{} for _ in samples]
    for idx, resp in enumerate(responses):
        sample_idx = idx // num_criteria
        criterion = criteria[idx % num_criteria]
        grades_list[sample_idx][criterion.name] = _get_score(resp)
    return grades_list


def _check_model(inference_engine: LLM) -> None:
    """Checks if model used for inference is based on meta-llama/Meta-Llama-3-8B-Instruct.

    Args:
        inference_engine (LLM): The vLLM inference engine.
    """
    if (
        inference_engine.llm_engine.get_model_config().model
        != "meta-llama/Meta-Llama-3-8B-Instruct"
    ):
        logger.warning(
            "This simulator was designed with `meta-llama/Meta-Llama-3-8B-Instruct`. Using another model may impact performance."
        )


class TREMA4Prompts(BaseSimulator):
    """This is a two stage approach to relevance assessment where:

    1) The model is prompted for criteria specific scoring
    2) Criteria evals are aggregated with another call to the model.
    """

    def __init__(self, inference_engine: LLM) -> None:
        """Creates a Trema4Prompts assessor instance

        Args:
            inference_engine (LLM): An instance of the vLLM inference engine.
        """
        _check_model(inference_engine)
        self.inference_engine = inference_engine

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        grades_list = _batch_grade_criteria(self.inference_engine, samples)

        batched_messages = [
            [
                {"role": "system", "content": _TREM4PROMPT_STAGE_TWO_SYS_PROMPT},
                {
                    "role": "user",
                    "content": Template(_TREM4PROMPT_STAGE_TWO_USER_PROMPT).substitute(
                        query=sample.query,
                        passage=sample.passage,
                        exactness_score=grades[_Criteria.EXACTNESS.name],
                        topicality_score=grades[_Criteria.TOPICALITY.name],
                        coverage_score=grades[_Criteria.COVERAGE.name],
                        contextual_fit_score=grades[_Criteria.CONTEXTUAL_FIT.name],
                    ),
                },
            ]
            for sample, grades in zip(samples, grades_list)
        ]

        responses = self.inference_engine.chat(
            messages=batched_messages,  # type: ignore[arg-type]
            sampling_params=SamplingParams(temperature=0),
        )

        for sample, resp in zip(samples, responses):
            score = _get_score(resp)
            sample.set_predicted_relevance(score)
        return samples


class TREMASumDecompose(BaseSimulator):
    """This is a two stage approach to relevance assessment where:

    1) The model is prompted for criteria specific scoring
    2) Scores from 1 are added up and mapped to a relevance score.
    """

    def __init__(self, inference_engine: LLM) -> None:
        """Creates an instance of TREMASumDecompose

        Args:
            inference_engine (LLM): An instance of vLLM's inference engine.
        """
        _check_model(inference_engine)
        self.inference_engine = inference_engine
        self._mapping = [(10, 3), (7, 2), (5, 1), (0, 0)]

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        grades_list = _batch_grade_criteria(self.inference_engine, samples)
        for sample, grades in zip(samples, grades_list):
            total = sum(grades.get(c.name, 0) for c in _Criteria)
            label = next((lbl for thresh, lbl in self._mapping if total >= thresh), 0)
            sample.set_predicted_relevance(label)
        return samples
