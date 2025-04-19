"""Implements simulators based on the TREMA approach in the SIGIR 2024 LLMJudge challenge.

For more details, see: https://arxiv.org/pdf/2410.14044.

Prompts are taken from https://github.com/llm4eval/LLMJudge-benchmark/blob/main/submissions/prompts/TREMA-4prompt.txt
"""

from enum import Enum
import logging
import re
from string import Template
from typing import Any

from vllm import LLM, RequestOutput, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer

from ..data_models.sample import Sample
from .base import BaseSimulator

logger = logging.getLogger(__name__)


class Criterion(Enum):
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
    def name(self) -> str:
        return self.value[0]

    @property
    def description(self) -> str:
        return self.value[1]


class PromptTemplates:
    """Centralized location for all prompt templates used in TREMA simulation."""

    STAGE_ONE_SYS_PROMPT = """
Please assess how well the provided passage meets specific criteria in relation to the query. Use the
following scoring scale (0-3) for evaluation:

0: Not relevant at all / No information provided.
1: Marginally relevant / Partially addresses the criterion.
2: Fairly relevant / Adequately addresses the criterion.
3: Highly relevant / Fully satisfies the criterion.
"""

    STAGE_ONE_USER_PROMPT = """
Please rate how well the given passage meets the $criterion criterion in relation to the query. The
output should be a single score (0-3) indicating $criterion_description.

Query: $query
Passage: $passage
$criterion:
"""

    STAGE_TWO_SYS_PROMPT = """
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

    STAGE_TWO_USER_PROMPT = """
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


class TREMAUtils:
    """Utility methods shared across TREMA implementations."""

    @staticmethod
    def extract_score(response: RequestOutput) -> int:
        """Extract the score from a model's output.

        Args:
            response (RequestOutput): Response output from the model

        Returns:
            int: Extracted score (0-3)
        """
        match = re.search(r"\b([0-3])\b", response.outputs[0].text)
        return int(match[1]) if match else 0

    @staticmethod
    def check_model(inference_engine: LLM) -> None:
        """Checks if model used for inference is based on meta-llama/Meta-Llama-3-8B-Instruct.

        Args:
            inference_engine (LLM): The vLLM inference engine.
        """
        expected_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        if inference_engine.llm_engine.get_model_config().model != expected_model:
            logger.warning(
                f"This simulator was designed with `{expected_model}`. "
                "Using another model may impact performance."
            )

    @staticmethod
    def create_tokenization_safe_messages(
        template: str,
        substitutions: dict[str, Any],
        sys_prompt: str,
        tokenizer: AnyTokenizer,
        max_model_len: int,
    ) -> list[dict[str, str]]:
        """Creates messages that fit within model token limits by truncating passage if needed.

        Args:
            template (str): The prompt template to use
            substitutions (Dict[str, Any]): Key-value pairs for template substitution
            sys_prompt (str): System prompt content
            tokenizer (AnyTokenizer): The model's tokenizer
            max_model_len (int): The model's tokenizer

        Returns:
            List[Dict[str, str]]: A list of message dictionaries suitable for model input
        """
        message = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": Template(template).substitute(**substitutions)},
        ]

        message_tokens = tokenizer.apply_chat_template(message, tokenize=True)  # type: ignore[union-attr, arg-type]

        # If message fits within limit, return as is
        if len(message_tokens) <= max_model_len:
            return message

        # Otherwise truncate the passage to fit
        passage = substitutions.get("passage", "")
        passage_tokens = tokenizer(passage).input_ids  # type: ignore[union-attr, operator]

        while len(message_tokens) > max_model_len and passage_tokens:
            excess = len(message_tokens) - max_model_len
            passage_tokens = passage_tokens[:-excess]
            updated_passage = tokenizer.decode(passage_tokens)  # type: ignore[union-attr]

            substitutions["passage"] = updated_passage

            message = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": Template(template).substitute(**substitutions)},
            ]
            message_tokens = tokenizer.apply_chat_template(message, tokenize=True)  # type: ignore[union-attr, arg-type]

        if not passage_tokens:
            raise RuntimeError(
                "Prompt is bigger than model's context window. Consider initialising inference engine with a higher `max_model_len`"
            )

        return message


class TREMAEvaluator:
    """Base class for TREMA evaluation strategies."""

    def __init__(self, inference_engine: LLM):
        """Initialize the TREMA evaluator.

        Args:
            inference_engine (LLM): An instance of the vLLM inference engine
        """
        TREMAUtils.check_model(inference_engine)
        self.inference_engine = inference_engine
        self.tokenizer = self.inference_engine.get_tokenizer()
        self.max_model_len = (
            self.inference_engine.llm_engine.model_config.max_model_len - 64
        )  # Buffer for tokens added from chat template.

    def batch_grade_criteria(self, samples: list[Sample]) -> list[dict[str, int]]:
        """Grade each sample on all criteria.

        Args:
            samples (list[Sample]): List of samples to evaluate

        Returns:
            list[dict[str, int]]: List of dictionaries with scores per criterion for each sample
        """
        criteria = list(Criterion)
        num_criteria = len(criteria)

        batched_messages = []

        # Create messages for each sample and criterion combination
        for sample in samples:
            for criterion in criteria:
                substitutions = {
                    "criterion": criterion.name,
                    "criterion_description": criterion.description,
                    "query": sample.query,
                    "passage": sample.passage,
                }

                message = TREMAUtils.create_tokenization_safe_messages(
                    template=PromptTemplates.STAGE_ONE_USER_PROMPT,
                    substitutions=substitutions,
                    sys_prompt=PromptTemplates.STAGE_ONE_SYS_PROMPT,
                    tokenizer=self.tokenizer,
                    max_model_len=self.max_model_len,
                )

                batched_messages.append(message)

        # Get responses for all criteria evaluations
        responses = self.inference_engine.chat(
            messages=batched_messages,  # type: ignore[arg-type]
            sampling_params=SamplingParams(temperature=0),
        )

        # Organize results by sample
        grades_list: list[dict] = [{} for _ in samples]
        for idx, response in enumerate(responses):
            sample_idx = idx // num_criteria
            criterion = criteria[idx % num_criteria]
            grades_list[sample_idx][criterion.name] = TREMAUtils.extract_score(response)

        return grades_list


class TREMA4Prompts(BaseSimulator):
    """Two-stage approach to relevance assessment where:

    1) The model is prompted for criteria-specific scoring
    2) Criteria evals are aggregated with another model call
    """

    def __init__(self, inference_engine: LLM) -> None:
        """Create a TREMA4Prompts assessor instance.

        Args:
            inference_engine (LLM): An instance of the vLLM inference engine
        """
        self.evaluator = TREMAEvaluator(inference_engine)

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        """Score samples using the two-stage TREMA 4 Prompt approach.

        Args:
            samples (list[Sample]): Samples to score

        Returns:
            list[Sample]: The same samples with predicted relevance scores set
        """
        # Stage 1: Get criteria-specific scores
        grades_list = self.evaluator.batch_grade_criteria(samples)

        # Stage 2: Aggregate scores via second model call
        batched_messages = []

        for sample, grades in zip(samples, grades_list):
            substitutions = {
                "query": sample.query,
                "passage": sample.passage,
                "exactness_score": grades[Criterion.EXACTNESS.name],
                "topicality_score": grades[Criterion.TOPICALITY.name],
                "coverage_score": grades[Criterion.COVERAGE.name],
                "contextual_fit_score": grades[Criterion.CONTEXTUAL_FIT.name],
            }

            message = TREMAUtils.create_tokenization_safe_messages(
                template=PromptTemplates.STAGE_TWO_USER_PROMPT,
                substitutions=substitutions,
                sys_prompt=PromptTemplates.STAGE_TWO_SYS_PROMPT,
                tokenizer=self.evaluator.tokenizer,
                max_model_len=self.evaluator.max_model_len,
            )

            batched_messages.append(message)

        responses = self.evaluator.inference_engine.chat(
            messages=batched_messages,  # type: ignore[arg-type]
            sampling_params=SamplingParams(temperature=0),
        )

        # Set predicted relevance for each sample
        for sample, response in zip(samples, responses):
            score = TREMAUtils.extract_score(response)
            sample.set_predicted_relevance(score)

        return samples


class TREMASumDecompose(BaseSimulator):
    """Two-stage approach to relevance assessment where:

    1) The model is prompted for criteria-specific scoring
    2) Scores are summed and mapped to a relevance score based on thresholds
    """

    def __init__(self, inference_engine: LLM) -> None:
        """Create a TREMASumDecompose assessor instance.

        Args:
            inference_engine (LLM): An instance of the vLLM inference engine
        """
        self.evaluator = TREMAEvaluator(inference_engine)
        # Mapping from total criterion score to relevance rating
        # Format: (threshold, label)
        self._mapping = [(10, 3), (7, 2), (5, 1), (0, 0)]

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        """Score samples by summing criteria scores and mapping to relevance.

        Args:
            samples (List[Sample]): Samples to score

        Returns:
            List[Sample]: The same samples with predicted relevance scores set
        """
        # Get criteria-specific scores
        grades_list = self.evaluator.batch_grade_criteria(samples)

        # Calculate final score based on sum of criteria scores
        for sample, grades in zip(samples, grades_list):
            total = sum(grades.get(criterion.name, 0) for criterion in Criterion)

            # Find appropriate label based on thresholds
            label = next((label for threshold, label in self._mapping if total >= threshold), 0)
            sample.set_predicted_relevance(label)

        return samples
