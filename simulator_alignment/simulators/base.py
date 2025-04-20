from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, fields
from datetime import datetime
import logging
from pathlib import Path
from string import Template

from openai import BadRequestError, OpenAI
from retry import retry  # type: ignore[import-untyped]

from ..data_models.sample import Sample

logger = logging.getLogger(__name__)


class BaseSimulator(ABC):
    """A simulator assesses how relevant a passage is to a query."""

    def assess(
        self,
        samples: list[Sample],
        write_to_file: bool = False,
        output_path: Path | str | None = None,
    ) -> list[Sample]:
        """Template method that produces relevance assessments of query-passage pairs in a list of samples.

        Each sample is simply updated with the predicted relevance score.

        Args:
            samples (list[Sample]): A list of `data_models.sample.Sample` objects to produce
                relevance assessments for.
            write_to_file (bool): Optionally write samples to file for analysis. Defaults to False.
            output_path (Path | str | None): The path to write samples out to. Defaults to None.

        Returns:
            list[Sample]: Updated `data_models.sample.Sample` objects, with the predicted_relevance
                fields populated.
        """
        scored_samples = self._score_samples(samples)

        if write_to_file:
            self._write_to_file(scored_samples, output_path)

        return scored_samples

    @abstractmethod
    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        """Predicts the relevance of each query, passage pair (i.e Sample)

        Args:
            samples (list[Sample]): A list of `data_models.sample.Sample` objects to produce
                relevance assessments for.

        Returns:
            list[Sample]: Updated `data_models.sample.Sample` objects, with the predicted_relevance
                fields populated.
        """
        ...

    def _write_to_file(self, samples: list[Sample], output_path: Path | str | None) -> None:
        """Write out the list of scored samples as a csv.

        If output_path is None, one is auto-generated in the form: ./<ClassName>_results_2025-04-18T14-23-00.csv

        Args:
            samples (list[Sample]): A list of `data_models.sample.Sample` objects with groundtruth and predicted
                relevance assesssments
            output_path (Path | str | None): Path to write out samples to. Defaults to None.
        """
        if output_path is None:
            folder = Path.cwd() / "data" / self.__class__.__name__
            folder.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            fname = f"{self.__class__.__name__}_results_{timestamp}.tsv"
            output_path = folder / fname
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(output_path)

        fieldnames = [field.name for field in fields(Sample)]

        with output_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()

            for s in samples:
                writer.writerow(asdict(s))

    @property
    def name(self):
        """Returns the class name."""
        return self.__class__.__name__


class OpenAIPromptedSimulator(BaseSimulator):
    """Base simulator for a subset of simulators that scores samples with OpenAI's GPT40 using a
    provided prompt template.

    Subclasses should set a prompt_template attribute that formats how the prompt is presented to
    the model. They should also define a _parse_output method that extracts the score from the
    model's outputs.
    """

    _PROMPT_TEMPLATE: str

    def __init__(self, inference_engine: OpenAI, model_name: str = "gpt-4o") -> None:
        if not getattr(self, "_PROMPT_TEMPLATE", None):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define a class-level _PROMPT_TEMPLATE."
            )

        if model_name != "gpt-4o":
            logger.warning(
                "This method was developed with gpt-4o. Using another model may affect performance."
            )

        self.inference_engine = inference_engine
        self.model_name = model_name

    @retry(tries=3, delay=0.1)
    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        for sample in samples:
            user_content = Template(self._PROMPT_TEMPLATE).substitute(
                query=sample.query, passage=sample.passage
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ]

            try:
                response = self.inference_engine.chat.completions.create(
                    messages=messages,  # type: ignore[arg-type]
                    model=self.model_name,
                    max_tokens=100,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.5,
                    presence_penalty=0,
                )
                output = (
                    response.choices[0].message.content.lower()
                    if response.choices[0].message.content
                    else ""
                )
            except BadRequestError as e:
                logger.error(f"Encountered {e} for {messages}, defaulting to empty string")
                output = ""

            score = self._parse_output(output)
            sample.set_predicted_relevance(score)
        return samples

    @abstractmethod
    def _parse_output(self, text: str) -> int:
        """Helper method to parse out the relevance scores from the model's output."""
        ...
