from abc import ABC, abstractmethod
import csv
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

from ..data_models.sample import Sample


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
