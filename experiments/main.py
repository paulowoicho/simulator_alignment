import datetime
import json
import logging
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import tqdm  # type: ignore[import-untyped]

from simulator_alignment.data_models.evaluation_output import EvaluationOutput
from simulator_alignment.dataloaders.base import BaseDataloader
from simulator_alignment.dataloaders.llm_judge_benchmark import LLMJudgeBenchmark
from simulator_alignment.evaluator import evaluate
from simulator_alignment.simulators.base import BaseSimulator
from simulator_alignment.simulators.dummy import CopyCatAssessor, RandomAssessor

logger = logging.getLogger(__name__)


class Result(TypedDict):
    simulator: str
    dataloader: str
    metrics: list[EvaluationOutput]


class ExperimentManager:
    """Runs one or more simulators against one or more datasets and returns metrics and charts that compare the simulators."""

    def __init__(
        self,
        simulators: list[BaseSimulator],
        dataloaders: list[BaseDataloader],
        splits: int | list[int] = 1,
    ) -> None:
        """Creates an ExperimentManager instance.

        Args:
            simulators (list[BaseSimulator]): The simulators to compare.
            dataloaders (list[BaseDataloader]): The datasets to compare the simulators on.
            splits (int | list[int], optional): Number of folds to use per dataloader.
                - If an integer is provided, *every* dataloader uses that many folds.
                - If a sequence of ints is provided, its length must equal the number of dataloaders,
                    and each entry specifies the folds for the corresponding dataloader in order.
                Defaults to 1 (i.e. one fold per dataloader, no split).
        """
        self.simulators = simulators
        self.dataloaders = dataloaders

        if isinstance(splits, int):
            self._splits = [splits] * len(dataloaders)
        elif len(splits) == len(dataloaders):
            self._splits = list(splits)
        else:
            raise ValueError(f"Expected {len(dataloaders)} fold‐counts but got {len(splits)}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = Path.cwd() / "reports" / f"experiment_results_{timestamp}"

        (self.output_path / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_path / "data").mkdir(parents=True, exist_ok=True)

    def run_experiments(self) -> None:
        """Runs the experiment suite and creates artefacts for analysis.

        Every simulator is run on every dataset.
        """
        results: list[Result] = []
        for simulator in tqdm.tqdm(self.simulators):
            for idx, dataloader in enumerate(self.dataloaders):
                logger.info(f"Running {simulator.name} with {dataloader.name}")

                # Averaging over multiple splits might introduce bias, so just do one split.
                metrics = evaluate(simulator, dataloader, num_folds=self._splits[idx])

                result = Result(
                    simulator=simulator.name,
                    dataloader=dataloader.name,
                    metrics=metrics,
                )

                results.append(result)

        self._save_results(results=results)
        results_df = self._results_to_dataframe(results)
        self._generate_comparison_plots(results_df)

    def _save_results(self, results: list[Result]):
        """Saves results as a jsonfile

        Args:
            results (list[Result]): A list of result objects to write out to a file.
        """
        json_path = f"{self.output_path}/data/results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4, default=str)

        logger.info(f"Results saved to {json_path}")

    @staticmethod
    def _results_to_dataframe(results: list[Result]) -> pd.DataFrame:
        """Converts a list of Result objects to a pandas Dataframe for easy manipulation.

        Args:
            results (list[Result]): A list of result objects to write out to a pandas dataframe.

        Returns:
            pd.DataFrame: The output dataframe.
        """
        rows = []
        for result in results:
            row = {"simulator": result["simulator"], "dataloader": result["dataloader"]}
            # Flatten metrics
            for idx, metrics in enumerate(result["metrics"], start=1):
                for metric_name, scores in metrics.items():
                    row[f"split_{idx}_{metric_name}"] = scores["score"]  # type: ignore[index]
            rows.append(row)

        return pd.DataFrame(rows)

    def _generate_comparison_plots(self, results_df: pd.DataFrame):
        """Given a pandas dataframe, creates several bar plots to compare alignment metrics.

        Args:
            results_df (pd.DataFrame): Dataframe with results.
        """
        metric_columns = [
            col for col in results_df.columns if col not in ["simulator", "dataloader"]
        ]

        for metric in metric_columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            pivot_df = results_df.pivot(index="dataloader", columns="simulator", values=metric)

            pivot_df.plot(kind="bar", ax=ax)
            ax.set_title(f"Comparison of {metric}")
            ax.set_ylabel(metric)
            ax.set_xlabel("Dataset")
            ax.legend(title="Simulator")

            plt.xticks(rotation=45, ha="right")

            plot_path = f"{self.output_path}/plots/{metric}_comparison.png"
            fig.tight_layout()
            fig.savefig(plot_path)
            plt.close(fig)


if __name__ == "__main__":
    simulators = [
        RandomAssessor(),
        CopyCatAssessor(),
        RandomAssessor(min_val=1, max_val=2),
        RandomAssessor(min_val=3, max_val=4),
    ]

    dataloaders: list[BaseDataloader] = [
        LLMJudgeBenchmark(
            qrels_path="data/llm_judge_benchmark/llm4eval_dev_qrel_2024.txt",
            queries_path="data/llm_judge_benchmark/llm4eval_query_2024.txt",
            passages_path="data/llm_judge_benchmark/llm4eval_document_2024.jsonl",
        )
    ]

    manager = ExperimentManager(simulators, dataloaders)
    manager.run_experiments()
