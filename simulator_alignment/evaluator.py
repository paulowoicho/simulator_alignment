from itertools import islice
from math import ceil
from typing import Iterator, cast

import krippendorff  # type: ignore[import-untyped]
from scipy.stats import kendalltau, spearmanr  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, cohen_kappa_score  # type: ignore[import-untyped]

from .data_models.evaluation_output import EvaluationOutput, Score, ScoreWithPValue
from .data_models.sample import Sample
from .dataloaders.base import BaseDataloader
from .simulators.base import BaseSimulator


def _compute_metrics(samples: list[Sample]) -> EvaluationOutput:
    """Computes inter annotator, accuracy, and correlation metrics on assessed Samples.

    Args:
        samples (list[Sample]): A list of samples to compute metrics on.

    Returns:
        EvaluationOutput: A dictionary with all computed metrics.
    """
    groundtruth = [sample.groundtruth_relevance for sample in samples]
    predictions = cast(list[int], [sample.predicted_relevance for sample in samples])

    accuracy = Score(score=accuracy_score(groundtruth, predictions))

    weighted_kappa = Score(score=cohen_kappa_score(groundtruth, predictions, weights="quadratic"))
    cohen_kappa = Score(score=cohen_kappa_score(groundtruth, predictions))

    krippendorff_alpha = Score(score=krippendorff.alpha([groundtruth, predictions]))

    spearman_stat, spearman_p = spearmanr(groundtruth, predictions)
    spearman_correlation = ScoreWithPValue(score=spearman_stat, p_value=spearman_p)

    kendall_stat, kendall_p = kendalltau(groundtruth, predictions)
    kendall_tau = ScoreWithPValue(score=kendall_stat, p_value=kendall_p)

    return EvaluationOutput(
        accuracy=accuracy,
        weighted_cohen_kappa=weighted_kappa,
        cohen_kappa=cohen_kappa,
        spearman_correlation=spearman_correlation,
        krippendorff_alpha=krippendorff_alpha,
        kendall_tau=kendall_tau,
    )


def _chunk(samples: Iterator[Sample], size: int) -> Iterator[list[Sample]]:
    """Yield successive chunks of at most size `size` from `samples`."""
    it = iter(samples)
    while chunk := list(islice(it, size)):
        yield chunk


def evaluate(
    simulator: BaseSimulator, dataloader: BaseDataloader, num_folds: int = 1
) -> list[EvaluationOutput]:
    """Split the data into `num_folds` batches, run `simulator.assess` on each batch,
        compute metrics, and return the list of results.

    Args:
        simulator (BaseSimulator): A child of BaseSimulator with an
            `assess(samples: list[Sample])` method.
        dataloader (BaseDataloader): A child of BaseDataloader exposing
            `num_samples` and a `.get_sample()` iterator.
        num_folds (int, optional): Number of folds to split the data into.
            Defaults to 1.

    Returns:
        EvaluationOutput: A list of EvaluationOutput, one per fold.
    """
    fold_size = ceil(dataloader.num_samples / num_folds)

    results: list[EvaluationOutput] = []
    for batch in _chunk(dataloader.get_sample(), fold_size):
        assessed = simulator.assess(samples=batch)
        metrics = _compute_metrics(assessed)
        results.append(metrics)

    return results
