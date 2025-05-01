from typing import TypedDict


class Score(TypedDict):
    score: float


class ScoreWithPValue(TypedDict):
    score: float
    p_value: float


class EvaluationOutput(TypedDict):
    accuracy: Score
    cohen_kappa: Score
    weighted_cohen_kappa: Score
    krippendorff_alpha: Score
    spearman_correlation: ScoreWithPValue
    kendall_tau: ScoreWithPValue
