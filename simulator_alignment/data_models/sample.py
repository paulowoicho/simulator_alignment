from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sample:
    """Represents a single sample in the dataset.

    Holds the query, passage, groundtruth relevance, and predicted relevance.
    """

    query: str
    passage: str
    groundtruth_relevance: int
    predicted_relevance: int | None = field(default=None, init=False)

    def set_predicted_relevance(self, value: int | None) -> None:
        """Bypasses class's frozen attribute by using object.__setattr__ to set predicted_releance.

        Args:
            value (int | None): The value to set for predicted_relevance.
        """
        object.__setattr__(self, "predicted_relevance", value)
