import torch
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from ..data_models.sample import Sample
from .base import BaseSimulator


class MonoT5Simulator(BaseSimulator):
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        device: torch.device | None = None,
        batch_size: int = 64,
        max_score: float = 3.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._device = device or next(model.parameters()).device
        self._batch_size = batch_size
        self._max_score = max_score

        self._false_id, self._true_id = self.tokenizer(
            ["false", "true"], add_special_tokens=False
        ).input_ids

    def _score_samples(self, samples: list[Sample]) -> list[Sample]:
        scored_samples = []

        for i in range(0, len(samples), self._batch_size):
            batch = samples[i : i + self._batch_size]
            prompts = [
                f"Query: {sample.query} Document: {sample.passage} Relevant:" for sample in batch
            ]
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512,
                return_attention_mask=True,
            )
            inputs = inputs.to(self._device)

            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
            )

            if not hasattr(gen_out, "scores") or gen_out.scores is None:
                raise RuntimeError(
                    "MonoT5Reranker.generate() did not return scores; cannot compute relevance probabilities."
                )

            batch_logits = gen_out.scores[0]
            batch_logits = batch_logits[:, [self._false_id, self._true_id]]
            batch_ps = torch.softmax(batch_logits, dim=1)
            batch_ps_list = batch_ps[:, 1].tolist()
            grades = [round(p[0] * self._max_score) for p in batch_ps_list]

            for sample, log_prob in zip(batch, grades):
                sample.set_predicted_relevance(log_prob)
            scored_samples.extend(batch)

        return scored_samples
