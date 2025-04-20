from .base import OpenAIPromptedSimulator

_USER_PROMPT = """
# Task
You are an expert in information retrieval. Your task is to judge the relevance
of a document to a query.

- Perfectly Relevant: The document meets the highest standard of relevance. It
  is specifically focused on the query and provides a complete, precise, and
  direct answer without needing further information. Every part of the document
  is dedicated to addressing the query, leaving no ambiguity about its
  relevance.
- Highly Relevant: This document addresses the query effectively but not as
  completely or directly as a "perfectly relevant" document. It includes useful
  and pertinent information related to the query. However, the answer might be
  embedded within additional content or presented in a less straightforward
  manner. The document should still provide a clear response to the query, even
  if the user needs to infer some details.
- Related Document: This document is relevant to the topic of the query but does
  not directly answer it. While the content is connected to the query's subject,
  it lacks the specific information needed to fully satisfy the query. It might
  provide context, background, or ancillary information that could be useful for
  understanding the query, but it does not fulfill the query's requirement for a
  direct answer.
- Irrelevant: The document does not relate to the query in any meaningful
  way. It lacks any significant connection to the query’s topic or content. This
  label is for documents that are off-topic or contain information that is
  unrelated to what the query is seeking.

# Output format
Answer "perfectly relevant", "highly relevant", "related", or "irrelevant" as labels

# Prediction
Query: $query
Doc: $passage
Label:
"""


class OlzGPT4o(OpenAIPromptedSimulator):
    """This approach prompts a model for relevance using a simple prompt."""

    _PROMPT_TEMPLATE = _USER_PROMPT

    def _parse_output(self, text: str) -> int:
        output_to_score = {
            "perfectly relevant": 3,
            "highly relevant": 2,
            "related": 1,
            "irrelevant": 0,
        }

        return output_to_score.get(text.strip().lower(), 0)
