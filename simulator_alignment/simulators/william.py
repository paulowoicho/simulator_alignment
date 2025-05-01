import re

from .base import OpenAIPromptedSimulator

_USER_PROMPT = """
Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query, 
1 = represents that the passage seems related to the query but does not answer it, 
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: $query
Passage: $passage

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
"""


class WilliamUmbrela1(OpenAIPromptedSimulator):
    """This approach prompts a model for relevance using the UMBRELA framework
    (https://arxiv.org/abs/2406.06519) based on the Bing relevance
    assessor technique (https://arxiv.org/abs/2309.10621)."""

    _PROMPT_TEMPLATE = _USER_PROMPT

    def _parse_output(self, text: str) -> int:
        match = re.search(r"##final score:\s*(\d+)", text.strip())
        return int(match[1]) if match else 0


class WilliamUmbrelaGPT4oMini(WilliamUmbrela1): ...
