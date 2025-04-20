import re

from .base import OpenAIPromptedSimulator

_USER_PROMPT = """
Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query, 
1 = represents that the passage seems related to the query but does not answer it, 
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Following are some of the examples of relevance categorizations for different query-passage pairs:

###
Query: what did colonial women wear?
Passage: Food historians tell us the first colonial/American cookbooks were not published until the late 18th century. 17th century Salem, MA/Puritan cooks would have been using (if they used at all) English cookbooks. The recipes would have been quite similar to what was prepared and eaten during Shakespeare's time .
##final score: 0
###

###
Query: what did colonial women wear?
Passage: What Did Renaissance Women Wear ? During the Renaissance, women wear was influenced by Italian and Germanic designs. The women's clothing was wide, barrel-shaped or square in shape. However, clothing was considered to be a symbol of status, and the wealthy were known to spend a lot of money on clothing, and were not averse to showing off their wealth via their clothes.
##final score: 0
###

###
Query: what did colonial women wear?
Passage: They protected tresses from grime and were more convenient to wash than the hair itself. Worn only indoors, they were covered by structured bonnets when women went out in public. In colonial America, mob caps were worn by all women, but the aristocratic versions were sometimes pleated and included bows.
##final score: 1
###

###
Query: what did colonial women wear?
Passage: As illustrated in her journal, Dress, the Jenness Miller Magazine, this system was similar to Kelloggâ€™s and included, leglettes and chemilettes, to replace petticoats, and a model bodice, to replace the corset.. The Jenness-Miller system also included a bosom support for stout women, a garment similar to a brassiere.
##final score: 1
###

###
Query: what did colonial women wear?
Passage: For many centuries women wore a variety of head-coverings which were called caps. For example, in the 18th and 19th centuries a cap was a kind of head covering made of a flimsy fabric such as muslin; it was worn indoors or under a bonnet by married women, or older unmarried women who were "on the shelf" (e.g. mob-cap ).
##final score: 2
###

###
Query: what did colonial women wear?
Passage: Spanish colonial era. Mestizos de Manila by Juan Ravenet showing the checkered narrow pares saya of native women in the 18th century Philippines. Also note the European-style clothing of the men. (c.1792-1794) Filipina, a 19th-century painting of a working-class woman in baro't saya by FabiÃ¡n de la Rosa. The Spanish clergy during the colonial period deemed the precolonial mode of dress as immodest for women and introduced the long skirt (known by the Spanish name saya or falda) to be worn under the tapis.
##final score: 3
###

Query: $query
Passage: $passage

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
"""


class H2looFewself(OpenAIPromptedSimulator):
    """This approach prompts a model for relevance using the UMBRELA framework
    (https://arxiv.org/abs/2406.06519) based on the Bing relevance
    assessor technique (https://arxiv.org/abs/2309.10621). The prompt utilises
    in-context examples.
    """

    _PROMPT_TEMPLATE = _USER_PROMPT

    def _parse_output(self, text: str) -> int:
        if match := re.search(r"final score:\s*(\d+)", text):
            return int(match[1])
        return 0
