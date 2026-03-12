from typing import Literal

INTENT_LABELS = Literal["factual", "summary", "comparison", "out_of_scope"]

# Keywords for each intent
INTENT_MAP = {
    "summary": [
        "summarize", "summary", "overview", "briefly describe",
        "give me an overview", "tldr", "main points", "key points",
        "what is this document about", "what does this paper cover"
    ],
    "comparison": [
        "compare", "contrast", "difference between", "similarities",
        "versus", "vs", "how does x differ", "which is better",
        "compare and contrast", "what are the differences"
    ],
    "out_of_scope": [
        "weather", "colour of sky", "color of sky", "who is the president",
        "what is the capital", "sports", "recipe", "joke", "sing",
        "write a poem", "tell me a story"
    ]
}


def classify_intent(question: str) -> str:
    """
    Classify question intent using keyword matching.
    Returns: factual | summary | comparison | out_of_scope
    No LLM needed — deterministic and fast.
    """
    q = question.lower().strip()

    # Check out_of_scope first — fastest exit
    for keyword in INTENT_MAP["out_of_scope"]:
        if keyword in q:
            return "out_of_scope"

    # Check summary intent
    for keyword in INTENT_MAP["summary"]:
        if keyword in q:
            return "summary"

    # Check comparison intent
    for keyword in INTENT_MAP["comparison"]:
        if keyword in q:
            return "comparison"

    # Default — treat as factual
    return "factual"