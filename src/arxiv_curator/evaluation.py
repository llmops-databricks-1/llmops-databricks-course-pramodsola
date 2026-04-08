"""Evaluation scorers and guidelines for Arxiv Curator agent."""

from __future__ import annotations

import mlflow
from mlflow.genai.judges import Guidelines


polite_tone_guideline = Guidelines(
    name="polite_tone",
    guidelines=[
        "The response must be polite and professional.",
        "The response must not be rude, dismissive, or condescending.",
        "The response must address the user's question directly.",
    ],
)

hook_in_post_guideline = Guidelines(
    name="hook_in_post",
    guidelines=[
        "The response must start with an engaging opening sentence.",
        "The opening must clearly signal what the response is about.",
    ],
)

scope_guideline = Guidelines(
    name="scope",
    guidelines=[
        "The response must stay on topic and answer the user's question.",
        "The response must not include unrelated information.",
        "The response must be relevant to AI/ML research papers.",
    ],
)


@mlflow.genai.scorer
def word_count_check(outputs: str) -> bool:  # type: ignore[return]
    """Check that the output is under 350 words."""
    text = str(outputs)
    return len(text.split()) < 350


@mlflow.genai.scorer
def mentions_papers(outputs: str) -> bool:  # type: ignore[return]
    """Check if the response mentions any research papers."""
    text = str(outputs).lower()
    return any(
        keyword in text
        for keyword in ["paper", "arxiv", "study", "research", "authors", "published"]
    )
