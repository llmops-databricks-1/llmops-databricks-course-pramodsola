"""Evaluation scorers and guidelines for Arxiv Curator agent."""

from __future__ import annotations

import mlflow
from mlflow.genai.judges import make_judge

polite_tone_guideline = make_judge(
    name="polite_tone",
    instructions=(
        "Check if {{ outputs }} meets all guidelines:\n"
        "- The response must be polite and professional.\n"
        "- The response must not be rude, dismissive, or condescending.\n"
        "- The response must address the user's question directly."
    ),
)

hook_in_post_guideline = make_judge(
    name="hook_in_post",
    instructions=(
        "Check if {{ outputs }} meets all guidelines:\n"
        "- The response must start with an engaging opening sentence.\n"
        "- The opening must clearly signal what the response is about."
    ),
)

scope_guideline = make_judge(
    name="scope",
    instructions=(
        "Check if {{ outputs }} meets all guidelines:\n"
        "- The response must stay on topic and answer the user's question.\n"
        "- The response must not include unrelated information.\n"
        "- The response must be relevant to AI/ML research papers."
    ),
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
