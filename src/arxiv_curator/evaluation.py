"""Evaluation scorers and guidelines for Arxiv Curator agent."""

from __future__ import annotations

import mlflow
from mlflow.genai.judges import make_judge

from arxiv_curator.config import ProjectConfig

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


def evaluate_agent(
    cfg: ProjectConfig,
    eval_inputs_path: str,
    max_questions: int = 3,
) -> mlflow.models.EvaluationResult:
    """Run evaluation on the ArxivAgent using code-based scorers.

    Args:
        cfg: Project configuration.
        eval_inputs_path: Path to file with one evaluation question per line.
        max_questions: Number of questions to evaluate (default 3 for speed).

    Returns:
        MLflow EvaluationResult with metrics.
    """
    from databricks.sdk import WorkspaceClient

    from arxiv_curator.agent import ArxivAgent

    w = WorkspaceClient()
    _user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "-")

    agent = ArxivAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=cfg.system_prompt,
        catalog=cfg.catalog,
        schema=cfg.schema,
        genie_space_id=None,  # skip Genie for faster eval
        lakebase_project_id=cfg.lakebase_project_id or f"{_user_prefix}-lakebase",
    )

    with open(eval_inputs_path) as f:
        eval_data = [
            {"inputs": {"question": line.strip()}}
            for line in f
            if line.strip()
        ][:max_questions]

    def predict_fn(question: str) -> str:
        request = {"input": [{"role": "user", "content": question}]}
        result = agent.predict(context=None, model_input=request)
        return result.output[-1].content[0]["text"]

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=[word_count_check, mentions_papers],
    )
