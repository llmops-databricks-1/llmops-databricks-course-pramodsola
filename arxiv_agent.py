"""MLflow pyfunc entry point for the ArxivAgent model."""

from __future__ import annotations

from arxiv_curator.agent import ArxivAgent


def load_context(context: object) -> ArxivAgent:  # type: ignore[return]
    """Load and return the ArxivAgent from model config."""
    import mlflow

    model_config = mlflow.models.ModelConfig(development_config="project_config.yml")

    return ArxivAgent(
        llm_endpoint=model_config.get("llm_endpoint"),
        system_prompt=model_config.get("system_prompt"),
        catalog=model_config.get("catalog"),
        schema=model_config.get("schema"),
        genie_space_id=model_config.get("genie_space_id"),
        lakebase_project_id=model_config.get("lakebase_project_id"),
    )
