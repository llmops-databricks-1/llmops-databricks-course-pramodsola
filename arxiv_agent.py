"""MLflow pyfunc entry point for the ArxivAgent model (code-based logging)."""

from __future__ import annotations

import mlflow

from arxiv_curator.agent import ArxivAgent


class _AgentLoader(mlflow.pyfunc.PythonModel):
    """Lazy-loading wrapper so ArxivAgent is only instantiated at serve time."""

    def load_context(self, context: object) -> None:
        config = mlflow.models.ModelConfig(development_config="project_config.yml")
        self._agent = ArxivAgent(
            llm_endpoint=config.get("llm_endpoint"),
            system_prompt=config.get("system_prompt"),
            catalog=config.get("catalog"),
            schema=config.get("schema"),
            genie_space_id=config.get("genie_space_id"),
            lakebase_project_id=config.get("lakebase_project_id"),
        )

    def predict(
        self,
        context: object,
        model_input: object,
        params: dict | None = None,
    ) -> object:
        return self._agent.predict(context, model_input, params)


mlflow.models.set_model(_AgentLoader())
