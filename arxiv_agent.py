"""MLflow pyfunc entry point for the ArxivAgent model (code-based logging).

Uses PythonModel with no type annotations on predict() so MLflow cannot
infer schema from type hints. The signature is set explicitly in log_model().
Initialization is fully lazy — _agent is created on first predict() call so
that any startup errors surface as traced prediction errors, not silent
container load failures.
"""

from __future__ import annotations

import mlflow


class _AgentLoader(mlflow.pyfunc.PythonModel):
    """Lazy-loading wrapper — ArxivAgent is created on first predict() call."""

    def load_context(self, context):
        # context.model_config is the dict passed to log_model(model_config=...)
        # and is always available in model serving — unlike ModelConfig() which
        # looks for project_config.yml on disk (absent in the serving container).
        self._model_config = context.model_config or {}
        self._agent = None

    def _ensure_agent(self):
        if self._agent is not None:
            return
        # Deferred import — keeps module-level clean of ResponsesAgentRequest
        from arxiv_curator.agent import ArxivAgent  # noqa: PLC0415

        self._agent = ArxivAgent(
            llm_endpoint=self._model_config.get("llm_endpoint"),
            system_prompt=self._model_config.get("system_prompt"),
            catalog=self._model_config.get("catalog"),
            schema=self._model_config.get("schema"),
            genie_space_id=self._model_config.get("genie_space_id"),
            lakebase_project_id=self._model_config.get("lakebase_project_id"),
        )

    def predict(self, context, model_input, params=None):
        # No type annotations — MLflow must use the explicit signature from log_model.
        # Returns ChatCompletionResponse format so agents.deploy() output schema check passes.
        import time

        self._ensure_agent()
        response = self._agent.predict(context, model_input, params)

        # Extract text from ResponsesAgentResponse
        text = ""
        if response.output:
            content = response.output[-1].content
            if content:
                item = content[0]
                text = item.get("text", "") if isinstance(item, dict) else str(item)

        return {
            "id": f"chatcmpl-agent-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._agent.llm_endpoint,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


mlflow.models.set_model(_AgentLoader())
