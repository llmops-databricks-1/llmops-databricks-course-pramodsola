"""MLflow pyfunc entry point for the ArxivAgent model (code-based logging).

Uses PythonModel with no type annotations on predict() so MLflow cannot
infer schema from type hints. The signature is set explicitly in log_model().
"""

from __future__ import annotations

import mlflow


class _AgentLoader(mlflow.pyfunc.PythonModel):
    """Lazy-loading wrapper so ArxivAgent is only instantiated at serve time.

    Import is deferred to load_context() to prevent MLflow from walking the
    ArxivAgent import graph and finding ResponsesAgentRequest type annotations
    that would override the explicitly-provided ChatCompletionRequest signature.
    """

    def load_context(self, context):
        # Deferred import — keeps module-level clean of ResponsesAgentRequest
        from arxiv_curator.agent import ArxivAgent  # noqa: PLC0415

        config = mlflow.models.ModelConfig(development_config="project_config.yml")
        self._agent = ArxivAgent(
            llm_endpoint=config.get("llm_endpoint"),
            system_prompt=config.get("system_prompt"),
            catalog=config.get("catalog"),
            schema=config.get("schema"),
            genie_space_id=config.get("genie_space_id"),
            lakebase_project_id=config.get("lakebase_project_id"),
        )

    def predict(self, context, model_input, params=None):
        # No type annotations — MLflow must use the explicit signature from log_model.
        # Returns ChatCompletionResponse format so agents.deploy() output schema check passes.
        import time

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
