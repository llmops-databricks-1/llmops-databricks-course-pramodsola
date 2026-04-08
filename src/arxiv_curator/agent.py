"""ArxivAgent — traced agentic loop with MCP tools and session memory."""

from __future__ import annotations

import random
from collections.abc import Iterator
from datetime import datetime
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import PythonModelContext
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI

from arxiv_curator.mcp import ToolInfo, create_mcp_tools
from arxiv_curator.memory import LakebaseMemory

GIT_SHA = "local"


class ArxivAgent(mlflow.pyfunc.PythonModel):
    """Research assistant agent with MLflow tracing, MCP tools, and session memory."""

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        catalog: str,
        schema: str,
        genie_space_id: str | None = None,
        lakebase_project_id: str | None = None,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.catalog = catalog
        self.schema = schema
        self.genie_space_id = genie_space_id
        self.lakebase_project_id = lakebase_project_id

        self.w = WorkspaceClient()
        self.tools: list[ToolInfo] = self._load_tools()
        self.memory: LakebaseMemory | None = self._init_memory()
        self.client = OpenAI(
            api_key=self._get_token(),
            base_url=f"{self.w.config.host.rstrip('/')}/serving-endpoints",
        )

    def _get_token(self) -> str:
        """Get Databricks auth token — works in serverless and classic clusters."""
        import os

        # 1. Explicit env var (CI, local dev)
        token = os.environ.get("DATABRICKS_TOKEN")
        if token:
            return token
        # 2. Databricks serverless: use SDK config which picks up OAuth credentials
        try:
            headers = self.w.config.authenticate()
            return headers.get("Authorization", "").removeprefix("Bearer ")
        except Exception:
            pass
        return ""

    def _load_tools(self) -> list[ToolInfo]:
        """Load MCP tools from Vector Search and Genie."""
        mcp_urls = [
            f"{self.w.config.host.rstrip('/')}/api/2.0/mcp/vector-search"
            f"/{self.catalog}/{self.schema}/arxiv_index",
        ]
        if self.genie_space_id:
            mcp_urls.append(
                f"{self.w.config.host.rstrip('/')}/api/2.0/mcp/genie"
                f"/{self.genie_space_id}"
            )
        return create_mcp_tools(self.w, mcp_urls)

    def _init_memory(self) -> LakebaseMemory | None:
        """Initialize session memory if project_id is configured."""
        if not self.lakebase_project_id:
            return None
        try:
            return LakebaseMemory(project_id=self.lakebase_project_id)
        except Exception:
            return None

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict[str, str]) -> str:
        """Execute a named tool and return its string output."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.exec_fn(**args)
        return f"Tool '{tool_name}' not found."

    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict]) -> object:
        """Call the LLM with the current message list."""
        tool_specs = [t.spec for t in self.tools] if self.tools else []
        kwargs: dict[str, object] = {
            "model": self.llm_endpoint,
            "messages": messages,
        }
        if tool_specs:
            kwargs["tools"] = tool_specs
        return self.client.chat.completions.create(**kwargs)

    @mlflow.trace(span_type=SpanType.CHAIN)
    def call_and_run_tools(self, messages: list[dict], max_iterations: int = 5) -> str:
        """Agentic loop: call LLM, execute tool calls, repeat until done."""
        for _ in range(max_iterations):
            response = self.call_llm(messages)
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message.model_dump(exclude_unset=True))
                for tc in choice.message.tool_calls:
                    import json

                    result = self.execute_tool(
                        tc.function.name, json.loads(tc.function.arguments)
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
            else:
                return choice.message.content or ""

        return "Max iterations reached."

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        context: PythonModelContext | None,
        model_input: ResponsesAgentRequest | dict,
        params: dict | None = None,
    ) -> ResponsesAgentResponse:
        """Handle a single agent request with full tracing."""
        if isinstance(model_input, dict):
            model_input = ResponsesAgentRequest(**model_input)

        # Extract custom inputs
        custom = model_input.custom_inputs or {}
        _ts = datetime.now().strftime("%Y%m%d%H%M%S")
        _rnd = random.randint(100000, 999999)
        session_id: str = custom.get("session_id", f"s-{_ts}-{_rnd}")
        request_id: str = custom.get("request_id", f"req-{_ts}-{_rnd}")

        # Attach trace metadata
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id},
            tags={
                "model_serving_endpoint_name": self.llm_endpoint,
                "git_sha": GIT_SHA,
            },
            client_request_id=request_id,
        )

        # Load memory
        past_messages: list[dict] = []
        if self.memory:
            past_messages = self.memory.load_messages(session_id)

        # Build message list
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += past_messages
        for msg in model_input.input:
            if hasattr(msg, "model_dump"):
                messages.append(msg.model_dump(exclude_unset=True))
            else:
                messages.append(dict(msg))

        # Run agentic loop
        assistant_response = self.call_and_run_tools(messages)

        # Save to memory
        if self.memory and model_input.input:
            new_messages = []
            for msg in model_input.input:
                if hasattr(msg, "model_dump"):
                    new_messages.append(msg.model_dump(exclude_unset=True))
                else:
                    new_messages.append(dict(msg))
            new_messages.append({"role": "assistant", "content": assistant_response})
            self.memory.save_messages(session_id, new_messages)

        return ResponsesAgentResponse(
            output=[
                {
                    "type": "message",
                    "id": str(uuid4()),
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": assistant_response,
                            "annotations": [],
                        }
                    ],
                }
            ]
        )

    def predict_stream(
        self,
        context: PythonModelContext | None,
        model_input: ResponsesAgentRequest | dict,
        params: dict | None = None,
    ) -> Iterator[ResponsesAgentStreamEvent]:
        """Streaming predict — yields a single event with the full response."""
        response = self.predict(context, model_input, params)
        yield ResponsesAgentStreamEvent(data=response)
