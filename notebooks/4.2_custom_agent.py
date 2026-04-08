# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.20.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 4.2: Custom Agent with Tracing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Integrating tracing into agents
# MAGIC - Tracing LLM calls, tool executions, and the agentic loop
# MAGIC - Session and request tracking
# MAGIC - End-to-end agent tracing with ArxivAgent
# MAGIC - Multi-turn conversation with persistent memory

# COMMAND ----------

import random
from datetime import datetime
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

mlflow.set_experiment(cfg.experiment_name)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Agent Architecture
# MAGIC
# MAGIC ```
# MAGIC User Request
# MAGIC     ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │  @mlflow.trace(AGENT)  predict()             │
# MAGIC │    ├─ Update trace metadata                   │
# MAGIC │    │  (session_id, request_id, git_sha)       │
# MAGIC │    ├─ Load session memory (past messages)     │
# MAGIC │    ├─ call_and_run_tools()  @CHAIN            │
# MAGIC │    │    ├─ call_llm()  @LLM                   │
# MAGIC │    │    └─ execute_tool()  @TOOL  (loop)      │
# MAGIC │    └─ Save new messages to memory             │
# MAGIC └──────────────────────────────────────────────┘
# MAGIC     ↓
# MAGIC Response + Complete Trace in MLflow UI
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create ArxivAgent

# COMMAND ----------

_user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "-")

agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
    lakebase_project_id=f"{_user_prefix}-lakebase",
)

logger.info("✓ ArxivAgent created")
logger.info(f"  LLM endpoint: {cfg.llm_endpoint}")
logger.info(f"  Tools loaded: {len(agent.tools)}")
logger.info(f"  Memory: {'enabled' if agent.memory else 'disabled'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Single-Turn Request with Tracing

# COMMAND ----------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

request = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Find papers about transformers and attention mechanisms"}],
    custom_inputs={"session_id": session_id, "request_id": request_id},
)

logger.info(f"Session ID: {session_id}")

response = agent.predict(context=None, model_input=request)
logger.info(f"Agent: {response.output[-1]['content'][:300]}...")
logger.info("✓ Trace created — check MLflow UI.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Multi-Turn Conversation

# COMMAND ----------

conv_session = f"s-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}"
logger.info(f"Conversation session: {conv_session}")

# Turn 1
req1 = ResponsesAgentRequest(
    input=[{"role": "user", "content": "What is RAG in LLMs?"}],
    custom_inputs={"session_id": conv_session, "request_id": f"req-1-{uuid4().hex[:8]}"},
)
resp1 = agent.predict(context=None, model_input=req1)
logger.info(f"Turn 1 — User: What is RAG in LLMs?")
logger.info(f"Turn 1 — Agent: {resp1.output[-1]['content'][:200]}...")

# COMMAND ----------

# Turn 2 — follow-up question (memory provides context)
req2 = ResponsesAgentRequest(
    input=[
        {"role": "user", "content": "What is RAG in LLMs?"},
        {"role": "assistant", "content": resp1.output[-1]["content"]},
        {"role": "user", "content": "What are the main components?"},
    ],
    custom_inputs={"session_id": conv_session, "request_id": f"req-2-{uuid4().hex[:8]}"},
)
resp2 = agent.predict(context=None, model_input=req2)
logger.info(f"Turn 2 — User: What are the main components?")
logger.info(f"Turn 2 — Agent: {resp2.output[-1]['content'][:200]}...")
logger.info(f"✓ Multi-turn conversation traced with session: {conv_session}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Analyzing Session Traces

# COMMAND ----------

session_traces_df = mlflow.search_traces(
    filter_string=f"request_metadata.`mlflow.trace.session` = '{conv_session}'",
    order_by=["timestamp_ms ASC"],
)
logger.info(f"Traces for session {conv_session}: {len(session_traces_df)}")

if len(session_traces_df) > 0:
    simple_cols = [
        c for c in session_traces_df.columns
        if c not in ["request", "response", "spans", "inputs", "outputs"]
    ]
    display(session_traces_df[simple_cols].head() if simple_cols else session_traces_df.head())  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Performance Analysis

# COMMAND ----------

recent_df = mlflow.search_traces(order_by=["timestamp_ms DESC"], max_results=20)

if len(recent_df) > 0:
    logger.info(f"Total traces: {len(recent_df)}")

    if "execution_time_ms" in recent_df.columns:
        durations = recent_df["execution_time_ms"].dropna()
        if len(durations) > 0:
            logger.info(f"Avg duration: {durations.mean():.0f}ms")
            logger.info(f"Min duration: {durations.min():.0f}ms")
            logger.info(f"Max duration: {durations.max():.0f}ms")

    if "status" in recent_df.columns:
        for status, count in recent_df["status"].value_counts().items():
            logger.info(f"  {status}: {count}")
