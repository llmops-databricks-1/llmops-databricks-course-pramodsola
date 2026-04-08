# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.20.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 4.1: MLflow Tracing Implementation
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - What is tracing?
# MAGIC - Why tracing matters for GenAI
# MAGIC - Using @mlflow.trace decorator
# MAGIC - Manual span creation
# MAGIC - Adding metadata and tags
# MAGIC - Searching and analyzing traces

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.entities import SpanType
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. What is Tracing?
# MAGIC
# MAGIC **Tracing** captures the execution flow of your GenAI application.
# MAGIC
# MAGIC ### Why Tracing Matters:
# MAGIC - **Observability**: See what your agent is doing
# MAGIC - **Debugging**: Find where things go wrong
# MAGIC - **Performance**: Identify bottlenecks
# MAGIC - **Cost**: Track token usage
# MAGIC - **Quality**: Analyze outputs
# MAGIC
# MAGIC ### Trace Structure:
# MAGIC ```
# MAGIC Trace (Root)
# MAGIC ├── Span: Agent Call
# MAGIC │   ├── Span: LLM Call
# MAGIC │   ├── Span: Tool Execution
# MAGIC │   │   └── Span: Vector Search
# MAGIC │   └── Span: LLM Call (with tool results)
# MAGIC └── Metadata: session_id, request_id, etc.
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simple Tracing with @mlflow.trace

# COMMAND ----------

mlflow.set_experiment(cfg.experiment_name)


@mlflow.trace
def add_numbers(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


result = add_numbers(5, 3)
logger.info(f"Result: {result}")
logger.info("✓ Trace created! Check MLflow UI to see the trace.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tracing with Span Types

# COMMAND ----------


@mlflow.trace(span_type=SpanType.LLM)
def call_llm_mock(prompt: str) -> str:
    """Simulate an LLM call."""
    return f"Response to: {prompt}"


@mlflow.trace(span_type=SpanType.TOOL)
def search_database(query: str) -> list:
    """Simulate a database search."""
    return [{"id": 1, "title": "Result 1"}, {"id": 2, "title": "Result 2"}]


@mlflow.trace(span_type=SpanType.CHAIN)
def process_query(user_query: str) -> str:
    """Process a user query with LLM and tools."""
    results = search_database(user_query)
    prompt = f"User asked: {user_query}\nResults: {results}"
    return call_llm_mock(prompt)


output = process_query("What are recent papers about transformers?")
logger.info(f"Output: {output}")
logger.info("✓ Multi-span trace created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Manual Span Creation

# COMMAND ----------


def complex_function(x: int, y: int) -> int:
    """Function with manual span control."""
    with mlflow.start_span("complex_function") as span:
        span.set_inputs({"x": x, "y": y})

        with mlflow.start_span("step_1_multiply") as step1:
            result1 = x * y
            step1.set_outputs({"result": result1})

        with mlflow.start_span("step_2_add") as step2:
            result2 = result1 + 10
            step2.set_outputs({"result": result2})

        span.set_outputs({"final_result": result2})
        return result2


result = complex_function(5, 3)
logger.info(f"Result: {result}")
logger.info("✓ Trace with nested spans created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Adding Metadata and Tags

# COMMAND ----------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"
git_sha = "abc123def456"


@mlflow.trace
def function_with_metadata(x: int, y: int) -> int:
    """Function with rich metadata."""
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.session": session_id,
            "user_id": "user_123",
            "environment": "production",
        },
        tags={
            "model_serving_endpoint_name": cfg.llm_endpoint,
            "model_version": "1",
            "git_sha": git_sha,
            "request_type": "calculation",
        },
        client_request_id=request_id,
    )
    return x + y


result = function_with_metadata(10, 20)
logger.info(f"Result: {result}")
logger.info(f"Session ID: {session_id}")
logger.info(f"Request ID: {request_id}")
logger.info(f"Git SHA: {git_sha}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Searching Traces

# COMMAND ----------

traces_df = mlflow.search_traces(
    filter_string=f"tags.git_sha = '{git_sha}'",
    max_results=5,
)
logger.info(f"Found {len(traces_df)} traces with git_sha={git_sha}")

if len(traces_df) > 0:
    cols = [c for c in ["request_id", "timestamp_ms", "status", "tags"] if c in traces_df.columns]
    display(traces_df[cols].head() if cols else traces_df.head())  # noqa: F821
else:
    logger.info("No traces found — run some traced functions first.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tracing Real LLM Calls

# COMMAND ----------

client = OpenAI(
    api_key=_token,
    base_url=f"{w.config.host.rstrip('/')}/serving-endpoints",
)


@mlflow.trace(span_type=SpanType.LLM)
def call_real_llm(prompt: str) -> str:
    """Call a real LLM with tracing."""
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
    )
    return response.choices[0].message.content


result = call_real_llm("What is machine learning in one sentence?")
logger.info(f"LLM Response: {result}")
logger.info("✓ Real LLM call traced!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Tracing Agent Interactions

# COMMAND ----------


@mlflow.trace(span_type=SpanType.AGENT)
def agent_interaction(user_message: str) -> dict:
    """Simulate a complete agent interaction."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    s_id = f"s-{ts}-{random.randint(100000, 999999)}"
    r_id = f"req-{ts}-{random.randint(100000, 999999)}"

    mlflow.update_current_trace(
        metadata={"mlflow.trace.session": s_id},
        tags={"agent_type": "research_assistant", "model_version": "1.0"},
        client_request_id=r_id,
    )

    with mlflow.start_span("analyze_query", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"query": user_message})
        analysis = {"intent": "search", "topic": "machine learning"}
        span.set_outputs(analysis)

    with mlflow.start_span("search_papers", span_type=SpanType.TOOL) as span:
        span.set_inputs({"query": analysis["topic"]})
        results = [
            {"title": "Paper 1", "relevance": 0.95},
            {"title": "Paper 2", "relevance": 0.87},
        ]
        span.set_outputs({"results": results})

    with mlflow.start_span("generate_response", span_type=SpanType.LLM) as span:
        span.set_inputs({"user_message": user_message, "search_results": results})
        response = f"I found {len(results)} relevant papers about {analysis['topic']}"
        span.set_outputs({"response": response})

    return {"response": response, "session_id": s_id, "request_id": r_id}


result = agent_interaction("What papers discuss machine learning?")
logger.info(f"Agent Response: {result['response']}")
logger.info(f"Session ID: {result['session_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Analyzing Recent Traces

# COMMAND ----------

recent_traces_df = mlflow.search_traces(
    order_by=["timestamp_ms DESC"],
    max_results=10,
)
logger.info(f"Recent Traces: {len(recent_traces_df)}")

if len(recent_traces_df) > 0:
    simple_cols = [
        c for c in recent_traces_df.columns
        if c not in ["request", "response", "spans", "inputs", "outputs"]
    ]
    display(recent_traces_df[simple_cols].head(10) if simple_cols else recent_traces_df.head(10))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Trace Filtering Examples

# COMMAND ----------

failed_traces = mlflow.search_traces(filter_string="status = 'ERROR'", max_results=5)
logger.info(f"Failed traces: {len(failed_traces)}")

endpoint_traces = mlflow.search_traces(
    filter_string=f"tags.model_serving_endpoint_name = '{cfg.llm_endpoint}'",
    max_results=5,
)
logger.info(f"Traces for endpoint {cfg.llm_endpoint}: {len(endpoint_traces)}")
