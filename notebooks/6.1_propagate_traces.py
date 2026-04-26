# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 6.1: Propagate Traces
# MAGIC
# MAGIC Sends sample arxiv queries to the deployed endpoint to generate MLflow traces.
# MAGIC Run this to seed data for the monitoring dashboard.
# MAGIC
# MAGIC **Note:** Traces appear in the Delta table after ~15 minutes (MLflow Delta sync job).

# COMMAND ----------

import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.23.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

import random
import time
from datetime import datetime

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

workspace = WorkspaceClient()
host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=3600).token_value

endpoint_name = f"arxiv-agent-pramodsola-{env}"

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

logger.info(f"Sending queries to: {endpoint_name}")

# COMMAND ----------

queries = [
    "What are recent papers about LLMs and reasoning?",
    "Find papers on transformer architectures published recently.",
    "What research exists on retrieval augmented generation?",
    "Show me papers about AI safety and alignment.",
    "What are the latest developments in multimodal models?",
    "Find papers about efficient fine-tuning methods like LoRA.",
    "What research covers agent-based systems and tool use?",
    "Are there papers on knowledge distillation for smaller models?",
    "What papers discuss evaluation benchmarks for language models?",
    "Find research on prompt engineering and in-context learning.",
    "What are recent papers on diffusion models?",
    "Show papers about code generation with LLMs.",
    "Find papers on reinforcement learning from human feedback.",
]

# COMMAND ----------

for i, query in enumerate(queries):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"

    logger.info(f"[{i + 1}/{len(queries)}] {query[:60]}...")
    try:
        client.chat.completions.create(
            model=endpoint_name,
            messages=[{"role": "user", "content": query}],
            extra_body={"custom_inputs": {"session_id": session_id}},
        )
    except Exception as e:
        logger.warning(f"Request {i + 1} failed: {e}")
    time.sleep(2)

logger.info("✓ All queries sent — traces appear in MLflow within ~15 min (Delta sync)")
