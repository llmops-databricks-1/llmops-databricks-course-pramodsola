# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.23.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 5.1: Agent Deployment & Testing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Deploying agents using `agents.deploy()`
# MAGIC - Configuring environment variables and secrets
# MAGIC - Testing deployed endpoints with the OpenAI-compatible client
# MAGIC
# MAGIC ## Prerequisites:
# MAGIC - Notebook 4.4 must have been run to register the model in Unity Catalog
# MAGIC - Secret scope `arxiv-agent-scope` with `client-id` and `client-secret` keys

# COMMAND ----------

import random
from datetime import datetime

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
client = MlflowClient()

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
endpoint_name = f"arxiv-agent-pramodsola-{env}"
secret_scope = "arxiv-agent-scope"

logger.info(f"Model: {model_name}")
logger.info(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Get Latest Registered Model Version

# COMMAND ----------

mv = client.get_model_version_by_alias(model_name, "latest-model")
model_version = mv.version
experiment = client.get_experiment_by_name(cfg.experiment_name)

import mlflow

model_uri = f"models:/{model_name}/{model_version}"
model_info = mlflow.models.get_model_info(model_uri)
input_fields = [f.name for f in model_info.signature.inputs.inputs] if model_info.signature else []
output_fields = [f.name for f in model_info.signature.outputs.inputs] if (model_info.signature and model_info.signature.outputs) else []

logger.info(f"Deploying version: {model_version}")
logger.info(f"Run ID: {mv.run_id}")
logger.info(f"Input schema fields: {input_fields}")
logger.info(f"Output schema fields: {output_fields}")
logger.info(f"Experiment ID: {experiment.experiment_id}")

# Hard stop — agents.deploy() validates both input and output schemas
if "messages" not in input_fields:
    raise ValueError(
        f"Input schema mismatch — got {input_fields}, need ['messages']. Re-run 4.4."
    )
if "choices" not in output_fields:
    raise ValueError(
        f"Output schema mismatch — got {output_fields}, need ChatCompletionResponse (choices). Re-run 4.4."
    )
logger.info("✓ Schema OK — input: messages, output: choices (ChatCompletionResponse)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy Agent
# MAGIC
# MAGIC `agents.deploy()` handles:
# MAGIC - Endpoint creation and configuration
# MAGIC - Inference tables for monitoring
# MAGIC - Environment variables and secrets
# MAGIC - Model versioning
# MAGIC
# MAGIC **Note:** Deployment takes 5–10 minutes. Check the Serving UI for status.

# COMMAND ----------

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": "local",
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client-id}}}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client-secret}}}}",
        "LAKEBASE_SP_HOST": w.config.host,
    },
)

logger.info(f"✓ Deployment triggered for endpoint: {endpoint_name}")
logger.info("Wait 5–10 minutes for the endpoint to become ready, then run the test cell below.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Deployed Endpoint
# MAGIC
# MAGIC Run this cell after the endpoint is ready (state = Ready in Serving UI).

# COMMAND ----------

host = w.config.host
token = w.tokens.create(lifetime_seconds=2000).token_value

openai_client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = openai_client.chat.completions.create(
    model=endpoint_name,
    messages=[{"role": "user", "content": "What are recent papers about LLMs and reasoning?"}],
)

logger.info(f"Session ID: {session_id}")
logger.info(f"Request ID: {request_id}")
logger.info("\nAssistant Response:")
logger.info("-" * 80)
logger.info(response.choices[0].message.content)
logger.info("-" * 80)
