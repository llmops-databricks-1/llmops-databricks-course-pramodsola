# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.23.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 4.4: MLflow Log & Register Agent
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Evaluating the agent with MLflow scorers
# MAGIC - Logging the agent as an MLflow pyfunc model
# MAGIC - Declaring Databricks resources (endpoints, indexes, warehouses)
# MAGIC - Registering to Unity Catalog with an alias

# COMMAND ----------

import random
from datetime import datetime

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from pyspark.sql import SparkSession

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import ProjectConfig, get_env, load_config
from arxiv_curator.evaluation import (
    mentions_papers,
    word_count_check,
)

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

mlflow.set_experiment(cfg.experiment_name)

w = WorkspaceClient()
_user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "-")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Initialize Agent

# COMMAND ----------

# Use Vector Search only for evaluation (no Genie) to keep eval fast
agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=None,
    lakebase_project_id=f"{_user_prefix}-lakebase",
)
logger.info("✓ ArxivAgent initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Evaluation Data

# COMMAND ----------

with open("../eval_inputs.txt") as f:
    eval_data = [
        {"inputs": {"question": line.strip()}}
        for line in f
        if line.strip()
    ][:3]  # Limit to 3 questions for a fast evaluation run

logger.info(f"✓ Loaded {len(eval_data)} evaluation questions")


def predict_fn(question: str) -> str:
    """Wrap agent for mlflow.genai.evaluate."""
    request = {"input": [{"role": "user", "content": question}]}
    result = agent.predict(context=None, model_input=request)
    return result.output[-1].content[0]['text']

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run Evaluation

# COMMAND ----------

results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=[word_count_check, mentions_papers],  # code-only scorers for fast eval
)
logger.info("✓ Evaluation complete")
logger.info(f"Metrics: {results.metrics}")
display(results.tables["eval_results"].astype(str))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Declare Databricks Resources

# COMMAND ----------

resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
    DatabricksVectorSearchIndex(
        index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index"
    ),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.arxiv_papers"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
]
# NOTE: Genie space intentionally excluded from resources.
# Including it causes agents.deploy() pre-deployment to fail with a
# permission metadata error (empty tree node ID) on this workspace.
# The agent still uses Genie at runtime via model_config; this only
# affects Databricks' auto-permission-grant during endpoint creation.

logger.info(f"✓ Declared {len(resources)} Databricks resources")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log Model to MLflow

# COMMAND ----------

try:
    git_sha = dbutils.widgets.get("git_sha") or "local"  # noqa: F821
except Exception:
    git_sha = "local"
try:
    run_id = dbutils.widgets.get("run_id") or "local"  # noqa: F821
except Exception:
    run_id = "local"

model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "genie_space_id": cfg.genie_space_id,
    "system_prompt": cfg.system_prompt,
    "llm_endpoint": cfg.llm_endpoint,
    "lakebase_project_id": f"{_user_prefix}-lakebase",
}

test_request = {
    "messages": [{"role": "user", "content": "What are recent papers about LLMs and reasoning?"}],
}

# Unity Catalog requires both input AND output signatures.
# agents.deploy() requires output to be ChatCompletionResponse or StringResponse.
test_response = {
    "id": "chatcmpl-example",
    "object": "chat.completion",
    "created": 0,
    "model": "example-endpoint",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Sample response."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}

signature = mlflow.models.infer_signature(model_input=test_request, model_output=test_response)
logger.info("✓ Model signature set (input: ChatCompletionRequest, output: ChatCompletionResponse)")

ts = datetime.now().strftime("%Y-%m-%d")
with mlflow.start_run(
    run_name=f"arxiv-agent-{ts}",
    tags={"git_sha": git_sha, "run_id": run_id},
) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../arxiv_agent.py",  # code-based logging (mlflow 3.x)
        resources=resources,
        input_example=test_request,
        signature=signature,
        model_config=model_config,
    )
    mlflow.log_metrics(results.metrics)
    logger.info(f"✓ Model logged: {model_info.model_uri}")
    logger.info(f"  Run ID: {run.info.run_id}")

# COMMAND ----------

# Verify the logged signature BEFORE registering — catch any MLflow override early
_logged_info = mlflow.models.get_model_info(model_info.model_uri)
_input_fields = [f.name for f in _logged_info.signature.inputs.inputs] if _logged_info.signature else []
logger.info(f"Logged input schema fields: {_input_fields}")
assert "messages" in _input_fields, (
    f"STOP: Schema has '{_input_fields}' not 'messages'. "
    "agents.deploy() will fail. Check arxiv_agent.py has no ResponsesAgentRequest imports at module level."
)
logger.info("✓ Schema OK — 'messages' field confirmed, safe to register")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register to Unity Catalog

# COMMAND ----------

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags={"git_sha": git_sha, "run_id": run_id},
    env_pack="databricks_model_serving",
)
logger.info(f"✓ Registered model: {model_name} v{registered_model.version}")

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)
logger.info(f"✓ Alias 'latest-model' → version {registered_model.version}")
