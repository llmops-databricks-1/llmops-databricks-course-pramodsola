# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.23.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 5.1 (Genie variant): Agent Deployment with Genie Space
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Deploying agents using `agents.deploy()` with Genie space resource
# MAGIC - How `DatabricksGenieSpace` enables auto-permission-grant at deploy time
# MAGIC
# MAGIC ## Prerequisites:
# MAGIC - Genie space must be accessible and have a valid tree node ID
# MAGIC - Run `5.2_spn_permissions.py` to grant SPN access to the Genie space first
# MAGIC - Secret scope `arxiv-agent-scope` with `client-id` and `client-secret` keys

# COMMAND ----------

import random
from datetime import datetime

import mlflow
from databricks import agents
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
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
client = MlflowClient()

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
endpoint_name = f"arxiv-agent-pramodsola-genie-{env}"
secret_scope = "arxiv-agent-scope"

logger.info(f"Model: {model_name}")
logger.info(f"Endpoint: {endpoint_name}")
logger.info(f"Genie space: {cfg.genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Re-register Model with Genie Space Resource
# MAGIC
# MAGIC This re-runs the log+register step with `DatabricksGenieSpace` included in resources.
# MAGIC Databricks will auto-grant the endpoint access to the Genie space at deploy time.
# MAGIC **Only run this cell if the Genie space is properly configured.**

# COMMAND ----------

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.evaluation import mentions_papers, word_count_check

try:
    git_sha = dbutils.widgets.get("git_sha") or "local"  # noqa: F821
except Exception:
    git_sha = "local"
try:
    run_id = dbutils.widgets.get("run_id") or "local"  # noqa: F821
except Exception:
    run_id = "local"

_user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "-")

resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
    DatabricksVectorSearchIndex(index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index"),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.arxiv_papers"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
    DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),  # included here
]
logger.info(f"✓ Declared {len(resources)} resources including Genie space")

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

mlflow.set_experiment(cfg.experiment_name)
ts = datetime.now().strftime("%Y-%m-%d")

with mlflow.start_run(
    run_name=f"arxiv-agent-genie-{ts}",
    tags={"git_sha": git_sha, "run_id": run_id},
) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../arxiv_agent.py",
        resources=resources,
        input_example=test_request,
        signature=signature,
        model_config=model_config,
    )
    logger.info(f"✓ Model logged: {model_info.model_uri}")

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags={"git_sha": git_sha, "run_id": run_id, "variant": "genie"},
    env_pack="databricks_model_serving",
)
logger.info(f"✓ Registered model: {model_name} v{registered_model.version}")

client.set_registered_model_alias(
    name=model_name,
    alias="latest-model-genie",
    version=registered_model.version,
)
logger.info(f"✓ Alias 'latest-model-genie' → version {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deploy Agent with Genie Space

# COMMAND ----------

mv = client.get_model_version_by_alias(model_name, "latest-model-genie")
model_version = mv.version
experiment = client.get_experiment_by_name(cfg.experiment_name)

_env_vars = {
    "GIT_SHA": git_sha,
    "MODEL_VERSION": model_version,
    "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
    "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
    "LAKEBASE_SP_HOST": w.config.host,
}
try:
    dbutils.secrets.list(secret_scope)  # noqa: F821
    _env_vars["LAKEBASE_SP_CLIENT_ID"] = f"{{{{secrets/{secret_scope}/client-id}}}}"
    _env_vars["LAKEBASE_SP_CLIENT_SECRET"] = f"{{{{secrets/{secret_scope}/client-secret}}}}"
    logger.info(f"✓ LakeBase secrets included")
except Exception:
    logger.warning("LakeBase secret scope not accessible — memory disabled")

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars=_env_vars,
)

logger.info(f"✓ Deployment triggered for endpoint: {endpoint_name}")
logger.info("Wait 5–10 minutes for the endpoint to become ready, then run the test cell below.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Deployed Endpoint
# MAGIC
# MAGIC Run this cell MANUALLY after the endpoint shows "Ready" in the Serving UI.

# COMMAND ----------

# NOTE: Run this cell MANUALLY after endpoint is Ready in Serving UI.
from databricks.sdk.service.serving import EndpointStateReady

host = w.config.host
token = w.tokens.create(lifetime_seconds=2000).token_value

openai_client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

endpoint_state = w.serving_endpoints.get(endpoint_name).state
if endpoint_state.ready != EndpointStateReady.READY:
    logger.warning(f"Endpoint not ready yet (state={endpoint_state.ready}). Wait and re-run this cell.")
else:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    response = openai_client.chat.completions.create(
        model=endpoint_name,
        messages=[{"role": "user", "content": "What are recent papers about LLMs and reasoning?"}],
    )
    logger.info("\nAssistant Response:")
    logger.info("-" * 80)
    logger.info(response.choices[0].message.content)
    logger.info("-" * 80)
