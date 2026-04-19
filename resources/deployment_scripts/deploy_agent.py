# Databricks notebook source
from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from loguru import logger
from mlflow import MlflowClient

from arxiv_curator.config import load_config
from arxiv_curator.utils.common import get_widget

# COMMAND ----------

env = get_widget("env", "dev")
git_sha = get_widget("git_sha", "local")
secret_scope = "arxiv-agent-scope"

cfg = load_config("../../project_config.yml", env=env)

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
endpoint_name = f"arxiv-agent-pramodsola-{env}"

client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "latest-model").version
experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info(f"Deploying model: {model_name} v{model_version}")
logger.info(f"Endpoint: {endpoint_name}")

# COMMAND ----------

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client-id}}}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client-secret}}}}",
        "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
    },
)

logger.info("✓ Deployment complete!")
