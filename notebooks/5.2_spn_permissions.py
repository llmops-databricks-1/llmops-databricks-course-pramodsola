# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.23.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 5.2: SPN Permissions for Deployment
# MAGIC
# MAGIC ## ⚠️ Reference Only — Run once when setting up CI/CD
# MAGIC
# MAGIC Grants a Service Principal (SPN) the permissions it needs to run the deployed agent.
# MAGIC The SPN is used by the CD pipeline to deploy to acc/prd environments.
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Granting SPN access to Genie spaces
# MAGIC - Granting SPN access to Vector Search endpoints
# MAGIC - Granting SPN access to SQL Warehouses

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Get SPN Client ID from Secret Scope

# COMMAND ----------

spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Grant Genie Space Access

# COMMAND ----------

if cfg.genie_space_id:
    w.permissions.update(
        request_object_type="genie",
        request_object_id=cfg.genie_space_id,
        access_control_list=[
            AccessControlRequest(
                service_principal_name=spn_app_id,
                permission_level=PermissionLevel.CAN_RUN,
            )
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Grant Vector Search Endpoint Access

# COMMAND ----------

vs_endpoint = w.vector_search_endpoints.get_endpoint(cfg.vector_search_endpoint)
w.permissions.update(
    request_object_type="vector-search-endpoints",
    request_object_id=vs_endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Grant SQL Warehouse Access

# COMMAND ----------

w.permissions.update(
    request_object_type="warehouses",
    request_object_id=cfg.warehouse_id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)
