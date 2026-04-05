# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.16.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 3.2b: Genie Space Integration
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating a SQL warehouse for Genie
# MAGIC - Configuring a Genie space with data sources
# MAGIC - Starting conversations with Genie
# MAGIC - Using Genie for natural language queries
# MAGIC
# MAGIC **What is Genie?**
# MAGIC - Databricks Genie is an AI-powered data analyst
# MAGIC - Converts natural language questions to SQL queries
# MAGIC - Executes queries and returns results
# MAGIC - Can be integrated with agents via MCP

# COMMAND ----------
from pyspark.sql import SparkSession

from arxiv_curator.config import load_config, get_env

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

catalog = cfg.catalog
schema = cfg.schema


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Check for Existing Genie Space
# MAGIC
# MAGIC First, check if we already have a Genie space configured.

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType
from loguru import logger

w = WorkspaceClient()

# Derive a personal warehouse/space name from the logged-in user
# so it doesn't clash with the shared course resources
_user_prefix = w.current_user.me().user_name.split("@")[0].replace(".", "_")
_warehouse_name = f"{_user_prefix}_arxiv_warehouse"
_space_title = f"{_user_prefix}-arxiv-curator-space"

logger.info(f"Personal warehouse name: {_warehouse_name}")
logger.info(f"Personal Genie space title: {_space_title}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create SQL Warehouse
# MAGIC
# MAGIC Create a personal SQL warehouse for your Genie space.
# MAGIC Scales to 0 when idle so it costs nothing when not in use.

# COMMAND ----------

# Use the shared warehouse from config (user doesn't have permission to create new ones)
warehouse_id = cfg.warehouse_id
logger.info(f"✓ Using shared warehouse from config: {warehouse_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configure Genie Space
# MAGIC
# MAGIC Create a personal Genie space pointing at your arxiv_papers table.

# COMMAND ----------

# Configure the Genie space with arxiv_papers table
serialized_space = {
    "version": 1,
    "data_sources": {
        "tables": [
            {
                "identifier": f"{catalog}.{schema}.arxiv_papers",
                "column_configs": [
                    {"column_name": "authors"},
                    {"column_name": "ingest_ts", "get_example_values": True},
                    {"column_name": "paper_id", "get_example_values": True},
                    {
                        "column_name": "pdf_url",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "processed", "get_example_values": True},
                    {"column_name": "published", "get_example_values": True},
                    {
                        "column_name": "summary",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "title",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {
                        "column_name": "volume_path",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                ],
            }
        ]
    },
}

# Delete existing space if present so it is always recreated with the current warehouse
existing_spaces = {s.title: s for s in (w.genie.list_spaces().spaces or [])}

if _space_title in existing_spaces:
    old_space_id = existing_spaces[_space_title].space_id
    try:
        w.genie.delete_space(space_id=old_space_id)
        logger.info(f"Deleted existing Genie Space {old_space_id} to recreate with current warehouse")
    except Exception as e:
        logger.warning(f"⚠️ Could not delete existing space: {type(e).__name__}: {e}")

space = w.genie.create_space(
    warehouse_id=warehouse_id,
    serialized_space=json.dumps(serialized_space),
    title=_space_title,
)
space_id = space.space_id
logger.info(f"✓ Created Genie Space: {space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify Genie Space

# COMMAND ----------

try:
    space = w.genie.get_space(space_id=space_id, include_serialized_space=True)
    logger.info(f"Genie Space ID: {space_id}")
    if space.serialized_space:
        logger.info(f"Space config: {json.loads(space.serialized_space)}")
except Exception as e:
    logger.warning(f"⚠️ Could not get space details (may need 'Can Edit' permission): {type(e).__name__}")
    logger.info(f"Genie Space ID: {space_id} — proceeding with conversations anyway")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Start a Conversation
# MAGIC
# MAGIC Ask Genie a natural language question about the data.

# COMMAND ----------

try:
    conversation = w.genie.start_conversation_and_wait(
        space_id=space_id,
        content="Find the last 10 papers published")
    logger.info(f"Conversation started: {conversation.conversation_id}")
    logger.info(conversation.as_dict())
except Exception as e:
    logger.warning(f"⚠️ Genie conversation failed: {type(e).__name__}: {e}")
    conversation = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Continue the Conversation
# MAGIC
# MAGIC Ask follow-up questions in the same conversation.

# COMMAND ----------

if conversation:
    try:
        message = w.genie.create_message_and_wait(
            space_id=space_id,
            conversation_id=conversation.conversation_id,
            content="Return the list of authors of the last 10 papers published")
        logger.info(message.as_dict())
    except Exception as e:
        logger.warning(f"⚠️ Genie follow-up message failed: {type(e).__name__}: {e}")
else:
    logger.info("Skipping follow-up — no active conversation.")

# COMMAND ----------
