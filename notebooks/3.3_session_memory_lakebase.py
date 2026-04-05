# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.17.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 3.3: Session Memory with Lakebase
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Lakebase (Databricks PostgreSQL) for session persistence
# MAGIC - PostgresAPI: project/branch/endpoint model
# MAGIC - Managing conversation history with JSONB
# MAGIC - Building stateful agents with LakebaseMemory

# COMMAND ----------

import json
import urllib.parse
from uuid import uuid4

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Project,
    ProjectDefaultEndpointSettings,
    ProjectSpec,
)
from google.protobuf.duration_pb2 import Duration
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.memory import LakebaseMemory

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Personal Lakebase Project
# MAGIC
# MAGIC **Lakebase** is Databricks' managed PostgreSQL service:
# MAGIC - Fully managed, scales to 0 when idle
# MAGIC - Integrated with Databricks authentication
# MAGIC - Project → Branch → Endpoint hierarchy
# MAGIC - Ideal for session state, caching, and metadata

# COMMAND ----------

project_id = cfg.lakebase_project_id

try:
    project = pg_api.get_project(name=f"projects/{project_id}")
    logger.info(f"✓ Using existing Lakebase project: {project_id}")
except Exception:
    logger.info(f"Creating new Lakebase project: {project_id}")
    project = pg_api.create_project(
        project_id=project_id,
        project=Project(
            spec=ProjectSpec(
                display_name=project_id,
                default_endpoint_settings=ProjectDefaultEndpointSettings(
                    autoscaling_limit_min_cu=1,
                    autoscaling_limit_max_cu=4,
                    suspend_timeout_duration=Duration(seconds=300),
                ),
            ),
        ),
    ).wait()
    logger.info(f"✓ Created Lakebase project: {project_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Connect and Create Table

# COMMAND ----------

default_branch = next(iter(pg_api.list_branches(parent=project.name)))
endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
host = endpoint.status.hosts.host

pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)
user = w.current_user.me()
username = urllib.parse.quote_plus(user.user_name)
conn_string = (
    f"postgresql://{username}:{pg_credential.token}@{host}:5432/"
    "databricks_postgres?sslmode=require"
)

logger.info(f"Lakebase host: {host}")

# COMMAND ----------

with psycopg.connect(conn_string) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
        ON session_messages(session_id)
    """)

logger.info("✓ session_messages table ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save and Load Messages

# COMMAND ----------

test_session_id = f"test-session-{uuid4()}"
test_messages = [
    {"role": "user", "content": "Hello, what can you help me with?"},
    {"role": "assistant", "content": "I can help you find research papers."},
    {"role": "user", "content": "Find papers about LLM reasoning"},
]

with psycopg.connect(conn_string) as conn:
    for msg in test_messages:
        conn.execute(
            "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
            (test_session_id, json.dumps(msg)),
        )

logger.info(f"✓ Saved {len(test_messages)} messages to session: {test_session_id}")

# COMMAND ----------

with psycopg.connect(conn_string) as conn:
    result = conn.execute(
        """
        SELECT message_data, created_at FROM session_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        """,
        (test_session_id,),
    ).fetchall()

logger.info(f"✓ Loaded {len(result)} messages:")
for row in result:
    logger.info(f"  [{row[1]}] {row[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Using LakebaseMemory Class

# COMMAND ----------

memory = LakebaseMemory(project_id=project_id)

# COMMAND ----------

session_id = f"memory-test-{uuid4()}"
messages = [
    {"role": "user", "content": "What papers discuss transformer architectures?"},
    {"role": "assistant", "content": "Here are some relevant papers..."},
]

memory.save_messages(session_id, messages)
logger.info(f"✓ Saved messages to session: {session_id}")

# COMMAND ----------

loaded = memory.load_messages(session_id)
logger.info(f"✓ Loaded {len(loaded)} messages:")
for msg in loaded:
    logger.info(f"  {msg['role']}: {msg['content'][:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Stateful Multi-Turn Conversation with LLM

# COMMAND ----------

_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821
client = OpenAI(api_key=_token, base_url=f"{w.config.host}/serving-endpoints")


def chat_with_memory(
    session_id: str, user_message: str, memory: LakebaseMemory
) -> str:
    """Chat with LLM using session memory for context."""
    previous_messages = memory.load_messages(session_id)
    messages = (
        [{"role": "system", "content": "You are a helpful research assistant."}]
        + previous_messages
        + [{"role": "user", "content": user_message}]
    )
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=messages,
    )
    assistant_response = response.choices[0].message.content
    memory.save_messages(
        session_id,
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ],
    )
    return assistant_response


logger.info("✓ Chat function with memory created")

# COMMAND ----------

agent_session_id = f"agent-session-{uuid4()}"

response1 = chat_with_memory(agent_session_id, "What is RAG in the context of LLMs?", memory)
logger.info(f"Response 1: {response1[:200]}...")

# COMMAND ----------

response2 = chat_with_memory(agent_session_id, "What are the main components?", memory)
logger.info(f"Response 2: {response2[:200]}...")

# COMMAND ----------

full_conversation = memory.load_messages(agent_session_id)
logger.info(f"✓ Full conversation ({len(full_conversation)} messages):")
for i, msg in enumerate(full_conversation, 1):
    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
    logger.info(f"  {i}. [{msg['role']}] {content}")

memory.close()
