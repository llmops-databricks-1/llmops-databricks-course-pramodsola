# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: PDF Parsing with AI Parse Documents
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Downloading and storing PDFs
# MAGIC - AI Parse Documents for intelligent parsing
# MAGIC - Comparison with other PDF parsing tools
# MAGIC - Storing parsed content in Delta tables

# COMMAND ----------

# Install arxiv_curator from bundle artifact (works for both bundle jobs and manual runs)
import subprocess, sys
from pyspark.sql import SparkSession as _SparkSession

_username = _SparkSession.builder.getOrCreate().sql("SELECT current_user()").first()[0]
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.16.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

from loguru import logger
from databricks.connect import DatabricksSession

from arxiv_curator.config import load_config, get_env
from arxiv_curator.data_processor import DataProcessor

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()
logger.info("✅ Using Databricks Connect Spark session")

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

logger.info(f"Catalog: {cfg.catalog}, Schema: {cfg.schema}, Volume: {cfg.volume}")

# COMMAND ----------

# Ensure the Unity Catalog Volume exists before downloading PDFs
spark.sql(
    f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema}.{cfg.volume}"
)
logger.info(f"✅ Volume ready: {cfg.catalog}.{cfg.schema}.{cfg.volume}")

# COMMAND ----------

# Initialize the DataProcessor (reusable class from arxiv_curator package)
processor = DataProcessor(spark=spark, config=cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. PDF Parsing Tools Comparison
# MAGIC
# MAGIC | Tool | Pros | Cons | Best For |
# MAGIC |------|------|------|----------|
# MAGIC | **AI Parse Documents** | - AI-powered<br>- Handles complex layouts<br>- Integrated with Databricks<br>- Preserves structure | - Databricks-specific<br>- Cost per page | Complex documents, tables, multi-column |
# MAGIC | **PyPDF2** | - Simple<br>- Free<br>- Pure Python | - Poor with complex layouts<br>- No table extraction | Simple text extraction |
# MAGIC | **pdfplumber** | - Good table extraction<br>- Layout analysis | - Slower<br>- Manual tuning needed | Tables and structured data |
# MAGIC | **Apache Tika** | - Multi-format support<br>- Metadata extraction | - Java dependency<br>- Heavy | Multi-format processing |
# MAGIC | **Unstructured.io** | - ML-powered<br>- Good chunking | - External service<br>- API costs | Modern RAG pipelines |
# MAGIC
# MAGIC **AI Parse Documents** is the recommended choice for Databricks users due to its integration and intelligent parsing capabilities.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Download PDFs, Parse, and Create Chunks
# MAGIC
# MAGIC The `DataProcessor` class from `arxiv_curator.data_processor` handles the full pipeline:
# MAGIC 1. Download PDFs for unprocessed papers from arXiv
# MAGIC 2. Update `arxiv_papers` table with processed timestamps and volume paths
# MAGIC 3. Parse PDFs using AI Parse Documents (`ai_parse_document`)
# MAGIC 4. Extract and clean text chunks, join with paper metadata
# MAGIC 5. Write chunks to `arxiv_chunks` table with Change Data Feed enabled
# MAGIC
# MAGIC The same class is used in `resources/deployment_scripts/process_data.py` for scheduled runs.

# COMMAND ----------

processor.process_and_save()
