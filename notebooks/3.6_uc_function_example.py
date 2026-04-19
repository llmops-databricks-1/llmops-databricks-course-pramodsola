# Databricks notebook source
import subprocess
import sys

_username = (
    spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
)
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.23.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 3.6: Unity Catalog Functions
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Creating Python UDFs in Unity Catalog
# MAGIC - Registering functions with `CREATE OR REPLACE FUNCTION`
# MAGIC - Calling UC functions from Spark SQL

# COMMAND ----------

from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

catalog = cfg.catalog
schema = cfg.schema

print(f"Using catalog: {catalog}, schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register UC Functions
# MAGIC
# MAGIC Register four arithmetic functions as Unity Catalog Python UDFs.

# COMMAND ----------

function_name = f"{catalog}.{schema}.add_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    return a + b
    $$
""")
print(f"Created: {function_name}")

# COMMAND ----------

function_name = f"{catalog}.{schema}.subtract_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    return a - b
    $$
""")
print(f"Created: {function_name}")

# COMMAND ----------

function_name = f"{catalog}.{schema}.multiply_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    return a * b
    $$
""")
print(f"Created: {function_name}")

# COMMAND ----------

function_name = f"{catalog}.{schema}.divide_numbers"
spark.sql(f"""
    CREATE OR REPLACE FUNCTION {function_name}(a DOUBLE, b DOUBLE)
    RETURNS DOUBLE
    LANGUAGE PYTHON AS
    $$
    if b == 0:
        return None
    return a / b
    $$
""")
print(f"Created: {function_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test the Functions

# COMMAND ----------

result = spark.sql(f"""
    SELECT
        {catalog}.{schema}.add_numbers(10, 3)      AS add_result,
        {catalog}.{schema}.subtract_numbers(10, 3) AS subtract_result,
        {catalog}.{schema}.multiply_numbers(10, 3) AS multiply_result,
        {catalog}.{schema}.divide_numbers(10, 3)   AS divide_result
""")
result.show()
