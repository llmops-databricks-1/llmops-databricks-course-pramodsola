# Databricks notebook source
# This notebook evaluates unevaluated agent traces and creates an aggregated view
# for monitoring agent performance (latency, token usage, quality scores).

import mlflow
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.evaluation import (
    mentions_papers,
    polite_tone_guideline,
    word_count_check,
)

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../../project_config.yml", env)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(cfg.experiment_name)

catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

# Discover the MLflow traces Delta table — name includes the experiment ID
experiment = MlflowClient().get_experiment_by_name(cfg.experiment_name)
experiment_id = experiment.experiment_id

traces_table = f"{catalog}.{schema}.trace_logs_{experiment_id}"
aggregated_view = f"{catalog}.{schema}.arxiv_traces_aggregated"

logger.info(f"Source traces table: {traces_table}")
logger.info(f"Target view: {aggregated_view}")

# COMMAND ----------

endpoint_name = f"arxiv-agent-pramodsola-{env}"

# Fetch traces not yet evaluated:
# - Filters to our personal serving endpoint only
# - Excludes traces that already have assessments (already scored)
# - Extracts the agent's final response from ChatCompletionResponse format
new_traces_df = spark.sql(f"""
    SELECT
        t.trace_id,
        t.request_preview,
        get_json_object(t.response, '$.choices[0].message.content') AS response_text
    FROM {traces_table} t
    WHERE tags['model_serving_endpoint_name'] = '{endpoint_name}'
      AND (t.assessments IS NULL OR size(t.assessments) = 0)
      AND get_json_object(t.response, '$.choices[0].message.content') IS NOT NULL
""")

traces_pdf = new_traces_df.toPandas()
logger.info(f"New traces to evaluate: {len(traces_pdf)}")

# COMMAND ----------

if len(traces_pdf) == 0:
    logger.info("No new traces to evaluate — skipping scoring.")
else:
    # Build eval DataFrame in the format mlflow.genai.evaluate expects
    eval_pdf = pd.DataFrame(
        {
            "trace_id": traces_pdf["trace_id"],
            "inputs": traces_pdf["request_preview"].apply(lambda x: {"query": x}),
            "outputs": traces_pdf["response_text"],
        }
    )

    # COMMAND ----------

    # Run heuristic scorers on ALL traces (no LLM cost)
    heuristic_result = mlflow.genai.evaluate(
        data=eval_pdf[["inputs", "outputs"]],
        scorers=[word_count_check, mentions_papers],
    )

    for trace_id, assessments in zip(
        eval_pdf["trace_id"],
        heuristic_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=a["assessment_name"],
                value=a["feedback"]["value"],
            )

    logger.info(f"Logged word_count_check + mentions_papers for {len(eval_pdf)} traces")

    # COMMAND ----------

    # Run LLM-judge scorer on 10% sample only (costs one LLM call per trace)
    sample_size = max(1, int(len(eval_pdf) * 0.1))
    sampled_pdf = eval_pdf.sample(n=sample_size, random_state=42)
    logger.info(f"Sampled {len(sampled_pdf)} traces for LLM-judge evaluation")

    llm_result = mlflow.genai.evaluate(
        data=sampled_pdf[["inputs", "outputs"]],
        scorers=[polite_tone_guideline],
    )

    for trace_id, assessments in zip(
        sampled_pdf["trace_id"],
        llm_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=a["assessment_name"],
                value=a["feedback"]["value"],
            )

    logger.info(f"Logged polite_tone for {len(sampled_pdf)} traces")

# COMMAND ----------

# Create/replace the aggregated SQL view — one clean row per trace for dashboarding.
#
# Per trace it computes:
# 1. Basic info: trace_id, request_time, request_preview, response_text, latency_seconds
# 2. Span metrics (via LATERAL VIEW explode):
#    - call_llm_exec_count — number of LLM calls the agent made
#    - tool_call_count — number of tool invocations
#    - total_tokens_used — sum of tokens from each call_llm span
# 3. Quality scores from assessments (0/1):
#    - word_count_check, mentions_papers (heuristic — 'true'/'false')
#    - polite_tone (LLM judge — 'Pass'/'Fail', NULL if not in 10% sample)
#
# Note: scores may show 0 for the first ~15 min after running the scorer —
# the MLflow Delta sync job ([<experiment_id>] Trace Archive Job) runs every 15 min.

spark.sql(f"""
    CREATE OR REPLACE VIEW {aggregated_view} AS
    SELECT
        t.trace_id,
        t.request_time,
        t.request_preview,
        get_json_object(t.response, '$.choices[0].message.content') AS response_text,
        CAST(t.execution_duration_ms / 1000.0 AS DOUBLE) AS latency_seconds,
        COUNT(IF(s.name = 'call_llm', 1, NULL)) AS call_llm_exec_count,
        COUNT(IF(s.name = 'execute_tool', 1, NULL)) AS tool_call_count,
        CAST(SUM(
            IF(
                s.name = 'call_llm',
                CAST(
                    get_json_object(
                        get_json_object(s.attributes['mlflow.spanOutputs'], '$.usage'),
                        '$.total_tokens'
                    ) AS INT
                ),
                0
            )
        ) AS LONG) AS total_tokens_used,
        current_timestamp() AS processed_ts,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'word_count_check'), 1
            ).feedback.value = 'true' THEN 1 ELSE 0
        END AS word_count_check,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'mentions_papers'), 1
            ).feedback.value = 'true' THEN 1 ELSE 0
        END AS mentions_papers,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'polite_tone'), 1
            ).feedback.value = 'Pass' THEN 1 ELSE 0
        END AS polite_tone
    FROM {traces_table} t
    LATERAL VIEW explode(spans) AS s
    WHERE tags['model_serving_endpoint_name'] = '{endpoint_name}'
    GROUP BY t.trace_id, t.request_time, t.execution_duration_ms,
             t.request_preview, t.response, t.assessments
""")

logger.info(f"✓ View {aggregated_view} created/updated")
