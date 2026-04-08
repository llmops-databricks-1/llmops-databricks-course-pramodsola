# Databricks notebook source
import subprocess
import sys

_username = spark.sql("SELECT current_user()").collect()[0][0]  # noqa: F821
_whl = f"/Workspace/Users/{_username}/.bundle/dev/course-code-hub/artifacts/.internal/arxiv_curator-0.20.0-py3-none-any.whl"
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", _whl, "-q"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Lecture 4.3: GenAI Evaluation Theory
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Why evaluation matters for GenAI
# MAGIC - Types of evaluation metrics
# MAGIC - MLflow evaluation framework
# MAGIC - Guidelines vs Judges
# MAGIC - Custom scorers
# MAGIC - Judge alignment with human feedback

# COMMAND ----------

from typing import Literal

import mlflow
from loguru import logger
from mlflow.genai.judges import make_judge
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.evaluation import (
    hook_in_post_guideline,
    mentions_papers,
    polite_tone_guideline,
    scope_guideline,
    word_count_check,
)

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

mlflow.set_experiment(cfg.experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Why Evaluation Matters for GenAI
# MAGIC
# MAGIC Traditional ML evaluation (accuracy, F1) doesn't work for GenAI:
# MAGIC - **Open-ended outputs** — no single correct answer
# MAGIC - **Subjective quality** — what's "good" varies by use case
# MAGIC - **Multiple dimensions** — accuracy, tone, safety, style
# MAGIC - **Context-dependent** — same output can be good or bad
# MAGIC
# MAGIC ### Why Evaluate?
# MAGIC 1. Quality assurance — ensure outputs meet standards
# MAGIC 2. Regression detection — catch degradation over time
# MAGIC 3. Model comparison — choose the best model/prompt
# MAGIC 4. Continuous improvement — identify areas to fix

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Available Scorers from arxiv_curator.evaluation

# COMMAND ----------

logger.info("Scorers available:")
logger.info(f"  polite_tone_guideline — {len(polite_tone_guideline.guidelines)} rules")
logger.info(f"  hook_in_post_guideline — {len(hook_in_post_guideline.guidelines)} rules")
logger.info(f"  scope_guideline — {len(scope_guideline.guidelines)} rules")
logger.info("  word_count_check — custom scorer (boolean, <350 words)")
logger.info("  mentions_papers — custom scorer (boolean)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Guidelines — Binary Pass/Fail

# COMMAND ----------

test_data = [
    {
        "inputs": {"question": "How do I deploy a model?"},
        "outputs": "Just figure it out yourself, it's not that hard.",
    },
    {
        "inputs": {"question": "How do I deploy a model?"},
        "outputs": "I'd be happy to help you deploy your model! Here are the steps...",
    },
]

results = mlflow.genai.evaluate(data=test_data, scorers=[polite_tone_guideline])
logger.info("Guidelines results:")
display(results.tables["eval_results"])  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Judges — Scored Evaluation (1–5)

# COMMAND ----------

quality_judge = make_judge(
    name="response_quality",
    instructions=(
        "Evaluate the quality of the response in {{ outputs }} to the question in {{ inputs }}. "
        "Score from 1 to 5:\n"
        "1 - Completely unhelpful\n"
        "2 - Partially helpful\n"
        "3 - Adequate\n"
        "4 - Good and clear\n"
        "5 - Excellent and comprehensive"
    ),
    model=f"databricks:/{cfg.llm_endpoint}",
    feedback_value_type=int,
)
logger.info(f"Judge created: {quality_judge.name} (1-5 scale)")

# COMMAND ----------

judge_test_data = [
    {
        "inputs": {"question": "What is machine learning?"},
        "outputs": "It's computers learning stuff.",
    },
    {
        "inputs": {"question": "What is machine learning?"},
        "outputs": (
            "Machine learning is a subset of AI where algorithms learn patterns "
            "from data to make predictions without being explicitly programmed."
        ),
    },
]

judge_results = mlflow.genai.evaluate(data=judge_test_data, scorers=[quality_judge])
logger.info("Judge results:")
display(judge_results.tables["eval_results"])  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Custom Code-Based Scorers

# COMMAND ----------

custom_test_data = [
    {
        "inputs": {"question": "How to use Python?"},
        "outputs": "Here's how:\n```python\nprint('Hello')\n```\nThis prints Hello.",
    },
    {
        "inputs": {"question": "How to use Python?"},
        "outputs": "Python is a programming language. " * 100,
    },
]

custom_results = mlflow.genai.evaluate(
    data=custom_test_data,
    scorers=[word_count_check, mentions_papers],
)
logger.info("Custom scorer results:")
display(custom_results.tables["eval_results"])  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Categorical Judges

# COMMAND ----------

sentiment_judge = make_judge(
    name="response_sentiment",
    instructions=(
        "Analyze the sentiment of the response in {{ outputs }}. "
        "Classify as: 'positive', 'neutral', or 'negative'"
    ),
    feedback_value_type=Literal["positive", "neutral", "negative"],
    model=f"databricks:/{cfg.llm_endpoint}",
)
logger.info(f"Categorical judge created: {sentiment_judge.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Combining Multiple Scorers

# COMMAND ----------

all_scorers = [polite_tone_guideline, quality_judge, word_count_check, sentiment_judge]

comprehensive_data = [
    {
        "inputs": {"question": "Explain transformers"},
        "outputs": (
            "Transformers are a neural network architecture using self-attention mechanisms. "
            "They've revolutionized NLP, enabling models like BERT and GPT."
        ),
    },
]

comprehensive_results = mlflow.genai.evaluate(
    data=comprehensive_data, scorers=all_scorers
)
logger.info("Comprehensive evaluation results:")
display(comprehensive_results.tables["eval_results"])  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Judge Alignment with Human Feedback
# MAGIC
# MAGIC LLM judges may not always align with human preferences.
# MAGIC Use **SIMBA alignment** to calibrate judges:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.judges.optimizers import SIMBAAlignmentOptimizer
# MAGIC
# MAGIC # Collect traces with both judge and human feedback
# MAGIC optimizer = SIMBAAlignmentOptimizer(model="databricks:/my-llm")
# MAGIC aligned_judge = my_judge.align(optimizer, traces_with_feedback)
# MAGIC ```
# MAGIC
# MAGIC ### Process:
# MAGIC 1. Run judge on test cases
# MAGIC 2. Collect human feedback on same cases
# MAGIC 3. Optimize judge instructions to match human preferences
# MAGIC 4. Use aligned judge for production evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluation Best Practices
# MAGIC
# MAGIC **Do:**
# MAGIC - Use multiple scorers for comprehensive evaluation
# MAGIC - Combine guidelines + judges + custom scorers
# MAGIC - Validate judges with human feedback periodically
# MAGIC - Track metrics over time for regression detection
# MAGIC - Include edge cases in evaluation data
# MAGIC
# MAGIC **Don't:**
# MAGIC - Rely on a single metric
# MAGIC - Use only automated metrics for GenAI
# MAGIC - Use the same model as both generator and judge
# MAGIC - Evaluate on too few examples
