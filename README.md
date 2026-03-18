# Arxiv Curator — LLMOps Course on Databricks

A hands-on LLMOps course project built on Databricks. The **Arxiv Curator** is an AI-powered agent that ingests, processes, and serves the latest AI/ML research papers from [arXiv](https://arxiv.org), demonstrating end-to-end LLMOps practices: model serving, data ingestion, vector search, and AI gateway governance.

---

## Project Structure

```
course-code-hub/
├── notebooks/                        # Databricks notebooks (weekly lectures)
│   ├── 1.1_foundation_models_overview.py
│   ├── 1.2_provisioned_throughput_deployment.py
│   ├── 1.3_arxiv_data_ingestion.py
│   └── 1.4_external_models_custom_provider.py
├── resources/                        # Databricks Asset Bundle job definitions
│   ├── arxiv_data_ingestion_job.yml
│   └── external_models_custom_provider_job.yml
├── src/
│   └── arxiv_curator/               # Shared Python package
│       ├── __init__.py
│       └── config.py                # Environment-aware config loader
├── tests/
├── databricks.yml                   # Databricks Asset Bundle configuration
├── project_config.yml               # Per-environment settings (dev/acc/prd)
├── pyproject.toml                   # Python project & dependency definitions
└── version.txt
```

---

## Week 1 — Foundation: Model Access & Data Ingestion

Week 1 establishes the infrastructure layer: how to access LLMs on Databricks, how to deploy your own model endpoint, and how to build the raw data pipeline for the Arxiv Curator agent.

### Notebook 1.1 — Foundation Models Overview

**What it covers:**
- The three ways to access AI models on Databricks: Foundation Model APIs (pay-per-token), Provisioned Throughput (dedicated capacity), and External Models (OpenAI, Anthropic, etc.)
- Live query to `databricks-llama-4-maverick` using the OpenAI-compatible SDK
- Pricing comparison and break-even analysis between model access types (DBU cost calculations)
- Decision guide: when to use each model access type

**Key concepts:** DBU pricing, Foundation Model APIs, model governance via Databricks AI Gateway

---

### Notebook 1.2 — Provisioned Throughput Deployment

**What it covers:**
- Deploying a dedicated LLaMA 3.2 1B serving endpoint with provisioned throughput
- Configuring inference tables: every request/response is automatically logged to a Delta table in Unity Catalog for auditing and monitoring
- Endpoint lifecycle: create → monitor → query → delete
- Cost estimation: model units, tokens/second, DBU/hour calculations

**Key concepts:** Provisioned throughput, model units, inference tables, AI Gateway, scale-to-zero

> **Note:** This notebook creates a real serving endpoint that incurs Databricks costs. Run the cleanup cell when done to delete the endpoint.

---

### Notebook 1.3 — Arxiv Data Ingestion

**What it covers:**
- Fetching the latest AI/ML research papers from the arXiv public API (`cs.AI`, `cs.LG` categories)
- Storing paper metadata (title, authors, abstract, categories, PDF URL) in a Delta table in Unity Catalog
- Schema design for downstream processing (fields reserved for Week 2: `processed`, `volume_path`)
- Data validation and statistics

**What it builds:**
- Delta table: `mlops_dev.{your_schema}.arxiv_papers` (50 most recent papers)

**Key concepts:** arXiv API, Apache Spark, Delta Lake, Unity Catalog, environment-aware config

---

### Notebook 1.4 — External Models with Custom Provider (Image Generation)

**What it covers:**
- Registering OpenAI's DALL-E 3 as a Databricks serving endpoint via the External Models API
- Databricks Secrets for secure API key management (no hardcoded keys in code)
- Querying the endpoint through the Databricks AI Gateway using the OpenAI SDK
- Generating and displaying AI images from text prompts (base64 and URL response formats)

**Key concepts:** External Models, Databricks Secrets, MLflow Deployments client, AI Gateway as a unified interface for all providers

---

## Environment Configuration

All environment-specific settings live in [`project_config.yml`](project_config.yml):

| Setting | dev | acc | prd |
|---|---|---|---|
| catalog | `mlops_dev` | `mlops_acc` | `mlops_prd` |
| schema | `{your_schema}` | `{your_schema}` | `{your_schema}` |
| LLM endpoint | `databricks-llama-4-maverick` | `databricks-meta-llama-3-1-70b-instruct` | `databricks-meta-llama-3-1-70b-instruct` |

The `env` is resolved automatically at runtime from the Databricks job widget, falling back to `"dev"`.

---

## Setup & Running

### Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv) for dependency management
- Databricks CLI configured (`databricks auth login`)

### Install dependencies

```bash
uv sync --extra dev
```

### Deploy to Databricks

```bash
databricks bundle deploy
```

This builds the `arxiv_curator` wheel, uploads it to your workspace, and deploys all job definitions.

### Run a notebook as a Databricks job

```bash
# Run the arXiv data ingestion pipeline
databricks bundle run arxiv_data_ingestion_job

# Run the external model (DALL-E) notebook
databricks bundle run external_models_custom_provider_job
```

### Run tests locally

```bash
uv run pytest
```

### Lint & format

```bash
uv run pre-commit run --all-files
```

---

## Secrets Setup (for Notebook 1.4)

Notebook 1.4 requires an OpenAI API key stored in a Databricks secret scope. The scope name is derived automatically from your Databricks username (e.g. `pramodk.sola@gmail.com` → `pramodk_secrets`).

To set up your secret scope:

```bash
# Create the scope (one-time setup, replace with your username prefix)
databricks secrets create-scope {your_username}_secrets

# Add your OpenAI API key
databricks secrets put-secret {your_username}_secrets openai_key --string-value sk-...
```

> Get your OpenAI API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Note that API usage is billed separately from ChatGPT Plus.

---

## Key Technologies

| Technology | Role |
|---|---|
| **Databricks Serverless** | Notebook & job execution environment |
| **Unity Catalog** | Governed data & model storage |
| **Databricks Asset Bundles** | Infrastructure as code for jobs & deployments |
| **Delta Lake** | ACID-compliant storage for arXiv paper metadata |
| **Databricks AI Gateway** | Unified interface for all LLM providers |
| **MLflow** | Model deployment and experiment tracking |
| **arXiv API** | Source of AI/ML research paper metadata |
| **OpenAI DALL-E 3** | Image generation via External Models |
| **Pydantic** | Configuration validation |
| **Loguru** | Structured logging across all notebooks |
