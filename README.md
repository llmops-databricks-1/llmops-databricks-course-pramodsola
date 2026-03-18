# Arxiv Curator

An AI-powered research assistant built on Databricks that ingests, processes, and serves the latest AI/ML research papers from [arXiv](https://arxiv.org). It leverages Databricks' full LLMOps stack — model serving, Unity Catalog, Delta Lake, Vector Search, and AI Gateway — to deliver a governed, production-ready pipeline.

---

## What It Does

- Fetches the latest AI/ML papers from the arXiv API (`cs.AI`, `cs.LG` categories)
- Stores paper metadata in Delta tables in Unity Catalog
- Serves LLMs via Databricks Foundation Model APIs and custom external model endpoints
- Uses Databricks AI Gateway as a unified interface for all model providers
- Supports image generation via OpenAI DALL-E 3 registered as an external endpoint

---

## Project Structure

```
course-code-hub/
├── src/
│   └── arxiv_curator/
│       ├── __init__.py
│       └── config.py            # Pydantic config model + env-aware YAML loader
├── notebooks/
│   ├── 1.1_foundation_models_overview.py     # Explore model access types & pricing
│   ├── 1.2_provisioned_throughput_deployment.py  # Deploy dedicated LLM endpoint
│   ├── 1.3_arxiv_data_ingestion.py           # Ingest papers → Delta table
│   └── 1.4_external_models_custom_provider.py    # DALL-E via External Models API
├── resources/
│   ├── arxiv_data_ingestion_job.yml          # Bundle job for notebook 1.3
│   └── external_models_custom_provider_job.yml   # Bundle job for notebook 1.4
├── tests/
├── databricks.yml               # Databricks Asset Bundle root config
├── project_config.yml           # Per-environment settings (dev / acc / prd)
├── pyproject.toml               # Dependencies & build config
└── version.txt
```

---

## Configuration

Environment-specific settings are managed in `project_config.yml` and loaded at runtime via the `arxiv_curator.config` module. The active environment (`dev`, `acc`, `prd`) is resolved from the Databricks job widget, falling back to `dev`.

```yaml
dev:
  catalog: mlops_dev
  schema: your_schema
  volume: arxiv_files
  llm_endpoint: databricks-llama-4-maverick
  embedding_endpoint: databricks-gte-large-en
  warehouse_id: ...
  vector_search_endpoint: llmops_course_vs_endpoint
```

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| **Databricks Asset Bundles** | Deploy notebooks as jobs; manage infrastructure as code |
| **Databricks Serverless (Env 4)** | Python 3.12 runtime for notebook & job execution |
| **Unity Catalog** | Governed storage for Delta tables, volumes, and models |
| **Delta Lake** | ACID-compliant table storage for arXiv paper metadata |
| **Databricks AI Gateway** | Unified API layer across all LLM providers |
| **Databricks Foundation Model APIs** | Pay-per-token access to Llama 4, Llama 3.x models |
| **Databricks Provisioned Throughput** | Dedicated capacity serving endpoint for LLaMA 3.2 1B |
| **MLflow Deployments** | Register and manage external model endpoints |
| **OpenAI DALL-E 3** | Image generation via Databricks External Models API |
| **Databricks Secrets** | Secure storage for API keys (no hardcoded credentials) |
| **arXiv Python API** | Fetch research paper metadata from arXiv |
| **Apache Spark** | Distributed data processing and Delta table writes |
| **Pydantic** | Config validation via `ProjectConfig` model |
| **Loguru** | Structured logging across all notebooks |
| **uv** | Dependency management and tool runner |

---

## Setup

### Prerequisites

- Python 3.12
- [`uv`](https://github.com/astral-sh/uv)
- Databricks CLI authenticated (`databricks auth login`)

### Install

```bash
uv sync --extra dev
```

### Deploy

```bash
databricks bundle deploy
```

Builds the `arxiv_curator` wheel and deploys all job resources to your Databricks workspace.

### Run

```bash
databricks bundle run arxiv_data_ingestion_job
databricks bundle run external_models_custom_provider_job
```

---

## Secrets Setup

Notebook 1.4 uses OpenAI's DALL-E 3 via Databricks External Models. The API key is stored in a Databricks secret scope — never hardcoded in any file.

The scope name is **derived automatically at runtime** from the logged-in user's Databricks email — no code changes needed:

| Email | Scope used |
|---|---|
| `john.doe@company.com` | `john_doe_secrets` |
| `pramodk.sola@gmail.com` | `pramodk_secrets` |

Each user must create their scope once and add their OpenAI key:

```bash
# Step 1 — derive your scope name:
#   take the part before @, replace dots with underscores, append _secrets
#   e.g. john.doe@company.com -> john_doe_secrets

# Step 2 — create the scope (one-time)
databricks secrets create-scope {your_username}_secrets

# Step 3 — add your OpenAI API key
databricks secrets put-secret {your_username}_secrets openai_key --string-value sk-...
```

> Get your OpenAI API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
> Note: OpenAI API usage is billed separately from ChatGPT Plus — add credits at `platform.openai.com/settings/organization/billing` before running.
