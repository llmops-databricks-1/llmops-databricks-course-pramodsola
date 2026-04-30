# Arxiv Curator

An AI-powered research assistant built on Databricks that ingests, processes, and serves the latest AI/ML research papers from [arXiv](https://arxiv.org). It leverages Databricks' full LLMOps stack — model serving, Unity Catalog, Delta Lake, Vector Search, Lakebase, MLflow Tracing, and AI Gateway — to deliver a governed, production-ready agentic pipeline.

---

## What It Does

- Fetches the latest AI/ML papers from the arXiv API (`cs.AI`, `cs.LG` categories)
- Parses PDFs, chunks text, and builds a searchable vector index
- Serves a traced, stateful agent via Databricks Model Serving
- Uses Genie for natural language SQL queries over paper metadata
- Persists conversation history in Lakebase (managed PostgreSQL)
- Evaluates agent quality with MLflow Guidelines, Judges, and custom scorers
- Registers the agent to Unity Catalog as a versioned, deployable model
- Runs a daily monitoring job that scores all production traces and surfaces quality metrics on a Lakeview dashboard

---

## Project Structure

```
course-code-hub/
├── src/
│   └── arxiv_curator/
│       ├── config.py            # Pydantic config model + env-aware YAML loader
│       ├── data_processor.py    # Download, parse, and chunk arXiv papers
│       ├── vector_search.py     # Vector search endpoint + index management
│       ├── mcp.py               # ToolInfo dataclass + create_mcp_tools()
│       ├── memory.py            # LakebaseMemory — PostgreSQL session persistence
│       ├── agent.py             # ArxivAgent — traced agentic loop with MCP tools
│       └── evaluation.py        # Guidelines, judges, and custom scorers
├── notebooks/
│   ├── 1.1–1.4                  # Foundation models, provisioned throughput, ingestion
│   ├── 2.1–2.4                  # Context engineering, PDF parsing, chunking, vector search
│   ├── 3.1–3.6                  # Agent tools, RAG, MCP, Genie, session memory, UC functions
│   ├── 4.1–4.4                  # Tracing, custom agent, evaluation, log & register
│   ├── 5.1–5.2                  # Endpoint deployment (standard + Genie), SPN permissions
│   └── 6.1                      # Trace propagation — send sample queries to live endpoint
├── resources/
│   ├── deployment_scripts/      # Notebooks run as Databricks jobs (not interactive)
│   │   └── update_traces_aggregated.py   # Daily evaluation + view refresh pipeline
│   ├── dashboard/               # Lakeview dashboard definition
│   │   ├── agent_monitoring_dashboard.lvdash.json
│   │   └── agent_monitoring_dashboard.yml
│   └── *.yml                    # Databricks Asset Bundle job definitions
├── arxiv_agent.py               # MLflow pyfunc entry point for model serving
├── eval_inputs.txt              # Evaluation questions for agent testing
├── databricks.yml               # Databricks Asset Bundle root config
├── project_config.yml           # Per-environment settings (dev / acc / prd)
├── pyproject.toml               # Dependencies & build config
└── version.txt
```

---

## Configuration

```yaml
dev:
  catalog: mlops_dev
  schema: your_schema
  volume: arxiv_files
  llm_endpoint: databricks-llama-4-maverick
  embedding_endpoint: databricks-gte-large-en
  warehouse_id: ...
  vector_search_endpoint: llmops_course_vs_endpoint
  genie_space_id: ...
  lakebase_project_id: your-name-lakebase
  experiment_name: /Shared/llmops-course-pramodsola  # personal experiment, one per student
```

---

## Week 1 — Foundation Models & Data Ingestion

### What Was Built
- Explored Databricks Foundation Model APIs (pay-per-token vs provisioned throughput)
- Deployed a dedicated LLM serving endpoint
- Ingested arXiv paper metadata into a Delta table in Unity Catalog
- Registered OpenAI DALL-E 3 as an external model endpoint via AI Gateway

### Notebooks

| Notebook | What It Covers |
|---|---|
| **1.1 Foundation Models Overview** | Model access types, pricing, pay-per-token vs provisioned throughput |
| **1.2 Provisioned Throughput Deployment** | Deploy a dedicated LLaMA endpoint with guaranteed capacity |
| **1.3 Arxiv Data Ingestion** | Fetch papers from arXiv API, store metadata in Delta table |
| **1.4 External Models Custom Provider** | Register DALL-E 3 via Databricks External Models / AI Gateway |

---

## Week 2 — Context Engineering, RAG & Vector Search

### What Was Built

#### `DataProcessor` (`src/arxiv_curator/data_processor.py`)
End-to-end processing pipeline:
1. **Download** — fetches arXiv PDFs, uploads to Unity Catalog Volume
2. **Parse** — uses `ai_parse_document()` to extract structured content
3. **Chunk** — explodes parsed JSON into clean text chunks, writes to Delta with CDF enabled

#### `VectorSearchManager` (`src/arxiv_curator/vector_search.py`)
- Creates and manages the `arxiv_index` Delta Sync index
- Handles endpoint/index lifecycle and exposes `search()` for similarity queries

### Notebooks

| Notebook | What It Covers |
|---|---|
| **2.1 Context Engineering Theory** | RAG theory, context window limits, query rewriting, lost-in-the-middle problem |
| **2.2 PDF Parsing with AI Parse** | `ai_parse_document()`, downloading PDFs to Volumes, storing parsed JSON |
| **2.3 Chunking Strategies** | Fixed-size vs semantic chunking, overlap, text cleaning |
| **2.4 Embeddings & Vector Search** | Creating vector indexes, similarity search, hybrid search, reranking |

### RAG Pipeline

```
arXiv API → PDFs in Volume + arxiv_papers table
         → ai_parsed_docs_table
         → arxiv_chunks_table (CDF enabled)
         → Vector Search Index
         → Similarity / Hybrid / Reranked results
```

---

## Week 3 — Agent Tools, MCP, Genie & Session Memory

### What Was Built

#### `mcp.py` — Tool Infrastructure
- `ToolInfo` dataclass: wraps a tool name, OpenAI-compatible spec, and execution function
- `create_mcp_tools()`: connects to MCP servers and converts their tools into `ToolInfo` objects

#### `memory.py` — Session Persistence
- `LakebaseMemory`: stores and retrieves conversation history per `session_id` in Lakebase
- Uses `PostgresAPI` project/branch/endpoint model — personal project scales to 0 when idle
- Table: `session_messages (id, session_id, message_data JSONB, created_at)`

### Notebooks

| Notebook | What It Covers |
|---|---|
| **3.1 Custom Functions & Tools** | `ToolInfo`, `ToolRegistry`, `SimpleAgent` with agentic loop |
| **3.1b Simple RAG** | Vector search retrieval + LLM generation + multi-turn conversation |
| **3.2 MCP Integration** | `DatabricksMCPClient`, `create_mcp_tools()`, graceful fallback |
| **3.2b Genie** | Personal Genie space per user, NL queries over `arxiv_papers` |
| **3.3 Session Memory** | Personal Lakebase project, `LakebaseMemory`, stateful LLM chat |
| **3.4 SPN Authentication** | Reference only — admin SPN setup for Lakebase |
| **3.5 SPN in Action** | Reference only — using SPN credentials with `LakebaseMemory` |
| **3.6 UC Function Example** | Register and call Python UDFs in Unity Catalog |

---

## Week 4 — Tracing, Custom Agent & MLflow Registration

### What Was Built

#### `agent.py` — Production-Ready ArxivAgent
Full `mlflow.pyfunc.PythonModel` with:
- `@mlflow.trace(AGENT)` on `predict()` — root span for every request
- `@mlflow.trace(LLM)` on `call_llm()` — tracks every model call
- `@mlflow.trace(TOOL)` on `execute_tool()` — tracks every tool invocation
- `@mlflow.trace(CHAIN)` on `call_and_run_tools()` — tracks the full agentic loop
- Automatic session + request ID metadata on every trace
- MCP tools (Vector Search + Genie) loaded at init
- `LakebaseMemory` for stateful multi-turn conversations

#### `evaluation.py` — Evaluation Scorers
- `polite_tone_guideline` — binary: polite and professional tone
- `hook_in_post_guideline` — binary: engaging opening sentence
- `scope_guideline` — binary: stays on topic
- `word_count_check` — custom scorer: response under 350 words
- `mentions_papers` — custom scorer: response references research papers

#### `arxiv_agent.py` — MLflow Pyfunc Entry Point
Root-level file loaded by `mlflow.pyfunc.log_model()` for model serving deployment.

### Notebooks

| Notebook | What It Covers |
|---|---|
| **4.1 Tracing Implementation** | `@mlflow.trace`, span types, manual spans, metadata/tags, searching traces |
| **4.2 Custom Agent** | `ArxivAgent` with full tracing, MCP tools, multi-turn conversation, performance analysis |
| **4.3 Evaluation Theory** | Guidelines vs Judges, custom scorers, categorical judges, SIMBA alignment |
| **4.4 MLflow Log & Register** | Evaluate agent, log as pyfunc model, declare resources, register to Unity Catalog |

### MLflow Model Registration Flow

```
ArxivAgent
    ↓  mlflow.genai.evaluate()  (word_count_check, mentions_papers)
Evaluation metrics
    ↓  mlflow.pyfunc.log_model()
MLflow Run (arxiv-agent-{date})
    ↓  mlflow.register_model()
Unity Catalog: {catalog}.{schema}.arxiv_agent  v1
    ↓  set_registered_model_alias("latest-model")
Alias → version ready for serving
```

---

## Week 5 — Deployment, CI/CD & SPN Permissions

### What Was Built

#### `serving.py` — Serving Endpoint Utilities
- `serve_model()`: Creates or updates a Databricks serving endpoint with AI Gateway inference tables
- `get_endpoint_status()`: Returns endpoint readiness state

#### `utils/common.py` (`src/arxiv_curator/utils/common.py`) — Shared Utilities
- `get_widget()`: Safe widget access with fallback default — works in both interactive notebooks and job runs

#### Deployment Pipeline (`resources/register_deploy_agent.yml`)
Two-task job chaining `log_register_agent` → `deploy_agent`:
- Evaluates the agent, logs to MLflow, registers to Unity Catalog, then deploys to a serving endpoint
- Triggered automatically by the CD pipeline on merge to main

#### `agent.py` — `log_register_agent()` function
Extracted from notebook 4.4 into a reusable function:
- Declares all Databricks resources (LLM endpoint, embedding endpoint, vector index, table, warehouse, Genie)
- Logs model as pyfunc with code-based logging
- Registers to Unity Catalog and sets `latest-model` alias

#### `evaluation.py` — `evaluate_agent()` function
Reusable evaluation runner called by the deployment pipeline before logging.

### Notebooks

| Notebook | What It Covers |
|---|---|
| **5.1 Endpoint Deployment** | `agents.deploy()`, environment vars, secrets injection, schema validation, testing the live endpoint |
| **5.1 Endpoint Deployment (Genie)** | Variant that re-registers with `DatabricksGenieSpace` resource for auto-permission grant at deploy time |
| **5.2 SPN Permissions** | Reference — grant SPN access to Genie, Vector Search, SQL Warehouse |

### CI/CD Pipeline

```
PR to main
    ↓
CI (.github/workflows/ci.yml)
  - pre-commit + ruff linting
  - pytest
    ↓ (on merge)
CD (.github/workflows/cd.yml)
  - databricks bundle deploy → acc
  - databricks bundle deploy → prd + git tag
```

**GitHub Actions setup required (one-time):**
- Create environments `acc` and `prd` in GitHub repo Settings → Environments
- Add secrets per environment: `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`
- Add variable per environment: `DATABRICKS_HOST`
- Course workspace SPN secret scopes already created: `dev_SPN`, `acc_SPN`, `prd_SPN`

---

## Week 6 — Monitoring Pipeline & Lakeview Dashboard

### What Was Built

#### `notebooks/6.1_propagate_traces.py` — Trace Generator
Dispatches 13 representative research queries to the live serving endpoint (`arxiv-agent-pramodsola-{env}`) with a 2-second gap between calls. Generates a statistically diverse trace population for the evaluation pipeline to score.

#### `resources/deployment_scripts/update_traces_aggregated.py` — Evaluation Pipeline
End-to-end daily job that:
1. Reads all unevaluated traces from `trace_logs_{experiment_id}` (WHERE assessments IS NULL)
2. Scores **100% of traces** with heuristic scorers (`word_count_check`, `mentions_papers`) — zero LLM cost
3. Scores a **10% random sample** with the LLM judge (`polite_tone_guideline`) — cost-controlled
4. Writes scores back via `mlflow.log_feedback()` per trace
5. Issues `CREATE OR REPLACE VIEW arxiv_traces_aggregated` — one clean row per trace for dashboarding

#### `resources/update_traces_aggregated.yml` — Scheduled Job
Cron schedule `0 0 7 * * ?` (07:00 Amsterdam). Controlled by the `schedule_pause_status` DAB variable:
- `PAUSED` in dev and acc (run on demand)
- `UNPAUSED` in prd (runs automatically every day)

#### `resources/dashboard/agent_monitoring_dashboard.lvdash.json` — Lakeview Dashboard
Version-controlled JSON dashboard definition deployed via `databricks bundle deploy`. Reads the live `arxiv_traces_aggregated` view on every page load.

| Widget | Type | Metric |
|---|---|---|
| Total Traces | Counter | `COUNT(trace_id)` |
| Total Tokens Used | Counter | `SUM(total_tokens_used)` |
| Avg Latency (s) | Counter | `AVG(latency_seconds)` |
| Avg Tokens / Trace | Counter | `AVG(total_tokens_used)` |
| Token Usage Over Time | Line chart | Daily token sum by `DATE(request_time)` |
| Avg Latency Over Time | Line chart | Daily avg latency — detects regressions |
| Assessment Pass Rates | Pivot table | Date × word_count_check / mentions_papers / polite_tone |
| LLM Calls vs Tool Calls | Scatter plot | call_llm_exec_count vs tool_call_count, coloured by tokens |
| Trace Log Details | Table (25/page) | Per-trace drill-down with all quality scores |

### Notebooks

| Notebook | What It Covers |
|---|---|
| **6.1 Propagate Traces** | Send sample queries to live endpoint · populate trace table for monitoring |

### Monitoring Pipeline

```
6.1_propagate_traces.py  →  arxiv-agent-pramodsola-dev endpoint
                              │
                              │  AI Gateway auto-captures every request
                              ▼
                         MLflow trace store
                              │
                              │  [experiment_id] Trace Archive Job (every 15 min)
                              ▼
                    trace_logs_{experiment_id} Delta table
                              │
                              │  update_traces_aggregated.py (daily at 07:00)
                              ▼
              ┌───────────────────────────────────────┐
              │  Score 100%: word_count_check,        │
              │              mentions_papers           │
              │  Score  10%: polite_tone (LLM judge)  │
              │  mlflow.log_feedback() per trace       │
              └───────────────────────────────────────┘
                              │
                              ▼
              arxiv_traces_aggregated (live SQL view)
                              │
                              ▼
              [dev] Arxiv Agent Monitoring Dashboard
```

### Live Results (April 2026)

| Metric | Value |
|---|---|
| Total traces | 58 |
| Total tokens served | 94,392 |
| Avg latency | 8.13 s |
| Avg tokens / trace | 1,627 |
| Avg LLM calls / trace | 1.3 |
| Avg tool calls / trace | 0.3 |
| Mentions papers pass rate | 67.2% |
| Word count pass rate | 8.6% (threshold set at 350 words — requires recalibration) |

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| **Databricks Asset Bundles** | Deploy notebooks as jobs; infrastructure as code |
| **Databricks Serverless (Env 4/5)** | Python 3.12 runtime for notebooks and jobs |
| **Unity Catalog** | Governed storage for tables, volumes, and registered models |
| **Delta Lake** | ACID table storage with Change Data Feed |
| **Databricks AI Gateway** | Unified API layer across all LLM providers |
| **Databricks Foundation Model APIs** | Pay-per-token access to Llama 4, Llama 3.x |
| **Databricks Vector Search** | Managed ANN index with Delta Sync |
| **Databricks Genie** | Natural language SQL queries over Delta tables |
| **Databricks Lakebase** | Managed PostgreSQL (project/branch/endpoint), scales to 0 |
| **MLflow Tracing** | Span-level observability for GenAI apps |
| **MLflow genai.evaluate** | Guidelines, LLM judges, and custom scorers |
| **MLflow Model Registry** | Version, alias, and govern models in Unity Catalog |
| **MCP (Model Context Protocol)** | Standard interface for agent tool integration |
| **OpenAI SDK** | Client for Databricks-hosted LLM endpoints |
| **psycopg** | PostgreSQL driver for Lakebase connections |
| **Apache Spark** | Distributed data processing and Delta writes |
| **Pydantic** | Config validation via `ProjectConfig` |
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

### Run a notebook
```bash
# Week 4
databricks bundle run tracing_implementation_job
databricks bundle run custom_agent_job
databricks bundle run evaluation_theory_job
databricks bundle run mlflow_log_register_job

# Week 5
databricks bundle run endpoint_deployment_job           # deploy to serving endpoint
databricks bundle run endpoint_deployment_genie_job     # deploy Genie variant

# Week 6
databricks bundle run update-and-evaluate-traces        # run evaluation + view refresh manually
```

---

## Secrets Setup

Notebook 1.4 uses OpenAI DALL-E 3. Store your key in a Databricks secret scope derived from your email:

```bash
# e.g. pramodk.sola@gmail.com → pramodk_secrets
databricks secrets create-scope pramodk_secrets
databricks secrets put-secret pramodk_secrets openai_key --string-value sk-...
```
