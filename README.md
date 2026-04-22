# ⚡ DocForge AI & CiteRAG

> AI-powered document generation with automatic Notion publishing, version control, and a conversational RAG support agent with built-in RAGAS evaluation.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-blue?style=flat-square)
![Notion](https://img.shields.io/badge/Notion-API-black?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-Caching-red?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Version Control & Publishing](#version-control--publishing)
- [Flowchart Rendering](#flowchart-rendering)
- [Redis Caching](#redis-caching)
- [Departments & Document Types](#departments--document-types)
- [Notion Database Schema](#notion-database-schema)
- [CiteRAG — Conversational RAG Agent](#citerag--conversational-rag-agent)
- [RAGAS Evaluation & Exports](#ragas-evaluation--exports)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

**DocForge AI** is a full-stack AI document generation platform. It uses **Azure OpenAI (GPT-4.1 Mini)** to generate professional, department-specific documents and publishes them to a **Notion database** with full metadata tracking and automatic version control.

A second major subsystem — **CiteRAG** — is a tool-calling conversational RAG agent. It lets users ask questions about ingested documents, automatically creates Notion support tickets when confidence is low, and manages the entire ticket lifecycle through natural language.

---

## Features

### Document Generation
- **AI-Powered Generation**: Powered by Azure OpenAI (GPT-4.1 Mini).
- **100+ Document Types**: NDA, Privacy Policy, SLA, Employment Contract, and more.
- **Multi-Department**: Custom section structures for HR, Finance, Legal, Sales, IT, Operations, and more.
- **Notion Sync**: One-click publishing with auto-versioning and full metadata tracking.
- **Version Control**: Persistent Notion-based version history per Department + Doc Type combination.
- **Dynamic Flowcharts**: Mermaid syntax auto-detected and rendered as PNG image blocks in Notion.
- **Redis Caching**: Caches department lists, section templates, LLM-generated content, and Notion library responses.

### CiteRAG Support Agent
- **Persistent Session Memory**: 30-day conversation history backed by Redis, synchronized across cache hits and live agent turns.
- **Unanswered Questions Queue**: Automatically tracks questions the AI cannot answer with high confidence; users can file tickets for these later.
- **Direct-Shot Retrieval**: Vector search → LLM in a single round-trip for minimum latency.
- **LLM-First Routing**: All tool selection, security, and intent routing handled by the LLM via LangGraph — no hardcoded keyword lists.
- **Multi-language Support**: Handles Hindi, Hinglish, Marathi, and Gujarati queries natively.
- **3-Layer Security**: Azure Content Filter → LLM System Guard → Action-Specific Cache Guard.
- **Two-Stage Deduplication**: Prevents duplicate tickets using a fast Vector Pre-filter followed by an accurate LLM Judge.
- **Urgency Detection**: Automatically classifies ticket priority (High/Low) based on legal, security, or financial risk.
- **Streaming Responses**: `/api/rag/ask` supports NDJSON token streaming for low-latency feedback.

### RAGAS Evaluation
- **Faithfulness & Answer Relevancy**: Checks if RAG outputs are grounded in the retrieved context.
- **Context Precision & Recall**: Benchmarked against 15 curated ground truth pairs in `qa_dataset.json`.
- **Run History**: Last 50 evaluation runs stored in Redis with full per-run snapshots.
- **Export Options**: Evaluation data exportable to DOCX reports (`ragas_report_YYYYMMDD.docx`) or JSON.

---

## Project Structure

```
docForge_AI-main/
├── backend/
│   ├── agents/
│   │   └── agent_graph.py          # LangGraph orchestrator: intent routing + tools
│   ├── api/
│   │   ├── routes.py               # DocForge document generation endpoints
│   │   ├── agent_routes.py         # Ticket lifecycle & session memory routes
│   │   └── rag_routes.py           # CiteRAG Q&A and Evaluation routes
│   ├── core/
│   │   ├── config.py               # Settings loaded via Pydantic Settings
│   │   ├── llm.py                  # Azure OpenAI client singletons
│   │   ├── logger.py               # Structured logging configuration
│   │   └── vector.py               # Vector store & embedding client helpers
│   ├── models/
│   │   └── document_model.py       # SQLAlchemy/SQLModel data entities
│   ├── prompts/
│   │   ├── prompts.py              # LLM templates for 100+ doc types
│   │   └── quality_gates.py        # Post-generation content validation
│   ├── rag/
│   │   ├── ingest_service.py       # Notion → ChromaDB ingestion pipeline
│   │   ├── rag_service.py          # Core RAG retrieval tools
│   │   ├── ragas_scorer.py         # RAGAS metric calculation
│   │   ├── system_prompt.py        # CiteRAG agent identity & instructions
│   │   └── ticket_dedup.py         # Two-stage duplicate detection logic
│   ├── schemas/
│   │   ├── document_schema.py      # Pydantic request/response models
│   │   └── notion_schema.py        # Notion-specific API schemas
│   ├── services/
│   │   ├── db_service.py           # PostgreSQL database operations
│   │   ├── document_utils.py       # Formatting and metadata helpers
│   │   ├── generator.py            # LLM generation orchestration
│   │   ├── notion_service.py       # Notion API wrapper & versioning
│   │   └── redis_service.py        # Redis cache with graceful fallback
│   └── main.py                     # FastAPI entry point & lifespan manager
├── ui/
│   ├── app.py                      # Main Streamlit entry point
│   ├── components/                 # UI tabs (Chat, Generate, Library, etc.)
│   ├── services/                   # API clients for backend communication
│   └── utils/                      # Session and UI helper functions
├── chroma_db/                      # Persistent vector storage
├── docker-compose.yml              # Multi-container orchestration
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment variable template
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database
- Redis (optional — app works without it, caching is disabled)
- Azure OpenAI resource with `GPT-4.1 Mini` and `text-embedding-3-large` deployments
- Notion Internal Integration token — [notion.so/my-integrations](https://www.notion.so/my-integrations)
- Imgur Client ID (optional, for flowchart rendering) — [api.imgur.com/oauth2/addclient](https://api.imgur.com/oauth2/addclient)

### 1. Clone the repository
```bash
git clone https://github.com/Tilakvasani/docForge_AI.git
cd docForge_AI
```

### 2. Create a virtual environment
```bash
python3.11 -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Edit `.env` with your credentials — see [Environment Variables](#environment-variables) below.

### 5. Start the backend
```bash
uvicorn backend.main:app --reload
```

### 6. Start the frontend (new terminal)
```bash
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Docker Deployment

Deploy the full stack (FastAPI, Redis, Streamlit) in one command:
```bash
docker-compose up --build
```

| Service | Port |
|---------|------|
| FastAPI backend | `8000` |
| Streamlit frontend | `8501` |
| Redis | `6379` |

---

## Environment Variables

```env
# ── Notion ────────────────────────────────────────────────────────────────────
NOTION_TOKEN=secret_...
NOTION_DATABASE_ID=...           # Published documents DB (also used as RAG source)
NOTION_TICKET_DB_ID=...          # Support ticket tracking DB (CiteRAG)

# ── Azure OpenAI — LLM ────────────────────────────────────────────────────────
AZURE_OPENAI_LLM_KEY=...
AZURE_LLM_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_DEPLOYMENT_41_MINI=gpt-4.1-mini
AZURE_LLM_API_VERSION=2024-12-01-preview

# ── Azure OpenAI — Embeddings ─────────────────────────────────────────────────
AZURE_OPENAI_EMB_KEY=...
AZURE_EMB_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_EMB_DEPLOYMENT=text-embedding-3-large
AZURE_EMB_API_VERSION=2024-02-01

# ── Storage & Infrastructure ──────────────────────────────────────────────────
DATABASE_URL=postgresql://user:pass@localhost:5432/docforge
REDIS_URL=redis://localhost:6379/0   # Optional — app degrades gracefully without it
CHROMA_PATH=./chroma_db              # Optional — defaults to project root

# ── Optional & Misc ──────────────────────────────────────────────────────────
IMGUR_CLIENT_ID=...              # Enables Mermaid flowchart rendering in Notion
CORS_ALLOWED_ORIGINS=http://localhost:8501
APP_ENV=development              # development | production
LOG_LEVEL=INFO                   # INFO | DEBUG | WARNING
```

---

## API Reference

### Document Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/departments` | List all available departments |
| `GET` | `/api/sections/{doc_type}` | Get sections for a document type |
| `POST` | `/api/questions/generate` | Generate context questions for a section |
| `POST` | `/api/answers/save` | Save user answers for a section |
| `POST` | `/api/section/generate` | Generate section content via LLM + quality gate |
| `POST` | `/api/section/edit` | Re-generate or edit a specific section |
| `POST` | `/api/document/save` | Persist generated document to PostgreSQL |
| `POST` | `/api/document/publish` | Publish to Notion with auto version bump |
| `GET` | `/api/library/notion` | Fetch all previously published documents |

### CiteRAG
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/rag/ask` | Ask a question — routes to the correct tool, supports streaming |
| `POST` | `/api/rag/ingest` | Ingest Notion database into ChromaDB via embeddings |
| `GET` | `/api/rag/status` | ChromaDB chunk count, doc count, ingest lock status |
| `DELETE` | `/api/rag/cache` | Flush all RAG retrieval + session + answer caches |
| `POST` | `/api/rag/eval` | Run a manual RAGAS evaluation on a question |
| `GET` | `/api/rag/scores?key=` | Poll async RAGAS task scores by run key |
| `GET` | `/api/rag/eval/runs` | Browse last 50 stored evaluation run summaries |
| `GET` | `/api/rag/eval/runs/{run_id}` | Full snapshot of a specific evaluation run |

### Agent / Tickets
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/agent/tickets` | Fetch all Notion tickets (60s Redis cache) |
| `POST` | `/api/agent/tickets/update` | Update ticket status (Open / In Progress / Resolved) |
| `POST` | `/api/agent/ticket/create` | Create a new support ticket in Notion |
| `POST` | `/api/agent/memory` | Save or merge session memory for an agent session |
| `DELETE` | `/api/agent/dedup/flush` | No-op — dedup uses live Notion data, no cache to flush |

### Example — Publish a Document
```bash
curl -X POST "http://localhost:8000/api/document/publish" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc-123",
    "doc_type": "Employee Offer Letter",
    "department": "HR",
    "gen_doc_full": "Full document content here...",
    "company_context": {
      "company_name": "Acme Corp",
      "industry": "SaaS",
      "region": "India",
      "company_size": "50-200"
    }
  }'
```

**Response:**
```json
{
  "notion_url": "https://notion.so/page-id",
  "notion_page_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "version": 2
}
```

---

## Version Control & Publishing

DocForge implements automatic **Notion-based version control**. Every publish queries whether a document with the same **Department + Doc Type** already exists, then assigns the next version number automatically.

```mermaid
graph TD
    A[Publish: Dept = HR, Doc Type = Employee Offer Letter] --> B(Query Notion — filter by Dept + Doc Type, sort by Version desc, limit 1)
    B -->|No match found| C[Version = 1]
    B -->|Match found| D[Version = match + 1]
    C --> E[Upload image blocks + text content to Notion]
    D --> E
```

| Publish # | Department | Doc Type | Auto-Assigned Version |
|-----------|------------|----------|---------|
| 1st | HR | Employee Offer Letter | v1 |
| 2nd | HR | Employee Offer Letter | v2 |
| 1st | Finance | Invoice Template | v1 |

Each Department + Doc Type combination holds an independent version counter in Notion.

---

## Flowchart Rendering

Documents containing Mermaid blocks are parsed at publish time. DocForge detects ` ```mermaid ` syntax, renders it to PNG via Python, uploads the image to Imgur to obtain a public URL, then embeds it as a Notion image block.

If `IMGUR_CLIENT_ID` is not set, the system safely falls back to a formatted numbered list representation.

---

## Redis Caching

All cache operations degrade gracefully to no-ops if Redis is unavailable — the app continues working normally without it.

| Cache Key | Data Cached | TTL |
|-----------|-------------|-----|
| `docforge:departments` | Department + doc type list | 1 hour |
| `docforge:sections:{doc_type}` | Section list for a doc type | 1 hour |
| `docforge:questions:{sec_id}` | LLM-generated questions | 24 hours |
| `docforge:section:{sec_id}` | LLM-generated section content | 24 hours |
| `docforge:notion_library` | Published document library | 5 minutes |
| `docforge:rag:answer:{hash}` | RAG answer for a question+filter hash | 1 hour |
| `docforge:rag:session:{id}` | Multi-turn conversation history | 30 days |
| `docforge:agent:tickets` | Fetched Notion ticket list | 60 seconds |
| `docforge:agent:memory:{id}` | Agent session memory (user name, role, etc.) | 24 hours |
| `ragas:runs:{run_id}` | Full RAGAS evaluation run snapshot | 7 days |

---

## Departments & Document Types

### Supported Departments
`HR`, `Finance`, `Legal`, `Sales`, `Marketing`, `IT`, `Operations`, `Customer Support`, `Product Management`, `Procurement`

### Popular Document Types
| Doc Type | Description |
|----------|-------------|
| NDA | Non-Disclosure Agreement |
| Privacy Policy | GDPR/CCPA compliant external-facing policy |
| Terms of Service | Software/platform terms and conditions |
| Employment Contract | Full-time/part-time employment agreement |
| Employee Offer Letter | Formal job offer detailing compensation |
| SLA | Service Level Agreement with uptime commitments |
| Business Proposal | Client-facing sales proposal |
| Compliance Report | SOC2/HIPAA internal audit documentation |
| Invoice Template | Finance billing invoice structure |

---

## Notion Database Schema

Two Notion databases are required. Share both with your Notion integration.

### Published Documents DB (`NOTION_DATABASE_ID`)
| Property Name | Property Type | Notes |
|---------------|---------------|-------|
| Title | Title | Auto-generated: `{Doc Type} — {Company}` |
| Department | Select | HR, Legal, Finance, etc. |
| Doc Type | Rich Text | String name of the document type |
| Industry | Rich Text | Set via session state |
| Version | Number | Auto-incremented per Dept + Doc Type |
| Status | Select | `Generated` / `Draft` / `Published` |
| Created By | Rich Text | System identifier |

### Ticket DB (`NOTION_TICKET_DB_ID`)
| Property Name | Property Type | Notes |
|---------------|---------------|-------|
| Question | Title | The user's unanswered question |
| Ticket ID | Rich Text | 8-digit numeric string |
| Status | Select | Open / In Progress / Resolved |
| Priority | Select | High / Medium / Low |
| Summary | Rich Text | LLM-generated ticket summary |
| Session ID | Rich Text | Links ticket back to the originating session |
| Attempted Sources | Multi-select | Document sources tried during retrieval |
| Created | Date | Date of ticket creation |
| Assigned Owner | Rich Text | Defaults to "Support Team" |
| User Info | Rich Text | User identifier (defaults to "Anonymous") |

---

## CiteRAG — Conversational RAG Agent

A single LLM prompt governs all of CiteRAG: it selects a tool, executes it, and returns the answer in one pass — no pre-classification step, no separate rewrite step.

### Agent Tools

| Intent / Trigger | Tool | Description |
|------------------|------|-------------|
| Policy or knowledge questions | `search` | Standard vector retrieval + cited answer |
| "Compare Policy A to B" | `compare` | Retrieves both docs, builds a differences table |
| "Compare A, B, and C" | `multi_compare` | Pairwise comparison across 3+ documents |
| "Find risks in..." | `analyze` | Deep gap, risk, and contradiction analysis |
| "Summarize the SLA" | `summarize` | HyDE-guided summarization for executive summaries |
| "Read the full handbook" | `full_doc` | Max vector returns to reconstruct a complete document |
| Multi-part questions | `multi_query` | Splits complex queries into 2-5 sub-tasks |
| Low-confidence answers | `create_ticket` | Create ticket for a specific or pending question |
| "Resolve ticket #12345" | `update_ticket` | Updates status (Open / In Progress / Resolved) |
| "What did we talk about?" | `chat_history_summary` | Answers questions about the current conversation |
| Off-topic or injection attempts | `block_off_topic` | Security layer — returns a safe fallback response |
| "Nevermind" / "Stop" | `cancel` | Aborts a pending ticket creation flow |

### Deduplication Engine

CiteRAG uses a robust two-stage process to ensure no duplicate tickets are created in Notion:

1. **Stage 1 — Vector Pre-filter**: The new question is normalized and embedded. ChromaDB quickly finds the top 10 most similar open/in-progress tickets.
2. **Stage 2 — LLM Judge**: The LLM receives the candidates and performs a deep intent comparison. It only flags a duplicate if the *exact same entity* (person, policy, document) is being queried.

### Ingestion Pipeline

1. Call `POST /api/rag/ingest` (or let it auto-trigger on first question when ChromaDB is empty).
2. Reads all pages and blocks from the Notion database.
3. Formats lists, toggles, and tables as paragraph-aware text chunks.
4. Embeds chunks using `text-embedding-3-large` and stores them in ChromaDB.
5. Subsequent ingest calls skip unchanged chunks automatically via ID tracking.

---

## RAGAS Evaluation & Exports

The RAGAS tab in the UI provides a batch evaluation interface:

1. Import questions from `qa_dataset.json` or enter them manually.
2. Each question runs through the full RAG pipeline and is scored on four metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision**, and **Context Recall**.
3. Results are stored in Redis as a run snapshot for session history browsing.
4. Export the full report as a **DOCX file** (`ragas_report_YYYYMMDD.docx`) with an executive summary table followed by per-question breakdowns, or as raw **JSON**.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| AI / LLM | Azure OpenAI (GPT-4.1 Mini) | Document generation, agent routing, RAGAS scoring |
| Orchestration | LangGraph | Complex agent tool-calling and state management |
| Embeddings | Azure OpenAI (text-embedding-3-large) | Semantic vector generation for RAG |
| Vector Store | ChromaDB | Chunk storage and similarity retrieval |
| Backend | FastAPI + Pydantic Settings | REST API, request validation, configuration |
| Frontend | Streamlit | Multi-tab UI (Chat, Generate, Library, Agent, RAGAS) |
| Relational DB | PostgreSQL (asyncpg) | Persistent storage for departments, sections, Q&A |
| Document Store | Notion API | Published document library + ticket tracking |
| Cache | Redis | Session history, answer cache, rate limiting |
| Flowchart Render | Mermaid + Imgur | Converts Mermaid syntax to PNG for Notion |
| Reporting | python-docx | RAGAS report and document export to .docx |

---

## License
MIT License. Open for modification, iteration, and enterprise scaling.

---
Built with ⚡ by [Tilak Vasani](https://github.com/Tilakvasani)