# ⚡ DocForge AI

> AI-powered document generation with automatic Notion publishing, version control, and a conversational RAG support agent.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-blue?style=flat-square)
![Notion](https://img.shields.io/badge/Notion-API-black?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-Caching-red?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Version Control](#version-control)
- [Flowchart Rendering](#flowchart-rendering)
- [Redis Caching](#redis-caching)
- [Departments & Document Types](#departments--document-types)
- [Notion Database Schema](#notion-database-schema)
- [CiteRAG — Conversational RAG Agent](#citerag--conversational-rag-agent)
- [Tech Stack](#tech-stack)

---

## Overview

**DocForge AI** is a full-stack AI document generation platform. It uses **Azure OpenAI** to generate professional, department-specific documents and publishes them to a **Notion database** with full metadata tracking and automatic version control.

A second major subsystem — **CiteRAG** — is a tool-calling conversational RAG agent. It lets users ask questions about ingested documents, automatically creates Notion support tickets when confidence is low, and manages the entire ticket lifecycle through natural language.

---

## Features

### Document Generation
- AI-powered generation via Azure OpenAI (GPT-4.1 Mini)
- 100 document types — NDA, Privacy Policy, SLA, Employment Contract, and more
- Multi-department support — HR, Finance, Legal, Sales, IT, Operations, and more
- One-click Notion publishing with full metadata (title, type, department, version, word count)
- Automatic version control — same Department + Doc Type increments automatically
- Redis caching for fast repeated generation and deduplication
- Mermaid flowchart rendering → PNG → Imgur → Notion image blocks
- DOCX export via `docx_builder.py`
- Modern dark UI built with Streamlit
- Docker-ready with `docker-compose.yml`

### CiteRAG Agent
- Conversational Q&A over ingested Notion documents
- Full chat history maintained in Redis across turns
- Understands contextual references ("the first one", "both of them")
- Auto-creates support tickets in Notion when answer confidence is low
- Ticket lifecycle management — create, update status, bulk-update — via natural language
- LLM-based duplicate ticket detection before creating new tickets
- RAGAS evaluation for answer quality scoring

---

## Project Structure

```
docForge_AI-main/
├── backend/
│   ├── agents/
│   │   └── agent_graph.py          # Tool-calling agent: LLM loop, tool executors, Redis history
│   ├── api/
│   │   ├── routes.py               # Core document generation endpoints
│   │   ├── agent_routes.py         # Ticket CRUD + Notion REST helpers
│   │   └── rag_routes.py           # /ask, /ingest, /status, /eval, /cache
│   ├── core/
│   │   ├── config.py               # Settings & .env loading
│   │   └── logger.py               # Logging setup
│   ├── models/
│   │   └── document_model.py       # Document dataclass
│   ├── prompts/
│   │   ├── prompts.py              # Prompt templates per doc type
│   │   └── quality_gates.py        # Output quality validation
│   ├── rag/
│   │   ├── rag_service.py          # Core RAG: vector retrieval + answer generation
│   │   ├── ingest_service.py       # Notion → ChromaDB ingest pipeline
│   │   ├── ragas_scorer.py         # RAGAS evaluation scoring
│   │   ├── ticket_dedup.py         # LLM-based duplicate ticket detection
│   │   └── system_prompt.py        # RAG system prompt
│   ├── schemas/
│   │   ├── document_schema.py      # Request/response Pydantic schemas
│   │   └── notion_schema.py        # Notion publish schema
│   ├── services/
│   │   ├── generator.py            # Azure OpenAI generation logic
│   │   ├── notion_service.py       # Notion API + version control
│   │   ├── redis_service.py        # Redis caching layer
│   │   ├── document_utils.py       # Document utility helpers
│   │   └── db_service.py           # Database service
│   └── main.py                     # FastAPI app entry point
├── ui/
│   └── streamlit_app.py            # Streamlit frontend
├── all_document/                   # Sample documents by department
├── chroma_db/                      # ChromaDB vector store
├── docx_builder.py                 # DOCX export helper
├── flowchart_renderer.py           # Mermaid → PNG renderer
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, for caching)
- Azure OpenAI resource with GPT-4.1 Mini and text-embedding-3-large deployments
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

Edit `.env` with your credentials (see [Environment Variables](#environment-variables)).

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

### 6. Start the frontend (new terminal)

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Docker Deployment

```bash
docker-compose up --build
```

This starts:

| Service | Port |
|---------|------|
| FastAPI backend | `8000` |
| Streamlit frontend | `8501` |
| Redis | `6379` |

---

## Environment Variables

```env
# Notion
NOTION_TOKEN=secret_...
NOTION_DATABASE_ID=...           # Source documents / published docs DB
NOTION_TICKET_DB_ID=...          # Support ticket tracking DB

# Redis
REDIS_URL=redis://localhost:6379

# Azure OpenAI — LLM (document generation + agent)
AZURE_OPENAI_LLM_KEY=...
AZURE_LLM_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_DEPLOYMENT_41_MINI=gpt-4o-mini
AZURE_LLM_API_VERSION=2024-12-01-preview

# Azure OpenAI — Embeddings (RAG)
AZURE_OPENAI_EMB_KEY=...
AZURE_EMB_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_EMB_DEPLOYMENT=text-embedding-3-large
AZURE_EMB_API_VERSION=2023-05-15

# Optional
IMGUR_CLIENT_ID=...              # Enables flowchart image rendering
DATABASE_URL=...                 # If using a relational DB
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
| `POST` | `/api/questions/generate` | Generate questions for a section |
| `POST` | `/api/answers/save` | Save user answers |
| `POST` | `/api/section/generate` | Generate section content via LLM |
| `POST` | `/api/section/edit` | Edit a generated section |
| `POST` | `/api/document/save` | Save full document to DB |
| `POST` | `/api/document/publish` | Publish document to Notion (with auto-versioning) |
| `GET` | `/api/library/notion` | Fetch all published documents from Notion |

### CiteRAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/rag/ask` | Conversational Q&A via tool-calling agent |
| `POST` | `/api/rag/ingest` | Ingest Notion docs into ChromaDB `{ "force": true }` |
| `GET` | `/api/rag/status` | ChromaDB collection stats |
| `DELETE` | `/api/rag/cache` | Flush retrieval/answer/session caches |
| `POST` | `/api/rag/eval` | Run manual RAGAS evaluation |
| `GET` | `/api/rag/scores?key=` | Poll RAGAS scores |

### Agent / Tickets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/agent/tickets` | List all Notion tickets (cached 60s) |
| `POST` | `/api/agent/tickets/update` | Update ticket `{ "ticket_id": "...", "status": "Resolved" }` |
| `GET` | `/api/agent/memory?session_id=` | Agent memory for a session |
| `POST` | `/api/agent/ticket/create` | Create ticket directly (internal) |
| `DELETE` | `/api/agent/dedup/flush` | Clear dedup cache (admin) |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

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

## Version Control

DocForge AI implements automatic Notion-based version control. Every publish checks whether a document with the same **Department + Doc Type** already exists.

```
User publishes: Department = HR, Doc Type = Employee Offer Letter
        │
        ▼
_get_next_version("HR", "Employee Offer Letter")
        │
        ▼
Query Notion: filter by Department + Doc Type, sort Version desc, limit 1
        │
        ├── No existing doc  →  version = 1
        └── Existing doc     →  version = existing_version + 1
        │
        ▼
Publish new Notion page with Version = version
```

**Example scenario:**

| Publish # | Department | Doc Type | Version |
|-----------|------------|----------|---------|
| 1st | HR | Employee Offer Letter | v1 |
| 2nd | HR | Employee Offer Letter | v2 |
| 3rd | HR | Employee Offer Letter | v3 |
| 1st | Finance | Invoice Template | v1 |
| 2nd | Finance | Invoice Template | v2 |

Each Department + Doc Type combination has its own independent version counter.

---

## Flowchart Rendering

Documents containing Mermaid blocks are automatically rendered as images in Notion.

```
Document content with ```mermaid ... ``` blocks
        │
        ▼
flowchart_renderer.py  →  mermaid_to_png_bytes()
        │
        ▼
PNG bytes uploaded to Imgur (anonymous)
        │
        ├── Upload success  →  Notion image block with Imgur URL
        └── Upload fail / no IMGUR_CLIENT_ID
                         →  Numbered step-list callout (fallback)
```

Set `IMGUR_CLIENT_ID` in `.env` to enable image rendering. Without it, flowcharts fall back to readable step lists.

---

## Redis Caching

| Cache Key | What's Cached | Invalidated When |
|-----------|---------------|-----------------|
| `departments` | All department list | Never (stable) |
| `sections:{doc_type}` | Sections for a doc type | Never (stable) |
| `questions:{sec_id}` | Generated questions | Never per session |
| `section_content:{sec_id}` | LLM-generated section text | On answer save or section edit |
| `notion_library` | All published Notion docs | On new document publish (5-min TTL) |

---

## Departments & Document Types

### Supported Departments

HR, Finance, Legal, Sales, Marketing, IT, Operations, Customer Support, Product Management, Procurement

### Supported Document Types

| Type | Description |
|------|-------------|
| NDA | Non-Disclosure Agreement |
| Privacy Policy | GDPR/CCPA compliant privacy policy |
| Terms of Service | Platform terms and conditions |
| Employment Contract | Full-time/part-time employment agreement |
| Employee Offer Letter | Formal job offer with compensation details |
| SLA | Service Level Agreement with uptime clauses |
| Business Proposal | Investment or partnership proposal |
| Technical Spec | API or system technical specification |
| Project Charter | Project scope and objectives document |
| Risk Assessment | Security or operational risk report |
| Compliance Report | SOC2, GDPR, HIPAA compliance audit |
| Invoice Template | Professional billing invoice |
| Partnership Agreement | Revenue-share partnership contract |

---

## Notion Database Schema

### Published Documents DB

| Field | Type | Description |
|-------|------|-------------|
| Title | Title | `{Doc Type} — {Company Name}` |
| Department | Select | HR, Finance, Legal, etc. |
| Doc Type | Rich Text | Document type string |
| Industry | Rich Text | Company industry |
| Version | Number | Auto-incremented per dept + doc type |
| Status | Select | Generated / Draft / Reviewed / Archived |
| Created By | Rich Text | "DocForge AI" |
| Word Count | Number | Auto-calculated from content |

### Ticket DB (CiteRAG)

| Property | Type | Options |
|----------|------|---------|
| Question | Title | — |
| Status | Select | Open, In Progress, Resolved |
| Priority | Select | High, Medium, Low |
| Summary | Rich Text | — |
| Attempted Sources | Multi-select | — |
| Session ID | Rich Text | — |
| Created | Date | — |
| Assigned Owner | Rich Text | — |
| User Info | Rich Text | — |

---

## CiteRAG — Conversational RAG Agent

CiteRAG handles every user turn in one LLM call, maintaining full chat history in Redis so it understands contextual references across turns.

### Architecture

```
User message
    │
    ▼
POST /api/rag/ask
    │
    ├─ RAG pipeline → document answer + confidence score
    │
    └─ run_agent() ──► LLM sees: [system prompt] + [chat history] + [user message]
                           │
                           └─ LLM picks one tool:
                                search()
                                create_ticket()
                                select_ticket(index)
                                create_all_tickets()
                                update_ticket(status, ticket_index?)
                                cancel()
                           │
                           └─ Tool executes → reply saved to Redis → response returned
```

### Agent Tools

| Tool | Triggered When | Action |
|------|---------------|--------|
| `search(question)` | Any doc/policy question | Returns RAG answer; silently saves unanswered questions |
| `create_ticket()` | "create ticket", "ticket banao" | Shows list if 2+ unanswered; creates directly if 1 |
| `select_ticket(index)` | User picks number from list | Creates ticket for that specific question |
| `create_all_tickets()` | "all", "both", "sabhi" | Creates tickets for all saved unanswered questions |
| `update_ticket(status, index?)` | "resolved", "in progress" | Updates one ticket or shows selection list |
| `cancel()` | "cancel", "no", "chodo" | Cancels flow; keeps saved questions for later |

### Ingest Pipeline

The `/api/rag/ingest` endpoint triggers `ingest_service.py`, which:

1. Fetches all pages from `NOTION_DATABASE_ID`
2. Extracts text block-by-block (headings, paragraphs, lists, tables, toggles)
3. Chunks at ~800 chars with 150-char overlap (paragraph-aware)
4. Embeds in batches of 64 using `text-embedding-3-large`
5. Upserts into ChromaDB with deterministic IDs (safe to re-run)
6. Stores `{ total_docs, total_chunks, ingested_at }` in Redis

```bash
curl -X POST http://localhost:8000/api/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

### Testing Checklist

- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `POST /api/rag/ingest` with `{"force": true}` — check logs for chunk count
- [ ] `GET /api/rag/status` — shows `total_chunks > 0`
- [ ] Ask a question that IS in the docs → answer returned, no ticket prompt
- [ ] Ask a question NOT in docs → low-confidence response, question saved silently
- [ ] Say "create ticket" → if 1 question: created directly; if 2+: numbered list shown
- [ ] Say a number → that ticket created; remaining questions stay in queue
- [ ] Say "all" → all remaining questions get tickets
- [ ] Create 2 tickets, say "resolved" → selection list shown
- [ ] Say "1" → only first ticket updated in Notion
- [ ] Say "all" after list → both tickets updated
- [ ] `GET /api/agent/tickets` → tickets visible with correct status

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Generation | Azure OpenAI GPT-4.1 Mini |
| Embeddings | Azure OpenAI text-embedding-3-large |
| Vector Store | ChromaDB |
| Backend | FastAPI + Python 3.11 |
| Frontend | Streamlit |
| Document Store | Notion API |
| Caching / History | Redis |
| Flowcharts | Mermaid → Imgur → Notion |
| Containerization | Docker + Docker Compose |
| DOCX Export | python-docx via `docx_builder.py` |

---

## License

MIT License — feel free to use, modify, and distribute.

---

Built with ⚡ by [Tilak Vasani](https://github.com/Tilakvasani)
