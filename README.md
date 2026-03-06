# 📄 AI Document Generator

Generate 100+ industry-ready documents using LangChain + Groq and store them in Notion.

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/ai-doc-generator.git
cd ai-doc-generator

# 2. Copy env file and add your keys
cp .env.example .env

# 3. Run everything with Docker
docker-compose up --build
```

- **Streamlit UI** → http://localhost:8501  
- **FastAPI Docs** → http://localhost:8000/docs  

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key |
| `NOTION_API_KEY` | Your Notion integration token |
| `NOTION_DATABASE_ID` | Your Notion database ID |
| `REDIS_URL` | Redis connection URL |

## 📁 Project Structure

```
ai-doc-generator/
├── backend/          # FastAPI backend
├── prompts/          # LangChain prompt templates
├── ui/               # Streamlit frontend
├── postman/          # API collection
├── tests/            # Unit tests
└── docker-compose.yml
```

## 🏭 Supported Industries
telecom | saas | healthcare | finance | retail

## 📋 Document Types
SOP | Policy | Proposal | SOW | Incident Report | FAQ | Business Case | Security Policy | KPI Report | Runbook
