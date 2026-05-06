"""
Document MCP Server — Tools + Resources + Prompts (all TODOs completed)
"""
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DocumentMCP", log_level="ERROR")

docs = {
    "deposition.md":  "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf":     "The report details the state of a 20m condenser tower.",
    "financials.docx":"These financials outline the project's budget and expenditures.",
    "outlook.pdf":    "This document presents the projected future performance of the system.",
    "plan.md":        "The plan outlines the steps for the project's implementation.",
    "spec.txt":       "These specifications define the technical requirements for the equipment.",
}

# ── TOOLS ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def read_doc(doc_id: str) -> str:
    """Read the full contents of a document by its ID (filename)."""
    if doc_id not in docs:
        return json.dumps({"error": f"Document '{doc_id}' not found. Available: {list(docs.keys())}"})
    return json.dumps({"doc_id": doc_id, "content": docs[doc_id]})


@mcp.tool()
def edit_doc(doc_id: str, new_content: str) -> str:
    """Edit/replace the content of an existing document by its ID."""
    if doc_id not in docs:
        return json.dumps({"error": f"Document '{doc_id}' not found."})
    docs[doc_id] = new_content
    return json.dumps({"doc_id": doc_id, "status": "updated", "content": new_content})


# ── RESOURCES ─────────────────────────────────────────────────────────────────

@mcp.resource("docs://documents")
def list_documents() -> str:
    """Returns the list of all available document IDs."""
    return json.dumps(list(docs.keys()))


@mcp.resource("docs://documents/{doc_id}")
def get_document(doc_id: str) -> str:
    """Returns the content of a specific document by ID."""
    if doc_id not in docs:
        return json.dumps({"error": f"Document '{doc_id}' not found."})
    return docs[doc_id]


# ── PROMPTS ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def rewrite_as_markdown(doc_id: str) -> str:
    """
    /rewrite_as_markdown — Rewrite a document in clean Markdown format.
    Usage: /rewrite_as_markdown report.pdf
    """
    content = docs.get(doc_id, "")
    return f"""Rewrite the following document in clean, well-structured Markdown format.
Use proper headings, bullet points, bold for key terms, and code blocks where appropriate.

Document ID: {doc_id}
Content:
{content}

Output ONLY the rewritten Markdown — no preamble or explanation."""


@mcp.prompt()
def summarize(doc_id: str) -> str:
    """
    /summarize — Produce a concise summary of a document.
    Usage: /summarize financials.docx
    """
    content = docs.get(doc_id, "")
    return f"""Summarize the following document concisely.
Provide:
1. A 2-sentence executive summary
2. 3-5 key bullet points
3. Any action items or important dates mentioned

Document ID: {doc_id}
Content:
{content}"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
