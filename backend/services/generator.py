"""
DocForge AI — generator.py  v6.0
════════════════════════════════════════════════════════════════
Section-type-aware generation pipeline.

Section types handled:
  text        → 0–3 contextual questions → plain-text prose
  table       → 1–3 data questions       → pipe-format markdown table
  flowchart   → 1–3 process questions    → Mermaid flowchart (```mermaid block)
  raci        → 1–2 role questions       → pipe-format RACI matrix table
  signature   → 0 questions              → formatted sign-off block (plain text)

Detection priority:
  1. DOC_STRUCTURE_METADATA from docforge_prompts.py  (per doc_type, per flag)
  2. SECTION_TYPE_PATTERNS keyword fallback            (section name heuristics)
  3. Default → text
"""

import re
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.core.config import settings
from backend.core.logger import logger
from backend.services.db_service import (
    save_questions, save_answers, get_qa_by_sec_id,
    update_section_content, get_generated_document,
)
from backend.services.document_utils import (
    markdown_to_plain_text, get_words_per_section,
)
from backend.schemas.document_schema import (
    GenerateQuestionsRequest, SaveAnswersRequest,
    GenerateSectionRequest, EditSectionRequest,
)

# Import your metadata dict from the prompts file
from prompts.templates import DOC_STRUCTURE_METADATA


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION TYPE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SECTION_TYPE_TEXT       = "text"
SECTION_TYPE_TABLE      = "table"
SECTION_TYPE_FLOWCHART  = "flowchart"
SECTION_TYPE_RACI       = "raci"
SECTION_TYPE_SIGNATURE  = "signature"


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD FALLBACK PATTERNS
#  Used when doc_type is not in DOC_STRUCTURE_METADATA or as a secondary check
#  for which SECTION within a doc needs which type.
# ─────────────────────────────────────────────────────────────────────────────

SECTION_TYPE_PATTERNS = {
    SECTION_TYPE_SIGNATURE: [
        "sign", "approval", "sign-off", "signoff", "authoris", "authoriz",
        "witness", "acknowledgement", "acknowledgment",
    ],
    SECTION_TYPE_RACI: [
        "raci", "responsibility matrix", "responsibility chart",
        "roles and responsibilities", "responsibility assignment",
    ],
    SECTION_TYPE_FLOWCHART: [
        "process flow", "workflow", "flowchart", "procedure flow",
        "escalation path", "escalation flow", "approval flow",
        "step-by-step process", "lifecycle", "onboarding journey",
        "implementation timeline", "release pipeline", "response flow",
        "clearance process", "exit process", "incident response",
    ],
    SECTION_TYPE_TABLE: [
        "table", "schedule", "matrix", "register", "checklist",
        "scorecard", "log", "budget", "cost breakdown", "fee schedule",
        "rate card", "pricing", "comparison", "summary table",
        "kpi", "milestone", "inventory", "asset list", "line items",
        "commission", "reimbursement", "leave entitlement", "competency",
        "color palette", "brand colors", "data collection",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  LLM FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.7) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_LLM_ENDPOINT,
        api_key=settings.AZURE_OPENAI_LLM_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        api_version="2024-12-01-preview",
        temperature=temperature,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION TYPE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detect_section_type(doc_type: str, section_name: str) -> str:
    """
    Determine the rendering type for a section.

    Priority:
      1. DOC_STRUCTURE_METADATA flags keyed by doc_type
         — but only if the section_name matches the expected section for that flag
         (uses keyword fallback to confirm, except for signature which always wins)
      2. Section name keyword patterns
      3. Default: text
    """
    sec_lower = section_name.lower()

    # ── Priority 1: metadata flags ───────────────────────────────────────────
    meta = DOC_STRUCTURE_METADATA.get(doc_type, {})

    # Signature check first — section name is the deciding signal
    if _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_SIGNATURE]):
        return SECTION_TYPE_SIGNATURE

    # For table/flowchart/raci, the metadata flag tells us the doc CAN have it,
    # but the section name confirms WHICH section it applies to.
    if meta.get("has_raci") and _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_RACI]):
        return SECTION_TYPE_RACI

    if meta.get("has_flowchart") and _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_FLOWCHART]):
        return SECTION_TYPE_FLOWCHART

    if meta.get("has_table") and _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_TABLE]):
        return SECTION_TYPE_TABLE

    # ── Priority 2: section name keywords alone ───────────────────────────────
    for stype in [SECTION_TYPE_RACI, SECTION_TYPE_FLOWCHART, SECTION_TYPE_TABLE]:
        if _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[stype]):
            return stype

    return SECTION_TYPE_TEXT


def _matches_keywords(text: str, keywords: list) -> bool:
    return any(kw in text for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
#  QUESTION GENERATION PROMPTS  (one per section type)
# ─────────────────────────────────────────────────────────────────────────────

# ── Text ─────────────────────────────────────────────────────────────────────

TEXT_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "company_size", "region"],
    template="""You are an expert enterprise documentation specialist.

Decide how many questions (0, 1, 2, or 3) are needed to fill this section, then write exactly that many.

Document Type : {doc_type}
Department    : {department}
Section       : {section_name}
Company       : {company_name} | Industry: {industry} | Size: {company_size} | Region: {region}

Decision rules:
- 0 questions: Purely structural section — intro boilerplate, version history, disclaimer.
  Respond with exactly: NONE
- 1 question: One concrete detail unlocks the whole section (e.g. effective date, policy owner)
- 2 questions: Two distinct pieces of context needed
- 3 questions: Complex section needing multiple specifics (maximum — do not exceed 3)

Quality rules:
- Ask for SPECIFIC data: names, dates, numbers, percentages, policy details
- Do NOT ask generic questions like "describe the company" or "what is the purpose"
- Each question must unlock a DIFFERENT piece of information
- Questions must be directly relevant to writing the {section_name} of a {doc_type}

Output: one question per line, no numbering, no bullet points, no extra text.
If 0 questions: respond NONE

Respond now:"""
)


# ── Table ─────────────────────────────────────────────────────────────────────

TABLE_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "table_hint"],
    template="""You are an expert enterprise documentation specialist.

This section will be rendered as a DATA TABLE. Write 1–3 questions to collect the exact
row data that should appear in the table.

Document Type : {doc_type}
Department    : {department}
Section       : {section_name}
Company       : {company_name} | Industry: {industry}
Table hint    : {table_hint}

Rules:
- Ask for the ACTUAL DATA ROWS, not descriptions or explanations
- Be specific about the format you expect
  Good: "List each expense item with: date, category, description, and amount (one item per line)"
  Bad:  "What expenses were incurred?"
- If the table has a natural primary key (employee name, vendor name, product), ask for it explicitly
- Maximum 3 questions
- One question per line, no numbering, no bullet points

Respond now:"""
)


# ── Flowchart ─────────────────────────────────────────────────────────────────

FLOWCHART_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "flowchart_hint"],
    template="""You are an expert enterprise documentation specialist.

This section will be rendered as a PROCESS FLOWCHART (Mermaid diagram).
Write 1–3 questions to collect the process steps and decision points.

Document Type  : {doc_type}
Department     : {department}
Section        : {section_name}
Company        : {company_name} | Industry: {industry}
Flowchart hint : {flowchart_hint}

Rules:
- Ask about the SEQUENCE OF STEPS in the process
- Ask about DECISION POINTS (yes/no branches, approvals, conditions)
- Ask about the ROLES or SYSTEMS involved at each step
- Maximum 3 questions
- One question per line, no numbering, no bullet points

Good example questions for a process flow:
  "List the sequential steps in this process from start to finish (e.g. Step 1: Submit request, Step 2: Manager review...)"
  "At which steps does the process branch based on a yes/no decision? What are the two outcomes?"
  "Which team or role is responsible for each step?"

Respond now:"""
)


# ── RACI ──────────────────────────────────────────────────────────────────────

RACI_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "raci_hint"],
    template="""You are an expert enterprise documentation specialist.

This section will be rendered as a RACI RESPONSIBILITY MATRIX TABLE.
Write 1–2 questions to collect role and activity information.

Document Type : {doc_type}
Department    : {department}
Section       : {section_name}
Company       : {company_name} | Industry: {industry}
RACI hint     : {raci_hint}

Rules:
- Question 1: Ask for the list of ROLES or JOB TITLES involved in this process
- Question 2 (optional): Ask for the key ACTIVITIES or TASKS to include in the matrix
- Maximum 2 questions
- One question per line, no numbering, no bullet points

Respond now:"""
)


# ── Signature ─────────────────────────────────────────────────────────────────

# No question prompt needed — signature sections need 0 questions.
# We go directly to generation.


# ─────────────────────────────────────────────────────────────────────────────
#  QUESTION GENERATION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def generate_questions(req: GenerateQuestionsRequest) -> dict:
    ctx       = req.company_context or {}
    doc_type  = req.doc_type
    sec_name  = req.section_name
    sec_type  = detect_section_type(doc_type, sec_name)
    meta      = DOC_STRUCTURE_METADATA.get(doc_type, {})

    logger.info(f"Section type detected: '{sec_type}' for '{sec_name}' in '{doc_type}'")

    # Signature: always 0 questions
    if sec_type == SECTION_TYPE_SIGNATURE:
        questions = []

    elif sec_type == SECTION_TYPE_TABLE:
        chain     = TABLE_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = chain.invoke({
            "section_name": sec_name,
            "doc_type":     doc_type,
            "department":   req.department,
            "company_name": ctx.get("company_name", "the company"),
            "industry":     ctx.get("industry", "general"),
            "table_hint":   meta.get("table_hint", f"Standard table for {sec_name}"),
        }).strip()
        questions = _parse_questions(raw, max_q=3)

    elif sec_type == SECTION_TYPE_FLOWCHART:
        chain     = FLOWCHART_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = chain.invoke({
            "section_name":   sec_name,
            "doc_type":       doc_type,
            "department":     req.department,
            "company_name":   ctx.get("company_name", "the company"),
            "industry":       ctx.get("industry", "general"),
            "flowchart_hint": meta.get("flowchart_hint", f"Standard process flow for {sec_name}"),
        }).strip()
        questions = _parse_questions(raw, max_q=3)

    elif sec_type == SECTION_TYPE_RACI:
        chain     = RACI_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = chain.invoke({
            "section_name": sec_name,
            "doc_type":     doc_type,
            "department":   req.department,
            "company_name": ctx.get("company_name", "the company"),
            "industry":     ctx.get("industry", "general"),
            "raci_hint":    meta.get("raci_hint", f"Standard RACI for {sec_name}"),
        }).strip()
        questions = _parse_questions(raw, max_q=2)

    else:  # SECTION_TYPE_TEXT (default)
        chain     = TEXT_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = chain.invoke({
            "section_name": sec_name,
            "doc_type":     doc_type,
            "department":   req.department,
            "company_name": ctx.get("company_name", "the company"),
            "industry":     ctx.get("industry", "general"),
            "company_size": ctx.get("company_size", "not specified"),
            "region":       ctx.get("region", "not specified"),
        }).strip()
        questions = _parse_questions(raw, max_q=3)

    sec_id = await save_questions(
        doc_sec_id=req.doc_sec_id, doc_id=req.doc_id,
        section_name=sec_name, questions=questions,
    )

    logger.info(f"[{sec_type.upper()}] {len(questions)} questions saved for '{sec_name}'")
    return {
        "sec_id":       sec_id,
        "doc_sec_id":   req.doc_sec_id,
        "doc_id":       req.doc_id,
        "section_name": sec_name,
        "section_type": sec_type,
        "questions":    questions,
    }


def _parse_questions(raw: str, max_q: int) -> list:
    """Parse LLM question output, strip non-questions, cap at max_q."""
    if not raw or raw.strip().upper() == "NONE":
        return []
    lines = [
        re.sub(r'^[\d\-\.\*\•]+\s*', '', line).strip()
        for line in raw.split("\n")
        if line.strip() and len(line.strip()) > 10
    ]
    return lines[:max_q]


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE ANSWERS
# ─────────────────────────────────────────────────────────────────────────────

async def save_user_answers(req: SaveAnswersRequest) -> dict:
    await save_answers(
        sec_id=req.sec_id, questions=req.questions,
        answers=req.answers, section_name=req.section_name
    )
    return {"sec_id": req.sec_id, "section_name": req.section_name, "saved": True}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION CONTENT GENERATION PROMPTS  (one per section type)
# ─────────────────────────────────────────────────────────────────────────────

# ── Text ─────────────────────────────────────────────────────────────────────

SECTION_TEXT_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "company_size", "region", "qa_block", "target_words"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}

User-provided information:
{qa_block}

STRICT RULES:
1. Write EXACTLY {target_words} words — hard limit, do NOT exceed
2. PLAIN TEXT ONLY — zero markdown, no asterisks (*), no # symbols, no backticks
3. Paragraphs separated by one blank line
4. Lists: use "1. Item" or "- Item" syntax only
5. If an answer is "not answered" — write realistic, industry-appropriate placeholder content
6. Do NOT include the section heading in your output
7. Professional {department} department tone throughout
8. Do NOT begin with "This section..." or "In this section..."

Write now:"""
)


# ── Table ─────────────────────────────────────────────────────────────────────

SECTION_TABLE_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "region", "qa_block", "table_hint"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}. This section MUST contain a data table.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}
Table guidance: {table_hint}

User-provided data:
{qa_block}

OUTPUT FORMAT — follow EXACTLY:
1. One professional sentence introducing the table (plain text, no markdown)
2. A blank line
3. A properly formatted pipe table using the user's data:

Column1 | Column2 | Column3
------- | ------- | -------
value1  | value2  | value3
value2  | value2  | value3

STRICT RULES:
- Column names must be industry-standard for {section_name} in a {doc_type}
- Use the user's data to populate rows; if data is missing, use realistic placeholders in [brackets]
- Minimum 3 data rows, maximum 10 rows
- NO markdown outside the table — no **, no ##, no backticks
- The intro sentence goes BEFORE the table, never inside a cell
- Do NOT include the section heading in output

Write now:"""
)


# ── Flowchart ─────────────────────────────────────────────────────────────────

SECTION_FLOWCHART_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "region", "qa_block", "flowchart_hint"],
    template="""You are a professional enterprise documentation writer and process designer.

Write the "{section_name}" section of a {doc_type}. This section MUST contain a Mermaid flowchart.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}
Process hint: {flowchart_hint}

User-provided process information:
{qa_block}

OUTPUT FORMAT — follow EXACTLY:
1. One professional sentence describing the process (plain text, no markdown)
2. A blank line
3. A Mermaid flowchart using this EXACT format:

```mermaid
flowchart TD
    A[Start: Step Name] --> B[Step Name]
    B --> C{{Decision Point?}}
    C -->|Yes| D[Step if Yes]
    C -->|No| E[Step if No]
    D --> F[Next Step]
    E --> F
    F --> G([End])
```

STRICT RULES:
- Use TD direction (top-down)
- Minimum 6 nodes, maximum 12 nodes
- Use [Rectangle] for regular steps
- Use {{Diamond}} for decision/approval steps (Yes/No branches)
- Use ([Rounded]) for Start and End nodes
- Label all arrow branches on decision nodes with |Yes| or |No| or relevant label
- Node text must be SHORT — max 5 words per node
- Use ACTUAL steps from the user's answers; if no data, use standard steps for {section_name} of a {doc_type}
- Close the mermaid block with ``` on its own line
- NO other markdown outside the mermaid block — no **, no ##
- Do NOT include the section heading in output

Write now:"""
)


# ── RACI ──────────────────────────────────────────────────────────────────────

SECTION_RACI_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "region", "qa_block", "raci_hint"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}. This section MUST contain a RACI matrix.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}
RACI guidance: {raci_hint}

User-provided role information:
{qa_block}

OUTPUT FORMAT — follow EXACTLY:
1. One professional sentence about accountability for this process (plain text, no markdown)
2. A blank line
3. A RACI table in this EXACT pipe format:

Activity | [Role 1] | [Role 2] | [Role 3] | [Role 4]
-------- | -------- | -------- | -------- | --------
Activity Name | R | A | C | I
Activity Name | C | R | A | I
Activity Name | I | C | R | A

RACI KEY (add this after the table):
R = Responsible | A = Accountable | C = Consulted | I = Informed

STRICT RULES:
- Replace [Role N] with actual role names from user's answers, or use standard roles for {doc_type}
- Minimum 6 activities, maximum 10 activities
- Every row must have exactly one R and exactly one A
- Activities must be specific to {section_name} of a {doc_type} — not generic
- Do NOT use markdown outside the table — no **, no ##, no backticks
- Do NOT include the section heading in output

Write now:"""
)


# ── Signature ─────────────────────────────────────────────────────────────────

SECTION_SIGNATURE_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "company_name", "section_name"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type} for {company_name} ({department} department).

This section is a formal approval and sign-off block.

OUTPUT FORMAT — follow EXACTLY:
1. One sentence stating this document requires the following authorised signatures (plain text)
2. A blank line
3. A pipe-format signature table:

Role | Name | Signature | Date
---- | ---- | --------- | ----
[Relevant Role 1] | __________________ | __________________ | __________
[Relevant Role 2] | __________________ | __________________ | __________
[Relevant Role 3] | __________________ | __________________ | __________

STRICT RULES:
- Use 3–5 role rows appropriate for a {doc_type} in the {department} department
- Role names must be specific and relevant (e.g., "HR Manager", "Chief People Officer", "Employee")
- Name, Signature, Date fields must be blank lines (__________) for manual completion
- NO other markdown — no **, no ##, no backticks
- Do NOT include the section heading in output

Write now:"""
)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION CONTENT GENERATION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def generate_section_content(req: GenerateSectionRequest) -> dict:
    qa_row = await get_qa_by_sec_id(req.sec_id)
    if not qa_row:
        raise ValueError(f"No Q&A found for sec_id={req.sec_id}")

    qa_data     = qa_row["doc_sec_que_ans"]
    questions   = qa_data.get("questions", [])
    answers     = qa_data.get("answers", [])
    # Prefer the type saved at question-gen time; re-detect as fallback
    sec_type    = qa_data.get("section_type") or detect_section_type(req.doc_type, req.section_name)
    ctx         = req.company_context or {}
    meta        = DOC_STRUCTURE_METADATA.get(req.doc_type, {})

    company_name = ctx.get("company_name", "the company")
    industry     = ctx.get("industry", "general")
    region       = ctx.get("region", "not specified")
    department   = req.department

    qa_block = _build_qa_block(questions, answers)

    logger.info(f"Generating [{sec_type.upper()}] section: '{req.section_name}'")

    # ── Dispatch to correct generator ────────────────────────────────────────

    if sec_type == SECTION_TYPE_SIGNATURE:
        chain = SECTION_SIGNATURE_PROMPT | get_llm(0.4) | StrOutputParser()
        raw   = chain.invoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "company_name": company_name,
            "section_name": req.section_name,
        })
        clean = _clean_preserve_tables(raw.strip())

    elif sec_type == SECTION_TYPE_TABLE:
        chain = SECTION_TABLE_PROMPT | get_llm(0.5) | StrOutputParser()
        raw   = chain.invoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "section_name": req.section_name,
            "company_name": company_name,
            "industry":     industry,
            "region":       region,
            "qa_block":     qa_block,
            "table_hint":   meta.get("table_hint", f"Standard data table for {req.section_name}"),
        })
        clean = _clean_preserve_tables(raw.strip())
        logger.info(f"Table section '{req.section_name}' generated")

    elif sec_type == SECTION_TYPE_FLOWCHART:
        chain = SECTION_FLOWCHART_PROMPT | get_llm(0.5) | StrOutputParser()
        raw   = chain.invoke({
            "doc_type":       req.doc_type,
            "department":     department,
            "section_name":   req.section_name,
            "company_name":   company_name,
            "industry":       industry,
            "region":         region,
            "qa_block":       qa_block,
            "flowchart_hint": meta.get("flowchart_hint", f"Standard process flow for {req.section_name}"),
        })
        clean = _clean_preserve_flowcharts(raw.strip())
        logger.info(f"Flowchart section '{req.section_name}' generated")

    elif sec_type == SECTION_TYPE_RACI:
        chain = SECTION_RACI_PROMPT | get_llm(0.4) | StrOutputParser()
        raw   = chain.invoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "section_name": req.section_name,
            "company_name": company_name,
            "industry":     industry,
            "region":       region,
            "qa_block":     qa_block,
            "raci_hint":    meta.get("raci_hint", f"Standard RACI matrix for {req.section_name}"),
        })
        clean = _clean_preserve_tables(raw.strip())
        logger.info(f"RACI section '{req.section_name}' generated")

    else:  # SECTION_TYPE_TEXT
        target_words = get_words_per_section(req.doc_type, req.num_sections or 10)
        chain        = SECTION_TEXT_PROMPT | get_llm(0.7) | StrOutputParser()
        raw          = chain.invoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "section_name": req.section_name,
            "company_name": company_name,
            "industry":     industry,
            "company_size": ctx.get("company_size", "not specified"),
            "region":       region,
            "qa_block":     qa_block,
            "target_words": target_words,
        })
        clean = _clean_preserve_tables(raw.strip())
        clean = _enforce_word_limit(clean, target_words)
        logger.info(f"Text section '{req.section_name}' — {len(clean.split())} words")

    return {
        "sec_id":       req.sec_id,
        "section_name": req.section_name,
        "section_type": sec_type,
        "content":      clean,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  EDIT SECTION
# ─────────────────────────────────────────────────────────────────────────────

EDIT_PROMPT = PromptTemplate(
    input_variables=["section_name", "section_type", "current_content", "edit_instruction"],
    template="""Professional enterprise document editor.

Section      : {section_name}
Section Type : {section_type}

Current Content:
{current_content}

Edit Instruction: {edit_instruction}

OUTPUT RULES based on section type:
- text      → PLAIN TEXT ONLY, no markdown, no asterisks, no # symbols
- table     → Keep pipe-format table intact; plain text intro sentence only
- flowchart → Keep the ```mermaid ... ``` block intact; update steps if instructed
- raci      → Keep pipe-format RACI table intact; update roles/activities if instructed
- signature → Keep pipe-format signature table intact

Apply the edit instruction to the content above and return ONLY the updated content.
Do not add explanations, preambles, or notes about what changed."""
)


async def edit_section(req: EditSectionRequest) -> dict:
    # Re-detect type for editing context
    sec_type = detect_section_type(req.doc_type, req.section_name) if hasattr(req, "doc_type") else "text"

    chain   = EDIT_PROMPT | get_llm(0.6) | StrOutputParser()
    raw     = chain.invoke({
        "section_name":     req.section_name,
        "section_type":     sec_type,
        "current_content":  req.current_content,
        "edit_instruction": req.edit_instruction,
    }).strip()

    if sec_type == SECTION_TYPE_FLOWCHART:
        updated = _clean_preserve_flowcharts(raw)
    else:
        updated = _clean_preserve_tables(raw)

    gen_doc = await get_generated_document(req.gen_id)
    if gen_doc:
        full_doc = gen_doc.get("gen_doc_full", "").replace(req.current_content, updated)
        await update_section_content(req.gen_id, gen_doc.get("gen_doc_sec_dec", []), full_doc)

    return {
        "sec_id":          req.sec_id,
        "section_name":    req.section_name,
        "section_type":    sec_type,
        "updated_content": updated,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _build_qa_block(questions: list, answers: list) -> str:
    """Format Q&A pairs for injection into any generation prompt."""
    if not questions:
        return "No specific input provided — use professional industry-standard placeholder content."
    pairs = []
    for i, q in enumerate(questions):
        a = answers[i] if i < len(answers) else "not answered"
        pairs.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(pairs)


def _clean_preserve_tables(text: str) -> str:
    """
    Strip markdown formatting from non-table lines.
    Pipe-format tables are preserved exactly.
    """
    lines  = text.split("\n")
    result = []
    for line in lines:
        if "|" in line:
            result.append(line.rstrip())          # table row — preserve as-is
        else:
            result.append(markdown_to_plain_text(line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()


def _clean_preserve_flowcharts(text: str) -> str:
    """
    Preserve ```mermaid ... ``` blocks exactly.
    Strip markdown from everything outside those blocks.

    Safety net: if LLM output contains 'flowchart TD' without backtick fences,
    automatically wraps it so docx_builder can detect and render it as an image.
    """
    # Auto-wrap bare flowchart blocks (LLM forgot the fences)
    if re.search(r'flowchart\s+(?:TD|LR|BT|RL)', text) and "```mermaid" not in text:
        text = re.sub(
            r'(flowchart\s+(?:TD|LR|BT|RL).*?)(\n\n|\Z)',
            lambda m: "```mermaid\n" + m.group(1).rstrip() + "\n```" + m.group(2),
            text,
            flags=re.DOTALL
        )

    mermaid_pattern = re.compile(r"(```mermaid.*?```)", re.DOTALL)
    parts = mermaid_pattern.split(text)
    cleaned = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Odd index = inside a mermaid block — preserve verbatim
            cleaned.append(part)
        else:
            # Even index = outside mermaid block — strip markdown
            cleaned.append(_clean_preserve_tables(part))
    result = "\n".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _enforce_word_limit(text: str, target_words: int) -> str:
    """Hard-truncate text to target_words at the nearest sentence boundary."""
    words = text.split()
    if len(words) <= int(target_words * 1.2):
        return text
    truncated   = " ".join(words[:target_words])
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > len(truncated) * 0.6:
        return truncated[:last_period + 1].strip()
    return truncated.strip()