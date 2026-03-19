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

# Import DOC_STRUCTURE_METADATA + ALL prompts from templates.py (single source of truth)
from prompts.prompts import (
    DOC_STRUCTURE_METADATA,
    TEXT_QUESTIONS_PROMPT,
    TABLE_QUESTIONS_PROMPT,
    FLOWCHART_QUESTIONS_PROMPT,
    RACI_QUESTIONS_PROMPT,
    SECTION_TEXT_PROMPT,
    SECTION_TABLE_PROMPT,
    SECTION_FLOWCHART_PROMPT,
    SECTION_RACI_PROMPT,
    SECTION_SIGNATURE_PROMPT,
    EDIT_PROMPT,
    PHASE_3A_SECTION_ENHANCE_PROMPT,
    ADD_SECTION_PROMPT,
    SECTION_RELEVANCE_PROMPT,
)


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

# ─────────────────────────────────────────────────────────────────────────────
#  ENHANCE FULL DOCUMENT  (section by section, returns all updated contents)
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.prompts import PromptTemplate as _PT


async def enhance_full_document(req) -> dict:
    """
    Enhance every section one by one using PHASE_3A_SECTION_ENHANCE_PROMPT.
    Returns dict of {section_name: enhanced_content} for all sections.
    """
    enhanced = {}
    for section_name, current_content in req.sections.items():
        prompt_text = PHASE_3A_SECTION_ENHANCE_PROMPT.format(
            doc_type=req.doc_type,
            department=req.department,
            section_name=section_name,
            current_section_content=current_content,
        )
        chain = _PT.from_template("{prompt}") | get_llm(0.6) | StrOutputParser()
        raw   = chain.invoke({"prompt": prompt_text}).strip()

        if "```mermaid" in current_content or "```mermaid" in raw:
            enhanced[section_name] = _clean_preserve_flowcharts(raw)
        else:
            enhanced[section_name] = _clean_preserve_tables(raw)

        logger.info(f"Enhanced section '{section_name}'")

    # Rebuild full document
    doc_lines = []
    for sec in req.section_order:
        c = enhanced.get(sec, req.sections.get(sec, "")).strip()
        if c:
            doc_lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
    full_doc = "\n".join(doc_lines).strip()

    logger.info(f"Full document enhanced — {len(full_doc.split())} words")
    return {"sections": enhanced, "full_document": full_doc}


# ─────────────────────────────────────────────────────────────────────────────
#  ADD SECTION  (LLM generates content for a brand new section)
# ─────────────────────────────────────────────────────────────────────────────

async def add_new_section(req) -> dict:
    """
    First checks if the section name is relevant to the document type.
    If not relevant → returns {"relevant": False, "message": "..."}.
    If relevant → generates content using ADD_SECTION_PROMPT from templates.py.
    """
    ctx          = req.company_context or {}
    company_name = ctx.get("company_name", "the company")
    industry     = ctx.get("industry", "general")
    region       = ctx.get("region", "not specified")

    # Build context block from existing sections (first 60 words each)
    existing_block = "\n".join(
        f"- {name}: {' '.join(content.split()[:60])}..."
        for name, content in req.existing_sections.items()
        if content.strip()
    ) or "No existing sections yet."

    existing_names = "\n".join(f"- {name}" for name in req.existing_sections)

    # ── Step 1: Relevance check ───────────────────────────────────────────────
    relevance_chain = SECTION_RELEVANCE_PROMPT | get_llm(0.0) | StrOutputParser()
    relevance_raw   = relevance_chain.invoke({
        "section_name":      req.section_name,
        "doc_type":          req.doc_type,
        "department":        req.department,
        "existing_sections": existing_names,
    }).strip().upper()

    is_relevant = relevance_raw.startswith("YES")

    if not is_relevant:
        logger.warning(f"Section '{req.section_name}' rejected as irrelevant for '{req.doc_type}'")
        return {
            "relevant": False,
            "message":  (
                f"'{req.section_name}' is not a valid section for a {req.doc_type}. "
                f"Please enter a section name that is relevant to this document."
            ),
        }

    # ── Step 2: Generate content ──────────────────────────────────────────────
    target_words = get_words_per_section(req.doc_type, len(req.existing_sections) + 1)

    gen_chain = ADD_SECTION_PROMPT | get_llm(0.7) | StrOutputParser()
    raw       = gen_chain.invoke({
        "doc_type":          req.doc_type,
        "department":        req.department,
        "section_name":      req.section_name,
        "company_name":      company_name,
        "industry":          industry,
        "region":            region,
        "existing_sections": existing_block,
        "target_words":      target_words,
    }).strip()

    content = _clean_preserve_tables(raw)
    content = _enforce_word_limit(content, target_words)

    logger.info(f"New section '{req.section_name}' generated — {len(content.split())} words")
    return {
        "relevant":     True,
        "section_name": req.section_name,
        "content":      content,
    }