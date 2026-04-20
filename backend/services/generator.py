"""
Section-type-aware document generation pipeline for DocForge AI.

Orchestrates the full generation flow for individual document sections:
    1. Detect the section type (text, table, flowchart, raci, signature).
    2. Generate LLM questions appropriate for that type.
    3. Accept and persist user answers.
    4. Generate section content using the matching prompt template.
    5. Apply post-processing (markdown stripping, word-count enforcement).

All LLM prompts are imported from `backend/prompts/prompts.py` as the
single source of truth for prompt templates.
"""

import re
from langchain_core.output_parsers import StrOutputParser

from backend.core.logger import logger
from backend.core.llm import get_llm
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

from backend.prompts.prompts import (
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
)


SECTION_TYPE_TEXT       = "text"
SECTION_TYPE_TABLE      = "table"
SECTION_TYPE_FLOWCHART  = "flowchart"
SECTION_TYPE_RACI       = "raci"
SECTION_TYPE_SIGNATURE  = "signature"


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


def detect_section_type(doc_type: str, section_name: str) -> str:
    """
    Determine the appropriate rendering type for a document section.

    Resolution order:
        1. Signature keywords in the section name always win.
        2. If the document metadata flags a type (e.g. `has_raci`) AND the
           section name confirms it via keyword matching, that type is used.
        3. Section name keyword matching alone (for docs without metadata flags).
        4. Default: `"text"`.

    Args:
        doc_type:     The document type (e.g. "Standard Operating Procedure").
        section_name: The section name to classify.

    Returns:
        One of: `"text"`, `"table"`, `"flowchart"`, `"raci"`, `"signature"`.
    """
    sec_lower = section_name.lower()
    meta = DOC_STRUCTURE_METADATA.get(doc_type, {})

    if _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_SIGNATURE]):
        return SECTION_TYPE_SIGNATURE

    if meta.get("has_raci") and _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_RACI]):
        return SECTION_TYPE_RACI

    if meta.get("has_flowchart") and _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_FLOWCHART]):
        return SECTION_TYPE_FLOWCHART

    if meta.get("has_table") and _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[SECTION_TYPE_TABLE]):
        return SECTION_TYPE_TABLE

    for stype in [SECTION_TYPE_RACI, SECTION_TYPE_FLOWCHART, SECTION_TYPE_TABLE]:
        if _matches_keywords(sec_lower, SECTION_TYPE_PATTERNS[stype]):
            return stype

    return SECTION_TYPE_TEXT


def _matches_keywords(text: str, keywords: list) -> bool:
    """Return True if any keyword from `keywords` appears as a substring of `text`."""
    return any(kw in text for kw in keywords)


async def generate_questions(req: GenerateQuestionsRequest) -> dict:
    """
    Generate contextual questions for a document section via the LLM.

    The section type is detected first to select the appropriate prompt
    template. Signature sections always produce zero questions. The
    generated questions are persisted to the database before returning.

    Args:
        req: A `GenerateQuestionsRequest` with section and company metadata.

    Returns:
        A dict with keys: `sec_id`, `doc_sec_id`, `doc_id`, `section_name`,
        `section_type`, and `questions`.
    """
    ctx       = req.company_context or {}
    doc_type  = req.doc_type
    sec_name  = req.section_name
    sec_type  = detect_section_type(doc_type, sec_name)
    meta      = DOC_STRUCTURE_METADATA.get(doc_type, {})

    logger.info(f"Section type detected: '{sec_type}' for '{sec_name}' in '{doc_type}'")

    if sec_type == SECTION_TYPE_SIGNATURE:
        questions = []

    elif sec_type == SECTION_TYPE_TABLE:
        chain     = TABLE_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = (await chain.ainvoke({
            "section_name": sec_name,
            "doc_type":     doc_type,
            "department":   req.department,
            "company_name": ctx.get("company_name", "the company"),
            "industry":     ctx.get("industry", "general"),
            "table_hint":   meta.get("table_hint", f"Standard table for {sec_name}"),
        })).strip()
        questions = _parse_questions(raw, max_q=3)

    elif sec_type == SECTION_TYPE_FLOWCHART:
        chain     = FLOWCHART_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = (await chain.ainvoke({
            "section_name":   sec_name,
            "doc_type":       doc_type,
            "department":     req.department,
            "company_name":   ctx.get("company_name", "the company"),
            "industry":       ctx.get("industry", "general"),
            "flowchart_hint": meta.get("flowchart_hint", f"Standard process flow for {sec_name}"),
        })).strip()
        questions = _parse_questions(raw, max_q=3)

    elif sec_type == SECTION_TYPE_RACI:
        chain     = RACI_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = (await chain.ainvoke({
            "section_name": sec_name,
            "doc_type":     doc_type,
            "department":   req.department,
            "company_name": ctx.get("company_name", "the company"),
            "industry":     ctx.get("industry", "general"),
            "raci_hint":    meta.get("raci_hint", f"Standard RACI for {sec_name}"),
        })).strip()
        questions = _parse_questions(raw, max_q=2)

    else:
        chain     = TEXT_QUESTIONS_PROMPT | get_llm(0.3) | StrOutputParser()
        raw       = (await chain.ainvoke({
            "section_name": sec_name,
            "doc_type":     doc_type,
            "department":   req.department,
            "company_name": ctx.get("company_name", "the company"),
            "industry":     ctx.get("industry", "general"),
            "company_size": ctx.get("company_size", "not specified"),
            "region":       ctx.get("region", "not specified"),
        })).strip()
        questions = _parse_questions(raw, max_q=3)

    sec_id = await save_questions(
        doc_sec_id=req.doc_sec_id, doc_id=req.doc_id,
        section_name=sec_name, questions=questions,
        section_type=sec_type
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
    """
    Parse raw LLM output into a clean list of question strings.

    Strips leading numbering, bullets, and whitespace. Skips lines shorter
    than 10 characters. Caps output at `max_q` questions.

    Args:
        raw:   Raw text from the LLM output parser.
        max_q: Maximum number of questions to return.

    Returns:
        A list of clean question strings, capped at `max_q`.
    """
    if not raw or raw.strip().upper() == "NONE":
        return []
    lines = [
        re.sub(r'^[\d\-\.\*\•]+\s*', '', line).strip()
        for line in raw.split("\n")
        if line.strip() and len(line.strip()) > 10
    ]
    return lines[:max_q]


async def save_user_answers(req: SaveAnswersRequest) -> dict:
    """
    Persist user-provided answers for a document section.

    Args:
        req: A `SaveAnswersRequest` with section ID, questions, and answers.

    Returns:
        A dict with `sec_id`, `section_name`, and `saved: True`.
    """
    await save_answers(
        sec_id=req.sec_id, questions=req.questions,
        answers=req.answers, section_name=req.section_name
    )
    return {"sec_id": req.sec_id, "section_name": req.section_name, "saved": True}


async def generate_section_content(req: GenerateSectionRequest) -> dict:
    """
    Generate content for a document section using the appropriate LLM prompt.

    Retrieves the stored Q&A data, selects the matching generation prompt
    based on the section type, invokes the LLM, and applies post-processing
    (markdown stripping, flowchart fencing, word-count enforcement for text).

    Args:
        req: A `GenerateSectionRequest` with section metadata and company context.

    Returns:
        A dict with `sec_id`, `section_name`, `section_type`, and `content`.

    Raises:
        ValueError: If no Q&A record exists for `req.sec_id`.
    """
    qa_row = await get_qa_by_sec_id(req.sec_id)
    if not qa_row:
        raise ValueError(f"No Q&A found for sec_id={req.sec_id}")

    qa_data      = qa_row["doc_sec_que_ans"]
    questions    = qa_data.get("questions", [])
    answers      = qa_data.get("answers", [])
    sec_type     = qa_data.get("section_type") or detect_section_type(req.doc_type, req.section_name)
    ctx          = req.company_context or {}
    meta         = DOC_STRUCTURE_METADATA.get(req.doc_type, {})

    company_name = ctx.get("company_name", "the company")
    industry     = ctx.get("industry", "general")
    region       = ctx.get("region", "not specified")
    department   = req.department

    qa_block = _build_qa_block(questions, answers)

    logger.info(f"Generating [{sec_type.upper()}] section: '{req.section_name}'")

    if sec_type == SECTION_TYPE_SIGNATURE:
        chain = SECTION_SIGNATURE_PROMPT | get_llm(0.4) | StrOutputParser()
        raw   = (await chain.ainvoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "company_name": company_name,
            "section_name": req.section_name,
        })).strip()
        clean = _clean_preserve_tables(raw)

    elif sec_type == SECTION_TYPE_TABLE:
        chain = SECTION_TABLE_PROMPT | get_llm(0.5) | StrOutputParser()
        raw   = (await chain.ainvoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "section_name": req.section_name,
            "company_name": company_name,
            "industry":     industry,
            "region":       region,
            "qa_block":     qa_block,
            "table_hint":   meta.get("table_hint", f"Standard data table for {req.section_name}"),
        })).strip()
        clean = _clean_preserve_tables(raw)
        logger.info(f"Table section '{req.section_name}' generated")

    elif sec_type == SECTION_TYPE_FLOWCHART:
        chain = SECTION_FLOWCHART_PROMPT | get_llm(0.5) | StrOutputParser()
        raw   = (await chain.ainvoke({
            "doc_type":       req.doc_type,
            "department":     department,
            "section_name":   req.section_name,
            "company_name":   company_name,
            "industry":       industry,
            "region":         region,
            "qa_block":       qa_block,
            "flowchart_hint": meta.get("flowchart_hint", f"Standard process flow for {req.section_name}"),
        })).strip()
        clean = _clean_preserve_flowcharts(raw)
        logger.info(f"Flowchart section '{req.section_name}' generated")

    elif sec_type == SECTION_TYPE_RACI:
        chain = SECTION_RACI_PROMPT | get_llm(0.4) | StrOutputParser()
        raw   = (await chain.ainvoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "section_name": req.section_name,
            "company_name": company_name,
            "industry":     industry,
            "region":       region,
            "qa_block":     qa_block,
            "raci_hint":    meta.get("raci_hint", f"Standard RACI matrix for {req.section_name}"),
        })).strip()
        clean = _clean_preserve_tables(raw)
        logger.info(f"RACI section '{req.section_name}' generated")

    else:
        target_words = get_words_per_section(req.doc_type, req.num_sections or 10)
        chain        = SECTION_TEXT_PROMPT | get_llm(0.7) | StrOutputParser()
        raw          = (await chain.ainvoke({
            "doc_type":     req.doc_type,
            "department":   department,
            "section_name": req.section_name,
            "company_name": company_name,
            "industry":     industry,
            "company_size": ctx.get("company_size", "not specified"),
            "region":       region,
            "qa_block":     qa_block,
            "target_words": target_words,
        })).strip()
        clean = _clean_preserve_tables(raw)
        clean = _enforce_word_limit(clean, target_words)
        logger.info(f"Text section '{req.section_name}' — {len(clean.split())} words")

    return {
        "sec_id":       req.sec_id,
        "section_name": req.section_name,
        "section_type": sec_type,
        "content":      clean,
    }


async def edit_section(req: EditSectionRequest) -> dict:
    """
    Apply a user-provided edit instruction to existing section content.

    The section type is read from the stored Q&A record (if available) to
    ensure the correct post-processing pipeline is applied. Falls back to
    detection from `doc_type` + `section_name` if the stored type is absent.

    After editing, updates the full document text in the database by
    replacing the first occurrence of the old content.

    Args:
        req: An `EditSectionRequest` with gen_id, section metadata,
             current content, and the edit instruction.

    Returns:
        A dict with `sec_id`, `section_name`, `section_type`, and
        `updated_content`.
    """
    sec_type = "text"
    if req.sec_id:
        qa_row = await get_qa_by_sec_id(req.sec_id)
        if qa_row:
            sec_type = qa_row.get("doc_sec_que_ans", {}).get("section_type", "text") or "text"
    if sec_type == "text" and hasattr(req, "doc_type") and req.doc_type:
        sec_type = detect_section_type(req.doc_type, req.section_name)

    chain   = EDIT_PROMPT | get_llm(0.6) | StrOutputParser()
    raw     = (await chain.ainvoke({
        "section_name":     req.section_name,
        "section_type":     sec_type,
        "current_content":  req.current_content,
        "edit_instruction": req.edit_instruction,
    })).strip()

    if sec_type == SECTION_TYPE_FLOWCHART:
        updated = _clean_preserve_flowcharts(raw)
    else:
        updated = _clean_preserve_tables(raw)

    gen_doc = await get_generated_document(req.gen_id)
    if gen_doc:
        full_doc = gen_doc.get("gen_doc_full", "")
        if req.current_content in full_doc:
            full_doc = full_doc.replace(req.current_content, updated, 1)
        else:
            logger.warning(
                "edit_section: current_content not found in gen_doc gen_id=%s — "
                "content may have been updated by another session. Skipping full_doc update.",
                req.gen_id,
            )
        await update_section_content(req.gen_id, gen_doc.get("gen_doc_sec_dec", []), full_doc)

    return {
        "sec_id":          req.sec_id,
        "section_name":    req.section_name,
        "section_type":    sec_type,
        "updated_content": updated,
    }


def _build_qa_block(questions: list, answers: list) -> str:
    """
    Format question-answer pairs as a plain text block for prompt injection.

    Returns a fallback message when no questions are present.

    Args:
        questions: List of question strings.
        answers:   List of answer strings aligned by index.

    Returns:
        A formatted string with Q/A pairs, or a placeholder if empty.
    """
    if not questions:
        return "No specific input provided — use professional industry-standard placeholder content."
    pairs = []
    for i, q in enumerate(questions):
        a = answers[i] if i < len(answers) else "not answered"
        pairs.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(pairs)


def _clean_preserve_tables(text: str) -> str:
    """
    Strip markdown formatting from non-table content while preserving pipe tables.

    Lines containing `|` are kept verbatim. All other lines are passed through
    `markdown_to_plain_text`. Consecutive blank lines are collapsed to two.

    Args:
        text: Raw LLM output that may contain markdown and pipe-format tables.

    Returns:
        Cleaned plain text with tables intact.
    """
    lines  = text.split("\n")
    result = []
    for line in lines:
        if "|" in line:
            result.append(line.rstrip())
        else:
            result.append(markdown_to_plain_text(line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()


def _clean_preserve_flowcharts(text: str) -> str:
    """
    Preserve Mermaid fenced blocks while stripping markdown from surrounding text.

    If the LLM produced a bare `flowchart TD` block without backtick fences,
    the function automatically wraps it so downstream renderers can detect it.

    Args:
        text: Raw LLM output that may contain Mermaid flowchart syntax.

    Returns:
        Cleaned text with all Mermaid blocks fenced and markdown stripped
        from non-Mermaid content.
    """
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
            cleaned.append(part)
        else:
            cleaned.append(_clean_preserve_tables(part))
    result = "\n".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _enforce_word_limit(text: str, target_words: int) -> str:
    """
    Hard-truncate `text` to approximately `target_words` at the nearest sentence boundary.

    Truncation is only applied when the word count exceeds `target_words * 1.2`.
    The cut is made at the last sentence-ending punctuation (`.`, `!`, `?`)
    found in the first `target_words` words, provided it falls past 60% of
    the truncation point.

    Args:
        text:         The text to truncate.
        target_words: Maximum desired word count.

    Returns:
        The text unchanged if within limit, otherwise the truncated version.
    """
    words = text.split()
    if len(words) <= int(target_words * 1.2):
        return text
    truncated   = " ".join(words[:target_words])
    last_period = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_period > len(truncated) * 0.6:
        return truncated[:last_period + 1].strip()
    return truncated.strip()