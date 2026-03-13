"""
DocForge AI — generator.py  v4.0
- Azure OpenAI GPT-4.1-mini (replaces Groq)
- Industry-standard document lengths
- Smart table detection — sections that need tables get them
- Plain text output (tables rendered as ASCII/text grids)
- Smart question count 0-3
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
    markdown_to_plain_text, get_words_per_section, SECTIONS_NEEDING_TABLES
)
from backend.schemas.document_schema import (
    GenerateQuestionsRequest, SaveAnswersRequest,
    GenerateSectionRequest, EditSectionRequest,
)


def get_llm(temperature: float = 0.7) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_LLM_ENDPOINT,
        api_key=settings.AZURE_OPENAI_LLM_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT_41_MINI,
        api_version="2024-12-01-preview",
        temperature=temperature,
    )


# ─── Smart Question Generation (0-3) ─────────────────────────────────────────

SMART_QUESTION_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "company_size", "region"],
    template="""You are an expert enterprise documentation specialist.

Decide how many questions (0, 1, 2, or 3) are needed to fill this section, then write exactly that many.

Document Type: {doc_type}
Department: {department}
Section: {section_name}
Company: {company_name} | Industry: {industry}

Rules:
- 0 questions: Purely structural — signature blocks, document title, date stamps, version stamps, header metadata. Respond: NONE
- 1 question: Simple single-value — one date, one name, one role
- 2 questions: Needs 2 distinct pieces of context
- 3 questions: Complex section needing multiple details (maximum)

Output:
- If 0 questions: respond NONE
- Otherwise: one question per line, no numbering, no extra text

Respond now:"""
)


async def generate_questions(req: GenerateQuestionsRequest) -> dict:
    ctx = req.company_context or {}
    chain = SMART_QUESTION_PROMPT | get_llm(0.3) | StrOutputParser()

    raw = chain.invoke({
        "section_name": req.section_name, "doc_type": req.doc_type,
        "department":   req.department,
        "company_name": ctx.get("company_name", "the company"),
        "industry":     ctx.get("industry", "general"),
        "company_size": ctx.get("company_size", "not specified"),
        "region":       ctx.get("region", "not specified"),
    }).strip()

    questions = [] if (not raw or raw.upper() == "NONE") else [
        q.strip() for q in raw.split("\n") if q.strip()
    ][:3]

    sec_id = await save_questions(
        doc_sec_id=req.doc_sec_id, doc_id=req.doc_id,
        section_name=req.section_name, questions=questions
    )
    logger.info(f"Questions: {len(questions)} for '{req.section_name}' sec_id={sec_id}")
    return {"sec_id": sec_id, "doc_sec_id": req.doc_sec_id, "doc_id": req.doc_id,
            "section_name": req.section_name, "questions": questions}


# ─── Save Answers ─────────────────────────────────────────────────────────────

async def save_user_answers(req: SaveAnswersRequest) -> dict:
    await save_answers(
        sec_id=req.sec_id, questions=req.questions,
        answers=req.answers, section_name=req.section_name
    )
    return {"sec_id": req.sec_id, "section_name": req.section_name, "saved": True}


# ─── Generate Section Content ─────────────────────────────────────────────────
# Two prompts: one for plain text sections, one for table sections

SECTION_TEXT_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "company_size", "region", "qa_block", "target_words"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}

User answers:
{qa_block}

STRICT RULES:
1. Write EXACTLY {target_words} words — this is a hard limit, do NOT exceed it
2. PLAIN TEXT ONLY — zero markdown, no asterisks, no # symbols, no backticks
3. Regular paragraphs separated by blank lines
4. Lists: use "1. Item" or "- Item" only
5. "not answered" = write realistic professional placeholder content
6. No section heading in output
7. Professional {department} department tone

Write now:"""
)

SECTION_TABLE_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "company_size", "region", "qa_block", "target_words"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}. This section REQUIRES a data table.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}

User answers:
{qa_block}

STRICT RULES:
1. Write EXACTLY {target_words} words total — hard limit, do NOT exceed
2. Start with 1-2 sentences of plain text introduction
3. Then include a TABLE formatted EXACTLY like this — pipe-separated, no extra spaces:
   Column1 | Column2 | Column3
   ------- | ------- | -------
   Value1  | Value2  | Value3
4. After the table, add 1-2 sentences of plain text summary/notes if needed
5. NO other markdown — no **, no ##, no backticks — ONLY the table uses pipes and dashes
6. Table must have realistic industry-standard data for a {doc_type}
7. "not answered" = use realistic professional placeholder values
8. No section heading in output

Write now:"""
)


def _needs_table(doc_type: str, section_name: str) -> bool:
    """Check if this doc_type+section combination should have a table."""
    key = f"{doc_type}|{section_name}".lower()
    for pattern in SECTIONS_NEEDING_TABLES:
        if pattern.lower() in key:
            return True
    return False


async def generate_section_content(req: GenerateSectionRequest) -> dict:
    qa_row = await get_qa_by_sec_id(req.sec_id)
    if not qa_row:
        raise ValueError(f"No Q&A found for sec_id={req.sec_id}")

    qa_data   = qa_row["doc_sec_que_ans"]
    questions = qa_data.get("questions", [])
    answers   = qa_data.get("answers", [])

    qa_block = (
        "No specific questions — write professional standard content."
        if not questions else
        "\n".join(f"Q: {q}\nA: {a}\n" for q, a in zip(questions, answers))
    )

    target_words = get_words_per_section(req.doc_type, req.num_sections or 10)
    ctx   = req.company_context or {}
    needs_table = _needs_table(req.doc_type, req.section_name)

    prompt = SECTION_TABLE_PROMPT if needs_table else SECTION_TEXT_PROMPT
    chain  = prompt | get_llm(0.7) | StrOutputParser()

    raw = chain.invoke({
        "doc_type":     req.doc_type,
        "department":   req.department,
        "section_name": req.section_name,
        "company_name": ctx.get("company_name", "the company"),
        "industry":     ctx.get("industry", "general"),
        "company_size": ctx.get("company_size", "not specified"),
        "region":       ctx.get("region", "not specified"),
        "qa_block":     qa_block,
        "target_words": target_words,
    })

    # Strip markdown, preserve tables
    clean = _clean_preserve_tables(raw.strip())

    # Hard enforce word limit — truncate at sentence boundary if LLM overshoots by >20%
    words = clean.split()
    if len(words) > target_words * 1.2:
        # Cut to target_words, then extend to next sentence end
        truncated = " ".join(words[:target_words])
        last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_period > len(truncated) * 0.6:
            clean = truncated[:last_period + 1].strip()
        else:
            clean = truncated.strip()

    logger.info(f"Generated section '{req.section_name}' ({len(clean.split())} words, table={needs_table})")
    return {"sec_id": req.sec_id, "section_name": req.section_name, "content": clean}


def _clean_preserve_tables(text: str) -> str:
    """
    Strip markdown from text but preserve table lines (lines containing |).
    Table lines are kept as-is for the docx builder to detect and render as Word tables.
    """
    lines = text.split('\n')
    result = []
    for line in lines:
        # Preserve table rows and separator rows exactly
        if '|' in line:
            result.append(line.rstrip())
        else:
            # Apply full markdown stripping to non-table lines
            clean = markdown_to_plain_text(line)
            result.append(clean)
    out = '\n'.join(result)
    out = re.sub(r'\n{3,}', '\n\n', out)
    return out.strip()


# ─── Edit Section ─────────────────────────────────────────────────────────────

EDIT_PROMPT = PromptTemplate(
    input_variables=["section_name", "current_content", "edit_instruction"],
    template="""Professional enterprise document editor.

Section: {section_name}
Current Content:
{current_content}

Instruction: {edit_instruction}

Apply the instruction. Professional tone.
PLAIN TEXT ONLY — no markdown, no asterisks, no # symbols.
If a table exists, keep it in pipe format.
Return ONLY the updated content:"""
)


async def edit_section(req: EditSectionRequest) -> dict:
    chain   = EDIT_PROMPT | get_llm(0.6) | StrOutputParser()
    raw     = chain.invoke({
        "section_name":     req.section_name,
        "current_content":  req.current_content,
        "edit_instruction": req.edit_instruction,
    }).strip()

    updated = _clean_preserve_tables(raw)

    gen_doc = await get_generated_document(req.gen_id)
    if gen_doc:
        full_doc = gen_doc.get("gen_doc_full", "").replace(req.current_content, updated)
        await update_section_content(req.gen_id, gen_doc.get("gen_doc_sec_dec", []), full_doc)

    return {"sec_id": req.sec_id, "section_name": req.section_name, "updated_content": updated}