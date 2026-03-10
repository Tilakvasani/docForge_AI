from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.config import settings
from backend.core.logger import logger
from backend.schemas.document_schema import DocumentRequest, DocumentResponse
from backend.models.document_model import DocumentModel
from backend.services.db_service import save_document_with_sections
from prompts.templates import get_prompt_template, get_sections_for_doc_type, build_answers_block
from prompts.quality_gates import check_quality
from datetime import datetime
import uuid


def get_model(temperature: float = 0.7):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=settings.GROQ_API_KEY
    )


def get_value(field) -> str:
    if hasattr(field, "value"):
        return field.value
    return str(field)


async def generate_document(request: DocumentRequest) -> DocumentResponse:
    industry_val  = get_value(request.industry)
    doc_type_val  = get_value(request.doc_type)
    dept_val      = request.department or "General"

    logger.info(f"Generating: {doc_type_val} | {dept_val} | {industry_val}")

    # ── Get sections for this doc type ────────────────────────────────────────
    sections = get_sections_for_doc_type(doc_type_val)

    # ── Build the answers block from user input ───────────────────────────────
    answers      = request.section_answers or {}
    answers_block = build_answers_block(answers, sections)

    # ── Build prompt ──────────────────────────────────────────────────────────
    template_str = get_prompt_template(doc_type_val)
    prompt = PromptTemplate(
        template=template_str,
        input_variables=["industry", "department", "document_type", "answers_block"]
    )

    chain   = prompt | get_model() | StrOutputParser()
    content = chain.invoke({
        "industry":      industry_val,
        "department":    dept_val,
        "document_type": doc_type_val,
        "answers_block": answers_block,
    })

    # ── Quality gate ──────────────────────────────────────────────────────────
    passed, reason = check_quality(content, request.doc_type)
    if not passed:
        logger.warning(f"Quality gate failed: {reason} — regenerating")
        content = chain.invoke({
            "industry":      industry_val,
            "department":    dept_val,
            "document_type": doc_type_val,
            "answers_block": answers_block + "\n\nIMPORTANT: Ensure all sections are complete.",
        })

    # ── Build document record ─────────────────────────────────────────────────
    tags   = request.tags or [industry_val, doc_type_val, dept_val]
    doc_id = str(uuid.uuid4())
    doc    = DocumentModel(
        doc_id=doc_id,
        title=request.title,
        industry=industry_val,
        doc_type=doc_type_val,
        content=content,
        tags=tags,
        created_by=request.created_by or "admin",
        created_at=datetime.utcnow(),
    )

    # ── Save to PostgreSQL ────────────────────────────────────────────────────
    try:
        await save_document_with_sections(
            doc_id=doc_id,
            title=request.title,
            industry=industry_val,
            department=dept_val,
            doc_type=doc_type_val,
            version=doc.version,
            content=content,
            tags=tags,
            created_by=request.created_by or "admin",
            answers=answers,
        )
        logger.info(f"Saved to PostgreSQL: {doc_id}")
    except Exception as e:
        logger.warning(f"PostgreSQL save failed (non-fatal): {e}")

    logger.info(f"Document generated: {doc_id}")

    return DocumentResponse(
        doc_id=doc.doc_id,
        title=doc.title,
        industry=doc.industry,
        department=request.department,
        doc_type=doc.doc_type,
        content=doc.content,
        tags=doc.tags,
        created_by=doc.created_by,
        created_at=doc.created_at,
        version=doc.version,
        published=False,
        section_answers=answers,
    )