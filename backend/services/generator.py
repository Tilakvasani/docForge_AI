from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.core.config import settings
from backend.core.logger import logger
from backend.schemas.document_schema import DocumentRequest, DocumentResponse
from backend.models.document_model import DocumentModel
from prompts.templates import get_prompt_template
from prompts.quality_gates import check_quality
from datetime import datetime
import uuid

def get_model(temperature: float = 0.7):
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=settings.GROQ_API_KEY
    )

async def generate_document(request: DocumentRequest) -> DocumentResponse:
    """Generate a document using LangChain + Groq"""
    logger.info(f"Starting generation: {request.doc_type} | {request.industry}")

    template_str = get_prompt_template(request.doc_type)

    prompt = PromptTemplate(
        template=template_str,
        input_variables=["title", "industry", "description"]
    )

    parser = StrOutputParser()
    model = get_model()
    chain = prompt | model | parser

    content = chain.invoke({
        "title": request.title,
        "industry": request.industry.value,
        "description": request.description or f"A professional {request.doc_type} document for {request.industry} industry"
    })

    # Quality gate check
    passed, reason = check_quality(content, request.doc_type)
    if not passed:
        logger.warning(f"Quality gate failed: {reason}. Regenerating...")
        content = chain.invoke({
            "title": request.title,
            "industry": request.industry.value,
            "description": f"{request.description}. Make sure to include all required sections."
        })

    doc = DocumentModel(
        doc_id=str(uuid.uuid4()),
        title=request.title,
        industry=request.industry.value,
        doc_type=request.doc_type.value,
        content=content,
        tags=request.tags or [request.industry.value, request.doc_type.value],
        created_by=request.created_by or "admin",
        created_at=datetime.utcnow(),
    )

    logger.info(f"Document generated successfully: {doc.doc_id}")

    return DocumentResponse(
        doc_id=doc.doc_id,
        title=doc.title,
        industry=doc.industry,
        doc_type=doc.doc_type,
        content=doc.content,
        tags=doc.tags,
        created_by=doc.created_by,
        created_at=doc.created_at,
        version=doc.version,
        published=False
    )
