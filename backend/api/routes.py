from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from backend.core.logger import logger
from backend.services.db_service import (
    get_all_departments, get_sections_by_doc_type,
    save_generated_document,
)
from backend.services.generator import (
    generate_questions, save_user_answers,
    generate_section_content, edit_section,
    enhance_full_document, add_new_section,
)
from backend.services.notion_service import publish_to_notion, fetch_library_from_notion
from backend.schemas.document_schema import (
    GenerateQuestionsRequest, SaveAnswersRequest,
    GenerateSectionRequest, EditSectionRequest,
    NotionPublishRequest,
)
from backend.services.redis_service import cache

router = APIRouter()


class SaveDocRequest(BaseModel):
    doc_id: int
    doc_sec_id: int
    sec_id: int
    gen_doc_sec_dec: List[str]
    gen_doc_full: str


class EnhanceFullDocRequest(BaseModel):
    doc_type: str
    department: str
    company_context: Dict = {}
    sections: Dict[str, str]      # {section_name: current_content}
    section_order: List[str]       # ordered list of section names


class AddSectionRequest(BaseModel):
    section_name: str
    doc_type: str
    department: str
    company_context: Dict = {}
    existing_sections: Dict[str, str]  # {section_name: content}


# ── Departments & Sections ────────────────────────────────────────────────────

@router.get("/departments")
async def get_departments():
    try:
        # Try cache first
        cached = await cache.get_departments()
        if cached is not None:
            logger.info("✅ [CACHE HIT] departments")
            return {"departments": cached, "total": len(cached), "cached": True}

        depts = await get_all_departments()

        # Store in cache
        await cache.set_departments(depts)
        logger.info("💾 [CACHE SET] departments (%d items)", len(depts))

        return {"departments": depts, "total": len(depts), "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sections/{doc_type}")
async def get_sections(doc_type: str):
    decoded = doc_type.replace("%2F", "/").replace("%28", "(").replace("%29", ")")
    try:
        # Try cache first
        cached = await cache.get_sections(decoded)
        if cached is not None:
            logger.info("✅ [CACHE HIT] sections:%s", decoded)
            return {**cached, "cached": True}

        sections = await get_sections_by_doc_type(decoded)
        if not sections:
            raise HTTPException(status_code=404, detail=f"No sections for: {decoded}")

        # Store in cache
        await cache.set_sections(decoded, sections)
        logger.info("💾 [CACHE SET] sections:%s", decoded)

        return {**sections, "cached": False}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Questions & Answers ───────────────────────────────────────────────────────

@router.post("/questions/generate")
async def api_generate_questions(req: GenerateQuestionsRequest):
    try:
        result = await generate_questions(req)

        # Cache by sec_id so re-generating the same section is instant
        sec_id = result.get("sec_id")
        if sec_id:
            await cache.set_questions(sec_id, result)
            logger.info("💾 [CACHE SET] questions:%s", sec_id)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answers/save")
async def api_save_answers(req: SaveAnswersRequest):
    try:
        result = await save_user_answers(req)

        # Invalidate cached section content — answers changed, regenerate fresh
        if req.sec_id:
            await cache.invalidate_section_content(req.sec_id)
            logger.info("🗑️  [CACHE DEL] section content:%s (answers updated)", req.sec_id)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Section Generation & Edit ─────────────────────────────────────────────────

@router.post("/section/generate")
async def api_generate_section(req: GenerateSectionRequest):
    try:
        # Check cache — if same sec_id already generated, return instantly
        if req.sec_id:
            cached = await cache.get_section_content(req.sec_id)
            if cached is not None:
                logger.info("✅ [CACHE HIT] section content:%s", req.sec_id)
                return {**cached, "cached": True}

        result = await generate_section_content(req)

        # Cache the expensive LLM result
        if req.sec_id and result:
            await cache.set_section_content(req.sec_id, result)
            logger.info("💾 [CACHE SET] section content:%s", req.sec_id)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/section/edit")
async def api_edit_section(req: EditSectionRequest):
    try:
        result = await edit_section(req)

        # Invalidate old cached content after edit
        if req.sec_id:
            await cache.invalidate_section_content(req.sec_id)
            logger.info("🗑️  [CACHE DEL] section content:%s (edited)", req.sec_id)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Document Save ─────────────────────────────────────────────────────────────

@router.post("/document/save")
async def api_save_document(req: SaveDocRequest):
    try:
        gen_id = await save_generated_document(
            doc_id=req.doc_id, doc_sec_id=req.doc_sec_id, sec_id=req.sec_id,
            gen_doc_sec_dec=req.gen_doc_sec_dec, gen_doc_full=req.gen_doc_full,
        )
        return {"gen_id": gen_id, "saved": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Notion ────────────────────────────────────────────────────────────────────

@router.post("/document/publish")
async def api_publish_document(req: NotionPublishRequest):
    try:
        result = await publish_to_notion(req)

        # Invalidate library cache so new doc appears immediately
        await cache.invalidate_notion_library()
        logger.info("🗑️  [CACHE DEL] notion_library (new doc published)")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/library/notion")
async def api_notion_library():
    try:
        # Try cache first (5-min TTL)
        cached = await cache.get_notion_library()
        if cached is not None:
            logger.info("✅ [CACHE HIT] notion_library (%d docs)", len(cached))
            return {"total": len(cached), "documents": cached, "cached": True}

        docs = await fetch_library_from_notion()

        await cache.set_notion_library(docs)
        logger.info("💾 [CACHE SET] notion_library (%d docs)", len(docs))

        return {"total": len(docs), "documents": docs, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Enhance Full Document ─────────────────────────────────────────────────────

@router.post("/document/enhance")
async def api_enhance_full_document(req: EnhanceFullDocRequest):
    try:
        result = await enhance_full_document(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Add Section ───────────────────────────────────────────────────────────────

@router.post("/section/add")
async def api_add_section(req: AddSectionRequest):
    try:
        result = await add_new_section(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))