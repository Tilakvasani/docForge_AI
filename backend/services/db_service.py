"""
DocForge AI — db_service.py
All PostgreSQL operations for the full workflow.
Tables: depart · document_section · section_que_ans · gen_doc
"""
import json
import asyncpg
from typing import Optional, List, Dict, Any
from backend.core.config import settings
from backend.core.logger import logger


# ─── Connection Pool ──────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=2,
            max_size=10
        )
        logger.info("PostgreSQL connection pool created")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


# ─── TABLE: depart ────────────────────────────────────────────────────────────

async def get_all_departments() -> List[Dict]:
    """Load all departments and their doc types."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, department, doc_types FROM depart ORDER BY doc_id"
        )
    return [dict(r) for r in rows]


async def get_department_by_id(doc_id: int) -> Optional[Dict]:
    """Get a single department row."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT doc_id, department, doc_types FROM depart WHERE doc_id = $1",
            doc_id
        )
    return dict(row) if row else None


# ─── TABLE: document_section ─────────────────────────────────────────────────

async def get_sections_by_doc_type(doc_type: str) -> Optional[Dict]:
    """Get sections for a given document type."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT doc_sec_id, doc_type, doc_sec FROM document_section WHERE doc_type = $1",
            doc_type
        )
    return dict(row) if row else None


async def get_sections_by_id(doc_sec_id: int) -> Optional[Dict]:
    """Get document_section row by primary key."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT doc_sec_id, doc_type, doc_sec FROM document_section WHERE doc_sec_id = $1",
            doc_sec_id
        )
    return dict(row) if row else None


# ─── TABLE: section_que_ans ──────────────────────────────────────────────────

async def save_questions(
    doc_sec_id: int,
    doc_id: int,
    section_name: str,
    questions: List[str],
    section_type: str = "text"        # ← add this
) -> int:
    pool = await get_pool()
    qa_data = json.dumps({
        "section_name": section_name,
        "section_type": section_type,  # ← add this
        "questions": questions,
        "answers": []
    })
    async with pool.acquire() as conn:
        sec_id = await conn.fetchval(
            """
            INSERT INTO section_que_ans (doc_sec_id, doc_id, doc_sec_que_ans)
            VALUES ($1, $2, $3::jsonb)
            RETURNING sec_id
            """,
            doc_sec_id, doc_id, qa_data
        )
    logger.info(f"Questions saved: sec_id={sec_id}, section={section_name}")
    return sec_id


async def save_answers(
    sec_id: int,
    questions: List[str],
    answers: List[str],
    section_name: str
) -> bool:
    """
    Update section_que_ans with user answers.
    """
    pool = await get_pool()
    qa_data = json.dumps({
        "section_name": section_name,
        "questions": questions,
        "answers": answers
    })
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE section_que_ans
            SET doc_sec_que_ans = $1::jsonb
            WHERE sec_id = $2
            """,
            qa_data, sec_id
        )
    logger.info(f"Answers saved: sec_id={sec_id}")
    return True


async def get_qa_by_sec_id(sec_id: int) -> Optional[Dict]:
    """Fetch the full Q&A row for a section."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT sec_id, doc_sec_id, doc_id, doc_sec_que_ans
            FROM section_que_ans
            WHERE sec_id = $1
            """,
            sec_id
        )
    if not row:
        return None
    result = dict(row)
    # Parse JSONB
    if isinstance(result["doc_sec_que_ans"], str):
        result["doc_sec_que_ans"] = json.loads(result["doc_sec_que_ans"])
    return result


async def get_all_qa_for_document(doc_sec_id: int, doc_id: int) -> List[Dict]:
    """Fetch all Q&A rows for a full document (all sections)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT sec_id, doc_sec_id, doc_id, doc_sec_que_ans
            FROM section_que_ans
            WHERE doc_sec_id = $1 AND doc_id = $2
            ORDER BY sec_id
            """,
            doc_sec_id, doc_id
        )
    result = []
    for row in rows:
        r = dict(row)
        if isinstance(r["doc_sec_que_ans"], str):
            r["doc_sec_que_ans"] = json.loads(r["doc_sec_que_ans"])
        result.append(r)
    return result


# ─── TABLE: gen_doc ───────────────────────────────────────────────────────────

async def save_generated_document(
    doc_id: int,
    doc_sec_id: int,
    sec_id: int,
    gen_doc_sec_dec: List[str],
    gen_doc_full: str
) -> int:
    """
    Insert a generated document into gen_doc.
    Returns the new gen_id (SERIAL PK).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        gen_id = await conn.fetchval(
            """
            INSERT INTO gen_doc (doc_id, doc_sec_id, sec_id, gen_doc_sec_dec, gen_doc_full)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING gen_id
            """,
            doc_id, doc_sec_id, sec_id,
            gen_doc_sec_dec,
            gen_doc_full
        )
    logger.info(f"Document saved: gen_id={gen_id}")
    return gen_id


async def update_section_content(gen_id: int, updated_sections: List[str], full_doc: str) -> bool:
    """Update gen_doc after section edit/enhance."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE gen_doc
            SET gen_doc_sec_dec = $1,
                gen_doc_full = $2
            WHERE gen_id = $3
            """,
            updated_sections, full_doc, gen_id
        )
    logger.info(f"Document updated: gen_id={gen_id}")
    return True


async def get_generated_document(gen_id: int) -> Optional[Dict]:
    """Fetch a generated document by gen_id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT gen_id, doc_id, doc_sec_id, sec_id,
                   gen_doc_sec_dec, gen_doc_full
            FROM gen_doc
            WHERE gen_id = $1
            """,
            gen_id
        )
    return dict(row) if row else None


async def get_all_generated_documents(doc_id: Optional[int] = None) -> List[Dict]:
    """Fetch all generated documents, optionally filtered by department doc_id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if doc_id:
            rows = await conn.fetch(
                "SELECT gen_id, doc_id, doc_sec_id, sec_id, gen_doc_full FROM gen_doc WHERE doc_id = $1 ORDER BY gen_id DESC",
                doc_id
            )
        else:
            rows = await conn.fetch(
                "SELECT gen_id, doc_id, doc_sec_id, sec_id, gen_doc_full FROM gen_doc ORDER BY gen_id DESC"
            )
    return [dict(r) for r in rows]