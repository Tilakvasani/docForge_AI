"""
PostgreSQL data access layer for DocForge AI.

Manages all database interactions across the four core tables:

    depart              — Departments and their available document types.
    document_section    — Standard sections per document type.
    section_que_ans     — LLM-generated questions and user answers per section.
    gen_doc             — Final generated documents (full text + per-section breakdown).

A shared asyncpg connection pool is maintained as a module-level singleton
and lazily initialized on first use.
"""

import json
import asyncpg
from typing import Optional, List, Dict, Any
from backend.core.config import settings
from backend.core.logger import logger


_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """
    Return the shared asyncpg connection pool, creating it on first call.

    The pool is initialized with a minimum of 2 and maximum of 10
    concurrent connections. Subsequent calls return the same instance.

    Returns:
        The active `asyncpg.Pool`.
    """
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
    """
    Gracefully close the connection pool and reset the module-level singleton.

    Should be called once during application shutdown via the FastAPI lifespan.
    """
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


async def get_all_departments() -> List[Dict]:
    """
    Fetch all department records ordered by `doc_id`.

    Returns:
        A list of dicts with keys: `doc_id`, `department`, `doc_types`.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT doc_id, department, doc_types FROM depart ORDER BY doc_id"
        )
    return [dict(r) for r in rows]


async def get_sections_by_doc_type(doc_type: str) -> Optional[Dict]:
    """
    Fetch the section definition for a specific document type.

    Args:
        doc_type: The exact document type string (e.g. "Standard Operating Procedure").

    Returns:
        A dict with `doc_sec_id`, `doc_type`, and `doc_sec`, or `None` if not found.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT doc_sec_id, doc_type, doc_sec FROM document_section WHERE doc_type = $1",
            doc_type
        )
    return dict(row) if row else None


async def save_questions(
    doc_sec_id: int,
    doc_id: int,
    section_name: str,
    questions: List[str],
    section_type: str = "text",
) -> int:
    """
    Persist LLM-generated questions for a document section.

    Creates a new row in `section_que_ans` with an empty answers list.
    The `section_type` is stored alongside questions so it can be
    retrieved at content-generation time without re-detection.

    Args:
        doc_sec_id:   FK referencing the document section record.
        doc_id:       FK referencing the department/doc-type record.
        section_name: Display name of the section.
        questions:    List of generated question strings.
        section_type: Detected section type (text, table, flowchart, raci, signature).

    Returns:
        The `sec_id` primary key of the newly inserted row.
    """
    pool = await get_pool()
    qa_data = json.dumps({
        "section_name": section_name,
        "section_type": section_type,
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
    Update a section's Q&A row with user-provided answers.

    Preserves the `section_type` that was stored during question generation
    so that the content generator can use the correct prompt template.

    Args:
        sec_id:       Primary key of the target `section_que_ans` row.
        questions:    The original list of generated questions.
        answers:      User-provided answers aligned by index to `questions`.
        section_name: Display name of the section (for logging).

    Returns:
        `True` on success.
    """
    existing = await get_qa_by_sec_id(sec_id)
    section_type = (existing or {}).get("doc_sec_que_ans", {}).get("section_type", "text")

    pool = await get_pool()
    qa_data = json.dumps({
        "section_name": section_name,
        "section_type": section_type,
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
    """
    Retrieve the full Q&A record for a section by its primary key.

    The `doc_sec_que_ans` column is stored as JSONB and is automatically
    deserialized to a Python dict.

    Args:
        sec_id: Primary key of the `section_que_ans` row.

    Returns:
        A dict with `sec_id`, `doc_sec_id`, `doc_id`, and `doc_sec_que_ans`,
        or `None` if no record exists.
    """
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
    if isinstance(result["doc_sec_que_ans"], str):
        result["doc_sec_que_ans"] = json.loads(result["doc_sec_que_ans"])
    return result


async def save_generated_document(
    doc_id: int,
    doc_sec_id: int,
    sec_id: int,
    gen_doc_sec_dec: List[str],
    gen_doc_full: str
) -> int:
    """
    Insert a fully generated document into the `gen_doc` table.

    Args:
        doc_id:          FK to the department record.
        doc_sec_id:      FK to the document section definition.
        sec_id:          FK to the section Q&A record.
        gen_doc_sec_dec: List of per-section content strings.
        gen_doc_full:    The complete assembled document text.

    Returns:
        The `gen_id` primary key of the newly created record.
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
    """
    Update the content of an existing generated document after a section edit.

    Args:
        gen_id:           Primary key of the `gen_doc` record to update.
        updated_sections: New list of per-section content strings.
        full_doc:         New full assembled document text.

    Returns:
        `True` on success.
    """
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
    """
    Fetch a generated document record by its primary key.

    Args:
        gen_id: Primary key of the `gen_doc` record.

    Returns:
        A dict with `gen_id`, `doc_id`, `doc_sec_id`, `sec_id`,
        `gen_doc_sec_dec`, and `gen_doc_full`, or `None` if not found.
    """
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