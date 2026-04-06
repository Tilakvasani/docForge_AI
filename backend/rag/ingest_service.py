"""
ingest_service.py — Notion → Chunks → Embeddings → ChromaDB
=============================================================

POST /api/rag/ingest triggers this module.

Pipeline:
  1. Fetch all pages from the configured Notion database
  2. Extract + clean text from each page (blocks → plain text)
  3. Chunk text with overlap (respects paragraph boundaries)
  4. Embed chunks using Azure OpenAI text-embedding-3-large
  5. Upsert into ChromaDB with rich metadata
  6. Store ingest metadata in Redis for /status endpoint

Idempotent: chunks are upserted by deterministic ID (md5 of page_id+heading+chunk_index),
so re-running ingest updates changed content without duplicating.

force=True skips the "already ingested" Redis lock check.

FIXES applied:
  - Heading text is now included in the section body (was silently dropped)
  - Headings-only sections (no body blocks) are no longer lost
  - MIN_CHUNK_LEN lowered to 40 to avoid discarding short-but-valid chunks
  - Per-page section/chunk log upgraded from DEBUG → INFO so it's always visible
  - NOTION_DATABASE_ID is auto-stripped of any '?v=...' view suffix on startup
  - Notion token fallback: tries NOTION_TOKEN then NOTION_API_KEY
  - Redis lock is force-cleared if previous ingest never released it
"""

import asyncio
import hashlib
import re
import time
from typing import Optional

import chromadb
from langchain_openai import AzureOpenAIEmbeddings
from notion_client import Client

from backend.core.config import settings
from backend.core.logger import logger
from backend.services.redis_service import cache

COLLECTION_NAME   = "rag_chunks"
CHUNK_SIZE        = 350     # target chars per chunk
CHUNK_OVERLAP     = 100     # overlap between consecutive chunks
MIN_CHUNK_LEN     = 50      # discard very short fragments (lowered from 80)
BATCH_EMBED_SIZE  = 16      # reduced from 64 — Azure times out on large batches
INGEST_LOCK_KEY   = "docforge:rag:ingest_lock"
INGEST_META_KEY   = "docforge:rag:ingest_meta"
INGEST_LOCK_TTL   = 1800    # 30 min — prevent concurrent ingests


# ── Singleton clients (reused across calls) ───────────────────────────────────

_embedder_instance   = None
_collection_instance = None


def _get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = AzureOpenAIEmbeddings(
            azure_endpoint=settings.AZURE_EMB_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMB_KEY,
            azure_deployment=settings.AZURE_EMB_DEPLOYMENT,
            api_version=settings.AZURE_EMB_API_VERSION,
            timeout=60,        # FIX: prevent silent hang on slow Azure response
            max_retries=2,     # FIX: retry on transient errors instead of hanging
        )
    return _embedder_instance


def _get_collection():
    global _collection_instance
    if _collection_instance is None:
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        _collection_instance = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' ready at %s", COLLECTION_NAME, settings.CHROMA_PATH)
    return _collection_instance


# ── Notion helpers ─────────────────────────────────────────────────────────────

def _get_notion_token() -> str:
    """
    Return the Notion token, trying NOTION_TOKEN first then NOTION_API_KEY fallback.
    Raises a clear error if neither is set.
    """
    token = settings.NOTION_TOKEN or settings.NOTION_API_KEY
    if not token:
        raise ValueError(
            "Notion token not set. Add NOTION_TOKEN=secret_xxx to your .env file."
        )
    return token


def _get_db_id() -> str:
    """
    Return the Notion database UUID, stripping any '?v=...' view suffix that
    users accidentally copy from the browser URL bar.

    Example:
      Input:  32212206f265800cb9d1fa5bd2f4566f?v=32212206f265814b8846000cf4d96197
      Output: 32212206f265800cb9d1fa5bd2f4566f
    """
    raw = settings.NOTION_DATABASE_ID or ""
    # Strip full URL prefix if someone pasted the whole URL
    if "notion.so/" in raw:
        raw = raw.split("notion.so/")[-1]
    # Strip query params (?v=... or ?pvs=...)
    db_id = raw.split("?")[0].strip().rstrip("/")
    if not db_id:
        raise ValueError(
            "NOTION_DATABASE_ID is not set. Add it to your .env — use only the UUID, "
            "not the full URL. Example: NOTION_DATABASE_ID=32212206f265800cb9d1fa5bd2f4566f"
        )
    if db_id != raw.split("?")[0].strip():
        logger.warning(
            "NOTION_DATABASE_ID had a view suffix that was stripped: '%s' → '%s'",
            settings.NOTION_DATABASE_ID, db_id,
        )
    return db_id


def _get_notion():
    return Client(auth=_get_notion_token())


def _fetch_all_notion_pages() -> list[dict]:
    """
    Query all pages from the configured Notion source database.
    Handles Notion pagination automatically.

    NOTE: NOTION_DATABASE_ID must be just the UUID (no '?v=...' suffix).
          _get_db_id() strips it automatically, but fix your .env anyway.
    """
    notion  = _get_notion()
    db_id   = _get_db_id()
    results = []
    cursor  = None

    logger.info("Querying Notion DB: %s", db_id)

    while True:
        kwargs: dict = {"database_id": db_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp    = notion.databases.query(**kwargs)
        results.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    logger.info("Fetched %d pages from Notion DB %s", len(results), db_id)

    if not results:
        logger.warning(
            "Notion returned 0 pages for DB '%s'. "
            "Check: (1) NOTION_DATABASE_ID is correct UUID only, "
            "(2) your integration is shared with the database in Notion "
            "(open DB → ... → Connections → add your integration).",
            db_id,
        )

    return results


def _fetch_page_blocks(page_id: str) -> list[dict]:
    """Recursively fetch all blocks (including children) for a Notion page."""
    notion = _get_notion()
    blocks = []
    cursor = None

    while True:
        kwargs: dict = {"block_id": page_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp   = notion.blocks.children.list(**kwargs)
        blocks.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")

    # Recurse into child blocks (toggles, callouts, etc.)
    expanded = []
    for block in blocks:
        expanded.append(block)
        if block.get("has_children"):
            try:
                child_blocks = _fetch_page_blocks(block["id"])
                expanded.extend(child_blocks)
            except Exception as e:
                logger.warning("Failed to fetch children of block %s: %s", block["id"], e)

    return expanded


# ── Text extraction ────────────────────────────────────────────────────────────

def _rich_text_to_str(rich_text_items: list) -> str:
    return "".join(t.get("plain_text", "") for t in rich_text_items)


def _block_to_text(block: dict) -> tuple[str, str]:
    """
    Extract (heading_label, text) from a Notion block.

    heading_label — non-empty only for heading block types; signals a new section.
    text          — the actual text content; ALWAYS returned so it can be added
                    to the section body (fixes the original bug where heading text
                    was set as both heading and text, then the text branch was
                    skipped because heading was truthy).

    Returns ("", "") for unsupported / empty block types.
    """
    btype = block.get("type", "")
    bdata = block.get(btype, {})

    heading = ""
    text    = ""

    if btype in ("heading_1", "heading_2", "heading_3"):
        heading = _rich_text_to_str(bdata.get("rich_text", []))
        text    = heading  # heading text is also kept as body content

    elif btype == "paragraph":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype in ("bulleted_list_item", "numbered_list_item", "to_do"):
        text = "• " + _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "toggle":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "quote":
        text = "> " + _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "callout":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "code":
        text = _rich_text_to_str(bdata.get("rich_text", []))

    elif btype == "table":
        # Parent table block — actual data is in table_row children.
        # We handle reconstruction in _extract_page_content.
        # Return empty here; rows will be collected there.
        text = ""

    elif btype == "table_row":
        cells = bdata.get("cells", [])
        row   = " | ".join(_rich_text_to_str(cell) for cell in cells)
        text  = f"| {row} |"

    elif btype == "divider":
        text = "---"

    return heading, text


def _extract_page_content(page: dict, blocks: list[dict]) -> dict:
    """
    Extract metadata and full text sections from a Notion page + its blocks.
    Returns a dict with title, doc_type, department, text sections with headings.

    FIX 1: heading text is now appended to current_texts so it's included in
            the section body and not silently discarded.
    FIX 2: the final section is always saved even if it only contains a heading
            with no following body blocks.
    """
    props = page.get("properties", {})

    def _prop_text(prop_name: str) -> str:
        prop  = props.get(prop_name, {})
        ptype = prop.get("type", "")
        if ptype == "title":
            return _rich_text_to_str(prop.get("title", []))
        if ptype == "rich_text":
            return _rich_text_to_str(prop.get("rich_text", []))
        if ptype == "select":
            sel = prop.get("select") or {}
            return sel.get("name", "")
        if ptype == "multi_select":
            return ", ".join(s.get("name", "") for s in prop.get("multi_select", []))
        return ""

    # Try common property name variants for title
    title = (
        _prop_text("Name") or _prop_text("Title") or _prop_text("Document Name")
        or page.get("url", "").split("/")[-1].replace("-", " ")
    )

    doc_type   = _prop_text("Doc Type") or _prop_text("Type") or _prop_text("Document Type") or ""
    department = _prop_text("Department") or _prop_text("Team") or ""
    version    = _prop_text("Version") or "v1"
    status     = _prop_text("Status") or ""

    # Walk blocks and build sections
    sections: list[dict] = []
    current_heading  = title or "General"
    current_texts: list[str] = []

    for i, block in enumerate(blocks):
        btype = block.get("type", "")
        heading, text = _block_to_text(block)

        # ── Table reconstruction: group table_row blocks into a markdown table ─
        if btype == "table":
            # Collect all subsequent table_row children into a markdown table
            table_rows = []
            for j in range(i + 1, len(blocks)):
                child = blocks[j]
                if child.get("type") == "table_row":
                    _, row_text = _block_to_text(child)
                    if row_text.strip():
                        table_rows.append(row_text.strip())
                else:
                    break  # stop at first non-row block

            if table_rows:
                # Build markdown table with header separator
                header = table_rows[0]
                col_count = header.count("|") - 1  # minus outer pipes
                separator = "| " + " | ".join(["---"] * max(col_count, 1)) + " |"
                md_table = "\n".join([header, separator] + table_rows[1:])
                # Mark table with special delimiter so chunker keeps it together:
                # Using \x00TABLE\x00 as a preserve marker
                current_texts.append(f"\x00TABLE_START\x00\n{md_table}\n\x00TABLE_END\x00")
            continue

        # Skip table_row blocks — they were consumed by the parent 'table' handler above
        if btype == "table_row":
            continue

        if not text.strip():
            continue

        if heading:
            # Save the previous section before starting a new one
            if current_texts:
                sections.append({
                    "heading": current_heading,
                    "text":    "\n".join(current_texts).strip(),
                })
                current_texts = []
            current_heading = heading
            # FIX 1: include the heading text in the new section's body
            current_texts.append(text)
        else:
            current_texts.append(text)

    # FIX 2: always flush the last section
    if current_texts:
        sections.append({
            "heading": current_heading,
            "text":    "\n".join(current_texts).strip(),
        })

    # ✅ FIX 3 (was debug → now info): always visible in default INFO log level
    logger.info(
        "  Page '%s': %d blocks → %d sections %s",
        title,
        len(blocks),
        len(sections),
        [(s["heading"][:40], len(s["text"])) for s in sections],
    )

    return {
        "page_id":    page["id"],
        "title":      title,
        "doc_type":   doc_type,
        "department": department,
        "version":    version,
        "status":     status,
        "url":        page.get("url", ""),
        "sections":   sections,
    }


# ── Chunker ────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, preferring paragraph/sentence boundaries.
    Short text that fits within one chunk is returned as-is (single-item list).
    TABLE PRESERVATION: Tables marked with TABLE_START/TABLE_END are never split.
    """
    if not text or len(text) < MIN_CHUNK_LEN:
        return [text] if text.strip() else []

    # ── Extract tables first so they're never split across chunks ────────────
    import re as _re
    table_pattern = _re.compile(r"\x00TABLE_START\x00\n(.*?)\n\x00TABLE_END\x00", _re.DOTALL)
    tables = table_pattern.findall(text)
    # Replace tables with a placeholder
    placeholder_text = table_pattern.sub("\x00TABLE_PLACEHOLDER\x00", text)

    # Split on double newlines first (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n\n+", placeholder_text) if p.strip()]
    chunks: list[str] = []
    current = ""
    table_idx = 0

    for para in paragraphs:
        # If paragraph is a table placeholder, emit current chunk then table as its own chunk
        if "\x00TABLE_PLACEHOLDER\x00" in para:
            if current:
                chunks.append(current)
                current = ""
            if table_idx < len(tables):
                chunks.append(tables[table_idx])
                table_idx += 1
            continue

        if len(current) + len(para) + 1 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If single paragraph exceeds size, split by sentence
            if len(para) > size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sent_buf  = ""
                for sent in sentences:
                    if len(sent_buf) + len(sent) + 1 <= size:
                        sent_buf = (sent_buf + " " + sent).strip()
                    else:
                        if sent_buf:
                            chunks.append(sent_buf)
                        sent_buf = sent
                if sent_buf:
                    current = sent_buf
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply overlap: prepend tail of previous chunk to next (skip table chunks)
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            # Don't apply overlap to/from table chunks
            if "|" in chunks[i] and "---" in chunks[i]:
                overlapped.append(chunks[i])
            elif "|" in chunks[i-1] and "---" in chunks[i-1]:
                overlapped.append(chunks[i])
            else:
                tail    = chunks[i - 1][-overlap:]
                merged  = (tail + " " + chunks[i]).strip()
                overlapped.append(merged)
        chunks = overlapped

    return [c for c in chunks if len(c) >= MIN_CHUNK_LEN]


# ── Chunk ID generator ─────────────────────────────────────────────────────────

def _chunk_id(page_id: str, heading: str, chunk_index: int) -> str:
    raw = f"{page_id}::{heading}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Citation formatter ─────────────────────────────────────────────────────────

def _format_citation(title: str, heading: str, doc_type: str) -> str:
    parts = [title]
    if doc_type and doc_type.lower() != title.lower():
        parts.append(doc_type)
    if heading and heading.lower() != title.lower():
        parts.append(f"§ {heading}")
    return " › ".join(parts)


# ── Embedding + upsert ─────────────────────────────────────────────────────────

def _embed_and_upsert(chunks_batch: list[dict], collection) -> int:
    """
    Embed a batch of chunk dicts and upsert into ChromaDB.
    Each chunk dict has: id, text, metadata.
    Returns number of chunks successfully upserted.
    """
    if not chunks_batch:
        return 0

    embedder = _get_embedder()
    texts    = [c["text"] for c in chunks_batch]

    logger.info("Embedding %d chunks via Azure OpenAI...", len(texts))
    try:
        embeddings = embedder.embed_documents(texts)
        logger.info("Embedding done — got %d vectors", len(embeddings))
    except Exception as e:
        logger.error("Embedding failed for batch of %d: %s", len(texts), e)
        return 0

    ids        = [c["id"]       for c in chunks_batch]
    metadatas  = [c["metadata"] for c in chunks_batch]

    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks_batch)
    except Exception as e:
        logger.error("ChromaDB upsert failed: %s", e)
        return 0


# ── Main ingest function ───────────────────────────────────────────────────────

async def ingest_from_notion(force: bool = False) -> dict:
    """
    Full ingest pipeline:
      Notion pages → extract → chunk → embed → ChromaDB upsert

    Args:
        force: if True, skip the Redis lock check (allow re-ingest).
               Always use force=True during development.

    Returns:
        dict with status, total_docs, total_chunks, elapsed_s, and optionally skipped
    """
    # ── Lock check ──────────────────────────────────────────────────────────────
    if not force:
        locked = await cache.exists(INGEST_LOCK_KEY)
        if locked:
            logger.warning(
                "Ingest lock is active — skipping. "
                "If a previous ingest crashed, clear it with: "
                "redis-cli DEL %s  OR call ingest with force=True",
                INGEST_LOCK_KEY,
            )
            return {
                "status":       "skipped",
                "reason":       "ingest_locked",
                "total_docs":   0,
                "total_chunks": 0,
            }
    else:
        # force=True: clear any stale lock before starting
        await cache.delete(INGEST_LOCK_KEY)
        logger.info("force=True — cleared any stale ingest lock")

    # Set lock
    await cache.set(INGEST_LOCK_KEY, "1", ttl=INGEST_LOCK_TTL)
    t_start = time.time()

    try:
        collection = _get_collection()

        # ── Step 1: Fetch pages ────────────────────────────────────────────────
        logger.info("═══ INGEST START ═══")
        logger.info("Fetching pages from Notion (DB: %s)...", _get_db_id())
        loop  = asyncio.get_event_loop()
        pages = await loop.run_in_executor(None, _fetch_all_notion_pages)

        if not pages:
            logger.warning(
                "No pages returned from Notion.\n"
                "  → Fix 1: Make sure NOTION_DATABASE_ID in .env is ONLY the UUID:\n"
                "           NOTION_DATABASE_ID=32212206f265800cb9d1fa5bd2f4566f\n"
                "           (no ?v=... suffix, no full URL)\n"
                "  → Fix 2: In Notion open the database → ... menu → Connections\n"
                "           → confirm your integration is listed there."
            )
            return {
                "status":       "done",
                "total_docs":   0,
                "total_chunks": 0,
                "elapsed_s":    round(time.time() - t_start, 1),
            }

        total_docs   = len(pages)
        total_chunks = 0
        all_chunks:  list[dict] = []

        # ── Step 2: Extract + chunk ────────────────────────────────────────────
        logger.info("Extracting + chunking %d pages...", total_docs)
        for page in pages:
            page_id = page["id"]
            try:
                blocks  = await loop.run_in_executor(None, _fetch_page_blocks, page_id)
                content = _extract_page_content(page, blocks)
            except Exception as e:
                logger.error("Failed to extract page %s: %s", page_id, e)
                continue

            title      = content["title"]
            doc_type   = content["doc_type"]
            department = content["department"]
            version    = content["version"]
            url        = content["url"]

            page_chunk_count = 0
            for section in content["sections"]:
                heading  = section["heading"]
                text     = section["text"]
                chunks   = _chunk_text(text)
                citation = _format_citation(title, heading, doc_type)

                for idx, chunk_text in enumerate(chunks):
                    cid = _chunk_id(page_id, heading, idx)
                    all_chunks.append({
                        "id":   cid,
                        "text": chunk_text,
                        "metadata": {
                            "notion_page_id": page_id,
                            "doc_title":      title,
                            "doc_type":       doc_type,
                            "department":     department,
                            "version":        version,
                            "heading":        heading,
                            "chunk_index":    idx,
                            "citation":       citation,
                            "source_url":     url,
                        },
                    })
                    page_chunk_count += 1

            logger.info(
                "  ✔ Page '%s': %d sections → %d chunks",
                title, len(content["sections"]), page_chunk_count,
            )

        logger.info("Prepared %d total chunks from %d pages", len(all_chunks), total_docs)

        if not all_chunks:
            logger.warning(
                "All pages extracted but produced 0 chunks. "
                "Possible causes: pages have no text content, or all text was below "
                "MIN_CHUNK_LEN=%d characters.", MIN_CHUNK_LEN
            )

        # ── Step 3: Embed + upsert in batches ──────────────────────────────────
        num_batches = (len(all_chunks) + BATCH_EMBED_SIZE - 1) // BATCH_EMBED_SIZE
        for i in range(0, len(all_chunks), BATCH_EMBED_SIZE):
            batch        = all_chunks[i : i + BATCH_EMBED_SIZE]
            upserted     = await loop.run_in_executor(
                None, _embed_and_upsert, batch, collection
            )
            total_chunks += upserted
            logger.info(
                "  Embedded batch %d/%d — %d/%d chunks upserted so far",
                i // BATCH_EMBED_SIZE + 1,
                num_batches,
                total_chunks,
                len(all_chunks),
            )

        elapsed = round(time.time() - t_start, 1)

        # ── Step 4: Save ingest metadata to Redis ──────────────────────────────
        meta = {
            "total_docs":   total_docs,
            "total_chunks": total_chunks,
            "elapsed_s":    elapsed,
            "ingested_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        await cache.set(INGEST_META_KEY, meta)

        logger.info(
            "═══ INGEST COMPLETE: %d docs, %d chunks, %.1fs ═══",
            total_docs, total_chunks, elapsed,
        )
        return {"status": "done", **meta}

    except Exception as e:
        logger.error("Ingest pipeline failed: %s", e, exc_info=True)
        raise

    finally:
        # Always release the lock regardless of success or failure
        await cache.delete(INGEST_LOCK_KEY)