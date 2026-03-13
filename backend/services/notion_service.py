"""
DocForge AI — notion_service.py
- Publishes plain-text documents to Notion with metadata callout block at top
- gen_doc_full contains ONLY section content (no metadata)
- Notion page gets: metadata callout + section headings + content
- Library fetch for sidebar
"""
import httpx
import asyncio
from backend.core.config import settings
from backend.core.logger import logger
from backend.schemas.document_schema import NotionPublishRequest

NOTION_API_URL = "https://api.notion.com/v1"

DEPT_MAP = {
    "HR": "HR", "Human Resources": "HR",
    "Finance": "Finance", "Finance / Accounting": "Finance",
    "Legal": "Legal", "Sales": "Sales", "Marketing": "Marketing",
    "IT": "IT", "Information Technology": "IT",
    "Operations": "Operations", "Customer Support": "Customer Support",
    "Product Management": "Product Management", "Procurement": "Procurement",
}


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }


def _txt(content: str, bold: bool = False) -> dict:
    """Shorthand for a rich_text text object."""
    obj = {"type": "text", "text": {"content": content}}
    if bold:
        obj["annotations"] = {"bold": True}
    return obj


def _para(content: str, bold: bool = False) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [_txt(content, bold)]}}


def _heading2(content: str) -> dict:
    return {"object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [_txt(content)], "color": "default"}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _callout(lines: list[str]) -> dict:
    """Build a grey callout block with metadata lines."""
    rich = []
    for i, line in enumerate(lines):
        if ':' in line:
            key, _, val = line.partition(':')
            rich.append(_txt(key.strip() + ": ", bold=True))
            rich.append(_txt(val.strip() + ("\n" if i < len(lines) - 1 else "")))
        else:
            rich.append(_txt(line + ("\n" if i < len(lines) - 1 else "")))
    return {
        "object": "block", "type": "callout",
        "callout": {
            "rich_text": rich,
            "icon": {"type": "emoji", "emoji": "📋"},
            "color": "gray_background",
        }
    }


def _table_to_notion(table_lines: list[str]) -> dict | None:
    """Convert pipe-format table lines to a Notion table block."""
    rows = []
    for line in table_lines:
        if all(c in '-|: ' for c in line):
            continue   # skip separator row
        if '|' not in line:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells:
            rows.append(cells)

    if not rows:
        return None

    col_count = max(len(r) for r in rows)
    # Pad shorter rows
    rows = [r + [''] * (col_count - len(r)) for r in rows]

    return {
        "object": "block", "type": "table",
        "table": {
            "table_width": col_count,
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                {
                    "object": "block", "type": "table_row",
                    "table_row": {
                        "cells": [[_txt(cell)] for cell in row]
                    }
                }
                for row in rows
            ]
        }
    }


def _plain_text_to_blocks(plain_text: str, meta_callout: dict) -> list[dict]:
    """
    Convert plain-text document (sections only, no metadata) to Notion blocks.
    Structure expected:
      SECTION NAME
      -----------
      content text...

      NEXT SECTION
      ------------
      content...
    """
    blocks = [meta_callout, _divider()]
    lines  = plain_text.split('\n')
    i      = 0

    while i < len(lines):
        line     = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Section heading: ALL CAPS line followed by a dash-only line
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        is_heading = (stripped.isupper() and len(stripped) > 2
                      and next_line and all(c == '-' for c in next_line))

        if is_heading:
            blocks.append(_heading2(stripped))
            i += 2  # skip heading + dash line
            continue

        # Pure dash separator line — skip (already handled above or standalone)
        if all(c == '-' for c in stripped):
            i += 1
            continue

        # Table block: collect all contiguous pipe lines
        if '|' in stripped:
            table_lines = []
            while i < len(lines) and ('|' in lines[i] or
                  (lines[i].strip() and all(c in '-|: ' for c in lines[i]))):
                table_lines.append(lines[i])
                i += 1
            tbl = _table_to_notion(table_lines)
            if tbl:
                blocks.append(tbl)
            continue

        # Numbered list line
        import re
        num_match = re.match(r'^(\d+)[.)]\s+(.+)$', stripped)
        if num_match:
            blocks.append({
                "object": "block", "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [_txt(num_match.group(2))]}
            })
            i += 1
            continue

        # Bullet line
        bullet_match = re.match(r'^[-•]\s+(.+)$', stripped)
        if bullet_match:
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [_txt(bullet_match.group(1))]}
            })
            i += 1
            continue

        # Regular paragraph — collect until blank line or heading
        para_lines = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur:
                break
            # Stop if next looks like a heading
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if cur.isupper() and nxt and all(c == '-' for c in nxt):
                break
            para_lines.append(cur)
            i += 1

        if para_lines:
            text = ' '.join(para_lines)
            # Chunk at 1900 chars for Notion's limit
            for chunk in [text[j:j+1900] for j in range(0, len(text), 1900)]:
                blocks.append(_para(chunk))

    return blocks


async def _post_blocks_in_batches(page_id: str, blocks: list[dict]):
    """Append blocks to a Notion page in batches of 100 (API limit)."""
    async with httpx.AsyncClient(timeout=30) as client:
        for start in range(0, len(blocks), 100):
            batch = blocks[start:start + 100]
            for attempt in range(4):
                resp = await client.patch(
                    f"{NOTION_API_URL}/blocks/{page_id}/children",
                    headers=_headers(),
                    json={"children": batch},
                )
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                break
            if resp.status_code not in (200, 201):
                logger.error(f"Block append error {resp.status_code}: {resp.text[:200]}")


async def publish_to_notion(request: NotionPublishRequest) -> dict:
    ctx        = request.company_context or {}
    company    = ctx.get("company_name", "Company")
    industry   = ctx.get("industry", "")
    region     = ctx.get("region", "")
    company_sz = ctx.get("company_size", "")
    title      = f"{request.doc_type} — {company}"
    dept       = DEPT_MAP.get(request.department, "Operations")
    word_count = len(request.gen_doc_full.split())

    logger.info(f"Publishing to Notion: '{title}' | dept={dept} | words={word_count}")

    # Build metadata callout (shown in Notion, NOT in downloaded doc)
    meta_lines = [
        f"Organization: {company}",
        f"Department: {request.department}",
        f"Industry: {industry}",
        f"Region: {region}",
        f"Company Size: {company_sz}",
        "Version: v1.0",
        "Classification: Internal Use Only",
        "Generated by: DocForge AI",
    ]
    meta_callout = _callout([l for l in meta_lines if l.split(': ', 1)[-1].strip()])

    # Convert plain text to Notion blocks
    all_blocks = _plain_text_to_blocks(request.gen_doc_full, meta_callout)

    # Step 1: Create page with properties only (no children — avoids 100-block limit on create)
    payload = {
        "parent":     {"database_id": settings.NOTION_DATABASE_ID},
        "properties": {
            "Title":      {"title":     [{"text": {"content": title}}]},
            "Department": {"select":    {"name": dept}},
            "Doc Type":   {"rich_text": [{"text": {"content": request.doc_type}}]},
            "Industry":   {"rich_text": [{"text": {"content": industry}}]},
            "Status":     {"select":    {"name": "Generated"}},
            "Created By": {"rich_text": [{"text": {"content": "DocForge AI"}}]},
            "Version":    {"number": 1},
            "Word Count": {"number": word_count},
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(4):
            resp = await client.post(
                f"{NOTION_API_URL}/pages", headers=_headers(), json=payload
            )
            if resp.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            break

    if resp.status_code not in (200, 201):
        logger.error(f"Notion create error {resp.status_code}: {resp.text}")
        raise Exception(f"Notion API {resp.status_code}: {resp.text[:300]}")

    data    = resp.json()
    page_id = data.get("id", "")
    url     = data.get("url", "")

    # Step 2: Append content blocks in batches of 100
    if all_blocks:
        await _post_blocks_in_batches(page_id, all_blocks)

    logger.info(f"Published: {url}")
    return {"notion_url": url, "notion_page_id": page_id}


async def fetch_library_from_notion() -> list[dict]:
    """Fetch all pages from the Notion database for the library tab."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{NOTION_API_URL}/databases/{settings.NOTION_DATABASE_ID}/query",
            headers=_headers(),
            json={"sorts": [{"property": "Created At", "direction": "descending"}],
                  "page_size": 50},
        )

    if resp.status_code != 200:
        logger.error(f"Library fetch error: {resp.status_code}")
        return []

    library = []
    for page in resp.json().get("results", []):
        props = page.get("properties", {})

        def get_text(k):
            p     = props.get(k, {})
            items = p.get("title", []) if p.get("type") == "title" else p.get("rich_text", [])
            return "".join(i.get("text", {}).get("content", "") for i in items)

        def get_select(k):
            sel = props.get(k, {}).get("select")
            return sel.get("name", "") if sel else ""

        library.append({
            "title":      get_text("Title"),
            "doc_type":   get_text("Doc Type"),
            "department": get_select("Department"),
            "industry":   get_text("Industry"),
            "status":     get_select("Status"),
            "notion_url": page.get("url", ""),
            "created_at": page.get("created_time", "")[:10],
        })

    return library