"""
DocForge AI — notion_service.py  v3.0
════════════════════════════════════════════════════════════════
Flowchart rendering strategy:
  - Mermaid blocks → PNG via flowchart_renderer.py
  - PNG uploaded to Imgur (anonymous, free, permanent URL)
  - Notion image block embeds the Imgur URL → renders as real visual
  - Fallback: if upload fails → numbered step list callout

Setup (one time):
  1. Go to https://api.imgur.com/oauth2/addclient
  2. Register → "Anonymous usage without user authorization"
  3. Add IMGUR_CLIENT_ID=your_client_id to your .env file
  4. If not set → falls back to step-list rendering (no image)
"""

import re
import base64
import httpx
import asyncio
from typing import Optional
from backend.core.config import settings
from backend.core.logger import logger
from backend.schemas.document_schema import NotionPublishRequest

try:
    from flowchart_renderer import mermaid_to_png_bytes
    FLOWCHART_RENDERER_AVAILABLE = True
except ImportError:
    FLOWCHART_RENDERER_AVAILABLE = False

NOTION_API_URL  = "https://api.notion.com/v1"
IMGUR_UPLOAD_URL = "https://api.imgur.com/3/image"

DEPT_MAP = {
    "HR": "HR", "Human Resources": "HR",
    "Finance": "Finance", "Finance / Accounting": "Finance",
    "Legal": "Legal", "Sales": "Sales", "Marketing": "Marketing",
    "IT": "IT", "Information Technology": "IT",
    "Operations": "Operations", "Customer Support": "Customer Support",
    "Product Management": "Product Management", "Procurement": "Procurement",
}



# ─────────────────────────────────────────────────────────────────────────────
#  FIX: TOKEN HELPER — tries NOTION_TOKEN first, then NOTION_API_KEY
# ─────────────────────────────────────────────────────────────────────────────

def _get_notion_token() -> str:
    token = settings.NOTION_TOKEN or settings.NOTION_API_KEY
    if not token:
        raise ValueError("Notion token not set. Add NOTION_TOKEN=secret_xxx to your .env")
    return token


# ─────────────────────────────────────────────────────────────────────────────
#  FIX: DB ID HELPER — strips ?v=... view suffix from NOTION_DATABASE_ID
# ─────────────────────────────────────────────────────────────────────────────

def _get_notion_db_id() -> str:
    raw = settings.NOTION_DATABASE_ID or ""
    if "notion.so/" in raw:
        raw = raw.split("notion.so/")[-1]
    db_id = raw.split("?")[0].strip().rstrip("/")
    if not db_id:
        raise ValueError("NOTION_DATABASE_ID is not set in your .env")
    return db_id


# ─────────────────────────────────────────────────────────────────────────────
#  NOTION HEADERS
# ─────────────────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_notion_token()}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  IMGUR UPLOAD — PNG bytes → public image URL
# ─────────────────────────────────────────────────────────────────────────────

async def _upload_png_to_imgur(png_bytes: bytes, title: str = "") -> Optional[str]:
    """
    Upload PNG to Imgur anonymously and return the direct image URL.
    Returns None if IMGUR_CLIENT_ID is not set or upload fails.
    """
    client_id = getattr(settings, "IMGUR_CLIENT_ID", None)
    if not client_id:
        logger.warning("IMGUR_CLIENT_ID not configured — flowchart will render as step list")
        return None

    try:
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                IMGUR_UPLOAD_URL,
                headers={"Authorization": f"Client-ID {client_id}"},
                json={
                    "image": b64,
                    "type": "base64",
                    "title": title or "DocForge AI Flowchart",
                    "description": f"Process flow diagram — {title}",
                }
            )
        if resp.status_code == 200:
            url = resp.json().get("data", {}).get("link", "")
            if url:
                logger.info(f"Imgur upload OK: {url}")
                return url
        logger.warning(f"Imgur upload failed {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        logger.warning(f"Imgur upload exception: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  NOTION BLOCK HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _txt(content: str, bold=False, italic=False, code=False, color="default") -> dict:
    return {
        "type": "text",
        "text": {"content": content},
        "annotations": {"bold": bold, "italic": italic, "code": code, "color": color}
    }


def _para(content: str, bold=False) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [_txt(content, bold=bold)]}}


def _heading2(content: str) -> dict:
    return {"object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [_txt(content)], "color": "default"}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _image_block(url: str, caption: str = "") -> dict:
    """Notion external image block — renders as a real visual image."""
    block = {
        "object": "block",
        "type": "image",
        "image": {
            "type": "external",
            "external": {"url": url},
        }
    }
    if caption:
        block["image"]["caption"] = [_txt(caption, italic=True)]
    return block


def _callout(lines: list, emoji: str = "📋", color: str = "gray_background") -> dict:
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
        "callout": {"rich_text": rich,
                    "icon": {"type": "emoji", "emoji": emoji},
                    "color": color}
    }


def _table_to_notion(table_lines: list) -> Optional[dict]:
    rows = []
    for line in table_lines:
        if all(c in '-|: ' for c in line):
            continue
        if '|' not in line:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells:
            rows.append(cells)
    if not rows:
        return None
    col_count = max(len(r) for r in rows)
    rows = [r + [''] * (col_count - len(r)) for r in rows]
    return {
        "object": "block", "type": "table",
        "table": {
            "table_width": col_count,
            "has_column_header": True,
            "has_row_header": False,
            "children": [
                {"object": "block", "type": "table_row",
                 "table_row": {"cells": [[_txt(cell)] for cell in row]}}
                for row in rows
            ]
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MERMAID → NOTION BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_mermaid_steps(mermaid_text: str) -> list:
    steps, seen = [], set()
    rounded_re  = re.compile(r'\w+\(\[([^\]\)]+)\]\)')
    diamond_re  = re.compile(r'\w+\{([^\}]+)\}')
    rect_re     = re.compile(r'\w+\[([^\]]+)\]')
    for line in mermaid_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('flowchart') or line.startswith('graph'):
            continue
        for m in rounded_re.finditer(line):
            lbl = m.group(1).strip()
            if lbl not in seen:
                seen.add(lbl)
                steps.append({'label': lbl, 'is_terminal': True, 'is_decision': False})
        for m in diamond_re.finditer(line):
            lbl = m.group(1).strip()
            if lbl not in seen:
                seen.add(lbl)
                steps.append({'label': lbl, 'is_terminal': False, 'is_decision': True})
        for m in rect_re.finditer(line):
            lbl = m.group(1).strip()
            if lbl not in seen:
                seen.add(lbl)
                steps.append({'label': lbl, 'is_terminal': False, 'is_decision': False})
    return steps


def _mermaid_fallback_blocks(mermaid_text: str, section_name: str = "") -> list:
    """Fallback: render flowchart as step-list callout when image upload unavailable."""
    blocks = [{
        "object": "block", "type": "callout",
        "callout": {
            "rich_text": [
                _txt("Process Flow Diagram", bold=True),
                _txt(f" — {section_name}" if section_name else ""),
            ],
            "icon": {"type": "emoji", "emoji": "🔀"},
            "color": "blue_background",
        }
    }]
    step_num = 1
    for step in _parse_mermaid_steps(mermaid_text):
        label = step['label']
        if step['is_terminal']:
            icon = "🟢" if step_num == 1 else "🏁"
            blocks.append({
                "object": "block", "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [_txt(f"{icon}  {label}", bold=True)]}
            })
        elif step['is_decision']:
            blocks.append({
                "object": "block", "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [_txt(f"❓ Decision: {label}", bold=True)]}
            })
            step_num += 1
        else:
            blocks.append({
                "object": "block", "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [_txt(label)]}
            })
            step_num += 1
    blocks.append(_divider())
    return blocks


async def _mermaid_to_notion_blocks(mermaid_text: str, section_name: str = "") -> list:
    """
    Convert Mermaid flowchart to Notion blocks.
    Tries: render PNG → upload to Imgur → Notion image block
    Falls back to: numbered step list callout
    """
    if FLOWCHART_RENDERER_AVAILABLE:
        try:
            png_bytes = mermaid_to_png_bytes(mermaid_text, title=section_name, dpi=180)
            img_url   = await _upload_png_to_imgur(png_bytes, title=section_name)
            if img_url:
                caption = f"Figure: {section_name} — Process Flow Diagram" if section_name else "Process Flow Diagram"
                return [
                    _divider(),
                    _image_block(img_url, caption=caption),
                    _divider(),
                ]
        except Exception as e:
            logger.warning(f"Flowchart image pipeline failed: {e}")

    return _mermaid_fallback_blocks(mermaid_text, section_name)


# ─────────────────────────────────────────────────────────────────────────────
#  PLAIN TEXT → NOTION BLOCKS  (async for image uploads)
# ─────────────────────────────────────────────────────────────────────────────

async def _plain_text_to_blocks(plain_text: str, meta_callout: dict) -> list:
    blocks          = [meta_callout, _divider()]
    mermaid_pattern = re.compile(r'```mermaid(.*?)```', re.DOTALL)
    segments        = mermaid_pattern.split(plain_text)
    current_section = ""

    for seg_idx, segment in enumerate(segments):

        # Mermaid block → image or fallback
        if seg_idx % 2 == 1:
            blocks.extend(await _mermaid_to_notion_blocks(segment.strip(), current_section))
            continue

        # Regular text
        lines = segment.split('\n')
        i     = 0
        while i < len(lines):
            line     = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            # Section heading (ALL CAPS + dash line)
            next_line  = lines[i + 1].strip() if i + 1 < len(lines) else ""
            is_heading = (stripped.isupper() and len(stripped) > 2
                          and next_line and all(c == '-' for c in next_line))
            if is_heading:
                current_section = stripped
                blocks.append(_heading2(stripped))
                i += 2
                continue

            # Standalone dash separator
            if all(c == '-' for c in stripped) and len(stripped) > 2:
                i += 1
                continue

            # Pipe table
            if '|' in stripped:
                table_lines = []
                while i < len(lines) and (
                    '|' in lines[i] or
                    (lines[i].strip() and all(c in '-|: ' for c in lines[i]))
                ):
                    table_lines.append(lines[i])
                    i += 1
                tbl = _table_to_notion(table_lines)
                if tbl:
                    blocks.append(tbl)
                continue

            # Numbered list
            num_match = re.match(r'^(\d+)[.)]\s+(.+)$', stripped)
            if num_match:
                blocks.append({
                    "object": "block", "type": "numbered_list_item",
                    "numbered_list_item": {"rich_text": [_txt(num_match.group(2))]}
                })
                i += 1
                continue

            # Bullet list
            bullet_match = re.match(r'^[-•]\s+(.+)$', stripped)
            if bullet_match:
                blocks.append({
                    "object": "block", "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [_txt(bullet_match.group(1))]}
                })
                i += 1
                continue

            # Regular paragraph
            para_lines = []
            while i < len(lines):
                cur = lines[i].strip()
                if not cur:
                    break
                if '|' in cur or cur.startswith('```'):
                    break
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if cur.isupper() and nxt and all(c == '-' for c in nxt):
                    break
                para_lines.append(cur)
                i += 1
            if para_lines:
                text = ' '.join(para_lines)
                for chunk in [text[j:j + 1900] for j in range(0, len(text), 1900)]:
                    blocks.append(_para(chunk))

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH BLOCK APPENDER
# ─────────────────────────────────────────────────────────────────────────────

async def _post_blocks_in_batches(page_id: str, blocks: list):
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


# ─────────────────────────────────────────────────────────────────────────────
#  VERSION CONTROL — get next version for dept + doc_type combo
# ─────────────────────────────────────────────────────────────────────────────

async def _get_next_version(dept: str, doc_type: str) -> int:
    """
    Query Notion for existing pages with the same Department + Doc Type.
    Returns max existing version + 1, or 1 if none found.
    """
    query = {
        "filter": {
            "and": [
                {"property": "Department", "select": {"equals": dept}},
                {"property": "Doc Type",   "rich_text": {"equals": doc_type}},
            ]
        },
        "sorts": [{"property": "Version", "direction": "descending"}],
        "page_size": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{NOTION_API_URL}/databases/{_get_notion_db_id()}/query",
                headers=_headers(),
                json=query,
            )
        if resp.status_code != 200:
            logger.warning(f"Version check failed {resp.status_code} — defaulting to v1")
            return 1

        results = resp.json().get("results", [])
        if not results:
            logger.info(f"No existing doc for dept='{dept}' type='{doc_type}' — starting at v1")
            return 1

        existing_version = (
            results[0]
            .get("properties", {})
            .get("Version", {})
            .get("number", 1) or 1
        )
        next_version = int(existing_version) + 1
        logger.info(f"Existing version found: v{existing_version} → publishing as v{next_version}")
        return next_version

    except Exception as e:
        logger.warning(f"Version check exception: {e} — defaulting to v1")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLISH TO NOTION
# ─────────────────────────────────────────────────────────────────────────────

async def publish_to_notion(request: NotionPublishRequest) -> dict:
    ctx        = request.company_context or {}
    company    = ctx.get("company_name", "Company")
    industry   = ctx.get("industry", "")
    region     = ctx.get("region", "")
    company_sz = ctx.get("company_size", "")
    title      = f"{request.doc_type} — {company}"
    dept       = DEPT_MAP.get(request.department, "Operations")
    word_count = len(request.gen_doc_full.split())
    fc_count   = len(re.findall(r'```mermaid', request.gen_doc_full))

    # ── Version control: auto-increment if same dept + doc_type exists ──────
    version = await _get_next_version(dept, request.doc_type)

    logger.info(f"Publishing: '{title}' | dept={dept} | words={word_count} | flowcharts={fc_count} | version=v{version}")

    meta_lines = [
        f"Organization: {company}",
        f"Department: {request.department}",
        f"Industry: {industry}",
        f"Region: {region}",
        f"Company Size: {company_sz}",
        f"Version: v{version}.0",
        "Classification: Internal Use Only",
        "Generated by: DocForge AI",
    ]
    meta_callout = _callout(
        [l for l in meta_lines if l.split(': ', 1)[-1].strip()],
        emoji="📋", color="gray_background"
    )

    all_blocks = await _plain_text_to_blocks(request.gen_doc_full, meta_callout)

    payload = {
        "parent":     {"database_id": _get_notion_db_id()},
        "properties": {
            "Title":      {"title":     [{"text": {"content": title}}]},
            "Department": {"select":    {"name": dept}},
            "Doc Type":   {"rich_text": [{"text": {"content": request.doc_type}}]},
            "Industry":   {"rich_text": [{"text": {"content": industry}}]},
            "Status":     {"select":    {"name": "Generated"}},
            "Created By": {"rich_text": [{"text": {"content": "DocForge AI"}}]},
            "Version":    {"number": version},
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

    if all_blocks:
        await _post_blocks_in_batches(page_id, all_blocks)

    logger.info(f"Published: {url} | blocks={len(all_blocks)} | version=v{version}")
    return {"notion_url": url, "notion_page_id": page_id, "version": version}


# ─────────────────────────────────────────────────────────────────────────────
#  LIBRARY FETCH
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_library_from_notion() -> list:
    library, cursor = [], None
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            body = {
                "sorts": [{"property": "Created At", "direction": "descending"}],
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor
            resp = await client.post(
                f"{NOTION_API_URL}/databases/{_get_notion_db_id()}/query",
                headers=_headers(), json=body,
            )
            if resp.status_code != 200:
                logger.error(f"Library fetch error: {resp.status_code}")
                break
            data = resp.json()
            for page in data.get("results", []):
                props = page.get("properties", {})

                def get_text(k, _props=props):
                    p     = _props.get(k, {})
                    items = p.get("title", []) if p.get("type") == "title" else p.get("rich_text", [])
                    return "".join(i.get("text", {}).get("content", "") for i in items)

                def get_select(k, _props=props):
                    sel = _props.get(k, {}).get("select")
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
            cursor = data.get("next_cursor")
            if not data.get("has_more") or not cursor:
                break
    logger.info(f"Library fetched: {len(library)} documents")
    return library