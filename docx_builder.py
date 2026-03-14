"""
DocForge AI — docx_builder.py  v3.0  (pure Python, no Node.js)
Drop-in replacement for the old JS-based builder.

Features (1:1 match with generate_docx.js):
  ✅ Professional header:  Company  |  Doc Type  (right-aligned, grey)
  ✅ Footer:               Page N of M  (centred, grey)
  ✅ Title block:          Doc type (large, accent blue) + subtitle line
  ✅ Section headings:     Heading 2, accent blue
  ✅ Pipe-format tables → real Word tables (header row shaded, zebra rows)
  ✅ Bullet lines  (- text)
  ✅ Numbered lines (1. text)
  ✅ Plain paragraphs
  ✅ Returns raw bytes — no temp files, no subprocess

Install once:
    pip install python-docx
"""

import re
import io
from typing import List, Dict

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


# ── Colour palette (matches JS) ────────────────────────────────────────────────
ACCENT  = RGBColor(0x2E, 0x40, 0x57)   # dark blue
GRAY    = RGBColor(0x88, 0x88, 0x88)
LGRAY   = RGBColor(0xF3, 0xF4, 0xF6)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x22, 0x22, 0x22)
BORDER  = RGBColor(0xCC, 0xCC, 0xCC)


# ── Low-level XML helpers ──────────────────────────────────────────────────────

def _set_run_font(run, size_pt: float, bold=False, color: RGBColor = None,
                  font_name="Arial"):
    run.font.name   = font_name
    run.font.size   = Pt(size_pt)
    run.font.bold   = bold
    if color:
        run.font.color.rgb = color


def _hex(color: RGBColor) -> str:
    return f"{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def _cell_shade(cell, fill_color: RGBColor):
    """Apply solid background shading to a table cell."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  _hex(fill_color))
    tcPr.append(shd)


def _cell_borders(cell):
    """Apply thin light-grey borders to a table cell."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"),   "single")
        el.set(qn("w:sz"),    "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), _hex(BORDER))
        tcBorders.append(el)
    tcPr.append(tcBorders)


def _cell_margins(cell, top=80, bottom=80, left=120, right=120):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom),
                      ("left", left), ("right", right)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"),    str(val))
        el.set(qn("w:type"), "dxa")
        tcMar.append(el)
    tcPr.append(tcMar)


def _add_page_number_field(paragraph):
    """Add PAGE / NUMPAGES fields to a paragraph for 'Page N of M'."""
    def _field(fld_char_type):
        fldChar = OxmlElement("w:fldChar")
        fldChar.set(qn("w:fldCharType"), fld_char_type)
        return fldChar

    def _instr(text):
        instrText = OxmlElement("w:instrText")
        instrText.set(qn("xml:space"), "preserve")
        instrText.text = text
        return instrText

    p = paragraph._p
    # "Page "
    r1 = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    r1.append(rPr)
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = "Page "
    r1.append(t)
    p.append(r1)

    # PAGE field
    for part in ("begin", "PAGE", "end"):
        r = OxmlElement("w:r")
        if part in ("begin", "end"):
            r.append(_field(part))
        else:
            r.append(_instr(f" {part} "))
        p.append(r)

    # " of "
    r2 = OxmlElement("w:r")
    t2 = OxmlElement("w:t")
    t2.set(qn("xml:space"), "preserve")
    t2.text = " of "
    r2.append(t2)
    p.append(r2)

    # NUMPAGES field
    for part in ("begin", "NUMPAGES", "end"):
        r = OxmlElement("w:r")
        if part in ("begin", "end"):
            r.append(_field(part))
        else:
            r.append(_instr(f" {part} "))
        p.append(r)


def _add_bottom_border(paragraph):
    """Add a bottom border line under a paragraph (divider)."""
    pPr  = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot  = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), _hex(ACCENT))
    pBdr.append(bot)
    pPr.append(pBdr)


# ── Pipe-table parser ──────────────────────────────────────────────────────────

def _is_table_line(line: str) -> bool:
    return "|" in line

def _is_separator(line: str) -> bool:
    return bool(re.match(r"^\s*[\|\-\s:]+$", line)) and "-" in line

def _parse_pipe_table(lines: List[str]) -> List[List[str]]:
    """Parse pipe-formatted markdown table into list of rows (list of cell strings)."""
    rows = []
    for line in lines:
        if _is_separator(line):
            continue
        if not _is_table_line(line):
            continue
        parts = line.split("|")
        cells = [c.strip() for c in parts]
        # Strip empty leading/trailing from "| a | b |" format
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if cells:
            rows.append(cells)
    return rows


def _build_word_table(doc: Document, rows: List[List[str]]):
    """Insert a styled Word table into the document."""
    if not rows:
        return

    col_count = max(len(r) for r in rows)

    table = doc.add_table(rows=len(rows), cols=col_count)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT

    # Column widths — distribute across 6.5 inches (page width minus margins)
    page_width_twips = 9360  # ≈ 6.5 inches in twips
    col_width = page_width_twips // col_count

    for ri, row_data in enumerate(rows):
        is_header = ri == 0
        tr = table.rows[ri]

        for ci in range(col_count):
            cell = tr.cells[ci]
            cell_text = row_data[ci] if ci < len(row_data) else ""

            # Shading
            if is_header:
                _cell_shade(cell, ACCENT)
            elif ri % 2 == 0:
                _cell_shade(cell, LGRAY)

            _cell_borders(cell)
            _cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

            # Set column width
            cell.width = Twips(col_width)

            # Text
            p = cell.paragraphs[0]
            p.paragraph_format.space_after  = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            run = p.add_run(cell_text)
            _set_run_font(run, size_pt=10, bold=is_header,
                          color=WHITE if is_header else DARK)

    # Spacer after table
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(6)


# ── Section content parser ─────────────────────────────────────────────────────

def _render_section_content(doc: Document, content: str):
    """Parse content string and add paragraphs/tables to doc."""
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Collect a pipe-table block
        if _is_table_line(line):
            table_lines = []
            while i < len(lines) and (_is_table_line(lines[i]) or _is_separator(lines[i])):
                table_lines.append(lines[i])
                i += 1
            rows = _parse_pipe_table(table_lines)
            if rows:
                _build_word_table(doc, rows)
            continue

        stripped = line.strip()

        # Empty line → small spacer
        if not stripped:
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(4)
            i += 1
            continue

        # Numbered list: "1. text" or "1) text"
        num_match = re.match(r"^(\d+)[.)]\s+(.+)$", stripped)
        if num_match:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent  = Inches(0.25)
            p.paragraph_format.space_after  = Pt(4)
            run = p.add_run(f"{num_match.group(1)}. {num_match.group(2)}")
            _set_run_font(run, size_pt=11, color=DARK)
            i += 1
            continue

        # Bullet: "- text" or "• text"
        bullet_match = re.match(r"^[-•]\s+(.+)$", stripped)
        if bullet_match:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent         = Inches(0.25)
            p.paragraph_format.first_line_indent   = Inches(-0.15)
            p.paragraph_format.space_after         = Pt(4)
            run_dot = p.add_run("•  ")
            _set_run_font(run_dot, size_pt=11, color=ACCENT)
            run = p.add_run(bullet_match.group(1))
            _set_run_font(run, size_pt=11, color=DARK)
            i += 1
            continue

        # Plain text paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(stripped)
        _set_run_font(run, size_pt=11, color=DARK)
        i += 1


# ── Header / Footer helpers ────────────────────────────────────────────────────

def _add_header(section, company_name: str, doc_type: str):
    header = section.header
    header.is_linked_to_previous = False
    # Clear default empty paragraph
    for p in header.paragraphs:
        p.clear()
    p = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run(f"{company_name}  |  {doc_type}")
    _set_run_font(run, size_pt=8, color=GRAY)


def _add_footer(section):
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in footer.paragraphs:
        p.clear()
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Style the runs via XML field codes
    for run in p.runs:
        _set_run_font(run, size_pt=8, color=GRAY)
    # Add Page N of M via field codes
    _add_page_number_field(p)
    # Style all runs in that paragraph
    for run in p.runs:
        run.font.size  = Pt(8)
        run.font.color.rgb = GRAY
        run.font.name  = "Arial"


# ── Public API ─────────────────────────────────────────────────────────────────

def build_docx(
    doc_type: str,
    department: str,
    company_name: str,
    industry: str,
    region: str,
    sections: List[Dict],   # [{"name": str, "content": str}]
) -> bytes:
    """
    Build a professional .docx file and return raw bytes.
    Drop-in replacement for the Node.js generate_docx.js pipeline.

    Args:
        doc_type:     Document type name (e.g. "Budget Report")
        department:   Department name
        company_name: Company name
        industry:     Industry string
        region:       Region string
        sections:     List of dicts with "name" and "content" keys

    Returns:
        bytes: Raw .docx file content, ready for st.download_button()
    """
    doc = Document()

    # ── Page setup: Letter size, 1-inch margins ────────────────────────────────
    for section in doc.sections:
        section.page_width      = Twips(12240)   # 8.5 inches
        section.page_height     = Twips(15840)   # 11 inches
        section.top_margin      = Inches(1)
        section.bottom_margin   = Inches(1)
        section.left_margin     = Inches(1)
        section.right_margin    = Inches(1)
        _add_header(section, company_name, doc_type)
        _add_footer(section)

    # ── Default font ───────────────────────────────────────────────────────────
    style = doc.styles["Normal"]
    style.font.name = "Arial"
    style.font.size = Pt(11)

    # ── Title ──────────────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_after  = Pt(10)
    title_p.paragraph_format.space_before = Pt(0)
    title_run = title_p.add_run(doc_type)
    _set_run_font(title_run, size_pt=20, bold=True, color=ACCENT)

    # ── Subtitle line ──────────────────────────────────────────────────────────
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(12)
    sub_run = sub_p.add_run(f"{company_name}  ·  {department}  ·  {region}")
    _set_run_font(sub_run, size_pt=10, color=GRAY)

    # ── Divider ────────────────────────────────────────────────────────────────
    div_p = doc.add_paragraph()
    div_p.paragraph_format.space_after = Pt(14)
    _add_bottom_border(div_p)

    # ── Sections ───────────────────────────────────────────────────────────────
    for idx, sec in enumerate(sections):
        name    = sec.get("name", "")
        content = sec.get("content", "").strip()

        if not content:
            continue

        # Inter-section spacer (not before first)
        if idx > 0:
            sp = doc.add_paragraph()
            sp.paragraph_format.space_after = Pt(8)

        # Section heading
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after  = Pt(6)
        h_run = h.add_run(name)
        _set_run_font(h_run, size_pt=13, bold=True, color=ACCENT)

        # Section body
        _render_section_content(doc, content)

    # ── Footer note at end of document ─────────────────────────────────────────
    doc.add_paragraph().paragraph_format.space_after = Pt(20)
    note_p = doc.add_paragraph()
    note_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note_run = note_p.add_run(
        f"{doc_type}  ·  Generated by DocForge AI  ·  Confidential")
    _set_run_font(note_run, size_pt=8, color=GRAY)

    # ── Serialize to bytes ─────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()
