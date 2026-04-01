import streamlit as st
import httpx
import sys
import os
import importlib.util

# ── Load templates.py directly by absolute file path ─────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))          # .../ui/components
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))    # .../tset_project
_TEMPLATES    = os.path.join(_PROJECT_ROOT, "prompts", "templates.py")

if not os.path.exists(_TEMPLATES):
    st.error(f"❌ templates.py not found at: {_TEMPLATES}")
    st.stop()

_spec = importlib.util.spec_from_file_location("prompts.templates", _TEMPLATES)
_tmpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tmpl)

# ── Verify this is the NEW templates.py (not the old 12-doc version) ──────────
if not hasattr(_tmpl, "DOC_TYPES_BY_CATEGORY"):
    st.error(
        "❌ **Old `templates.py` detected.**\n\n"
        f"The file at `{_TEMPLATES}` is the old version (12 doc types).\n\n"
        "**Fix:** Replace `prompts/templates.py` with the new file from outputs. "
        "It must contain `DOC_TYPES_BY_CATEGORY`, `DEPARTMENTS`, and `get_sections_for_doc_type`."
    )
    st.stop()

get_sections_for_doc_type = _tmpl.get_sections_for_doc_type
DOC_TYPES_BY_CATEGORY     = _tmpl.DOC_TYPES_BY_CATEGORY
DEPARTMENTS               = _tmpl.DEPARTMENTS
# ─────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000/api"


def init_state():
    for k, v in {
        "gen_step":            "config",
        "gen_section":         0,
        "gen_answers":         {},
        "gen_config":          {},
        "last_doc":            None,
        "published":           False,
        "notion_url":          "",
        "cur_sections":        [],
        "gen_sections_done":   [],
        "gen_total_sections":  0,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _progress_bar(current, total, label=""):
    pct    = int((current / total) * 100) if total else 0
    filled = int((current / total) * 28)  if total else 0
    bar    = "█" * filled + "░" * (28 - filled)
    st.markdown(
        f'<div style="margin:1.2rem 0">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px">'
        f'<span style="font-family:monospace;font-size:0.62rem;color:#555;letter-spacing:0.1em;text-transform:uppercase">'
        f'{label or f"Section {current} of {total}"}</span>'
        f'<span style="font-family:monospace;font-size:0.62rem;color:#d4a64a">{pct}%</span>'
        f'</div>'
        f'<div style="font-family:monospace;font-size:0.7rem;color:#d4a64a">{bar}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def _breadcrumb(sections):
    current = st.session_state.gen_section
    dots = []
    for i, s in enumerate(sections):
        if i < current:
            c, sym = "#d4a64a", "●"
        elif i == current:
            c, sym = "#d4a64a", s["icon"]
        else:
            c, sym = "#2a2820", "○"
        dots.append(
            f'<span title="{s["name"]}" style="color:{c};font-size:0.95rem;margin:0 3px">{sym}</span>'
        )
    st.markdown(
        f'<div style="text-align:center;margin-bottom:1.2rem">{"".join(dots)}</div>',
        unsafe_allow_html=True
    )


def _section_header(sec):
    freq = sec.get("freq", "")
    freq_badge = (
        f'<span style="background:rgba(212,166,74,0.1);border:1px solid rgba(212,166,74,0.2);'
        f'border-radius:4px;padding:1px 7px;font-size:0.6rem;color:#d4a64a;'
        f'font-family:monospace;margin-left:8px">{freq}</span>'
    ) if freq else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:1.4rem">'
        f'<span style="font-size:1.4rem">{sec["icon"]}</span>'
        f'<span style="font-family:monospace;font-size:0.68rem;color:#d4a64a;'
        f'letter-spacing:0.14em;text-transform:uppercase">{sec["name"]}</span>'
        f'{freq_badge}</div>',
        unsafe_allow_html=True
    )


def _section_done_badge(name, word_count, auto=False):
    color = "#3a3830" if auto else "#d4a64a"
    icon  = "◈" if auto else "✓"
    label = "auto" if auto else f"{word_count}w"
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'background:#0a0908;border:1px solid #1e1c18;border-radius:8px;'
        f'padding:0.6rem 1rem;margin-bottom:0.4rem">'
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'<span style="color:{color};font-size:0.8rem">{icon}</span>'
        f'<span style="font-family:monospace;font-size:0.63rem;color:#5a5650;'
        f'letter-spacing:0.08em;text-transform:uppercase">{name}</span>'
        f'</div>'
        f'<span style="font-family:monospace;font-size:0.58rem;color:#3a3830">{label}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


def _reset():
    for k in ["gen_step","gen_section","gen_answers","gen_config","last_doc",
              "published","notion_url","cur_sections","gen_sections_done","gen_total_sections"]:
        st.session_state.pop(k, None)


def render_generator_form():
    init_state()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1 — CONFIG
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.gen_step == "config":

        st.markdown(
            '<p style="font-family:monospace;font-size:0.62rem;color:#555;'
            'letter-spacing:0.14em;text-transform:uppercase;margin-bottom:1.4rem">'
            '◈ Document Configuration</p>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            dept = st.selectbox("Department / Category", DEPARTMENTS, key="cfg_dept")
        with col2:
            dept_docs = DOC_TYPES_BY_CATEGORY.get(dept, [])
            doc_type  = st.selectbox("Document Type", dept_docs, key="cfg_doc_type")

        st.markdown(
            '<div style="background:#0a0908;border:1px solid #1a1812;border-radius:10px;'
            'padding:0.9rem 1.2rem;margin:1rem 0 0.5rem">'
            '<p style="font-family:monospace;font-size:0.56rem;color:#2e2c28;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">'
            '◈ always auto-generated — no questions needed</p>'
            '<p style="font-size:0.76rem;color:#2e2c28">'
            'Document Metadata &nbsp;·&nbsp; Version Control Table &nbsp;·&nbsp; Confidentiality Notice'
            '</p></div>',
            unsafe_allow_html=True
        )

        if doc_type:
            preview = get_sections_for_doc_type(doc_type)
            n = len(preview)
            names_html = " &nbsp;›&nbsp; ".join(
                f'<span style="color:#5a5650">{s["name"]}</span>' for s in preview
            )
            st.markdown(
                f'<div style="background:#0a0908;border:1px solid #1a1812;border-radius:10px;'
                f'padding:0.9rem 1.2rem;margin:0.5rem 0 1.2rem">'
                f'<p style="font-family:monospace;font-size:0.56rem;color:#d4a64a;'
                f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">'
                f'◉ {n} sections · {n * 2} questions · each section generated independently</p>'
                f'<p style="font-size:0.74rem;line-height:2">{names_html}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        col3, col4 = st.columns(2)
        with col3:
            author = st.text_input("Document Author", placeholder="e.g. Jane Smith")
        with col4:
            tags = st.text_input("Tags (comma-separated)", placeholder="e.g. SaaS, GDPR, Q2-2026")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Continue to Questions →", type="primary", use_container_width=True):
            if not doc_type:
                st.warning("Please select a document type.")
                return
            st.session_state.gen_config = {
                "industry":   "SaaS",
                "department": dept,
                "doc_type":   doc_type,
                "tags":       [t.strip() for t in tags.split(",") if t.strip()],
                "author":     author or "Admin",
                "title":      f"{doc_type} — {dept}",
            }
            st.session_state.cur_sections = get_sections_for_doc_type(doc_type)
            st.session_state.gen_section  = 0
            st.session_state.gen_answers  = {}
            st.session_state.gen_step     = "questions"
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2 — GUIDED QUESTIONS
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.gen_step == "questions":
        sections = st.session_state.cur_sections
        idx      = st.session_state.gen_section
        total    = len(sections)
        sec      = sections[idx]
        cfg      = st.session_state.gen_config

        _progress_bar(idx + 1, total)
        _breadcrumb(sections)
        _section_header(sec)

        st.markdown(
            f'<p style="font-family:monospace;font-size:0.58rem;color:#2a2820;'
            f'letter-spacing:0.1em;margin-bottom:1.4rem">'
            f'{cfg["doc_type"]} &nbsp;·&nbsp; {cfg["department"]}</p>',
            unsafe_allow_html=True
        )

        temp = {}
        for (key, label, ph) in sec["questions"]:
            temp[key] = st.text_area(
                label,
                value=st.session_state.gen_answers.get(key, ""),
                placeholder=ph,
                height=88,
                key=f"ta_{key}_{idx}"
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_back, col_next = st.columns([1, 3])

        with col_back:
            if st.button("← Back", use_container_width=True):
                st.session_state.gen_answers.update(temp)
                if idx > 0:
                    st.session_state.gen_section -= 1
                else:
                    st.session_state.gen_step = "config"
                st.rerun()

        with col_next:
            is_last   = (idx == total - 1)
            next_name = sections[idx + 1]["name"] if not is_last else ""
            btn_label = "⚡  Generate Document" if is_last else f"Next: {next_name} →"
            btn_type  = "primary" if is_last else "secondary"

            if st.button(btn_label, type=btn_type, use_container_width=True):
                st.session_state.gen_answers.update(temp)
                if is_last:
                    st.session_state.gen_sections_done  = []
                    st.session_state.gen_total_sections = len(sections) + 1
                    st.session_state.gen_step = "generating"
                else:
                    st.session_state.gen_section += 1
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3 — GENERATING WITH LIVE PROGRESS
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.gen_step == "generating":

        cfg      = st.session_state.gen_config
        answers  = st.session_state.gen_answers
        sections = st.session_state.cur_sections
        total    = st.session_state.gen_total_sections
        done     = st.session_state.gen_sections_done

        st.markdown(
            '<p style="font-family:monospace;font-size:0.62rem;color:#d4a64a;'
            'letter-spacing:0.14em;text-transform:uppercase;margin-bottom:1.2rem">'
            '⚡ Generating Document — Section by Section</p>',
            unsafe_allow_html=True
        )

        for s in done:
            _section_done_badge(s["name"], len(s["content"].split()), auto=s.get("auto", False))

        _progress_bar(len(done), total, label=f"Generating section {len(done)+1} of {total}…")

        section_answers = {
            key: answers.get(key, "")
            for sec in sections
            for (key, _label, _ph) in sec.get("questions", [])
        }

        payload = {
            "title":           cfg["title"],
            "industry":        cfg["industry"],
            "department":      cfg["department"],
            "doc_type":        cfg["doc_type"],
            "tags":            cfg["tags"],
            "created_by":      cfg["author"],
            "description":     f"{cfg['doc_type']} for {cfg['department']}",
            "section_answers": section_answers,
            "mode":            "section_by_section",
        }

        with st.spinner(f"Writing section {len(done)+1} of {total}…"):
            try:
                r = httpx.post(f"{API_URL}/generate", json=payload, timeout=180.0)
                r.raise_for_status()
                doc = r.json()

                if "sections" in doc:
                    st.session_state.gen_sections_done = [
                        {"name": s["name"], "content": s["content"], "auto": s.get("auto", False)}
                        for s in doc["sections"]
                    ]

                st.session_state.last_doc   = doc
                st.session_state.published  = False
                st.session_state.notion_url = ""
                st.session_state.gen_step   = "result"
                st.rerun()

            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
                st.session_state.gen_step = "questions"
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.session_state.gen_step = "questions"

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4 — RESULT WITH SECTION BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.gen_step == "result":
        doc = st.session_state.last_doc
        if not doc:
            st.session_state.gen_step = "config"
            st.rerun()

        cfg           = st.session_state.gen_config
        wc            = len(doc.get("content", "").split())
        sections_data = doc.get("sections", [])

        col_t, col_b = st.columns([3, 1])
        with col_t:
            st.markdown(
                f'<p style="font-size:1.05rem;font-weight:600;color:#f0ece3;margin-bottom:0.2rem">'
                f'{doc.get("title","Document")}</p>'
                f'<p style="font-family:monospace;font-size:0.6rem;color:#444">'
                f'{cfg["department"]} &nbsp;·&nbsp; {cfg["doc_type"]} &nbsp;·&nbsp; v{doc.get("version","1.0")}</p>',
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(
                f'<div style="text-align:right;padding-top:4px">'
                f'<span style="background:rgba(212,166,74,0.1);border:1px solid rgba(212,166,74,0.2);'
                f'border-radius:6px;padding:4px 10px;font-family:monospace;font-size:0.62rem;'
                f'color:#d4a64a">{wc} words</span></div>',
                unsafe_allow_html=True
            )

        if sections_data:
            st.markdown(
                f'<p style="font-family:monospace;font-size:0.54rem;color:#3a3830;'
                f'letter-spacing:0.1em;text-transform:uppercase;margin:1rem 0 0.6rem">'
                f'◉ {len(sections_data)} sections generated independently</p>',
                unsafe_allow_html=True
            )
            cols = st.columns(min(len(sections_data), 4))
            for i, s in enumerate(sections_data):
                with cols[i % 4]:
                    sw    = len(s.get("content", "").split())
                    label = "auto" if s.get("auto") else f"{sw}w"
                    color = "#3a3830" if s.get("auto") else "#d4a64a"
                    st.markdown(
                        f'<div style="background:#0a0908;border:1px solid #1a1812;border-radius:6px;'
                        f'padding:0.5rem 0.6rem;margin-bottom:0.4rem;text-align:center">'
                        f'<p style="font-family:monospace;font-size:0.52rem;color:{color};'
                        f'letter-spacing:0.05em;text-transform:uppercase;margin:0;'
                        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                        f'{s["name"][:22]}</p>'
                        f'<p style="font-family:monospace;font-size:0.58rem;color:#2a2820;margin:2px 0 0">'
                        f'{label}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:#0a0908;border:1px solid #1a1812;border-radius:12px;'
            f'padding:2rem;max-height:520px;overflow-y:auto;font-size:0.85rem;'
            f'line-height:1.9;color:#8a857a;white-space:pre-wrap">'
            f'{doc.get("content","")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if not st.session_state.published:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("◫  Publish to Notion", type="primary", use_container_width=True):
                    with st.spinner("Publishing..."):
                        try:
                            pr = httpx.post(
                                f"{API_URL}/publish",
                                json={"doc_id": doc.get("doc_id"), **doc},
                                timeout=30.0
                            )
                            pr.raise_for_status()
                            pd_data = pr.json()
                            st.session_state.published  = True
                            st.session_state.notion_url = pd_data.get("notion_url", "")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Publish failed: {e}")
            with c2:
                if st.button("◈  New Document", use_container_width=True):
                    _reset(); st.rerun()
        else:
            st.success("✓ Published to Notion successfully!")
            if st.session_state.notion_url:
                st.markdown(
                    f'<a href="{st.session_state.notion_url}" target="_blank" '
                    f'style="display:inline-flex;align-items:center;gap:8px;'
                    f'background:rgba(0,180,150,0.08);border:1px solid rgba(0,180,150,0.2);'
                    f'border-radius:8px;padding:8px 16px;color:#00b496;font-family:monospace;'
                    f'font-size:0.72rem;text-decoration:none">◫ &nbsp; View in Notion</a>',
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◈  Generate Another Document", type="primary", use_container_width=True):
                _reset(); st.rerun()