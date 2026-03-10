import streamlit as st
import httpx
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from prompts.templates import get_sections_for_doc_type

API_URL = "http://localhost:8000/api"

DEPARTMENTS = [
    "Human Resources (HR)", "Legal", "Finance / Accounting", "Sales",
    "Marketing", "Engineering / Development", "Product Management",
    "Operations", "Customer Support", "Compliance / Risk Management",
]

DOC_TYPES = [
    "Terms of Service", "Employment Contract", "Privacy Policy", "SOP",
    "SLA", "Product Requirement Document", "Technical Specification",
    "Incident Report", "Security Policy", "Customer Onboarding Guide",
    "Business Proposal", "NDA",
]

# Section count per doc type — shown in config preview
SECTION_COUNTS = {t: len(get_sections_for_doc_type(t)) for t in DOC_TYPES}


def init_state():
    for k, v in {
        "gen_step": "config", "gen_section": 0,
        "gen_answers": {}, "gen_config": {},
        "last_doc": None, "published": False,
        "notion_url": "", "cur_sections": [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _progress(current, total):
    pct = int((current / total) * 100) if total else 0
    filled = int((current / total) * 24) if total else 0
    bar = "█" * filled + "░" * (24 - filled)
    st.markdown(
        f'<div style="margin:1.2rem 0">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px">'
        f'<span style="font-family:monospace;font-size:0.62rem;color:#555;'
        f'letter-spacing:0.1em;text-transform:uppercase">Section {current} of {total}</span>'
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
    freq_html = (
        f'<span style="background:rgba(212,166,74,0.1);border:1px solid rgba(212,166,74,0.2);'
        f'border-radius:4px;padding:1px 7px;font-size:0.6rem;color:#d4a64a;'
        f'font-family:monospace;margin-left:8px">{freq}</span>'
    ) if freq else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:1.4rem">'
        f'<span style="font-size:1.4rem">{sec["icon"]}</span>'
        f'<span style="font-family:monospace;font-size:0.68rem;color:#d4a64a;'
        f'letter-spacing:0.14em;text-transform:uppercase">{sec["name"]}</span>'
        f'{freq_html}</div>',
        unsafe_allow_html=True
    )


def render_generator_form():
    init_state()

    # ══════════════════════════════════════════════════════════════
    #  STEP 1 — CONFIG
    # ══════════════════════════════════════════════════════════════
    if st.session_state.gen_step == "config":

        st.markdown(
            '<p style="font-family:monospace;font-size:0.62rem;color:#555;'
            'letter-spacing:0.14em;text-transform:uppercase;margin-bottom:1.4rem">'
            '◈ Document Configuration</p>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            department = st.selectbox("Department", DEPARTMENTS, key="cfg_dept")
        with col2:
            doc_type = st.selectbox("Document Type", DOC_TYPES, key="cfg_doc_type")

        # ── Live section preview ──────────────────────────────────
        preview = get_sections_for_doc_type(doc_type)
        n       = len(preview)

        # Auto-generated sections notice
        st.markdown(
            '<div style="background:#0d0c0a;border:1px solid #1a1812;border-radius:10px;'
            'padding:1rem 1.2rem;margin:1rem 0">'
            '<p style="font-family:monospace;font-size:0.58rem;color:#3a3830;'
            'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">'
            '◈ auto-generated (no questions needed)</p>'
            '<p style="font-size:0.78rem;color:#3a3830;line-height:1.7">'
            'Document Metadata &nbsp;·&nbsp; Version Control Table &nbsp;·&nbsp; Confidentiality Notice'
            '</p></div>',
            unsafe_allow_html=True
        )

        # User-answered sections preview
        names_html = " &nbsp;›&nbsp; ".join(
            f'<span style="color:#6a6560">{s["name"]}</span>' for s in preview
        )
        st.markdown(
            f'<div style="background:#0d0c0a;border:1px solid #1a1812;border-radius:10px;'
            f'padding:1rem 1.2rem;margin:0.5rem 0 1.2rem">'
            f'<p style="font-family:monospace;font-size:0.58rem;color:#d4a64a;'
            f'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem">'
            f'◉ {n} sections · you will answer {n * 2} questions</p>'
            f'<p style="font-size:0.76rem;line-height:1.9">{names_html}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        tags   = st.text_input("Tags (comma-separated)", placeholder="e.g. SaaS, GDPR, Q2-2026")
        author = st.text_input("Document Author", placeholder="e.g. Jane Smith")

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue to Questions →", type="primary", use_container_width=True):
            st.session_state.gen_config = {
                "industry":  "SaaS",
                "department": department,
                "doc_type":   doc_type,
                "tags":       [t.strip() for t in tags.split(",") if t.strip()],
                "author":     author or "Admin",
                "title":      f"{doc_type} — {department}",
            }
            st.session_state.cur_sections = preview
            st.session_state.gen_section  = 0
            st.session_state.gen_answers  = {}
            st.session_state.gen_step     = "questions"
            st.rerun()

    # ══════════════════════════════════════════════════════════════
    #  STEP 2 — GUIDED QUESTIONS
    # ══════════════════════════════════════════════════════════════
    elif st.session_state.gen_step == "questions":
        sections = st.session_state.cur_sections
        idx      = st.session_state.gen_section
        total    = len(sections)
        sec      = sections[idx]
        cfg      = st.session_state.gen_config

        _progress(idx + 1, total)
        _breadcrumb(sections)
        _section_header(sec)

        st.markdown(
            f'<p style="font-family:monospace;font-size:0.6rem;color:#2a2820;'
            f'letter-spacing:0.1em;margin-bottom:1.4rem">'
            f'{cfg["doc_type"]} &nbsp;·&nbsp; {cfg["department"]}</p>',
            unsafe_allow_html=True
        )

        # Render the 2 questions
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
            btn_label = "⚡  Generate Document" if is_last else f"Next: {sections[idx+1]['name']} →"
            btn_type  = "primary" if is_last else "secondary"

            if st.button(btn_label, type=btn_type, use_container_width=True):
                st.session_state.gen_answers.update(temp)
                if is_last:
                    st.session_state.gen_step = "generating"
                else:
                    st.session_state.gen_section += 1
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    #  STEP 3 — GENERATING
    # ══════════════════════════════════════════════════════════════
    elif st.session_state.gen_step == "generating":

        st.markdown(
            '<div style="text-align:center;padding:3rem 0">'
            '<p style="font-size:1.8rem;margin-bottom:0.8rem">⚡</p>'
            '<p style="font-family:monospace;font-size:0.68rem;color:#d4a64a;'
            'letter-spacing:0.16em;text-transform:uppercase">Generating Document</p>'
            '<p style="font-size:0.82rem;color:#444;margin-top:0.6rem">'
            'LLaMA 3.3 · 70B Versatile is writing your enterprise document…</p>'
            '</div>',
            unsafe_allow_html=True
        )

        cfg      = st.session_state.gen_config
        answers  = st.session_state.gen_answers
        sections = st.session_state.cur_sections

        # Collect all answer keys
        section_answers = {
            key: answers.get(key, "")
            for sec in sections
            for (key, _label, _ph) in sec.get("questions", [])
        }

        payload = {
            "title":            cfg["title"],
            "industry":         cfg["industry"],
            "department":       cfg["department"],
            "doc_type":         cfg["doc_type"],
            "tags":             cfg["tags"],
            "created_by":       cfg["author"],
            "description":      f"{cfg['doc_type']} for {cfg['department']} department",
            "section_answers":  section_answers,
        }

        with st.spinner("Writing your document..."):
            try:
                r = httpx.post(f"{API_URL}/generate", json=payload, timeout=120.0)
                r.raise_for_status()
                doc = r.json()
                st.session_state.last_doc  = doc
                st.session_state.published = False
                st.session_state.notion_url = ""
                st.session_state.gen_step  = "result"
                st.rerun()
            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
                st.session_state.gen_step = "questions"
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.session_state.gen_step = "questions"

    # ══════════════════════════════════════════════════════════════
    #  STEP 4 — RESULT
    # ══════════════════════════════════════════════════════════════
    elif st.session_state.gen_step == "result":
        doc = st.session_state.last_doc
        if not doc:
            st.session_state.gen_step = "config"
            st.rerun()

        cfg = st.session_state.gen_config
        wc  = len(doc.get("content", "").split())

        # Header
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

        st.markdown("<br>", unsafe_allow_html=True)

        # Document body
        st.markdown(
            f'<div style="background:#0d0c0a;border:1px solid #1a1812;border-radius:12px;'
            f'padding:2rem;max-height:520px;overflow-y:auto;font-size:0.85rem;'
            f'line-height:1.85;color:#8a857a;white-space:pre-wrap">'
            f'{doc.get("content","")}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        def _reset():
            for k in ["gen_step","gen_section","gen_answers","gen_config",
                      "last_doc","published","notion_url","cur_sections"]:
                st.session_state.pop(k, None)

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
                            pd = pr.json()
                            st.session_state.published  = True
                            st.session_state.notion_url = pd.get("notion_url", "")
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