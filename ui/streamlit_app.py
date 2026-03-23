"""
DocForge AI x CiteRAG Lab -- streamlit_app.py  v10.0
Clean Claude-style UI. Native Streamlit only. All bugs fixed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import httpx

try:
    from docx_builder import build_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

API_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="DocForge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global font ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f111a;
    border-right: 1px solid #1e2433;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem; }

/* ── Main background ── */
.main .block-container { background: #0d0f18; padding-top: 2rem; margin-top: 0.75rem; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #131722;
    border: 1px solid #1e2843;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #1a1f35;
    border-color: #2a3a6e;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #131722 !important;
    border: 1px solid #2a3a6e !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* ── Tables in chat (comparison) ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0;
    font-size: 0.85rem;
}
th {
    background: #1e2843;
    color: #93c5fd;
    padding: 8px 12px;
    text-align: left;
    border: 1px solid #2a3a6e;
    font-weight: 600;
}
td {
    padding: 7px 12px;
    border: 1px solid #1e2433;
    color: #cbd5e1;
}
tr:nth-child(even) td { background: #131722; }
tr:nth-child(odd) td { background: #0f111a; }
tr:hover td { background: #1a2035; }

/* ── Dividers ── */
hr { border-color: #1e2433 !important; margin: 0.5rem 0 !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    transition: all 0.15s;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
    background: #131722 !important;
    border: 1px solid #1e2843 !important;
    color: #94a3b8 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #3b82f6 !important;
    color: #93c5fd !important;
}

/* ── Info/warning boxes ── */
[data-testid="stInfo"] {
    background: #0c1a2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #93c5fd !important;
}
[data-testid="stWarning"] {
    background: #1c1200 !important;
    border: 1px solid #3d2a00 !important;
    border-radius: 8px !important;
    color: #fcd34d !important;
}

/* ── Caption text ── */
.stCaption { color: #475569 !important; font-size: 0.75rem !important; }

/* ── Confidence badge ── */
.cite-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.cite-high   { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.cite-medium { background: #1c1a00; color: #facc15; border: 1px solid #854d0e; }
.cite-low    { background: #1c0a0a; color: #f87171; border: 1px solid #7f1d1d; }

/* ── Source citations ── */
.cite-sources {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #1e2433;
}
.cite-sources a { color: #3b82f6 !important; text-decoration: none; }
.cite-sources a:hover { text-decoration: underline; }

/* ── Comparison columns ── */
.compare-col {
    background: #0f111a;
    border: 1px solid #1e2843;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ── API helpers ────────────────────────────────────────────────────────────────

def api_get(ep):
    try:
        r = httpx.get(f"{API_URL}{ep}", timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def api_post(ep, data, timeout=120):
    try:
        r = httpx.post(f"{API_URL}{ep}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


# ── Session ────────────────────────────────────────────────────────────────────

def init_session():
    defaults = dict(
        step=1, company_ctx={}, departments=[],
        selected_dept=None, selected_dept_id=None,
        selected_doc_type=None, doc_sec_id=None, sections=[],
        section_questions={}, section_answers={},
        section_contents={}, sec_ids_ordered=[],
        gen_id=None, full_document="",
        active_tab="ask",
        rag_chats={}, rag_active_chat=None,
        docx_bytes_cache=None, docx_cache_doc=None,
        _library_data=None, _answer_drafts={},
        _last_chunks=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    import time as _t, uuid as _u

    st.markdown("## ⚡ DocForge AI")
    st.caption("Generate · Ask · Discover")
    st.divider()

    tab = st.radio(
        "Mode",
        ["💬 CiteRAG", "⚡ DocForge", "📚 Library"],
        label_visibility="collapsed",
        key="main_tab",
        horizontal=False,
    )
    if "DocForge" in tab:
        st.session_state.active_tab = "generate"
    elif "Library" in tab:
        st.session_state.active_tab = "library"
    else:
        st.session_state.active_tab = "ask"

    st.divider()

    # ── CiteRAG: Chat history ──────────────────────────────────────────────────
    if st.session_state.active_tab == "ask":

        if not st.session_state.rag_chats:
            _c0 = _u.uuid4().hex[:8]
            st.session_state.rag_chats[_c0] = {
                "title": "New chat", "messages": [], "created": _t.time()
            }
            st.session_state.rag_active_chat = _c0

        # New chat button
        if st.button("＋  New Chat", use_container_width=True,
                     key="sb_new_chat", type="primary"):
            _cn = _u.uuid4().hex[:8]
            st.session_state.rag_chats[_cn] = {
                "title": "New chat", "messages": [], "created": _t.time()
            }
            st.session_state.rag_active_chat = _cn
            st.rerun()

        st.markdown("###### Recent")

        _sorted = sorted(
            st.session_state.rag_chats.items(),
            key=lambda x: x[1].get("created", 0),
            reverse=True,
        )

        for _cid, _chat in _sorted:
            _active = _cid == st.session_state.rag_active_chat
            _title  = _chat["title"][:22] + ("…" if len(_chat["title"]) > 22 else "")
            _msgs   = len([m for m in _chat["messages"] if m["role"] == "user"])
            _label  = f"{'💬' if _msgs else '🆕'}  {_title}"

            # Rename mode — inline input
            if st.session_state.get(f"renaming_{_cid}"):
                _new = st.text_input(
                    "", value=_chat["title"],
                    key=f"rename_input_{_cid}",
                    label_visibility="collapsed",
                    placeholder="Enter new name…",
                )
                _r1, _r2 = st.columns(2)
                with _r1:
                    if st.button("✅ Save", key=f"save_ren_{_cid}",
                                 use_container_width=True, type="primary"):
                        if _new.strip():
                            st.session_state.rag_chats[_cid]["title"] = _new.strip()
                        del st.session_state[f"renaming_{_cid}"]
                        st.rerun()
                with _r2:
                    if st.button("✕ Cancel", key=f"cancel_ren_{_cid}",
                                 use_container_width=True):
                        del st.session_state[f"renaming_{_cid}"]
                        st.rerun()

            # Normal mode
            else:
                # Title button — full width
                if st.button(
                    _label,
                    key=f"chat_{_cid}",
                    use_container_width=True,
                    type="primary" if _active else "secondary",
                ):
                    st.session_state.rag_active_chat = _cid
                    st.rerun()

                # Rename + Delete row — only for active chat
                if _active:
                    _a, _b = st.columns(2)
                    with _a:
                        if st.button("✏️ Rename", key=f"ren_{_cid}",
                                     use_container_width=True):
                            st.session_state[f"renaming_{_cid}"] = True
                            st.rerun()
                    with _b:
                        if st.button("🗑 Delete", key=f"del_{_cid}",
                                     use_container_width=True, type="primary"):
                            del st.session_state.rag_chats[_cid]
                            for _k in list(st.session_state.keys()):
                                if _k.endswith(f"_{_cid}"):
                                    st.session_state.pop(_k, None)
                            if st.session_state.rag_active_chat == _cid:
                                st.session_state.rag_active_chat = (
                                    next(iter(st.session_state.rag_chats))
                                    if st.session_state.rag_chats else None
                                )
                            st.rerun()

    # ── DocForge: Step progress ────────────────────────────────────────────────
    if st.session_state.active_tab == "generate":
        st.markdown("###### Steps")
        steps = [
            (1, "🏢", "Setup"),
            (2, "❓", "Questions"),
            (3, "✍️", "Answers"),
            (4, "⚙️", "Generate"),
            (5, "💾", "Export"),
        ]
        cur = st.session_state.step
        for n, emoji, lbl in steps:
            if n < cur:
                st.markdown(f"✅  ~~Step {n} — {lbl}~~")
            elif n == cur:
                st.markdown(f"**{emoji}  Step {n} — {lbl}**")
            else:
                st.markdown(f"⬜  Step {n} — {lbl}")

        st.divider()
        ctx = st.session_state.company_ctx
        if ctx.get("company_name"):
            st.markdown(f"🏢 **{ctx['company_name']}**")
            st.caption(f"{ctx.get('industry','')} · {ctx.get('region','')}")
        if st.session_state.selected_doc_type:
            st.caption(f"📄 {st.session_state.selected_doc_type}")
        if st.session_state.sections:
            done_n  = len(st.session_state.section_contents)
            total_n = len(st.session_state.sections)
            st.progress(
                done_n / total_n if total_n else 0,
                text=f"{done_n} / {total_n} sections done",
            )
        st.divider()
        if st.button("↺  Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  CITRAG LAB
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.active_tab == "ask":
    import uuid as _uuid, time as _time_mod

    if not st.session_state.rag_chats:
        _c0 = _uuid.uuid4().hex[:8]
        st.session_state.rag_chats[_c0] = {
            "title": "New chat", "messages": [], "created": _time_mod.time()
        }
        st.session_state.rag_active_chat = _c0

    if (not st.session_state.rag_active_chat
            or st.session_state.rag_active_chat not in st.session_state.rag_chats):
        st.session_state.rag_active_chat = next(iter(st.session_state.rag_chats))

    active_id   = st.session_state.rag_active_chat
    active_chat = st.session_state.rag_chats[active_id]
    messages    = active_chat["messages"]

    if not messages:
        st.markdown("""
<div style="text-align:center;padding:3rem 0 1rem">
    <div style="font-size:3rem;margin-bottom:1rem">⚡</div>
    <h2 style="color:#e2e8f0;font-weight:700;margin-bottom:0.5rem">CiteRAG Lab</h2>
    <p style="color:#475569;font-size:0.95rem">Ask questions about your documents · Cite sources · Compare clauses · Analyse legal risk</p>
</div>
""", unsafe_allow_html=True)
        st.divider()
        examples = [
            "What is the notice period in the employment contract?",
            "Compare SOW vs NDA confidentiality clauses",
            "What are the leave policy details?",
            "Summarise the HR policies",
        ]
        c1, c2 = st.columns(2)
        for i, ex in enumerate(examples):
            with (c1 if i % 2 == 0 else c2):
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state._prefill_q = ex
                    st.rerun()
    else:
        st.caption(f"💬  {active_chat.get('title', 'New chat')}")

    for msg in messages:
        role       = msg["role"]
        text       = msg["content"]
        citations  = msg.get("citations", [])
        confidence = msg.get("confidence", "")

        with st.chat_message(role):
            if role == "assistant" and confidence:
                badge_cls = "cite-high" if confidence == "high" else "cite-medium" if confidence == "medium" else "cite-low"
                badge_label = {"high": "CiteRAG · HIGH", "medium": "CiteRAG · MEDIUM", "low": "CiteRAG · LOW"}.get(confidence, "CiteRAG")
                st.markdown(f'<span class="cite-badge {badge_cls}">{badge_label}</span>', unsafe_allow_html=True)

            if msg.get("tool_used") == "compare" and msg.get("side_a"):
                raw = msg.get("content", "")

                def _sect(text, start_tag, end_tags):
                    if start_tag not in text:
                        return ""
                    part = text.split(start_tag, 1)[1]
                    for tag in end_tags:
                        if tag in part:
                            part = part.split(tag, 1)[0]
                    return part.strip()

                doc_a = msg.get("doc_a", "Document A")
                doc_b = msg.get("doc_b", "Document B")
                doc_a_tag = f"DOCUMENT A -- {doc_a}"
                doc_b_tag = f"DOCUMENT B -- {doc_b}"

                final_answer  = _sect(raw, "FINAL ANSWER", [doc_a_tag, "DOCUMENT_A:"])
                if doc_a_tag in raw:
                    doc_a_content = _sect(raw, doc_a_tag, [doc_b_tag, "COMPARISON TABLE", "GAP IDENTIFIED:"])
                    doc_b_content = _sect(raw, doc_b_tag, ["COMPARISON TABLE", "GAP IDENTIFIED:", "KEY DIFFERENCE:", "COMPARISON INSIGHT:", "SUMMARY:"])
                    comp_table    = _sect(raw, "COMPARISON TABLE", ["GAP IDENTIFIED:", "KEY DIFFERENCE:", "SYSTEMIC ISSUE", "COMPARISON INSIGHT:"])
                else:
                    doc_a_content = msg.get("side_a", "")
                    doc_b_content = msg.get("side_b", "")
                    comp_table    = msg.get("comp_table", "")

                gap_block  = _sect(raw, "GAP IDENTIFIED:", ["KEY DIFFERENCE:", "SYSTEMIC ISSUE", "COMPARISON INSIGHT:", "SUMMARY:"])
                key_diff   = _sect(raw, "KEY DIFFERENCE:", ["SYSTEMIC ISSUE", "COMPARISON INSIGHT:", "SUMMARY:"])
                systemic   = _sect(raw, "SYSTEMIC ISSUE", ["COMPARISON INSIGHT:", "SUMMARY:"])
                insight    = _sect(raw, "COMPARISON INSIGHT:", ["SUMMARY:"])
                summary    = _sect(raw, "SUMMARY:", ["END_NEVER_MATCHES"])

                # FINAL ANSWER
                if final_answer:
                    st.markdown("📋 **FINAL ANSWER:** " + final_answer)
                    st.divider()

                # Doc A | Doc B side by side
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**🔵 {doc_a}**")
                    st.markdown(doc_a_content or msg.get("side_a", ""))
                with col_b:
                    st.markdown(f"**🟢 {doc_b}**")
                    st.markdown(doc_b_content or msg.get("side_b", ""))

                # Comparison table
                if comp_table:
                    st.divider()
                    st.markdown("**📊 COMPARISON TABLE**")
                    st.markdown(comp_table)

                # Full-width sections
                if gap_block:
                    st.divider()
                    st.markdown("**🔴 GAP IDENTIFIED:**")
                    st.markdown(gap_block)
                if key_diff:
                    st.markdown("**🔑 KEY DIFFERENCE:** " + key_diff)
                if systemic:
                    st.warning("⚠️ **SYSTEMIC ISSUE:** " + systemic)
                if insight:
                    st.divider()
                    st.markdown("**💡 COMPARISON INSIGHT:**")
                    st.markdown(insight)
                if summary:
                    st.divider()
                    st.markdown("**📝 SUMMARY:** " + summary)

            else:
                st.markdown(text)

            if citations and role == "assistant":
                seen, unique = set(), []
                for c in citations:
                    key = c if isinstance(c, str) else c.get("text", "")
                    if key not in seen:
                        seen.add(key)
                        unique.append(c)
                parts = []
                for c in unique:
                    if isinstance(c, dict) and c.get("url"):
                        parts.append(f'<a href="{c["url"]}" target="_blank">{c["text"]}</a>')
                    else:
                        txt = c if isinstance(c, str) else c.get("text", str(c))
                        parts.append(f'<span>{txt}</span>')
                src_html = " &nbsp;·&nbsp; ".join(parts)
                st.markdown(f'<div class="cite-sources">📎 {src_html}</div>', unsafe_allow_html=True)

    _prefill = st.session_state.pop("_prefill_q", "")
    user_q   = st.chat_input("Ask anything about your documents...")

    if user_q or _prefill:
        question = (user_q or _prefill).strip()
        if not messages:
            st.session_state.rag_chats[active_id]["title"] = (
                question[:40] + ("..." if len(question) > 40 else "")
            )
        st.session_state.rag_chats[active_id]["messages"].append(
            {"role": "user", "content": question}
        )
        with st.spinner("Thinking..."):
            res = api_post("/rag/ask", {
                "question":   question,
                "session_id": active_id,
                "top_k":      15,
            }, timeout=120)
        if res:
            ai_msg = {
                "role":       "assistant",
                "content":    res.get("answer", "No answer returned."),
                "citations":  res.get("citations", []),
                "confidence": res.get("confidence", ""),
                "tool_used":  res.get("tool_used", ""),
            }
            if res.get("tool_used") == "compare":
                ai_msg.update({
                    "side_a":      res.get("side_a", ""),
                    "side_b":      res.get("side_b", ""),
                    "comp_table":  res.get("comp_table", ""),
                    "summary":     res.get("summary", ""),
                    "doc_a":       res.get("doc_a", "Document A"),
                    "doc_b":       res.get("doc_b", "Document B"),
                })
            st.session_state._last_chunks = res.get("chunks", [])
        else:
            ai_msg = {
                "role":      "assistant",
                "content":   "Could not reach the RAG service. Make sure the backend is running.",
                "citations": [],
            }
            st.session_state._last_chunks = []
        st.session_state.rag_chats[active_id]["messages"].append(ai_msg)
        st.rerun()

    chunks = st.session_state.get("_last_chunks", [])
    if chunks:
        if st.toggle("🔍 Show Sources", value=False, key="show_retrieval"):
            seen_docs = {}
            for c in chunks:
                title   = c.get("doc_title", "")
                score   = c.get("score", 0)
                page_id = c.get("notion_page_id", "")
                if title and title not in seen_docs:
                    seen_docs[title] = {"score": score, "page_id": page_id}
            top5 = sorted(
                seen_docs.items(), key=lambda x: x[1]["score"], reverse=True
            )[:5]
            if top5:
                cols = st.columns(len(top5))
                for i, (doc, info) in enumerate(top5):
                    score   = info["score"]
                    page_id = info["page_id"]
                    url     = f"https://www.notion.so/{page_id}" if page_id else ""
                    conf    = "🟢" if score >= 0.6 else "🟡" if score >= 0.4 else "🔴"
                    with cols[i]:
                        if url:
                            st.markdown(f"{conf} [{doc}]({url})")
                        else:
                            st.markdown(f"{conf} {doc}")
                        st.caption(f"Rank {i+1} · {score:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "library":
    st.markdown("## 📚 Document Library")
    st.divider()

    if st.button("↺ Refresh"):
        st.session_state["_library_data"] = None

    if st.session_state.get("_library_data") is None:
        with st.spinner("Loading from Notion..."):
            lib = api_get("/library/notion")
        st.session_state["_library_data"] = lib or {}

    lib  = st.session_state.get("_library_data", {})
    docs = lib.get("documents", []) if isinstance(lib, dict) else []

    if not docs:
        st.info("No documents yet. Generate your first one from the DocForge tab!")
    else:
        f1, f2 = st.columns([1, 2])
        with f1:
            dept_filter = st.selectbox(
                "Department",
                ["All"] + sorted({d.get("department", "") for d in docs if d.get("department")}),
            )
        with f2:
            search = st.text_input("Search", placeholder="Search documents...")

        filtered = [
            d for d in docs
            if (dept_filter == "All" or d.get("department") == dept_filter)
            and (not search or search.lower() in d.get("title", "").lower())
        ]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(docs))
        c2.metric("Departments", len({d.get("department") for d in docs}))
        c3.metric("Showing", len(filtered))
        st.divider()

        for doc in filtered:
            with st.container(border=True):
                ca, cb, cc = st.columns([4, 2, 1])
                with ca:
                    st.markdown(f"**{doc.get('title', '--')}**")
                    st.caption(f"{doc.get('doc_type', '--')} · {doc.get('department', '--')}")
                with cb:
                    st.caption(doc.get("created_at", "--"))
                    status = doc.get("status", "--")
                    badge  = {"Generated": "🟢", "Draft": "🟡", "Reviewed": "🔵"}.get(status, "⚪")
                    st.caption(f"{badge} {status}")
                with cc:
                    if doc.get("notion_url"):
                        st.link_button("Open →", doc["notion_url"], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DOCFORGE GENERATE
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "generate":

    # ── Step 1: Setup ──────────────────────────────────────────────────────────
    if st.session_state.step == 1:
        st.markdown("## ⚡ DocForge AI")
        st.caption("Enter your company details and pick a document to generate.")
        st.divider()

        if not st.session_state.departments:
            with st.spinner("Loading catalog..."):
                data = api_get("/departments")
                if data:
                    st.session_state.departments = data["departments"]

        depts = st.session_state.departments
        if not depts:
            st.warning("Backend not reachable — run: uvicorn backend.main:app --reload")
            st.stop()

        with st.container(border=True):
            st.markdown("**🏢 Company Info**")
            c1, c2 = st.columns(2)
            with c1:
                company_name = st.text_input(
                    "Company Name",
                    value=st.session_state.company_ctx.get("company_name", ""),
                    placeholder="e.g. Turabit Technologies",
                )
                industry = st.selectbox("Industry", [
                    "Technology / SaaS", "Finance / Banking", "Healthcare",
                    "Manufacturing", "Retail / E-Commerce", "Legal Services",
                    "Marketing / Media", "Logistics / Supply Chain", "Education", "Other",
                ])
            with c2:
                company_size = st.selectbox("Company Size", [
                    "1-10 employees", "11-50 employees", "51-200 employees",
                    "201-500 employees", "500+ employees",
                ], index=2)
                region = st.selectbox("Region", [
                    "India", "United States", "United Kingdom", "UAE / Middle East",
                    "Canada", "Australia", "Europe", "Other",
                ])

        with st.container(border=True):
            st.markdown("**📂 Select Document**")
            c3, c4 = st.columns(2)
            with c3:
                selected_dept = st.selectbox("Department", [d["department"] for d in depts])
            dept_data = next((d for d in depts if d["department"] == selected_dept), None)
            with c4:
                selected_doc_type = st.selectbox(
                    "Document Type",
                    dept_data["doc_types"] if dept_data else [],
                )

        st.write("")
        if st.button("Continue →", type="primary", use_container_width=True):
            if not company_name.strip():
                st.error("Please enter your company name.")
            else:
                with st.spinner("Loading sections..."):
                    safe = (selected_doc_type
                            .replace("/", "%2F")
                            .replace("(", "%28")
                            .replace(")", "%29"))
                    data = api_get(f"/sections/{safe}")
                if data:
                    st.session_state.company_ctx = {
                        "company_name": company_name.strip(),
                        "industry":     industry,
                        "company_size": company_size,
                        "region":       region,
                    }
                    st.session_state.selected_dept     = selected_dept
                    st.session_state.selected_dept_id  = dept_data["doc_id"]
                    st.session_state.selected_doc_type = selected_doc_type
                    st.session_state.doc_sec_id        = data["doc_sec_id"]
                    seen, deduped = set(), []
                    for s in data["doc_sec"]:
                        if s not in seen:
                            seen.add(s)
                            deduped.append(s)
                    st.session_state.sections          = deduped
                    st.session_state.section_questions = {}
                    st.session_state.section_answers   = {}
                    st.session_state.section_contents  = {}
                    st.session_state.full_document     = ""
                    st.session_state.gen_id            = None
                    st.session_state.docx_bytes_cache  = None
                    st.session_state._answer_drafts    = {}
                    st.session_state.step              = 2
                    st.rerun()

    # ── Step 2: Generate Questions ─────────────────────────────────────────────
    elif st.session_state.step == 2:
        sections = st.session_state.sections
        total    = len(sections)
        done     = len(st.session_state.section_questions)

        st.markdown("## ❓ Generate Questions")
        st.caption(f"{st.session_state.selected_doc_type} · {total} sections")
        st.divider()

        grid_slot    = st.empty()
        counter_slot = st.empty()

        def render_grid():
            q    = st.session_state.section_questions
            cols = st.columns(3)
            for i, s in enumerate(sections):
                mark = "✅" if s in q else "⭕"
                cols[i % 3].markdown(f"{mark}  {s[:28]}")

        def render_counter():
            d = len(st.session_state.section_questions)
            counter_slot.progress(
                d / total if total else 0,
                text=f"{d} / {total} ready",
            )

        with grid_slot.container():
            render_grid()
        render_counter()

        st.write("")
        if done < total:
            if st.button("⚡ Generate Questions for All Sections",
                         type="primary", use_container_width=True):
                status = st.empty()
                for sec in sections:
                    if sec in st.session_state.section_questions:
                        continue
                    status.caption(f"Working on: {sec}...")
                    res = api_post("/questions/generate", {
                        "doc_sec_id":      st.session_state.doc_sec_id,
                        "doc_id":          st.session_state.selected_dept_id,
                        "section_name":    sec,
                        "doc_type":        st.session_state.selected_doc_type,
                        "department":      st.session_state.selected_dept,
                        "company_context": st.session_state.company_ctx,
                    })
                    if res:
                        st.session_state.section_questions[sec] = {
                            "sec_id":       res["sec_id"],
                            "questions":    res.get("questions", []),
                            "section_type": res.get("section_type", "text"),
                        }
                    with grid_slot.container():
                        render_grid()
                    render_counter()
                status.empty()
                st.rerun()

        if done == total:
            st.success(f"✅ All {total} sections ready!")
            if st.button("Start Answering →", type="primary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()

    # ── Step 3: Answer Questions ───────────────────────────────────────────────
    elif st.session_state.step == 3:
        sections   = st.session_state.sections
        ans_map    = st.session_state.section_answers
        q_map      = st.session_state.section_questions
        unanswered = [s for s in sections if s not in ans_map]

        if not unanswered:
            st.markdown("## ✅ All Sections Answered")
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Answered", len(ans_map))
            c2.metric("Total Sections", len(sections))
            st.write("")

            if st.button("⚡ Generate Document", type="primary", use_container_width=True):
                active    = sections
                total_sec = len(active)
                progress  = st.progress(0, text="Starting...")
                ids       = []

                for i, sec in enumerate(active):
                    if sec in st.session_state.section_contents:
                        ids.append(q_map.get(sec, {}).get("sec_id", 0))
                        progress.progress(
                            (i + 1) / total_sec,
                            text=f"{i + 1}/{total_sec} done",
                        )
                        continue
                    progress.progress(
                        i / total_sec,
                        text=f"Writing: {sec}",
                    )
                    q_data = q_map.get(sec, {})
                    res = api_post("/section/generate", {
                        "sec_id":          q_data.get("sec_id"),
                        "doc_sec_id":      st.session_state.doc_sec_id,
                        "doc_id":          st.session_state.selected_dept_id,
                        "section_name":    sec,
                        "doc_type":        st.session_state.selected_doc_type,
                        "department":      st.session_state.selected_dept,
                        "company_context": st.session_state.company_ctx,
                        "num_sections":    total_sec,
                    }, timeout=120)
                    if res:
                        st.session_state.section_contents[sec] = res["content"]
                        ids.append(q_data.get("sec_id"))
                    progress.progress(
                        (i + 1) / total_sec,
                        text=f"{i + 1}/{total_sec} done",
                    )

                st.session_state.sec_ids_ordered = ids
                doc_lines = []
                for sec in active:
                    c = st.session_state.section_contents.get(sec, "").strip()
                    if c:
                        doc_lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
                full_doc = "\n".join(doc_lines).strip()

                save_res = api_post("/document/save", {
                    "doc_id":          st.session_state.selected_dept_id,
                    "doc_sec_id":      st.session_state.doc_sec_id,
                    "sec_id":          ids[-1] if ids else 0,
                    "gen_doc_sec_dec": list(st.session_state.section_contents.values()),
                    "gen_doc_full":    full_doc,
                })
                st.session_state.gen_id           = save_res.get("gen_id", 0) if save_res else 0
                st.session_state.full_document    = full_doc
                st.session_state.docx_bytes_cache = None
                st.session_state.step             = 4
                st.rerun()

        else:
            current   = unanswered[0]
            done_cnt  = len(ans_map)
            total     = len(sections)
            q_data    = q_map.get(current, {})
            questions = q_data.get("questions", [])
            sec_id    = q_data.get("sec_id")
            sec_type  = q_data.get("section_type", "text")
            icon_map  = {
                "table": "📊", "flowchart": "🔀",
                "raci": "👥", "signature": "✍️", "text": "✏️",
            }
            icon = icon_map.get(sec_type, "✏️")

            st.markdown(f"## {icon}  {current}")
            st.progress(
                done_cnt / total if total else 0,
                text=f"{done_cnt} / {total} answered",
            )
            st.divider()

            if "_answer_drafts" not in st.session_state:
                st.session_state._answer_drafts = {}
            if current not in st.session_state._answer_drafts:
                st.session_state._answer_drafts[current] = [""] * len(questions)

            user_answers = []
            if not questions:
                st.info("No questions needed — this section will be auto-generated.")
            else:
                st.caption("Leave blank to auto-fill with professional content.")
                for i, q in enumerate(questions):
                    cur_val = (
                        st.session_state._answer_drafts[current][i]
                        if i < len(st.session_state._answer_drafts[current]) else ""
                    )
                    a = st.text_area(
                        f"Q{i+1}: {q}",
                        value=cur_val,
                        key=f"draft_{current}_{i}",
                        height=85,
                        placeholder="Your answer (or leave blank)...",
                    )
                    if i < len(st.session_state._answer_drafts[current]):
                        st.session_state._answer_drafts[current][i] = a
                    user_answers.append(a)

            st.write("")
            if st.button("Save & Next →", type="primary", use_container_width=True):
                filled = [a.strip() if a.strip() else "not answered" for a in user_answers]
                if sec_id:
                    api_post("/answers/save", {
                        "sec_id":       sec_id,
                        "doc_sec_id":   st.session_state.doc_sec_id,
                        "doc_id":       st.session_state.selected_dept_id,
                        "section_name": current,
                        "questions":    questions,
                        "answers":      filled or ["not answered"],
                    })
                st.session_state.section_answers[current] = filled
                st.session_state._answer_drafts.pop(current, None)
                st.rerun()

            if ans_map:
                st.divider()
                st.caption("Answered so far")
                for s in sections:
                    if s in ans_map:
                        st.markdown(f"✅ {s}")

    # ── Step 4: Review & Edit ──────────────────────────────────────────────────
    elif st.session_state.step == 4:
        active   = st.session_state.sections
        contents = st.session_state.section_contents
        icon_map = {
            "table": "📊", "flowchart": "🔀",
            "raci": "👥", "signature": "✍️", "text": "✏️",
        }

        st.markdown("## 🔍 Review & Edit")
        st.divider()

        def rebuild_doc():
            lines = []
            for sec in active:
                c = contents.get(sec, "").strip()
                if c:
                    lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
            st.session_state.full_document = "\n".join(lines).strip()

        left, right = st.columns([1, 2])

        with left:
            st.caption("SECTIONS")
            sel = st.radio("", active, label_visibility="collapsed", key="sec_radio")

        with right:
            cur      = contents.get(sel, "")
            sec_type = st.session_state.section_questions.get(sel, {}).get("section_type", "text")
            icon     = icon_map.get(sec_type, "✏️")

            st.markdown(f"**{icon} {sel}**")

            with st.expander("Current Content", expanded=True):
                if cur:
                    st.text(cur)
                else:
                    st.caption("(empty)")

            st.write("")
            instr = st.text_area(
                "AI Edit Instruction",
                placeholder="e.g. Make more formal · Add detail · Shorten",
                height=60,
                key="edit_instr",
                label_visibility="collapsed",
            )
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("🤖 Apply AI Edit", type="primary", use_container_width=True):
                    if not instr.strip():
                        st.warning("Enter an instruction.")
                    else:
                        with st.spinner("Editing..."):
                            res = api_post("/section/edit", {
                                "gen_id":           st.session_state.gen_id or 0,
                                "sec_id":           st.session_state.section_questions.get(sel, {}).get("sec_id", 0),
                                "section_name":     sel,
                                "doc_type":         st.session_state.selected_doc_type,
                                "current_content":  cur,
                                "edit_instruction": instr,
                            }, timeout=120)
                        if res:
                            st.session_state.section_contents[sel] = res["updated_content"]
                            st.session_state.docx_bytes_cache = None
                            rebuild_doc()
                            st.success("✅ Updated!")
                            st.rerun()
            with ec2:
                manual = st.text_area(
                    "Manual Edit",
                    value=cur,
                    height=180,
                    key=f"manual_{sel}",
                    label_visibility="collapsed",
                )
                if st.button("💾 Save Manual", use_container_width=True, key=f"save_{sel}"):
                    st.session_state.section_contents[sel] = manual
                    st.session_state.docx_bytes_cache = None
                    rebuild_doc()
                    st.success("✅ Saved!")
                    st.rerun()

        st.divider()
        if st.button("Export →", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

    # ── Step 5: Export ─────────────────────────────────────────────────────────
    elif st.session_state.step == 5:
        ctx      = st.session_state.company_ctx
        doc_type = st.session_state.selected_doc_type
        full_doc = st.session_state.full_document
        active   = st.session_state.sections
        contents = st.session_state.section_contents

        st.markdown("## 💾 Export")
        st.divider()

        if not full_doc:
            st.error("No document found — go back and generate first.")
            if st.button("← Back"):
                st.session_state.step = 3
                st.rerun()
            st.stop()

        st.success(f"✅ {doc_type} is ready!")

        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Document", doc_type)
            c2.metric("Department", st.session_state.selected_dept)
            c3.metric("Company", ctx.get("company_name", "--"))
            c4, c5, c6 = st.columns(3)
            c4.metric("Industry", ctx.get("industry", "--"))
            c5.metric("Sections", len(active))
            c6.metric("Words", f"~{len(full_doc.split())}")

        st.divider()
        col_n, col_d = st.columns(2)

        with col_n:
            with st.container(border=True):
                st.markdown("**📓 Publish to Notion**")
                st.caption("Send to your Notion workspace.")
                if st.button("🚀 Publish to Notion", type="primary", use_container_width=True):
                    with st.spinner("Publishing..."):
                        res = api_post("/document/publish", {
                            "gen_id":          st.session_state.gen_id or 0,
                            "doc_type":        doc_type,
                            "department":      st.session_state.selected_dept,
                            "gen_doc_full":    full_doc,
                            "company_context": ctx,
                        })
                    if res:
                        url = res.get("notion_url", "")
                        st.success("✅ Published!")
                        if url:
                            st.link_button("🔗 Open in Notion", url, use_container_width=True)

        with col_d:
            with st.container(border=True):
                st.markdown("**📥 Download**")
                safe = (doc_type
                        .replace(" ", "_")
                        .replace("/", "-")
                        .replace("(", "")
                        .replace(")", ""))

                if DOCX_AVAILABLE:
                    if (st.session_state.get("docx_bytes_cache") is None
                            or st.session_state.get("docx_cache_doc") != doc_type):
                        try:
                            sections_data = [
                                {"name": sec, "content": contents.get(sec, "")}
                                for sec in active if contents.get(sec)
                            ]
                            st.session_state.docx_bytes_cache = build_docx(
                                doc_type=doc_type,
                                department=st.session_state.selected_dept,
                                company_name=ctx.get("company_name", "Company"),
                                industry=ctx.get("industry", ""),
                                region=ctx.get("region", ""),
                                sections=sections_data,
                            )
                            st.session_state.docx_cache_doc = doc_type
                        except Exception as e:
                            st.error(f"DOCX error: {e}")
                            st.session_state.docx_bytes_cache = None

                    if st.session_state.get("docx_bytes_cache"):
                        st.download_button(
                            "⬇️ Download .docx",
                            data=st.session_state.docx_bytes_cache,
                            file_name=f"{safe}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                            type="primary",
                        )
                else:
                    st.warning("docx_builder.py not found.")

                st.download_button(
                    "⬇️ Download .txt",
                    data=full_doc,
                    file_name=f"{safe}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        st.divider()
        with st.expander("📄 Preview Full Document"):
            st.text(full_doc)

        st.write("")
        if st.button("➕ Create Another Document", type="primary", use_container_width=True):
            saved_ctx   = st.session_state.company_ctx
            saved_depts = st.session_state.departments
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_session()
            st.session_state["company_ctx"]  = saved_ctx
            st.session_state["departments"]  = saved_depts
            st.session_state["step"]         = 1
            st.rerun()