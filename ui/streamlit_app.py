"""
DocForge AI x CiteRAG Lab -- streamlit_app.py  v11.0
Clean Claude-style UI. Native Streamlit only. RAGAS integrated into Show Sources toggle.
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

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/") + "/api"

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

/* ── RAGAS quality section ── */
.ragas-section {
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid #1e2433;
}
.ragas-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.ragas-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.ragas-badge {
    font-size: 0.65rem;
    font-weight: 700;
    background: rgba(29,158,117,0.15);
    color: #4ade80;
    padding: 2px 7px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    border: 1px solid rgba(29,158,117,0.25);
}
.rq-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 9px;
}
.rq-label {
    font-size: 0.75rem;
    color: #94a3b8;
    width: 138px;
    flex-shrink: 0;
}
.rq-hint {
    font-size: 0.68rem;
    color: #334155;
    width: 108px;
    flex-shrink: 0;
}
.rq-track {
    flex: 1;
    height: 5px;
    background: #1e2433;
    border-radius: 3px;
    overflow: hidden;
    min-width: 60px;
}
.rq-fill-g { height: 100%; border-radius: 3px; background: #22c55e; }
.rq-fill-a { height: 100%; border-radius: 3px; background: #f59e0b; }
.rq-fill-r { height: 100%; border-radius: 3px; background: #ef4444; }
.rq-score-g { font-size: 0.78rem; font-weight: 600; color: #4ade80; width: 34px; text-align: right; flex-shrink: 0; }
.rq-score-a { font-size: 0.78rem; font-weight: 600; color: #fbbf24; width: 34px; text-align: right; flex-shrink: 0; }
.rq-score-r { font-size: 0.78rem; font-weight: 600; color: #f87171; width: 34px; text-align: right; flex-shrink: 0; }
.rq-warn {
    margin-top: 10px;
    padding: 7px 11px;
    background: rgba(245,158,11,0.08);
    border-left: 2px solid #f59e0b;
    border-radius: 0 5px 5px 0;
    font-size: 0.72rem;
    color: #94a3b8;
    line-height: 1.5;
}
.rq-warn b { color: #fbbf24; }
.rq-ok {
    margin-top: 10px;
    padding: 7px 11px;
    background: rgba(34,197,94,0.06);
    border-left: 2px solid #22c55e;
    border-radius: 0 5px 5px 0;
    font-size: 0.72rem;
    color: #475569;
}
.source-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
    margin-bottom: 0;
}
.source-card {
    background: #0f111a;
    border: 1px solid #1e2843;
    border-radius: 9px;
    padding: 10px 11px;
}
.source-card-title {
    font-size: 0.75rem;
    font-weight: 500;
    color: #60a8f8;
    line-height: 1.35;
    margin-bottom: 5px;
}
.source-card-section {
    font-size: 0.65rem;
    color: #334155;
    margin-bottom: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.source-card-rank {
    font-size: 0.68rem;
    color: #475569;
}
.source-card-rank b { color: #94a3b8; }
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
        try:
            err_msg = e.response.json().get("detail", e.response.text[:200])
        except:
            err_msg = e.response.text[:200]
        st.session_state._last_api_error = err_msg
        return None
    except Exception as e:
        st.session_state._last_api_error = f"Connection error: {e}"
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
        active_tab="ask", main_tab="💬 CiteRAG",
        rag_chats={}, rag_active_chat=None,
        docx_bytes_cache=None, docx_cache_doc=None,
        _library_data=None, _answer_drafts={},
        _last_chunks=[],
        _ragas_history=[],         # [{question, scores, timestamp, tool_used}]
        _batch_progress=None,      # {done, total, current_q} — live batch progress
        # ── Agent / Tickets tab ────────────────────────────────────────────────
        agent_tickets=[],          # [{ticket_id, question, status, priority, url, created_at, summary, sources}]
        agent_tickets_loaded=False,
        _pending_ticket_idx=None,  # index of message awaiting ticket creation
        _ticket_created={},        # {msg_idx: ticket_url} — tracks created tickets per message
        agent_memory={},           # {user_name, industry, last_doc, last_intent}
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

    def _switch_tab():
        tab = st.session_state.main_tab
        if "DocForge" in tab:
            st.session_state.active_tab = "generate"
        elif "Library" in tab:
            st.session_state.active_tab = "library"
        elif "RAGAS" in tab:
            st.session_state.active_tab = "ragas"
        elif "Ticket" in tab:
            st.session_state.active_tab = "agent"
        else:
            st.session_state.active_tab = "ask"

    st.radio(
        "Mode",
        ["💬 CiteRAG", "⚡ DocForge", "📚 Library", "📊 RAGAS", "🎫 Tickets"],
        label_visibility="collapsed",
        key="main_tab",
        horizontal=False,
        on_change=_switch_tab,
    )

    st.divider()

    # ── CiteRAG: Chat history ──────────────────────────────────────────────────
    if st.session_state.active_tab == "ask":

        if not st.session_state.rag_chats:
            _c0 = _u.uuid4().hex[:8]
            st.session_state.rag_chats[_c0] = {
                "title": "New chat", "messages": [], "created": _t.time()
            }
            st.session_state.rag_active_chat = _c0

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
            else:
                if st.button(
                    _label,
                    key=f"chat_{_cid}",
                    use_container_width=True,
                    type="primary" if _active else "secondary",
                ):
                    st.session_state.rag_active_chat = _cid
                    st.rerun()

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
                            # Clear chunks + RAGAS so Show Sources toggle disappears
                            st.session_state._last_chunks       = []
                            st.session_state._last_ragas_scores = None
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

                if final_answer:
                    st.markdown("📋 **FINAL ANSWER:** " + final_answer)
                    st.divider()

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**🔵 {doc_a}**")
                    st.markdown(doc_a_content or msg.get("side_a", ""))
                with col_b:
                    st.markdown(f"**🟢 {doc_b}**")
                    st.markdown(doc_b_content or msg.get("side_b", ""))

                if comp_table:
                    st.divider()
                    st.markdown("**📊 COMPARISON TABLE**")
                    st.markdown(comp_table)

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

            # ── Agent ticket replies ───────────────────────────────────────────
            # tool_used=="agent": full reply is in content (already rendered above)
            # agent_reply set on RAG messages: show as info box below the answer
            if role == "assistant":
                agent_note = msg.get("agent_reply", "")
                if agent_note and msg.get("tool_used") != "agent":
                    # Ticket created / duplicate found / status updated
                    # Show as a distinct callout below the RAG answer
                    is_ticket_created = "✅" in agent_note and "Ticket" in agent_note
                    is_duplicate      = "🎫" in agent_note
                    box_bg  = "rgba(34,197,94,0.07)"  if is_ticket_created else "rgba(59,130,246,0.07)"
                    box_bdr = "#22c55e"                if is_ticket_created else "#3b82f6"
                    box_clr = "#4ade80"                if is_ticket_created else "#93c5fd"
                    st.markdown(
                        f'<div style="margin-top:10px;padding:10px 14px;' +
                        f'background:{box_bg};border-left:3px solid {box_bdr};' +
                        f'border-radius:0 8px 8px 0;font-size:0.83rem;color:{box_clr}">' +
                        agent_note.replace("\n", "<br>") +
                        '</div>',
                        unsafe_allow_html=True,
                    )
                elif msg.get("tool_used") == "agent":
                    st.markdown(
                        '<div style="margin-top:6px">'
                        '<span style="font-size:11px;background:rgba(99,102,241,0.15);color:#a5b4fc;'
                        'padding:2px 8px;border-radius:99px">🤖 Agent</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )



    _prefill = st.session_state.pop("_prefill_q", "")
    user_q   = st.chat_input("Ask anything...")


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
                "role":           "assistant",
                "content":        res.get("answer", "No answer returned."),
                "citations":      res.get("citations", []),
                "confidence":     res.get("confidence", ""),
                "tool_used":      res.get("tool_used", ""),
                "ticket_pending": res.get("ticket_pending", False),
                "orig_question":  question,
                "ticket_id":      res.get("ticket_id"),
                "ticket_url":     res.get("ticket_url"),
                "agent_reply":    res.get("agent_reply", ""),  # ticket create/update reply
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
            # Use _raw_chunks when chunks=[] (not-found) so Show Sources still appears
            st.session_state._last_chunks    = res.get("chunks") or res.get("_raw_chunks", [])
            st.session_state._last_not_found = (not res.get("chunks") and bool(res.get("_raw_chunks")))
        else:
            err = st.session_state.get("_last_api_error", "Could not reach the RAG service. Make sure the backend is running.")
            if "_last_api_error" in st.session_state:
                del st.session_state["_last_api_error"]

            ai_msg = {
                "role":      "assistant",
                "content":   f"⚠️ **Error:** {err}",
                "citations": [],
            }
            st.session_state._last_chunks    = []
            st.session_state._last_not_found = False
        st.session_state.rag_chats[active_id]["messages"].append(ai_msg)
        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # SHOW SOURCES — toggle section (works for found AND not-found answers)
    # ══════════════════════════════════════════════════════════════════════════

    chunks       = st.session_state.get("_last_chunks", [])
    not_found    = st.session_state.get("_last_not_found", False)
    toggle_label = "🔍 Show Sources" if not not_found else "🔍 Show Sources (searched but not found)"

    if chunks:
        if st.toggle(toggle_label, value=False, key="show_retrieval"):

            # ── Banner when answer was not found ─────────────────────────────
            if not_found:
                st.markdown(
                    '<div style="background:#1a1f2e;border:1px solid #ef444430;border-radius:8px;'
                    'padding:8px 12px;margin-bottom:10px;font-size:12px;color:#f87171">'
                    '⚠️ These were the closest documents found — none contained a confident answer.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            # ── Build top-5 unique docs ───────────────────────────────────────
            seen_docs = {}
            for c in chunks:
                title   = c.get("doc_title", "")
                score   = c.get("score", 0)
                page_id = c.get("notion_page_id", "")
                section = c.get("section", c.get("heading", ""))
                if title and title not in seen_docs:
                    seen_docs[title] = {
                        "score":   score,
                        "page_id": page_id,
                        "section": section,
                    }
            top5 = sorted(
                seen_docs.items(), key=lambda x: x[1]["score"], reverse=True
            )[:5]

            # ── Source cards (100% inline styles — no CSS class dependency) ──
            if top5:
                cards_html = (
                    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));'
                    'gap:8px;margin-bottom:12px">'
                )
                for i, (doc, info) in enumerate(top5):
                    score   = info["score"]
                    page_id = info["page_id"]
                    section = info.get("section", "")
                    url     = f"https://www.notion.so/{page_id}" if page_id else ""
                    dot_col = "#22c55e" if score >= 0.6 else "#f59e0b" if score >= 0.4 else "#ef4444"
                    title_html = (
                        f'<a href="{url}" target="_blank" '
                        f'style="color:#60a8f8;text-decoration:none;font-size:12px;'
                        f'font-weight:500;line-height:1.35">{doc}</a>'
                        if url else
                        f'<span style="color:#60a8f8;font-size:12px;font-weight:500">{doc}</span>'
                    )
                    cards_html += (
                        f'<div style="background:#0f111a;border:1px solid #1e2843;'
                        f'border-radius:9px;padding:10px 11px">'
                        f'<div style="display:flex;align-items:center;gap:5px;margin-bottom:5px">'
                        f'<span style="width:7px;height:7px;border-radius:50%;background:{dot_col};flex-shrink:0"></span>'
                        f'{title_html}'
                        f'</div>'
                        f'<div style="font-size:10px;color:#334155;margin-bottom:5px;white-space:nowrap;'
                        f'overflow:hidden;text-overflow:ellipsis">{section}</div>'
                        f'<div style="font-size:11px;color:#475569">Rank {i+1} · '
                        f'<b style="color:#94a3b8">{score:.3f}</b></div>'
                        f'</div>'
                    )
                cards_html += "</div>"
                st.markdown(cards_html, unsafe_allow_html=True)
            elif not_found:
                st.info("No documents were retrieved — this topic may not exist in any ingested document yet.")

# ══════════════════════════════════════════════════════════════════════════════
#  LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "library":
    ch, cb = st.columns([4,1])
    with ch: st.markdown("### 📚 Document Library")
    with cb:
        if st.button("↺ Refresh", use_container_width=True):
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
            sc = {"Generated":"#ff6b00","Draft":"#f59e0b","Reviewed":"#60a5fa","Archived":"#3a3a5a"}.get(doc.get("status",""),"#3a3a5a")
            a,b,c = st.columns([4,2,1])
            with a:
                st.markdown(f'<div class="lib-card"><div class="lib-title">{doc.get("title","—")}</div>'
                            f'<div class="lib-meta">{doc.get("doc_type","—")} · {doc.get("industry","—")}</div></div>',
                            unsafe_allow_html=True)
            with b:
                st.markdown(f'<div class="lib-card"><div class="lib-meta">🏢 {doc.get("department","—")}</div>'
                            f'<div class="lib-meta">📅 {doc.get("created_at","—")}</div>'
                            f'<div class="lib-meta" style="color:{sc}">● {doc.get("status","—")}</div></div>',
                            unsafe_allow_html=True)
            with c:
                if doc.get("notion_url"):
                    st.link_button("Open →", doc["notion_url"], use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
#  RAGAS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "ragas":

    st.markdown("""
<div style="padding:0 0 1rem">
    <h2 style="color:#e2e8f0;font-weight:700;margin-bottom:0.25rem">📊 RAGAS Evaluation</h2>
    <p style="color:#475569;font-size:0.9rem">Real answer quality scores — faithfulness, relevancy, precision, recall</p>
</div>
""", unsafe_allow_html=True)
    st.divider()

    def _ragas_bar_color(v):
        if v is None:
            return "#334155", "#475569"
        if v >= 0.85:
            return "#22c55e", "#4ade80"
        if v >= 0.70:
            return "#f59e0b", "#fbbf24"
        return "#ef4444", "#f87171"

    def _render_ragas_scores(scores, title="", timestamp=""):
        """Render a full RAGAS score panel."""
        if not scores:
            return

        metrics = [
            ("Faithfulness",      scores.get("faithfulness"),      "no hallucination",
             "Answer is grounded solely in the retrieved documents."),
            ("Answer Relevancy",  scores.get("answer_relevancy"),   "on-topic answer",
             "Answer directly addresses the question asked."),
            ("Context Precision", scores.get("context_precision"),  "clean retrieval",
             "Retrieved chunks are relevant — no noise from unrelated documents."),
            ("Context Recall",    scores.get("context_recall"),     "full coverage",
             "All relevant information from ground-truth was retrieved."),
        ]

        avg_vals   = [m[1] for m in metrics if m[1] is not None]
        avg_score  = round(sum(avg_vals) / len(avg_vals), 2) if avg_vals else None
        avg_col    = "#4ade80" if (avg_score or 0) >= 0.85 else "#fbbf24" if (avg_score or 0) >= 0.70 else "#f87171"

        header_html = (
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'margin-bottom:16px">'
            f'<div style="font-size:13px;font-weight:600;color:#cbd5e1">'
            f'{title or "Scores"}</div>'
            f'<div style="display:flex;align-items:center;gap:10px">'
        )
        if timestamp:
            header_html += f'<span style="font-size:11px;color:#475569">{timestamp}</span>'
        if avg_score is not None:
            header_html += (
                f'<span style="font-size:12px;font-weight:700;color:{avg_col};'
                f'background:rgba(255,255,255,0.04);padding:3px 10px;border-radius:5px;'
                f'border:1px solid {avg_col}33">avg {avg_score:.2f}</span>'
            )
        header_html += '</div></div>'

        rows_html  = ""
        warn_lines = []
        has_null   = False

        for label, val, hint, explanation in metrics:
            if val is None:
                has_null = True
                rows_html += (
                    f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;opacity:0.45">'
                    f'<span style="font-size:12px;color:#94a3b8;width:150px;flex-shrink:0">{label}</span>'
                    f'<span style="font-size:11px;color:#334155;width:110px;flex-shrink:0">{hint}</span>'
                    f'<div style="flex:1;height:5px;background:#1e2433;border-radius:3px;min-width:80px">'
                    f'<div style="width:0%;height:100%;background:#334155;border-radius:3px"></div></div>'
                    f'<span style="font-size:11px;color:#334155;width:38px;text-align:right;flex-shrink:0">n/a</span>'
                    f'</div>'
                )
                continue

            pct                = int(val * 100)
            bar_col, txt_col   = _ragas_bar_color(val)
            score_str          = f"{val:.2f}"

            if val < 0.70:
                if "faith" in label.lower():
                    warn_lines.append("&#9888; <b>Faithfulness low</b> — answer may contain claims not grounded in documents.")
                elif "precision" in label.lower():
                    warn_lines.append("&#9888; <b>Context precision low</b> — retriever fetched irrelevant chunks.")
                elif "recall" in label.lower():
                    warn_lines.append("&#9888; <b>Context recall low</b> — relevant chunks may have been missed.")
                elif "relev" in label.lower():
                    warn_lines.append("&#9888; <b>Answer relevancy low</b> — answer drifted off-topic.")

            rows_html += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">'
                f'<span style="font-size:12px;color:#94a3b8;width:150px;flex-shrink:0">{label}</span>'
                f'<span style="font-size:11px;color:#334155;width:110px;flex-shrink:0">{hint}</span>'
                f'<div style="flex:1;height:5px;background:#1e2433;border-radius:3px;overflow:hidden;min-width:80px">'
                f'<div style="width:{pct}%;height:100%;background:{bar_col};border-radius:3px"></div></div>'
                f'<span style="font-size:12px;font-weight:600;color:{txt_col};width:38px;text-align:right;flex-shrink:0">{score_str}</span>'
                f'</div>'
            )

        feedback_html = ""
        if warn_lines:
            feedback_html = (
                f'<div style="margin-top:10px;padding:8px 12px;background:rgba(245,158,11,0.08);'
                f'border-left:2px solid #f59e0b;border-radius:0 5px 5px 0;font-size:12px;color:#94a3b8;line-height:1.7">'
                + "<br>".join(warn_lines) + '</div>'
            )
        elif avg_vals:
            feedback_html = (
                '<div style="margin-top:10px;padding:8px 12px;background:rgba(34,197,94,0.06);'
                'border-left:2px solid #22c55e;border-radius:0 5px 5px 0;font-size:12px;color:#475569">'
                '✓ All quality metrics look good.</div>'
            )

        if has_null:
            feedback_html += (
                '<div style="margin-top:8px;font-size:11px;color:#334155">'
                '* Context recall shown as n/a — no matching ground truth in qa_dataset.json for this question.</div>'
            )

        st.markdown(
            f'<div style="background:#131722;border:1px solid #1e2843;border-radius:12px;padding:18px 20px;margin-bottom:12px">'
            + header_html + rows_html + feedback_html +
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Section 1: Latest scores (from most recent CiteRAG answer) ─────────────
    last_scores = st.session_state.get("_last_ragas_scores")

    if last_scores:
        st.markdown("#### 🔬 Latest Answer Quality")
        _render_ragas_scores(last_scores, title="Most recent question")



    # ── Section 3: Batch Evaluation ───────────────────────────────────────────
    st.divider()
    st.markdown("#### 🗂 Batch Evaluation")
    st.caption("Run RAGAS on multiple questions at once — add rows manually or import a JSON file.")

    # ── Init batch state ──────────────────────────────────────────────────────
    if "batch_rows" not in st.session_state:
        st.session_state.batch_rows = [{"question": "", "ground_truth": ""}]
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = []
    if "batch_running" not in st.session_state:
        st.session_state.batch_running = False

    with st.container(border=True):

        # ── JSON import ────────────────────────────────────────────────────────
        with st.expander("📥 Import from JSON", expanded=False):
            st.caption(
                'Expected format: `[{"question": "...", "ground_truth": "..."}, ...]`  '
                '— `ground_truth` is optional in each item.'
            )

            def _handle_json_upload():
                uploaded = st.session_state.get("batch_json_upload")
                if uploaded:
                    import json as _json
                    try:
                        raw_data = _json.loads(uploaded.read().decode("utf-8"))
                        if not isinstance(raw_data, list):
                            st.session_state["_batch_err"] = "JSON must be a list of objects."
                        else:
                            parsed_rows = []
                            for item in raw_data:
                                if isinstance(item, dict) and item.get("question", "").strip():
                                    parsed_rows.append({
                                        "question":     item.get("question", "").strip(),
                                        "ground_truth": item.get("ground_truth", "").strip(),
                                    })
                            if parsed_rows:
                                st.session_state.batch_rows = parsed_rows
                                st.session_state.batch_results = []
                                st.session_state["_batch_succ"] = f"Loaded {len(parsed_rows)} questions from JSON."
                            else:
                                st.session_state["_batch_err"] = "No valid questions found in JSON."
                    except Exception as _je:
                        st.session_state["_batch_err"] = f"JSON parse error: {_je}"

            st.file_uploader(
                "Upload JSON file",
                type=["json"],
                key="batch_json_upload",
                label_visibility="collapsed",
                on_change=_handle_json_upload,
            )
            
            if "_batch_succ" in st.session_state:
                st.success(st.session_state.pop("_batch_succ"))
            if "_batch_err" in st.session_state:
                st.error(st.session_state.pop("_batch_err"))

        # ── Manual rows ────────────────────────────────────────────────────────
        st.markdown("**Questions**")
        rows_to_delete = []
        for _ri, _row in enumerate(st.session_state.batch_rows):
            _rc1, _rc2, _rc3 = st.columns([3, 3, 0.5])
            with _rc1:
                _q_val = st.text_input(
                    f"Question {_ri + 1}",
                    value=_row["question"],
                    placeholder="e.g. What is the leave policy?",
                    key=f"batch_q_{_ri}",
                    label_visibility="collapsed",
                )
                st.session_state.batch_rows[_ri]["question"] = _q_val
            with _rc2:
                _gt_val = st.text_input(
                    f"Ground Truth {_ri + 1}",
                    value=_row["ground_truth"],
                    placeholder="Ground truth",
                    key=f"batch_gt_{_ri}",
                    label_visibility="collapsed",
                )
                st.session_state.batch_rows[_ri]["ground_truth"] = _gt_val
            with _rc3:
                if len(st.session_state.batch_rows) > 1:
                    if st.button("✕", key=f"batch_del_{_ri}", help="Remove row"):
                        rows_to_delete.append(_ri)

        if rows_to_delete:
            for _idx in sorted(rows_to_delete, reverse=True):
                st.session_state.batch_rows.pop(_idx)
            st.session_state.batch_results = []
            st.rerun()

        _ba1, _ba2 = st.columns([1, 3])
        with _ba1:
            if st.button("＋ Add Row", key="batch_add_row"):
                st.session_state.batch_rows.append({"question": "", "ground_truth": ""})
                st.rerun()
        with _ba2:
            st.caption(f"{len(st.session_state.batch_rows)} question(s) queued · Each takes 20–60s · No timeout limit")

        st.write("")
        _valid_rows = [r for r in st.session_state.batch_rows if r["question"].strip() and r['ground_truth'].strip()]

        # ── Live progress display (survives st.rerun) ─────────────────────────
        _bp = st.session_state.get("_batch_progress")
        if _bp and _bp.get("running"):
            _done  = _bp["done"]
            _total = _bp["total"]
            _frac  = _done / _total if _total else 0
            st.progress(_frac, text=f"⏳ Running {_done}/{_total}: {_bp.get('current_q', '')[:55]}…")
            st.caption(f"Question {_done} of {_total} complete — waiting for next result…")

        if st.button(
            f"▶ Run Batch ({len(_valid_rows)} questions)",
            type="primary",
            key="batch_run_btn",
            use_container_width=True,
            disabled=len(_valid_rows) == 0 or st.session_state.get("batch_running", False),
        ):
            if _valid_rows:
                import time as _bt
                st.session_state.batch_results = []
                st.session_state.batch_running = True
                _total = len(_valid_rows)
                st.session_state._batch_progress = {
                    "running": True, "done": 0, "total": _total, "current_q": ""
                }

                for _bi, _brow in enumerate(_valid_rows):
                    _bq  = _brow["question"].strip()
                    _bgt = _brow["ground_truth"].strip()

                    # Update progress state so the re-render above shows current status
                    st.session_state._batch_progress["current_q"] = _bq
                    st.session_state._batch_progress["done"]       = _bi

                    _bts  = _bt.strftime("%H:%M:%S")
                    _bres = api_post("/rag/eval", {
                        "question":     _bq,
                        "ground_truth": _bgt,
                        "top_k":        15,
                    }, timeout=600)

                    _bresult = {
                        "question":     _bq,
                        "ground_truth": _bgt,
                        "timestamp":    _bts,
                        "scores":       None,
                        "answer":       "",
                        "error":        None,
                        "tool_used":    "",
                    }
                    if _bres:
                        _bresult["scores"]    = _bres.get("ragas_scores")
                        _bresult["answer"]    = _bres.get("answer", "")
                        _bresult["error"]     = _bres.get("ragas_error")
                        _bresult["tool_used"] = _bres.get("tool_used", "")
                        if _bresult["scores"]:
                            st.session_state._ragas_history.append({
                                "question":  _bq,
                                "scores":    _bresult["scores"],
                                "tool_used": _bresult["tool_used"],
                                "timestamp": _bts,
                            })
                            st.session_state._ragas_history = st.session_state._ragas_history[-20:]
                    else:
                        _bresult["error"] = "API call failed — backend unreachable."

                    st.session_state.batch_results.append(_bresult)
                    st.session_state._batch_progress["done"] = _bi + 1

                st.session_state._batch_progress = {"running": False, "done": _total, "total": _total, "current_q": ""}
                st.session_state.batch_running = False
                st.rerun()

    # ── Batch results display ─────────────────────────────────────────────────
    if st.session_state.batch_results:
        _br = st.session_state.batch_results
        st.divider()
        st.markdown(f"**📊 Batch Results** — {len(_br)} questions")

        # Aggregate summary row
        _bscored = [r for r in _br if r["scores"]]
        if _bscored:
            def _bavg(key):
                vals = [r["scores"].get(key) for r in _bscored if r["scores"].get(key) is not None]
                return round(sum(vals) / len(vals), 3) if vals else None

            _bfa = _bavg("faithfulness")
            _bra = _bavg("answer_relevancy")
            _bpa = _bavg("context_precision")
            _bca = _bavg("context_recall")

            with st.container(border=True):
                st.caption(f"BATCH AVERAGES · {len(_bscored)}/{len(_br)} scored")
                _bc1, _bc2, _bc3, _bc4 = st.columns(4)
                def _bm(col, label, val):
                    if val is not None:
                        col.metric(label, f"{val:.3f}")
                    else:
                        col.metric(label, "n/a")
                _bm(_bc1, "Faithfulness",      _bfa)
                _bm(_bc2, "Ans. Relevancy",     _bra)
                _bm(_bc3, "Context Precision",  _bpa)
                _bm(_bc4, "Context Recall",     _bca)

        # Per-question expandable cards
        for _bri, _bentry in enumerate(_br):
            _blabel = f"Q{_bri+1}: {_bentry['question'][:65]}{'…' if len(_bentry['question']) > 65 else ''}"
            _bstatus = "✅" if _bentry["scores"] else ("❌" if _bentry["error"] else "⚠️")
            with st.expander(f"{_bstatus} {_blabel} · {_bentry['timestamp']}"):
                if _bentry["answer"]:
                    st.markdown(f"**📄 RAG Answer:**")
                    st.markdown(_bentry["answer"])
                if _bentry["scores"]:
                    _render_ragas_scores(_bentry["scores"], title=_bentry["question"])
                elif _bentry["error"]:
                    st.error(f"RAGAS error: {_bentry['error']}")
                else:
                    st.warning("No scores returned — check backend logs.")

        # Export batch results as JSON
        import json as _ejson, time as _et
        _export_data = [
            {
                "question":     r["question"],
                "ground_truth": r["ground_truth"],
                "timestamp":    r["timestamp"],
                "tool_used":    r["tool_used"],
                "answer":       r["answer"],
                "scores":       r["scores"],
                "error":        r["error"],
            }
            for r in _br
        ]
        st.download_button(
            "⬇️ Export Results as JSON",
            data=_ejson.dumps(_export_data, indent=2),
            file_name=f"ragas_batch_{_et.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="batch_export_btn",
        )

    # ── Section 4: Session history ────────────────────────────────────────────
    
    history = st.session_state.get("_ragas_history", [])
    if history:
        st.divider()
        st.markdown(f"#### 📈 Session History  <span style=\'font-size:12px;color:#475569;font-weight:400\'>{len(history)} evaluations</span>", unsafe_allow_html=True)

        # Aggregate trend
        all_faith  = [h["scores"].get("faithfulness")      for h in history if h["scores"].get("faithfulness")      is not None]
        all_relev  = [h["scores"].get("answer_relevancy")  for h in history if h["scores"].get("answer_relevancy")  is not None]
        all_prec   = [h["scores"].get("context_precision") for h in history if h["scores"].get("context_precision") is not None]
        all_rec    = [h["scores"].get("context_recall")    for h in history if h["scores"].get("context_recall")    is not None]

        def _avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else None

        fa, ra, pa, ca = _avg(all_faith), _avg(all_relev), _avg(all_prec), _avg(all_rec)

        c1, c2, c3, c4 = st.columns(4)
        def _metric_col(col, label, val):
            if val is not None:
                col.metric(label, f"{val:.2f}", delta=None)
            else:
                col.metric(label, "n/a")

        _metric_col(c1, "Avg Faithfulness",      fa)
        _metric_col(c2, "Avg Ans. Relevancy",     ra)
        _metric_col(c3, "Avg Context Precision",  pa)
        _metric_col(c4, "Avg Context Recall",     ca)

        st.write("")

        for entry in reversed(history):
            q_label  = entry["question"][:70] + ("…" if len(entry["question"]) > 70 else "")
            tool_tag = f" · `{entry['tool_used']}`" if entry.get("tool_used") else ""
            ts_tag   = f" · {entry['timestamp']}" if entry.get("timestamp") else ""
            with st.expander(f"**{q_label}**{tool_tag}{ts_tag}"):
                _render_ragas_scores(entry["scores"], title=entry["question"])

        if st.button("🗑 Clear History", key="ragas_clear_hist"):
            st.session_state._ragas_history = []
            st.rerun()

    # ── Section 4: Metric explanations ────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
| Metric | What it measures | Good threshold |
|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved documents? High = no hallucination. | ≥ 0.85 |
| **Answer Relevancy** | Does the answer actually address the question? Low = answer went off-topic. | ≥ 0.80 |
| **Context Precision** | Are the retrieved chunks relevant? Low = retriever is pulling in noise. | ≥ 0.75 |
| **Context Recall** | Did retrieval cover all the important facts? Only scored when a **Ground Truth** is provided in the batch or eval form. | ≥ 0.75 |
""")

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

# ══════════════════════════════════════════════════════════════════════════════
#  🎫 TICKETS — Knowledge-gap ticket viewer + status manager
#  Tickets are auto-created when CiteRAG retrieval confidence is low.
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "agent":
    import json as _aj

    def _ag_get(ep, timeout=30):
        try:
            r = httpx.get(f"{API_URL}{ep}", timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def _ag_post(ep, payload, timeout=60):
        try:
            r = httpx.post(f"{API_URL}{ep}", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Page header ────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding:0 0 0.5rem">
  <h2 style="color:#e2e8f0;font-weight:700;margin-bottom:0.2rem">🎫 Tickets</h2>
  <p style="color:#475569;font-size:0.88rem;margin:0">
    Knowledge-gap tickets — say <b style="color:#93c5fd">"create a ticket"</b>,
    <b style="color:#93c5fd">"raise a ticket"</b> or <b style="color:#93c5fd">"open a case"</b>
    in <b style="color:#93c5fd">💬 CiteRAG</b> to log a missing answer
  </p>
</div>
""", unsafe_allow_html=True)
    st.divider()

    # ── Two-column layout ──────────────────────────────────────────────────────
    col_mem, col_tix = st.columns([1, 2], gap="large")

    # ════════════════════════════════════════════════════════════
    #  LEFT — 🧠 Memory panel
    # ════════════════════════════════════════════════════════════
    with col_mem:
        st.markdown("""
<div style="font-size:11px;font-weight:600;color:#475569;letter-spacing:0.08em;
            text-transform:uppercase;margin-bottom:10px">🧠 Session Memory</div>
""", unsafe_allow_html=True)

        mem = st.session_state.agent_memory

        # Live memory chips
        if mem:
            chips_html = ""
            _chip = lambda ico, val: (
                f'<span style="display:inline-flex;align-items:center;gap:4px;'
                f'background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.2);'
                f'border-radius:20px;padding:3px 10px;font-size:11px;color:#93c5fd;'
                f'margin:2px 3px">{ico} {val}</span>'
            )
            if mem.get("user_name"):
                chips_html += _chip("👤", mem["user_name"])
            if mem.get("industry"):
                chips_html += _chip("🏭", mem["industry"])
            if mem.get("last_intent"):
                chips_html += _chip("🎯", mem["last_intent"])
            if mem.get("last_doc"):
                chips_html += _chip("📄", mem["last_doc"][:22])
            st.markdown(f'<div style="margin-bottom:14px">{chips_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("No memory yet — ask a question in 💬 CiteRAG to populate this.")

        st.divider()

        # Context hints (pre-fill memory)
        st.markdown("""
<div style="font-size:11px;font-weight:600;color:#475569;letter-spacing:0.08em;
            text-transform:uppercase;margin-bottom:8px">⚙️ Context Hints</div>
""", unsafe_allow_html=True)
        st.caption("Pre-fill memory so the agent knows who you are before the first question.")

        _hn = st.text_input("Your name", value=mem.get("user_name", ""),
                            placeholder="e.g. Rahul", key="ag_hint_name")
        _hi = st.text_input("Industry / department", value=mem.get("industry", ""),
                            placeholder="e.g. Technology / HR", key="ag_hint_industry")

        if st.button("💾 Save Hints", key="ag_save_hints", use_container_width=True):
            st.session_state.agent_memory["user_name"] = _hn.strip()
            st.session_state.agent_memory["industry"]  = _hi.strip()
            
            # ── SYNC TO BACKEND ──────────────────
            st.session_state._last_api_error = None  # Clear previous errors
            res_sync = api_post("/agent/memory", {
                "session_id": st.session_state.rag_active_chat,
                "memory": {
                    "user_name": _hn.strip(),
                    "industry":  _hi.strip()
                }
            })
            if res_sync:
                st.success("Hints saved & synced to backend!")
            else:
                st.error(f"Sync failed: {st.session_state.get('_last_api_error', 'Unknown error')}")
            st.rerun()

        st.divider()

        # Sync memory from backend (LangGraph checkpointer)
        if st.button("↺ Sync from Backend", key="ag_sync_mem", use_container_width=True):
            st.session_state._last_api_error = None  # Clear previous errors
            _mres = api_get(f"/agent/memory?session_id={st.session_state.rag_active_chat}")
            if _mres and "error" not in _mres:
                st.session_state.agent_memory.update(_mres.get("memory", {}))
                st.success("Memory synced!")
                st.rerun()
            else:
                st.warning(
                    f"⚠️ `/api/agent/memory` not yet implemented. "
                    "Memory will populate automatically once `agent_routes.py` is deployed."
                )

        # Quick stats
        st.divider()
        _open_n = len([t for t in st.session_state.agent_tickets
                       if t.get("status") == "Open"])
        _tot_n  = len(st.session_state.agent_tickets)
        _c1, _c2 = st.columns(2)
        _c1.metric("Total Tickets", _tot_n)
        _c2.metric("Open", _open_n)

    # ════════════════════════════════════════════════════════════
    #  RIGHT — 🎫 My Tickets
    # ════════════════════════════════════════════════════════════
    with col_tix:
        # Header row
        _th1, _th2, _th3 = st.columns([2, 1, 1])
        with _th1:
            st.markdown("""
<div style="font-size:11px;font-weight:600;color:#475569;letter-spacing:0.08em;
            text-transform:uppercase;margin-bottom:10px">🎫 Knowledge-Gap Tickets</div>
""", unsafe_allow_html=True)
        with _th2:
            _tix_filter = st.selectbox(
                "Status filter",
                ["All", "Open", "In Progress", "Resolved"],
                key="ag_tix_filter",
                label_visibility="collapsed",
            )
        with _th3:
            if st.button("↺ Refresh", key="ag_refresh_tix", use_container_width=True):
                st.session_state.agent_tickets_loaded = False

        # Load tickets from backend (Notion)
        if not st.session_state.agent_tickets_loaded:
            with st.spinner("Loading tickets from Notion…"):
                _td = _ag_get("/agent/tickets")
            if _td and "error" not in _td:
                st.session_state.agent_tickets = _td.get("tickets", [])
                st.session_state.agent_tickets_loaded = True
            else:
                _err_msg = _td.get("error", "Unknown") if _td else "Backend unreachable"
                st.info(
                    f"ℹ️ Tickets endpoint not yet active: `{_err_msg}`\n\n"
                    "Tickets will appear here automatically once `agent_routes.py` is deployed "
                    "and CiteRAG starts creating them on low-confidence answers."
                )

        _all_tix = st.session_state.agent_tickets
        _show_tix = (
            _all_tix if _tix_filter == "All"
            else [t for t in _all_tix if t.get("status", "") == _tix_filter]
        )

        if not _all_tix:
            st.markdown("""
<div style="text-align:center;padding:3rem 1rem;background:#0f111a;
            border:1px dashed #1e2843;border-radius:12px;margin-top:8px">
  <div style="font-size:2rem;margin-bottom:0.6rem">🎫</div>
  <div style="color:#475569;font-size:0.85rem;line-height:1.6">
    No tickets yet.<br>
<<<<<<< HEAD
    When <b style="color:#93c5fd">CiteRAG</b> can't find an answer in the documents,<br>
    the LangGraph agent will auto-create a ticket here<br>
    with the question, attempted sources, and priority.
=======
    Ask a question in <b style="color:#93c5fd">💬 CiteRAG</b>, then say:<br>
    <b style="color:#a5b4fc">"create a ticket"</b> · <b style="color:#a5b4fc">"raise a ticket"</b> · <b style="color:#a5b4fc">"open a case"</b><br>
    Duplicate questions are automatically detected before creating a new ticket.
>>>>>>> rag
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            # Summary metrics bar
            _op  = len([t for t in _all_tix if t.get("status") == "Open"])
            _ip  = len([t for t in _all_tix if t.get("status") == "In Progress"])
            _dn  = len([t for t in _all_tix if t.get("status") == "Resolved"])
            _sm1, _sm2, _sm3, _sm4 = st.columns(4)
            _sm1.metric("Total",       len(_all_tix))
            _sm2.metric("Open",        _op)
            _sm3.metric("In Progress", _ip)
            _sm4.metric("Resolved",    _dn)
            st.write("")

            if not _show_tix:
                st.info(f"No tickets with status '{_tix_filter}'.")
            else:
                for _t in _show_tix:
                    _ts   = _t.get("status", "Open")
                    _tp   = _t.get("priority", "Medium")
                    _tq   = _t.get("question", "—")
                    _tid  = _t.get("ticket_id", "—")
                    _tts  = _t.get("created_at", "")
                    _turl = _t.get("url", "")
                    _tsum = _t.get("summary", "")
                    _tsrc = _t.get("attempted_sources", [])

                    # Colour mappings
                    _s_col = {"Open": "#f87171", "In Progress": "#fbbf24",
                              "Resolved": "#4ade80"}.get(_ts, "#94a3b8")
                    _s_bg  = {"Open": "rgba(239,68,68,0.08)", "In Progress": "rgba(245,158,11,0.08)",
                              "Resolved": "rgba(34,197,94,0.08)"}.get(_ts, "rgba(148,163,184,0.08)")
                    _s_bd  = {"Open": "rgba(239,68,68,0.25)", "In Progress": "rgba(245,158,11,0.25)",
                              "Resolved": "rgba(34,197,94,0.25)"}.get(_ts, "rgba(148,163,184,0.25)")
                    _p_col = {"High": "#f87171", "Medium": "#fbbf24",
                              "Low": "#4ade80"}.get(_tp, "#94a3b8")

                    with st.container(border=True):
                        _tca, _tcb = st.columns([4, 1])
                        with _tca:
                            st.markdown(
                                f'<span style="background:{_s_bg};border:1px solid {_s_bd};'
                                f'border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;'
                                f'color:{_s_col}">{_ts}</span>'
                                f'&nbsp;<span style="font-size:11px;color:{_p_col};'
                                f'font-weight:600">{_tp}</span>'
                                f'&nbsp;<span style="background:rgba(51,65,85,0.06);'
                                f'padding:2px 6px;border-radius:4px;font-family:monospace;'
                                f'font-size:11px;color:#475569;border:1px solid rgba(51,65,85,0.12)">'
                                f'#{_tid}</span>'
                                + (f'&nbsp;<span style="font-size:11px;color:#64748b">'
                                   f' · {(_tts[:10] if _tts else "")}</span>'
                                   if _tts else ""),
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"**{_tq[:110]}{'…' if len(_tq) > 110 else ''}**"
                            )
                            if _tsum:
                                st.caption(_tsum[:200])

                        with _tcb:
                            if _turl:
                                st.link_button("Notion →", _turl,
                                               use_container_width=True)

                        # Attempted sources
                        if _tsrc:
                            with st.expander("📎 Attempted sources", expanded=False):
                                for _s in _tsrc:
                                    st.markdown(f"- {_s}")

                        # Inline status update
                        _opts   = ["Open", "In Progress", "Resolved"]
                        _curidx = _opts.index(_ts) if _ts in _opts else 0
                        # Use page_id for widget keys — always unique even if ticket_id is missing
                        _wkey   = _t.get("page_id", _tid).replace("-", "")[:16]
                        _col_sel, _col_btn = st.columns([2, 1])
                        with _col_sel:
                            _new_s = st.selectbox(
                                "Update status",
                                _opts,
                                index=_curidx,
                                key=f"tix_sel_{_wkey}",
                                label_visibility="collapsed",
                            )
                        with _col_btn:
                            if _new_s != _ts:
                                if st.button("✅ Update", key=f"tix_upd_{_wkey}",
                                             use_container_width=True, type="primary"):
                                    _ur = _ag_post("/agent/tickets/update",
                                                   {"ticket_id": _tid, "status": _new_s})
                                    if _ur and "error" not in _ur:
                                        st.success(f"Ticket #{_tid} → {_new_s}")
                                        st.session_state.agent_tickets_loaded = False
                                        st.rerun()
                                    else:
                                        st.error(_ur.get("error", "Update failed") if _ur
                                                 else "Backend unreachable")

    # ── Backend endpoint reference (collapsible) ───────────────────────────────
    st.divider()
    with st.expander("🔧 Backend endpoints needed to fully activate this tab"):
        st.markdown("""
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/agent/tickets` | GET | Fetch all tickets from Notion ticket DB |
| `/api/agent/tickets/update` | POST | Update ticket status in Notion |
| `/api/agent/memory` | GET | Read LangGraph thread memory for current user |


**How tickets get created (manual only):**
The LangGraph agent detects ticket-creation intent from the user's message:
- ✅ "create a ticket", "raise a ticket", "open a case", "generate a ticket"
- ✅ "escalate this", "log this", "file a ticket", "submit a request"
- ✅ "mark as resolved", "set to in progress", "close the ticket"

**Duplicate detection (2-layer):**
- Layer 1 — Exact: MD5 hash of normalised question → Redis lookup (instant)
- Layer 2 — Semantic: cosine similarity of embeddings ≥ 0.88 → blocks duplicate

**New file required:** Place `ticket_dedup.py` in `backend/services/rag/`

**New endpoint:** `DELETE /api/agent/dedup/flush` — clears dedup cache after bulk cleanup
""")