"""
DocForge AI × CiteRAG Lab — streamlit_app.py  v9.0
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

st.set_page_config(page_title="DocForge AI × CiteRAG Lab", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(ep):
    try:
        r = httpx.get(f"{API_URL}{ep}", timeout=30)
        r.raise_for_status(); return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def api_post(ep, data, timeout=120):
    try:
        r = httpx.post(f"{API_URL}{ep}", json=data, timeout=timeout)
        r.raise_for_status(); return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

TYPE_ICON = {"table":"📊","flowchart":"🔀","raci":"👥","signature":"✍️","text":"✏️"}
TYPE_CLS  = {"table":"tb-table","flowchart":"tb-flowchart","raci":"tb-raci","signature":"tb-signature"}

def tbadge(sec_type):
    cls = TYPE_CLS.get(sec_type, "")
    if not cls: return ""
    return f'<span class="tbadge {cls}">{TYPE_ICON.get(sec_type,"")} {sec_type}</span>'

def stat_box(num, lbl):
    return (f'<div class="stat-box"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>')


# ── Session ───────────────────────────────────────────────────────────────────

def init_session():
    defaults = dict(
        step=1, company_ctx={}, departments=[],
        selected_dept=None, selected_dept_id=None,
        selected_doc_type=None, doc_sec_id=None, sections=[],
        section_questions={}, section_answers={},
        section_contents={},
        sec_ids_ordered=[], gen_id=None, full_document="",
        active_tab="ask",
        rag_chats={},
        rag_active_chat=None,
        docx_bytes_cache=None, docx_cache_doc=None,
        _library_data=None,
        _answer_drafts={},
        _renaming=None,
        _last_chunks=[],
        _last_confidence="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ DocForge × CiteRAG")
    st.caption("Generate · Ask · Discover")
    st.divider()

    tab = st.radio("Navigation", ["💬 CiteRAG Lab", "⚡ DocForge AI", "📚 Library"],
                   label_visibility="collapsed", key="main_tab")
    if "DocForge" in tab:
        st.session_state.active_tab = "generate"
    elif "Library" in tab:
        st.session_state.active_tab = "library"
    else:
        st.session_state.active_tab = "ask"

    st.divider()

    if st.session_state.active_tab == "ask":
        import time as _st2, uuid as _su2
        if not st.session_state.rag_chats:
            _c0 = _su2.uuid4().hex[:8]
            st.session_state.rag_chats[_c0] = {
                "title": "New chat", "messages": [], "created": _st2.time()
            }
            st.session_state.rag_active_chat = _c0
        if st.button("＋ New Chat", use_container_width=True,
                     key="sb_new_chat", type="primary"):
            _cn = _su2.uuid4().hex[:8]
            st.session_state.rag_chats[_cn] = {
                "title": "New chat", "messages": [], "created": _st2.time()
            }
            st.session_state.rag_active_chat = _cn
            st.rerun()
        _sorted = sorted(st.session_state.rag_chats.items(),
                         key=lambda x: x[1].get("created", 0), reverse=True)
        for _cid, _chat in _sorted:
            _active = _cid == st.session_state.rag_active_chat
            _title  = _chat["title"][:20] + ("…" if len(_chat["title"]) > 20 else "")
            _nm     = len([m for m in _chat["messages"] if m["role"] == "user"])
            # Rename mode
            if st.session_state.get(f"renaming_{_cid}"):
                _new_name = st.text_input("", value=_chat["title"],
                                          key=f"rename_input_{_cid}",
                                          label_visibility="collapsed")
                _rc1, _rc2 = st.columns(2)
                with _rc1:
                    if st.button("Save", key=f"save_ren_{_cid}", use_container_width=True):
                        if _new_name.strip():
                            st.session_state.rag_chats[_cid]["title"] = _new_name.strip()
                        del st.session_state[f"renaming_{_cid}"]
                        st.rerun()
                with _rc2:
                    if st.button("Cancel", key=f"cancel_ren_{_cid}", use_container_width=True):
                        del st.session_state[f"renaming_{_cid}"]
                        st.rerun()
            else:
                _col1, _col2, _col3 = st.columns([4, 1, 1])
                with _col1:
                    if st.button(f"{'▶ ' if _active else ''}{_title}",
                                 key=f"chat_{_cid}", use_container_width=True,
                                 type="primary" if _active else "secondary"):
                        st.session_state.rag_active_chat = _cid
                        st.rerun()
                with _col2:
                    if st.button("✏️", key=f"ren_{_cid}", help="Rename"):
                        st.session_state[f"renaming_{_cid}"] = True
                        st.rerun()
                with _col3:
                    if st.button("🗑", key=f"del_{_cid}", help="Delete"):
                        del st.session_state.rag_chats[_cid]
                        if st.session_state.rag_active_chat == _cid:
                            st.session_state.rag_active_chat = (
                                next(iter(st.session_state.rag_chats))
                                if st.session_state.rag_chats else None)
                        st.rerun()
                if not _active:
                    st.caption(f"{_nm} msg(s)")

    if st.session_state.active_tab == "generate":
        steps = [(1,"Setup"),(2,"Questions"),(3,"Answers"),(4,"Review"),(5,"Export")]
        cur   = st.session_state.step
        for n, lbl in steps:
            icon = "✅" if n < cur else "🔵" if n == cur else "⚪"
            st.markdown(f"{icon} **Step {n}** — {lbl}" if n == cur else f"{icon} Step {n} — {lbl}")
        st.divider()
        ctx = st.session_state.company_ctx
        if ctx.get("company_name"):
            st.caption(f"🏢 {ctx['company_name']}")
        if st.session_state.selected_doc_type:
            st.caption(f"📄 {st.session_state.selected_doc_type}")
        if st.session_state.sections:
            done_n  = len(st.session_state.section_contents)
            total_n = len(st.session_state.sections)
            st.progress(done_n / total_n if total_n else 0,
                        text=f"{done_n}/{total_n} sections")
        st.divider()
        if st.button("↺ Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

if st.session_state.active_tab == "ask":
    import uuid as _uuid, time as _time_mod

    if not st.session_state.rag_chats:
        _c0 = _uuid.uuid4().hex[:8]
        st.session_state.rag_chats[_c0] = {
            "title": "New chat", "messages": [], "created": _time_mod.time()
        }
        st.session_state.rag_active_chat = _c0

    if not st.session_state.rag_active_chat or             st.session_state.rag_active_chat not in st.session_state.rag_chats:
        st.session_state.rag_active_chat = next(iter(st.session_state.rag_chats))

    active_id   = st.session_state.rag_active_chat
    active_chat = st.session_state.rag_chats[active_id]
    messages    = active_chat["messages"]

    st.caption(f"💬 {active_chat.get('title', 'New chat')}")

    # Empty state
    if not messages:
        st.write("")
        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            st.markdown("### ⚡ CiteRAG Lab")
            st.caption("Ask · Cite · Compare · Discover")
            st.write("")
            examples = [
                "🔍 What is the notice period in the employment contract?",
                "⚖️ Compare SOW vs NDA confidentiality clauses",
                "📋 What are the leave policy details?",
                "📄 Summarise the HR policies",
            ]
            for i, ex in enumerate(examples):
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state._prefill_q = ex.split(" ", 1)[1]
                    st.rerun()

    # Render messages
    for msg in messages:
        role       = msg["role"]
        text       = msg["content"]
        citations  = msg.get("citations", [])
        confidence = msg.get("confidence", "")
        tool_used  = msg.get("tool_used", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(text)
        else:
            with st.chat_message("assistant"):
                if confidence and confidence != "low":
                    st.caption(f"CiteRAG Lab · {confidence.upper()}")
                else:
                    st.caption("CiteRAG Lab")

                if tool_used == "compare" and msg.get("side_a"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**📄 {msg.get('doc_a', 'Document A')}**")
                        st.markdown(msg["side_a"])
                    with col_b:
                        st.markdown(f"**📄 {msg.get('doc_b', 'Document B')}**")
                        st.markdown(msg["side_b"])
                    if msg.get("summary"):
                        st.info(f"**Key difference:** {msg['summary']}")
                else:
                    st.markdown(text)

                if citations:
                    seen, unique = set(), []
                    for c in citations:
                        key = c if isinstance(c, str) else c.get("text", "")
                        if key not in seen:
                            seen.add(key)
                            unique.append(c)
                    parts = []
                    for c in unique:
                        if isinstance(c, dict) and c.get("url"):
                            parts.append(f"[📄 {c['text']}]({c['url']})")
                        else:
                            text_c = c if isinstance(c, str) else c.get("text", str(c))
                            parts.append(f"📄 {text_c}")
                    st.markdown("**Sources:** " + " · ".join(parts))

    # Chat input
    _prefill = st.session_state.pop("_prefill_q", "")
    user_q   = st.chat_input("Ask anything about your documents...")

    if user_q or _prefill:
        question = (user_q or _prefill).strip()
        if not messages:
            st.session_state.rag_chats[active_id]["title"] = (
                question[:40] + ("…" if len(question) > 40 else ""))
        st.session_state.rag_chats[active_id]["messages"].append(
            {"role": "user", "content": question})
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
                    "side_a":  res.get("side_a", ""),
                    "side_b":  res.get("side_b", ""),
                    "summary": res.get("summary", ""),
                    "doc_a":   res.get("doc_a", "Document A"),
                    "doc_b":   res.get("doc_b", "Document B"),
                })
            st.session_state._last_chunks = res.get("chunks", [])
        else:
            ai_msg = {
                "role":    "assistant",
                "content": "Could not reach the RAG service. Make sure the backend is running.",
                "citations": [],
            }
            st.session_state._last_chunks = []
        st.session_state.rag_chats[active_id]["messages"].append(ai_msg)
        st.rerun()

    # ── Retrieval panel (toggle, top 5) ───────────────────────────────────────
    chunks = st.session_state.get("_last_chunks", [])
    if chunks:
        show_ret = st.toggle("🔍 Retrieval", value=False, key="show_retrieval")
        if show_ret:
            seen_docs = {}
            for c in chunks:
                title   = c.get("doc_title", "")
                score   = c.get("score", 0)
                page_id = c.get("notion_page_id", "")
                if title and title not in seen_docs:
                    seen_docs[title] = {"score": score, "page_id": page_id}
            top5 = sorted(seen_docs.items(),
                          key=lambda x: x[1]["score"], reverse=True)[:5]
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


elif st.session_state.active_tab == "library":
    if st.button("↺ Refresh", use_container_width=True):
        st.session_state["_library_data"] = None

    if st.session_state.get("_library_data") is None:
        with st.spinner("Loading from Notion..."):
            lib = api_get("/library/notion")
        st.session_state["_library_data"] = lib or {}

    lib  = st.session_state.get("_library_data", {})
    docs = lib.get("documents", []) if isinstance(lib, dict) else []

    if not docs:
        st.info("📭 No documents yet. Generate your first one from the DocForge AI tab!")
    else:
        f1, f2 = st.columns([2, 3])
        with f1:
            dept_filter = st.selectbox("Dept",
                ["All"] + sorted({d.get("department", "") for d in docs if d.get("department")}),
                label_visibility="collapsed")
        with f2:
            search = st.text_input("Search", placeholder="🔍 Search documents...",
                                   label_visibility="collapsed")

        filtered = [d for d in docs
            if (dept_filter == "All" or d.get("department") == dept_filter)
            and (not search or search.lower() in d.get("title", "").lower())]

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total", len(docs))
        with c2: st.metric("Departments", len({d.get("department") for d in docs}))
        with c3: st.metric("Showing", len(filtered))

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        for doc in filtered:
            sc = {"Generated": "#ff6b00", "Draft": "#f59e0b",
                  "Reviewed": "#60a5fa", "Archived": "#3a3a5a"}.get(doc.get("status", ""), "#3a3a5a")
            a, b, c = st.columns([4, 2, 1])
            with a:
                st.markdown(
                    f'<div class="lib-card"><div class="lib-title">{doc.get("title","—")}</div>'
                    f'<div class="lib-meta">{doc.get("doc_type","—")} · {doc.get("department","—")}</div></div>',
                    unsafe_allow_html=True)
            with b:
                st.markdown(
                    f'<div class="lib-card">'
                    f'<div class="lib-meta">📅 {doc.get("created_at","—")}</div>'
                    f'<div class="lib-meta" style="color:{sc}">● {doc.get("status","—")}</div></div>',
                    unsafe_allow_html=True)
            with c:
                if doc.get("notion_url"):
                    st.link_button("Open →", doc["notion_url"], use_container_width=True)


# ── GENERATE ──────────────────────────────────────────────────────────────────

elif st.session_state.active_tab == "generate":

    if st.session_state.step == 1:
        st.markdown('<div class="step-pill">⚡ Step 1 of 5 — Setup</div>', unsafe_allow_html=True)

        if not st.session_state.departments:
            with st.spinner("Loading catalog..."):
                data = api_get("/departments")
                if data: st.session_state.departments = data["departments"]

        depts = st.session_state.departments
        if not depts:
            st.warning("⚠️ Backend not reachable — run: uvicorn backend.main:app --reload")
            st.stop()

        st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">🏢 Company Information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            company_name = st.text_input("Company Name",
                value=st.session_state.company_ctx.get("company_name", ""),
                placeholder="e.g. Turabit Technologies")
            industry = st.selectbox("Industry",
                ["Technology / SaaS","Finance / Banking","Healthcare","Manufacturing",
                 "Retail / E-Commerce","Legal Services","Marketing / Media",
                 "Logistics / Supply Chain","Education","Other"], index=0)
        with c2:
            company_size = st.selectbox("Company Size",
                ["1-10 employees","11-50 employees","51-200 employees",
                 "201-500 employees","500+ employees"], index=2)
            region = st.selectbox("Region",
                ["India","United States","United Kingdom","UAE / Middle East",
                 "Canada","Australia","Europe","Other"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">📂 Select Document</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            selected_dept = st.selectbox("Department", [d["department"] for d in depts] if depts else [])
        dept_data = next((d for d in depts if d["department"] == selected_dept), None) if depts else None
        with c4:
            selected_doc_type = st.selectbox("Document Type", dept_data["doc_types"] if dept_data else [])
        if selected_dept and selected_doc_type:
            st.markdown(f'<div style="margin-top:6px;font-size:0.75rem;color:#ff6b00;">▸ {selected_dept} → {selected_doc_type}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Continue →", type="primary", use_container_width=True):
            if not company_name.strip():
                st.error("Please enter your company name.")
            else:
                with st.spinner("Loading sections..."):
                    safe = selected_doc_type.replace("/", "%2F").replace("(", "%28").replace(")", "%29")
                    data = api_get(f"/sections/{safe}")
                if data:
                    st.session_state.company_ctx       = {"company_name": company_name.strip(),
                        "industry": industry, "company_size": company_size, "region": region}
                    st.session_state.selected_dept     = selected_dept
                    st.session_state.selected_dept_id  = dept_data["doc_id"]
                    st.session_state.selected_doc_type = selected_doc_type
                    st.session_state.doc_sec_id        = data["doc_sec_id"]
                    seen, deduped = set(), []
                    for s in data["doc_sec"]:
                        if s not in seen: seen.add(s); deduped.append(s)
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

    elif st.session_state.step == 2:
        sections = st.session_state.sections
        total    = len(sections)
        st.markdown('<div class="step-pill">❓ Step 2 of 5 — Generate Questions</div>', unsafe_allow_html=True)
        grid_slot   = st.empty()
        status_slot = st.empty()

        def render_grid():
            live_q = st.session_state.section_questions
            done   = len(live_q)
            pct    = int(done / total * 100) if total else 0
            rows   = [f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1rem">' +
                      stat_box(total, "Sections") + stat_box(done, "Ready") + stat_box(str(pct)+"%", "Complete") + '</div>',
                      '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">']
            for s in sections:
                if s in live_q:
                    st_type = live_q[s].get("section_type", "text")
                    rows.append(f'<div class="sec-done">✓ <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{s[:28]}</span>{tbadge(st_type)}</div>')
                else:
                    rows.append(f'<div class="sec-pend">○ {s[:32]}</div>')
            rows.append('</div>')
            grid_slot.markdown("".join(rows), unsafe_allow_html=True)

        def render_status(done, pct, status_text=""):
            status_part = f'<span style="color:#f97316;font-size:0.78rem">⏳ {status_text}</span>' if status_text else ""
            html = (f'<div style="display:flex;align-items:center;justify-content:space-between;margin-top:10px;margin-bottom:6px">'
                    f'<span style="color:#ea580c;font-size:0.82rem;font-weight:700">{done} of {total} sections ready</span>'
                    f'{status_part}</div>'
                    f'<div style="background:#fee2e2;border-radius:999px;height:6px;margin-bottom:10px">'
                    f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);height:6px;border-radius:999px;width:{pct}%"></div></div>')
            status_slot.markdown(html, unsafe_allow_html=True)

        render_grid()
        if len(st.session_state.section_questions) < total:
            if st.button("⚡ Generate Questions for All Sections", type="primary", use_container_width=True):
                for i, sec in enumerate(sections):
                    live_done = len(st.session_state.section_questions)
                    live_pct  = int(live_done / total * 100)
                    if sec in st.session_state.section_questions:
                        render_grid(); render_status(live_done, live_pct); continue
                    render_status(live_done, live_pct, status_text=sec)
                    res = api_post("/questions/generate", {
                        "doc_sec_id": st.session_state.doc_sec_id,
                        "doc_id": st.session_state.selected_dept_id,
                        "section_name": sec, "doc_type": st.session_state.selected_doc_type,
                        "department": st.session_state.selected_dept,
                        "company_context": st.session_state.company_ctx,
                    })
                    if res:
                        st.session_state.section_questions[sec] = {
                            "sec_id": res["sec_id"], "questions": res.get("questions", []),
                            "section_type": res.get("section_type", "text"),
                        }
                    render_grid()
                    new_done = len(st.session_state.section_questions)
                    render_status(new_done, int(new_done / total * 100))
                status_slot.empty()
                st.rerun()
        if len(st.session_state.section_questions) == total:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Start Answering →", type="primary", use_container_width=True):
                st.session_state.step = 3; st.rerun()

    elif st.session_state.step == 3:
        sections   = st.session_state.sections
        ans_map    = st.session_state.section_answers
        q_map      = st.session_state.section_questions
        unanswered = [s for s in sections if s not in ans_map]

        if not unanswered:
            st.markdown('<div class="step-pill">✅ Step 3 of 5 — Ready to Generate</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: st.markdown(stat_box(len(ans_map), "Answered"), unsafe_allow_html=True)
            with c2: st.markdown(stat_box(len(sections), "Will Generate"), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("⚡ Generate Document", type="primary", use_container_width=True):
                active = sections
                total  = len(active)
                gen_status = st.empty()
                ids = []
                for i, sec in enumerate(active):
                    if sec in st.session_state.section_contents:
                        ids.append(q_map.get(sec, {}).get("sec_id", 0))
                        _pct = int((i+1)/total*100)
                        gen_status.markdown(f'<div style="background:#fee2e2;border-radius:999px;height:6px"><div style="background:linear-gradient(90deg,#ea580c,#f97316);height:6px;border-radius:999px;width:{_pct}%"></div></div>', unsafe_allow_html=True)
                        continue
                    gen_status.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">'
                        f'<span style="color:#ea580c;font-size:0.78rem;font-weight:600">{len(st.session_state.section_contents)} of {total} sections written</span>'
                        f'<span style="color:#f97316;font-size:0.78rem">✍️ Writing: {sec}</span></div>'
                        f'<div style="background:#fee2e2;border-radius:999px;height:6px">'
                        f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);height:6px;border-radius:999px;width:{int(len(st.session_state.section_contents)/total*100)}%"></div></div>',
                        unsafe_allow_html=True)
                    q_data = q_map.get(sec, {})
                    res = api_post("/section/generate", {
                        "sec_id": q_data.get("sec_id"), "doc_sec_id": st.session_state.doc_sec_id,
                        "doc_id": st.session_state.selected_dept_id, "section_name": sec,
                        "doc_type": st.session_state.selected_doc_type,
                        "department": st.session_state.selected_dept,
                        "company_context": st.session_state.company_ctx, "num_sections": total,
                    }, timeout=120)
                    if res:
                        st.session_state.section_contents[sec] = res["content"]
                        ids.append(q_data.get("sec_id"))
                    _pct = int((i+1)/total*100)
                    gen_status.markdown(f'<div style="background:#fee2e2;border-radius:999px;height:6px"><div style="background:linear-gradient(90deg,#ea580c,#f97316);height:6px;border-radius:999px;width:{_pct}%"></div></div>', unsafe_allow_html=True)

                st.session_state.sec_ids_ordered = ids
                doc_lines = []
                for sec in active:
                    c = st.session_state.section_contents.get(sec, "").strip()
                    if c: doc_lines += [sec.upper(), "-"*len(sec), "", c, "", ""]
                full_doc = "\n".join(doc_lines).strip()
                save_res = api_post("/document/save", {
                    "doc_id": st.session_state.selected_dept_id,
                    "doc_sec_id": st.session_state.doc_sec_id,
                    "sec_id": ids[-1] if ids else 0,
                    "gen_doc_sec_dec": list(st.session_state.section_contents.values()),
                    "gen_doc_full": full_doc,
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

            st.markdown('<div class="step-pill">✍️ Step 3 of 5 — Answer Questions</div>', unsafe_allow_html=True)
            pct3 = int(done_cnt / total * 100) if total else 0
            st.markdown(
                f'<div style="background:#fee2e2;border-radius:999px;height:6px;margin-bottom:6px">'
                f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);height:6px;border-radius:999px;width:{pct3}%;transition:width 0.3s"></div></div>'
                f'<div style="color:#ea580c;font-size:0.75rem;margin-bottom:1rem">{done_cnt} of {total} complete · {len(unanswered)} remaining</div>',
                unsafe_allow_html=True)

            hints = {
                "table":     ("📊","Data table — your answers will populate the rows.","#eff6ff"),
                "flowchart": ("🔀","Process flowchart — describe the steps and decisions.","#f0fdf4"),
                "raci":      ("👥","RACI matrix — list the roles involved.","#faf5ff"),
                "signature": ("✍️","Sign-off block — auto-generated, no input needed.","#fdf2f8"),
                "text":      ("✏️","",""),
            }
            icon, hint, bg = hints.get(sec_type, ("✏️","",""))
            st.markdown(
                f'<div class="df-card" style="background:{bg or "#ffffff"};margin-bottom:12px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:{"0.5rem" if hint else "0"}">'
                f'<span style="font-size:1rem;font-weight:700;color:#111827">{icon} {current}</span>'
                f'{tbadge(sec_type)}</div>'
                + (f'<div style="font-size:0.82rem;color:#6b7280;line-height:1.5">{hint}</div>' if hint else "")
                + '</div>', unsafe_allow_html=True)

            if "_answer_drafts" not in st.session_state:
                st.session_state._answer_drafts = {}
            if current not in st.session_state._answer_drafts:
                st.session_state._answer_drafts[current] = [""] * len(questions)

            user_answers = []
            if not questions:
                st.markdown('<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:10px 16px;color:#ea580c;font-size:0.82rem">✨ No questions needed — this section is auto-generated professionally.</div><div style="height:12px"></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:0.8rem;color:#6b7280;margin-bottom:10px">Leave blank to auto-fill with professional content</div>', unsafe_allow_html=True)
                for i, q in enumerate(questions):
                    current_val = st.session_state._answer_drafts[current][i] if i < len(st.session_state._answer_drafts[current]) else ""
                    a = st.text_area(f"Q{i+1}: {q}", value=current_val, key=f"draft_{current}_{i}",
                                     height=85, placeholder="Your answer (or leave blank for auto-fill)...")
                    if i < len(st.session_state._answer_drafts[current]):
                        st.session_state._answer_drafts[current][i] = a
                    user_answers.append(a)

            if st.button("Save & Next →", type="primary", use_container_width=True):
                filled = [a.strip() if a.strip() else "not answered" for a in user_answers]
                if sec_id:
                    api_post("/answers/save", {
                        "sec_id": sec_id, "doc_sec_id": st.session_state.doc_sec_id,
                        "doc_id": st.session_state.selected_dept_id, "section_name": current,
                        "questions": questions, "answers": filled or ["not answered"],
                    })
                st.session_state.section_answers[current] = filled
                st.session_state._answer_drafts.pop(current, None)
                st.rerun()

            if ans_map:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown('<div style="font-size:0.68rem;color:#ff6b00;font-weight:700;letter-spacing:0.08em;margin-bottom:6px">ANSWERED</div>', unsafe_allow_html=True)
                for s in sections:
                    if s in ans_map:
                        st.markdown(f'<div class="sec-answered">✓ <strong>{s}</strong></div>', unsafe_allow_html=True)

    elif st.session_state.step == 4:
        active   = st.session_state.sections
        contents = st.session_state.section_contents
        st.markdown('<div class="step-pill">🔍 Step 4 of 5 — Review & Edit</div>', unsafe_allow_html=True)

        def rebuild_doc():
            lines = []
            for sec in active:
                c = contents.get(sec, "").strip()
                if c: lines += [sec.upper(), "-"*len(sec), "", c, "", ""]
            st.session_state.full_document = "\n".join(lines).strip()

        left, right = st.columns([1, 2])
        with left:
            st.markdown('<div style="font-size:0.68rem;color:#ea580c;font-weight:700;letter-spacing:0.08em;margin-bottom:8px">SECTIONS</div>', unsafe_allow_html=True)
            sel = st.radio("", active, label_visibility="collapsed", key="sec_radio")
        with right:
            cur      = contents.get(sel, "")
            sec_type = st.session_state.section_questions.get(sel, {}).get("section_type", "text")
            icon     = TYPE_ICON.get(sec_type, "✏️")
            st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.75rem"><span style="font-size:0.95rem;font-weight:700;color:#111827">{icon} {sel}</span>{tbadge(sec_type)}</div>', unsafe_allow_html=True)
            with st.expander("📄 Current Content", expanded=True):
                if "```mermaid" in (cur or ""):
                    st.markdown(cur)
                elif not cur:
                    st.markdown('<div style="color:#9ca3af;font-style:italic;padding:1rem">(empty)</div>', unsafe_allow_html=True)
                else:
                    _lines = cur.split("\n")
                    _html  = '<div style="font-family:Georgia,serif;font-size:0.88rem;color:#1f2937;line-height:1.8;padding:1.2rem 1.5rem;background:#fafafa;border-radius:10px;border:1px solid #e5e7eb;">'
                    for _line in _lines:
                        if not _line.strip(): _html += '<div style="height:6px"></div>'
                        else: _html += f'<p style="margin:0 0 4px 0">{_line}</p>'
                    _html += '</div>'
                    st.markdown(_html, unsafe_allow_html=True)

            instr = st.text_area("AI Edit Instruction", placeholder="e.g. Make more formal · Add detail · Shorten",
                                  height=60, key="edit_instr", label_visibility="collapsed")
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("🤖 Apply AI Edit", type="primary", use_container_width=True):
                    if not instr.strip(): st.warning("Enter an instruction.")
                    else:
                        with st.spinner("Editing..."):
                            res = api_post("/section/edit", {
                                "gen_id": st.session_state.gen_id or 0,
                                "sec_id": st.session_state.section_questions.get(sel, {}).get("sec_id", 0),
                                "section_name": sel, "doc_type": st.session_state.selected_doc_type,
                                "current_content": cur, "edit_instruction": instr,
                            }, timeout=120)
                        if res:
                            st.session_state.section_contents[sel] = res["updated_content"]
                            st.session_state.docx_bytes_cache = None
                            rebuild_doc(); st.success("✅ Updated!"); st.rerun()
            with ec2:
                manual = st.text_area("Manual Edit", value=cur, height=180, key=f"manual_{sel}", label_visibility="collapsed")
                if st.button("💾 Save Manual", use_container_width=True, key=f"save_{sel}"):
                    st.session_state.section_contents[sel] = manual
                    st.session_state.docx_bytes_cache = None
                    rebuild_doc(); st.success("✅ Saved!"); st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Export →", type="primary", use_container_width=True):
            st.session_state.step = 5; st.rerun()

    elif st.session_state.step == 5:
        ctx      = st.session_state.company_ctx
        doc_type = st.session_state.selected_doc_type
        full_doc = st.session_state.full_document
        active   = st.session_state.sections
        contents = st.session_state.section_contents

        st.markdown('<div class="step-pill">💾 Step 5 of 5 — Export</div>', unsafe_allow_html=True)

        if not full_doc:
            st.markdown('<div class="df-card" style="text-align:center;padding:3rem 2rem"><div style="font-size:3rem;margin-bottom:1rem">📄</div><div style="font-size:1.2rem;font-weight:700;color:#1f2937;margin-bottom:0.5rem">No document found</div></div>', unsafe_allow_html=True)
            if st.button("🏠 Go to Home Page", type="primary", use_container_width=False):
                for k in list(st.session_state.keys()): del st.session_state[k]
                init_session(); st.rerun()
            st.stop()

        st.markdown(f"""
        <div class="df-card df-card-glow">
            <div class="df-card-title">📄 Document Ready</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:4px">
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Document</div><div style="font-size:0.9rem;color:#111827;font-weight:600">{doc_type}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Department</div><div style="font-size:0.9rem;color:#111827;font-weight:600">{st.session_state.selected_dept}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Company</div><div style="font-size:0.9rem;color:#111827;font-weight:600">{ctx.get("company_name","—")}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Industry</div><div style="font-size:0.9rem;color:#374151">{ctx.get("industry","—")}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Sections</div><div style="font-size:0.9rem;color:#374151">{len(active)} sections</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Words</div><div style="font-size:0.9rem;color:#374151">~{len(full_doc.split())} words</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        col_n, col_d = st.columns(2)
        with col_n:
            st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">📓 Publish to Notion</div><div style="font-size:0.78rem;color:#3a3a5a;margin-bottom:10px">Publish to your Notion workspace database.</div>', unsafe_allow_html=True)
            if st.button("🚀 Publish to Notion", type="primary", use_container_width=True):
                with st.spinner("Publishing..."):
                    res = api_post("/document/publish", {
                        "gen_id": st.session_state.gen_id or 0, "doc_type": doc_type,
                        "department": st.session_state.selected_dept, "gen_doc_full": full_doc,
                        "company_context": ctx,
                    })
                if res:
                    url = res.get("notion_url", "")
                    st.success("✅ Published to Notion!")
                    if url: st.link_button("🔗 Open in Notion", url, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">📥 Download</div>', unsafe_allow_html=True)
            safe = doc_type.replace(" ","_").replace("/","-").replace("(","").replace(")","")
            if DOCX_AVAILABLE:
                if (st.session_state.get("docx_bytes_cache") is None or
                        st.session_state.get("docx_cache_doc") != doc_type):
                    try:
                        sections_data = [{"name":sec,"content":contents.get(sec,"")} for sec in active if contents.get(sec)]
                        st.session_state.docx_bytes_cache = build_docx(
                            doc_type=doc_type, department=st.session_state.selected_dept,
                            company_name=ctx.get("company_name","Company"),
                            industry=ctx.get("industry",""), region=ctx.get("region",""),
                            sections=sections_data)
                        st.session_state.docx_cache_doc = doc_type
                    except Exception as e:
                        st.error(f"DOCX error: {e}")
                        st.session_state.docx_bytes_cache = None
                if st.session_state.get("docx_bytes_cache"):
                    st.download_button("⬇️ Download .docx", data=st.session_state.docx_bytes_cache,
                        file_name=f"{safe}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True, type="primary")
            else:
                st.warning("docx_builder.py not found.")
            st.download_button("⬇️ Download .txt", data=full_doc, file_name=f"{safe}.txt",
                               mime="text/plain", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📄 Preview Full Document", expanded=False):
            import re as _re
            def _render_doc(text):
                lines = text.split("\n")
                html  = '<div style="font-family:Georgia,serif;font-size:0.88rem;color:#1f2937;line-height:1.8;padding:1.5rem 2rem;background:#fff;border-radius:10px;border:1px solid #e5e7eb;">'
                i = 0
                while i < len(lines):
                    line = lines[i]
                    if i+1 < len(lines) and _re.match(r'^-{3,}$', lines[i+1].strip()) and line.strip() == line.strip().upper() and line.strip():
                        html += f'<h3 style="font-size:1rem;font-weight:700;color:#ea580c;border-bottom:2px solid #fed7aa;padding-bottom:4px;margin:1.5rem 0 0.5rem;">{line.strip().title()}</h3>'
                        i += 2; continue
                    elif _re.match(r'^-{3,}$', line.strip()):
                        i += 1; continue
                    elif not line.strip():
                        html += '<div style="height:4px"></div>'
                    else:
                        html += f'<p style="margin:0 0 4px 0">{line}</p>'
                    i += 1
                html += '</div>'
                return html
            st.markdown(_render_doc(full_doc), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("➕ Create Another Document", type="primary", use_container_width=True):
            saved_ctx   = st.session_state.company_ctx
            saved_depts = st.session_state.departments
            for k in list(st.session_state.keys()): del st.session_state[k]
            init_session()
            st.session_state["company_ctx"]  = saved_ctx
            st.session_state["departments"]  = saved_depts
            st.session_state["step"]         = 1
            st.rerun()