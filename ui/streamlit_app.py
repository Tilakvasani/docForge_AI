"""
streamlit_app.py — DocForge AI  v15

Changes vs v14:
  1. Blank-page flash eliminated — Step 2 & Step 3 generation loops now run
     entirely inside a single script execution using st.empty() placeholders.
     No mid-loop st.rerun() is issued; placeholders are mutated in-place so
     Streamlit never renders an empty screen.
  2. Section tiles turn ⚪→✅ green (st.success) in real-time as each section
     completes, both for question generation (Step 2) and document generation
     (Step 3b).
  3. Dead code removed — sec_ids_ordered session key was never referenced.
  4. All imports are at the top, directly after the docstring.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time as _time_mod
import uuid as _uuid
import base64 as _b64
import streamlit as st
import streamlit.components.v1 as _components
import httpx
import requests

try:
    from docx_builder import build_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/") + "/api"

_TAB_MAP = {
    "💬 CiteRAG": "ask",
    "⚡ DocForge": "generate",
    "📚 Library":  "library",
    "📊 RAGAS":    "ragas",
    "🎫 Tickets":  "agent",
}

# ── Page config & CSS ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocForge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
[data-testid="stSidebar"] {
    background-image: linear-gradient(180deg, #1e1e2f 0%, #12121e 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)


# ── RAGAS renderer ────────────────────────────────────────────────────────────

def _render_ragas_scores(scores: dict, title: str = "", timestamp: str = ""):
    if not scores:
        return
    metrics = [
        ("Faithfulness",      scores.get("faithfulness"),      "no hallucination"),
        ("Answer Relevancy",  scores.get("answer_relevancy"),  "on-topic answer"),
        ("Context Precision", scores.get("context_precision"), "clean retrieval"),
        ("Context Recall",    scores.get("context_recall"),    "full coverage"),
    ]
    avg_vals  = [m[1] for m in metrics if m[1] is not None]
    avg_score = round(sum(avg_vals) / len(avg_vals), 2) if avg_vals else None
    header    = (title or "RAGAS Scores") + (f"  ·  {timestamp}" if timestamp else "")
    if avg_score is not None:
        header += f"  ·  avg {avg_score:.2f}"

    with st.container(border=True):
        st.caption(header)
        warn_lines = []
        for label, val, hint in metrics:
            if val is None:
                st.caption(f"{label} — n/a ({hint})")
                continue
            icon = "🟢" if val >= 0.85 else "🟡" if val >= 0.70 else "🔴"
            st.progress(int(val * 100), text=f"{icon} **{label}** `{val:.2f}` — {hint}")
            if val < 0.70:
                if   "faith"  in label.lower(): warn_lines.append("⚠️ Faithfulness low — answer may contain unsupported claims.")
                elif "prec"   in label.lower(): warn_lines.append("⚠️ Context precision low — retriever fetched irrelevant chunks.")
                elif "recall" in label.lower(): warn_lines.append("⚠️ Context recall low — relevant chunks may have been missed.")
                elif "relev"  in label.lower(): warn_lines.append("⚠️ Answer relevancy low — answer drifted off-topic.")
        if warn_lines:
            for w in warn_lines:
                st.warning(w)
        elif avg_vals:
            st.success("All quality metrics look good.")


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(ep: str) -> dict | None:
    try:
        r = httpx.get(f"{API_URL}{ep}", timeout=30)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None


def api_post(ep: str, data: dict, timeout: int = 120) -> dict | None:
    try:
        r = httpx.post(f"{API_URL}{ep}", json=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        try:
            msg = e.response.json().get("detail", e.response.text[:200])
        except Exception:
            msg = e.response.text[:200]
        st.session_state._last_api_error = msg
        return None
    except Exception as e:
        st.session_state._last_api_error = f"Connection error: {e}"
        return None


# ── Live section-grid helper ──────────────────────────────────────────────────

def _render_section_grid(placeholder, sections: list, done_set: set, failed_set: set):
    """
    Rebuild the 3-column section grid inside a single st.empty() placeholder.
    Completed sections render as green ✅, failed as red ❌, pending as ⚪.
    """
    with placeholder.container():
        cols = st.columns(3)
        for i, s in enumerate(sections):
            label = s[:30]
            if s in done_set:
                cols[i % 3].success(f"✅ {label}")
            elif s in failed_set:
                cols[i % 3].error(f"❌ {label}")
            else:
                cols[i % 3].markdown(f"⚪ {label}")


# ── Session init ──────────────────────────────────────────────────────────────

def init_session():
    defaults = dict(
        step=1, company_ctx={}, departments=[],
        selected_dept=None, selected_dept_id=None,
        selected_doc_type=None, doc_sec_id=None, sections=[],
        section_questions={}, section_answers={},
        section_contents={}, gen_id=None, full_document="",
        main_tab="💬 CiteRAG",
        rag_chats={}, rag_active_chat=None,
        docx_bytes_cache=None, docx_cache_doc=None,
        _library_data=None, _answer_drafts={},
        _last_chunks=[], _last_not_found=False,
        _last_ragas_scores=None, _ragas_history=[],
        agent_tickets=[], agent_tickets_loaded=False,
        agent_memory={},
        _gen_failed_q_sections=set(),
        _gen_failed_doc_sections=set(),
        gen_questions_running=False,
        gen_doc_running=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ DocForge AI")
    st.caption("Generate · Ask · Discover")
    st.divider()

    st.radio("Mode", list(_TAB_MAP.keys()),
             label_visibility="collapsed", key="main_tab", horizontal=False)

    active_tab = _TAB_MAP.get(st.session_state.main_tab, "ask")

    st.divider()

    if active_tab == "ask":
        if not st.session_state.rag_chats:
            _c0 = _uuid.uuid4().hex[:8]
            st.session_state.rag_chats[_c0] = {"title": "New chat", "messages": [], "created": _time_mod.time()}
            st.session_state.rag_active_chat = _c0

        if st.button("＋  New Chat", use_container_width=True, key="sb_new_chat", type="primary"):
            _cn = _uuid.uuid4().hex[:8]
            st.session_state.rag_chats[_cn] = {"title": "New chat", "messages": [], "created": _time_mod.time()}
            st.session_state.rag_active_chat = _cn
            st.rerun()

        st.caption("Recent")
        _sorted = sorted(st.session_state.rag_chats.items(), key=lambda x: x[1].get("created", 0), reverse=True)
        for _cid, _chat in _sorted:
            _active = _cid == st.session_state.rag_active_chat
            _title  = _chat["title"][:22] + ("…" if len(_chat["title"]) > 22 else "")
            _msgs   = len([m for m in _chat["messages"] if m["role"] == "user"])
            _label  = f"{'💬' if _msgs else '🆕'}  {_title}"

            if st.session_state.get(f"renaming_{_cid}"):
                _new = st.text_input("", value=_chat["title"], key=f"rename_input_{_cid}",
                                     label_visibility="collapsed", placeholder="Enter new name…")
                r1, r2 = st.columns(2)
                with r1:
                    if st.button("✅ Save", key=f"save_ren_{_cid}", use_container_width=True, type="primary"):
                        if _new.strip():
                            st.session_state.rag_chats[_cid]["title"] = _new.strip()
                        del st.session_state[f"renaming_{_cid}"]
                        st.rerun()
                with r2:
                    if st.button("✕", key=f"cancel_ren_{_cid}", use_container_width=True):
                        del st.session_state[f"renaming_{_cid}"]
                        st.rerun()
            else:
                if st.button(_label, key=f"chat_{_cid}", use_container_width=True,
                             type="primary" if _active else "secondary"):
                    st.session_state.rag_active_chat = _cid
                    st.rerun()
                if _active:
                    _a, _b = st.columns(2)
                    with _a:
                        if st.button("✏️ Rename", key=f"ren_{_cid}", use_container_width=True):
                            st.session_state[f"renaming_{_cid}"] = True
                            st.rerun()
                    with _b:
                        if st.button("🗑 Delete", key=f"del_{_cid}", use_container_width=True, type="primary"):
                            del st.session_state.rag_chats[_cid]
                            if st.session_state.rag_active_chat == _cid:
                                st.session_state.rag_active_chat = (
                                    next(iter(st.session_state.rag_chats)) if st.session_state.rag_chats else None
                                )
                            st.session_state._last_chunks       = []
                            st.session_state._last_ragas_scores = None
                            st.rerun()

    if active_tab == "generate":
        st.caption("Steps")
        steps = [(1,"🏢","Setup"),(2,"❓","Questions"),(3,"✍️","Answers"),(4,"⚙️","Generate"),(5,"💾","Export")]
        cur = st.session_state.step
        for n, emoji, lbl in steps:
            if   n < cur:  st.markdown(f"✅  ~~Step {n} — {lbl}~~")
            elif n == cur: st.markdown(f"**{emoji}  Step {n} — {lbl}**")
            else:          st.markdown(f"⬜  Step {n} — {lbl}")
        st.divider()
        if st.button("↺  Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── CiteRAG prefill helper ────────────────────────────────────────────────────

def set_rag_prefill(question: str):
    st.session_state._prefill_q = question


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: CiteRAG
# ══════════════════════════════════════════════════════════════════════════════

if active_tab == "ask":
    if not st.session_state.rag_chats:
        _c0 = _uuid.uuid4().hex[:8]
        st.session_state.rag_chats[_c0] = {"title": "New chat", "messages": [], "created": _time_mod.time()}
        st.session_state.rag_active_chat = _c0

    if not st.session_state.rag_active_chat or st.session_state.rag_active_chat not in st.session_state.rag_chats:
        st.session_state.rag_active_chat = next(iter(st.session_state.rag_chats))

    active_id   = st.session_state.rag_active_chat
    active_chat = st.session_state.rag_chats[active_id]
    messages    = active_chat["messages"]

    if not messages:
        st.markdown("## ⚡ CiteRAG Lab")
        st.caption("Ask questions about your documents · Cite sources · Compare clauses · Analyse risk")
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
                st.button(ex, key=f"ex_{i}", use_container_width=True,
                          on_click=set_rag_prefill, args=(ex,))
    else:
        st.caption(f"💬  {active_chat.get('title', 'New chat')}")

    for idx, msg in enumerate(messages):
        role       = msg["role"]
        text       = msg["content"]
        confidence = msg.get("confidence", "")
        with st.chat_message(role):
            if role == "assistant" and confidence:
                conf_icon = "🟢" if confidence == "high" else "🟡" if confidence == "medium" else "🔴"
                st.caption(f"{conf_icon} CiteRAG  ·  confidence: {confidence}")

            if role == "user":
                st.markdown(text)
            else:
                display_text = text.replace("📋 FINAL ANSWER: ", "").strip()
                if display_text.startswith(": "):
                    display_text = display_text[2:]
                st.markdown(display_text)

                agent_note = msg.get("agent_reply", "")
                if agent_note and msg.get("tool_used") != "agent":
                    st.success(agent_note) if "✅" in agent_note else st.info(agent_note)

                _encoded = _b64.b64encode(text.encode("utf-8")).decode("ascii")
                _components.html(
                    f"""<button id="cpbtn_{idx}" onclick="
                        try {{
                            var t = atob('{_encoded}');
                            navigator.clipboard.writeText(t).then(function(){{
                                document.getElementById('cpbtn_{idx}').textContent='✅ Copied!';
                                setTimeout(function(){{document.getElementById('cpbtn_{idx}').textContent='📋 Copy';}},2000);
                            }}).catch(function(){{
                                var el=document.createElement('textarea');el.value=t;
                                document.body.appendChild(el);el.select();document.execCommand('copy');
                                document.body.removeChild(el);
                                document.getElementById('cpbtn_{idx}').textContent='✅ Copied!';
                                setTimeout(function(){{document.getElementById('cpbtn_{idx}').textContent='📋 Copy';}},2000);
                            }});
                        }} catch(e){{console.error(e);}}
                    " style="background:transparent;border:1px solid #334155;border-radius:5px;
                    color:#64748b;font-size:11px;padding:3px 10px;cursor:pointer;font-family:inherit;">
                    📋 Copy</button>""",
                    height=32,
                )

                f_ups = msg.get("followups", [])
                if f_ups:
                    st.caption("Suggested follow-ups:")
                    cols = st.columns(min(len(f_ups), 3))
                    for i, fq in enumerate(f_ups[:3]):
                        with cols[i]:
                            st.button(fq, key=f"fup_{idx}_{i}", use_container_width=True,
                                      on_click=set_rag_prefill, args=(fq,))

    _prefill = st.session_state.pop("_prefill_q", "")
    user_q   = st.chat_input("Ask anything about your documents…")

    if user_q or _prefill:
        question = (user_q or _prefill).strip()
        if not messages:
            st.session_state.rag_chats[active_id]["title"] = question[:40] + ("..." if len(question) > 40 else "")
        st.session_state.rag_chats[active_id]["messages"].append({"role": "user", "content": question})

        ai_msg = {"role": "assistant", "content": "", "citations": [],
                  "confidence": "", "tool_used": "", "agent_reply": "", "followups": []}

        with st.chat_message("assistant", avatar="🤖"):
            stream_placeholder = st.empty()
            res_box    = {}
            full_answer = ""
            try:
                with requests.post(
                    f"{API_URL}/rag/ask",
                    json={"question": question, "session_id": active_id, "top_k": 15, "stream": True},
                    timeout=120, stream=True,
                ) as resp:
                    resp.raise_for_status()

                    def _token_gen():
                        for line in resp.iter_lines():
                            if line:
                                data = json.loads(line)
                                if data.get("type") == "token":
                                    for char in data.get("content", ""):
                                        yield char
                                        _time_mod.sleep(0.008)
                                elif data.get("type") == "done":
                                    res_box["result"] = data.get("result", {})

                    full_answer = stream_placeholder.write_stream(_token_gen())
                    res = res_box.get("result")
                    if not full_answer and res:
                        full_answer = res.get("answer", "")
                        stream_placeholder.write(full_answer)

            except requests.exceptions.HTTPError as e:
                try:
                    err = e.response.json().get("detail", "Request rejected by security policy.")
                except Exception:
                    err = f"API Error: {e.response.status_code}"
                stream_placeholder.error(f"**Security Alert:** {err}")
                res = None
            except Exception as e:
                stream_placeholder.error(f"**Error:** Could not reach the RAG service. {e}")
                res = None

        if res:
            ai_msg.update({
                "content":     res.get("answer", full_answer or "No answer returned."),
                "citations":   res.get("citations", []),
                "confidence":  res.get("confidence", ""),
                "tool_used":   res.get("tool_used", ""),
                "agent_reply": res.get("agent_reply", ""),
                "followups":   res.get("followups", []),
            })
            st.session_state._last_chunks       = res.get("chunks") or res.get("_raw_chunks", [])
            st.session_state._last_not_found    = (not res.get("chunks") and bool(res.get("_raw_chunks")))
            st.session_state._last_ragas_scores = res.get("ragas_scores")
        else:
            err = st.session_state.pop("_last_api_error", "Could not reach the RAG service.")
            ai_msg = {"role": "assistant", "content": f"⚠️ **Error:** {err}", "citations": []}
            st.session_state._last_chunks       = []
            st.session_state._last_not_found    = False
            st.session_state._last_ragas_scores = None

        st.session_state.rag_chats[active_id]["messages"].append(ai_msg)
        st.rerun()

    chunks    = st.session_state.get("_last_chunks", [])
    not_found = st.session_state.get("_last_not_found", False)
    src_label = "🔍 Show Sources" if not not_found else "🔍 Show Sources (searched but not found)"

    if chunks and st.toggle(src_label, value=False, key="show_retrieval"):
        last_scores = st.session_state.get("_last_ragas_scores")
        if last_scores and not not_found:
            st.markdown("#### 🔬 Answer Quality (RAGAS)")
            _render_ragas_scores(last_scores, title="Automatic evaluation")
            st.divider()
        if not_found:
            st.warning("These were the closest documents found — none contained a confident answer.")

        seen = {}
        for c in chunks:
            title   = c.get("doc_title", "")
            score   = c.get("score", 0)
            page_id = c.get("notion_page_id", "")
            section = c.get("section", c.get("heading", ""))
            key     = f"{title}::{section}"
            if title and key not in seen:
                seen[key] = {"doc_title": title, "score": score, "page_id": page_id, "section": section}

        for i, info in enumerate(sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:5]):
            with st.container(border=True):
                sc   = info["score"]
                icon = "🟢" if sc >= 0.6 else "🟡" if sc >= 0.4 else "🔴"
                pid  = info["page_id"]
                url  = f"https://www.notion.so/{pid.replace('-', '')}" if pid else ""
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"{icon} **{info['doc_title']}**")
                    if info.get("section"):
                        st.caption(info["section"])
                    st.caption(f"Rank {i+1}  ·  score `{sc:.3f}`")
                with c2:
                    if url:
                        st.link_button("Open →", url, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Library
# ══════════════════════════════════════════════════════════════════════════════

elif active_tab == "library":
    ch, cb = st.columns([4, 1])
    with ch:
        st.markdown("### 📚 Document Library")
    with cb:
        if st.button("↺ Refresh", use_container_width=True):
            st.session_state["_library_data"] = None

    if st.session_state.get("_library_data") is None:
        with st.spinner("Loading from Notion…"):
            lib = api_get("/library/notion")
        st.session_state["_library_data"] = lib or {}

    lib  = st.session_state.get("_library_data", {})
    docs = lib.get("documents", []) if isinstance(lib, dict) else []

    if not docs:
        st.info("No documents yet. Generate your first one from the ⚡ DocForge tab!")
    else:
        f1, f2 = st.columns([1, 2])
        with f1:
            dept_filter = st.selectbox("Department",
                ["All"] + sorted({d.get("department", "") for d in docs if d.get("department")}))
        with f2:
            search = st.text_input("Search", placeholder="Search documents…")

        filtered = [d for d in docs
                    if (dept_filter == "All" or d.get("department") == dept_filter)
                    and (not search or search.lower() in d.get("title", "").lower())]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total",       len(docs))
        c2.metric("Departments", len({d.get("department") for d in docs}))
        c3.metric("Showing",     len(filtered))
        st.divider()

        for doc in filtered:
            with st.container(border=True):
                a, b, c = st.columns([4, 2, 1])
                with a:
                    st.markdown(f"**{doc.get('title', '—')}**")
                    st.caption(f"{doc.get('doc_type','—')}  ·  {doc.get('industry','—')}")
                with b:
                    st.caption(f"🏢 {doc.get('department','—')}")
                    st.caption(f"📅 {doc.get('created_at','—')}")
                    status      = doc.get("status", "—")
                    status_icon = {"Generated":"🟠","Draft":"🟡","Reviewed":"🔵","Archived":"⚪"}.get(status,"⚪")
                    st.caption(f"{status_icon} {status}")
                with c:
                    if doc.get("notion_url"):
                        st.link_button("Open →", doc["notion_url"], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: RAGAS
# ══════════════════════════════════════════════════════════════════════════════

elif active_tab == "ragas":
    st.markdown("## 📊 RAGAS Evaluation")
    st.caption("Real answer quality scores — faithfulness, relevancy, precision, recall")
    st.divider()

    st.markdown("#### 🗂 Batch Evaluation")
    st.caption("Run RAGAS on multiple questions. Add rows manually or import a JSON file.")

    if "batch_rows"    not in st.session_state: st.session_state.batch_rows    = [{"question": "", "ground_truth": ""}]
    if "batch_results" not in st.session_state: st.session_state.batch_results = []
    if "batch_running" not in st.session_state: st.session_state.batch_running = False

    with st.container(border=True):
        with st.expander("📥 Import from JSON", expanded=False):
            st.caption('Expected: `[{"question": "...", "ground_truth": "..."}, ...]`')

            def _handle_json_upload():
                uploaded = st.session_state.get("batch_json_upload")
                if uploaded:
                    try:
                        raw_data = json.loads(uploaded.read().decode("utf-8"))
                        if not isinstance(raw_data, list):
                            st.session_state["_batch_err"] = "JSON must be a list."
                        else:
                            parsed = [
                                {"question": i.get("question","").strip(),
                                 "ground_truth": i.get("ground_truth","").strip()}
                                for i in raw_data if isinstance(i, dict) and i.get("question","").strip()
                            ]
                            if parsed:
                                st.session_state.batch_rows    = parsed
                                st.session_state.batch_results = []
                                st.session_state["_batch_succ"] = f"Loaded {len(parsed)} questions."
                            else:
                                st.session_state["_batch_err"] = "No valid questions found."
                    except Exception as je:
                        st.session_state["_batch_err"] = f"JSON parse error: {je}"

            st.file_uploader("Upload JSON", type=["json"], key="batch_json_upload",
                             label_visibility="collapsed", on_change=_handle_json_upload)
            if "_batch_succ" in st.session_state: st.success(st.session_state.pop("_batch_succ"))
            if "_batch_err"  in st.session_state: st.error(st.session_state.pop("_batch_err"))

        st.markdown("**Questions**")
        rows_to_delete = []
        for ri, row in enumerate(st.session_state.batch_rows):
            rc1, rc2, rc3 = st.columns([3, 3, 0.5])
            with rc1:
                q_val = st.text_input(f"Q{ri+1}", value=row["question"],
                                      placeholder="e.g. What is the leave policy?",
                                      key=f"batch_q_{ri}", label_visibility="collapsed")
                st.session_state.batch_rows[ri]["question"] = q_val
            with rc2:
                gt_val = st.text_input(f"GT{ri+1}", value=row["ground_truth"],
                                       placeholder="Ground truth (optional)",
                                       key=f"batch_gt_{ri}", label_visibility="collapsed")
                st.session_state.batch_rows[ri]["ground_truth"] = gt_val
            with rc3:
                if len(st.session_state.batch_rows) > 1:
                    if st.button("✕", key=f"batch_del_{ri}", help="Remove row"):
                        rows_to_delete.append(ri)

        if rows_to_delete:
            for idx in sorted(rows_to_delete, reverse=True):
                st.session_state.batch_rows.pop(idx)
            st.session_state.batch_results = []
            st.rerun()

        ba1, ba2 = st.columns([1, 3])
        with ba1:
            if st.button("＋ Add Row", key="batch_add_row"):
                st.session_state.batch_rows.append({"question": "", "ground_truth": ""})
                st.rerun()
        with ba2:
            st.caption(f"{len(st.session_state.batch_rows)} question(s) queued · 20–60s each")

        _valid_rows = [r for r in st.session_state.batch_rows if r["question"].strip()]
        _bp = st.session_state.get("_batch_progress") or {}
        if _bp.get("running") and _bp.get("total", 0) > 0:
            st.progress(_bp["done"] / _bp["total"],
                        text=f"⏳ {_bp['done']}/{_bp['total']}: {str(_bp.get('current_q',''))[:55]}…")

        if st.button(f"▶ Run Batch ({len(_valid_rows)} questions)", type="primary",
                     key="batch_run_btn", use_container_width=True,
                     disabled=len(_valid_rows) == 0 or st.session_state.get("batch_running", False)):
            if _valid_rows:
                st.session_state.batch_results = []
                st.session_state.batch_running = True
                _total = len(_valid_rows)
                st.session_state._batch_progress = {"running": True, "done": 0, "total": _total, "current_q": ""}

                for bi, brow in enumerate(_valid_rows):
                    bq  = brow["question"].strip()
                    bgt = brow["ground_truth"].strip()
                    st.session_state._batch_progress.update({"current_q": bq, "done": bi})
                    bts  = _time_mod.strftime("%H:%M:%S")
                    bres = api_post("/rag/eval", {"question": bq, "ground_truth": bgt, "top_k": 15}, timeout=600)

                    entry = {"question": bq, "ground_truth": bgt, "timestamp": bts,
                             "scores": None, "answer": "", "error": None, "tool_used": ""}
                    if bres:
                        entry.update({"scores": bres.get("ragas_scores"), "answer": bres.get("answer",""),
                                      "error": bres.get("ragas_error"), "tool_used": bres.get("tool_used","")})
                        if entry["scores"]:
                            st.session_state._ragas_history.append(
                                {"question": bq, "scores": entry["scores"],
                                 "tool_used": entry["tool_used"], "timestamp": bts})
                            st.session_state._ragas_history = st.session_state._ragas_history[-20:]
                    else:
                        entry["error"] = "API call failed — backend unreachable."
                    st.session_state.batch_results.append(entry)
                    st.session_state._batch_progress["done"] = bi + 1

                st.session_state._batch_progress = {"running": False, "done": _total, "total": _total}
                st.session_state.batch_running = False
                st.rerun()

    if st.session_state.batch_results:
        br = st.session_state.batch_results
        st.divider()
        st.markdown(f"**📊 Results** — {len(br)} questions")
        _bscored = [r for r in br if r["scores"]]
        if _bscored:
            def _bavg(key):
                vals = [r["scores"].get(key) for r in _bscored if r["scores"].get(key) is not None]
                return round(sum(vals) / len(vals), 3) if vals else None
            with st.container(border=True):
                st.caption(f"AVERAGES · {len(_bscored)}/{len(br)} scored")
                bc1, bc2, bc3, bc4 = st.columns(4)
                for col, lbl, key in [
                    (bc1,"Faithfulness","faithfulness"),(bc2,"Ans. Relevancy","answer_relevancy"),
                    (bc3,"Ctx Precision","context_precision"),(bc4,"Ctx Recall","context_recall"),
                ]:
                    v = _bavg(key)
                    col.metric(lbl, f"{v:.3f}" if v is not None else "n/a")

        for bri, bentry in enumerate(br):
            blabel  = f"Q{bri+1}: {bentry['question'][:65]}{'…' if len(bentry['question'])>65 else ''}"
            bstatus = "✅" if bentry["scores"] else ("❌" if bentry["error"] else "⚠️")
            with st.expander(f"{bstatus} {blabel} · {bentry['timestamp']}"):
                if bentry["answer"]:
                    st.markdown("**RAG Answer:**")
                    st.markdown(bentry["answer"])
                if bentry["scores"]:
                    _render_ragas_scores(bentry["scores"], title=bentry["question"])
                elif bentry["error"]:
                    st.error(f"RAGAS error: {bentry['error']}")
                else:
                    st.warning("No scores returned.")

        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "⬇️ Export Results as JSON",
                data=json.dumps([{"question":r["question"],"ground_truth":r["ground_truth"],
                                  "timestamp":r["timestamp"],"tool_used":r["tool_used"],
                                  "answer":r["answer"],"scores":r["scores"],"error":r["error"]}
                                 for r in br], indent=2),
                file_name=f"ragas_report_{_time_mod.strftime('%Y%m%d')}.json",
                mime="application/json", key="batch_export_btn",
                use_container_width=True,
            )
        with ex2:
            if DOCX_AVAILABLE:
                _br_scored = [r for r in br if r["scores"]]
                def _get_avg(key):
                    vals = [r["scores"].get(key) for r in _br_scored if r["scores"].get(key) is not None]
                    return round(sum(vals) / len(vals), 3) if vals else None

                summ = "| Metric | Average Score |\n|---|---|\n"
                for lbl, key in [
                    ("Faithfulness", "faithfulness"), ("Answer Relevancy", "answer_relevancy"),
                    ("Context Precision", "context_precision"), ("Context Recall", "context_recall"),
                ]:
                    av = _get_avg(key)
                    summ += f"| {lbl} | {f'{av:.3f}' if av is not None else 'n/a'} |\n"
                
                rep_secs = [{"name": "Executive Summary", "content": summ}]
                for bri, bentry in enumerate(br):
                    q_det = f"**Question:** {bentry['question']}\n\n"
                    if bentry.get("ground_truth"):
                        q_det += f"**Ground Truth:** {bentry['ground_truth']}\n\n"
                    q_det += f"**RAG Answer:** {bentry['answer']}\n\n"
                    
                    if bentry.get("scores"):
                        q_det += "| Metric | Score |\n|---|---|\n"
                        for lbl, key in [
                            ("Faithfulness", "faithfulness"), ("Answer Relevancy", "answer_relevancy"),
                            ("Context Precision", "context_precision"), ("Context Recall", "context_recall"),
                        ]:
                            sv = bentry["scores"].get(key)
                            q_det += f"| {lbl} | {f'{sv:.3f}' if sv is not None else 'n/a'} |\n"
                    elif bentry.get("error"):
                        q_det += f"**Error:** {bentry['error']}\n"
                    rep_secs.append({"name": f"Evaluation Q{bri+1}", "content": q_det})

                try:
                    ctx = st.session_state.get("company_ctx", {})
                    docx_rep = build_docx(
                        doc_type="RAGAS Evaluation Report",
                        department="AI Quality Assurance",
                        company_name=ctx.get("company_name", "DocForge AI"),
                        industry=ctx.get("industry", "Evaluation"),
                        region=ctx.get("region", "Global"),
                        sections=rep_secs
                    )
                    st.download_button(
                        "⬇️ Download Report (.docx)",
                        data=docx_rep,
                        file_name=f"ragas_report_{_time_mod.strftime('%Y%m%d')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key="batch_export_docx", type="primary", use_container_width=True,
                    )
                except Exception as de:
                    st.error(f"DOCX Report error: {de}")
            else:
                st.info("DOCX builder not available.")

    history = st.session_state.get("_ragas_history", [])
    if history:
        st.divider()
        st.markdown(f"#### 📈 Session History  ·  {len(history)} evaluations")
        for entry in reversed(history):
            q_label = entry["question"][:70] + ("…" if len(entry["question"]) > 70 else "")
            with st.expander(f"**{q_label}**  ·  {entry.get('timestamp','')}"):
                _render_ragas_scores(entry["scores"], title=entry["question"])
        if st.button("🗑 Clear History", key="ragas_clear_hist"):
            st.session_state._ragas_history = []
            st.rerun()

    st.divider()
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
| Metric | What it measures | Good threshold |
|---|---|---|
| **Faithfulness**      | Every claim grounded in retrieved documents | ≥ 0.85 |
| **Answer Relevancy**  | Answer directly addresses the question      | ≥ 0.80 |
| **Context Precision** | Retrieved chunks are relevant (no noise)    | ≥ 0.75 |
| **Context Recall**    | All relevant facts were retrieved           | ≥ 0.75 |
""")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: DocForge Generate
# ══════════════════════════════════════════════════════════════════════════════

elif active_tab == "generate":

    # ── Step 1: Setup ──────────────────────────────────────────────────────────
    if st.session_state.step == 1:
        st.markdown("## ⚡ DocForge AI")
        st.caption("Enter company details and pick a document type to generate.")
        st.divider()

        if not st.session_state.departments:
            with st.spinner("Loading catalog…"):
                data = api_get("/departments")
                if data:
                    st.session_state.departments = data["departments"]

        depts = st.session_state.departments
        if not depts:
            st.warning("Backend not reachable — run: `uvicorn backend.main:app --reload`")
            st.stop()

        with st.container(border=True):
            st.markdown("**🏢 Company Info**")
            c1, c2 = st.columns(2)
            with c1:
                company_name = st.text_input("Company Name",
                    value=st.session_state.company_ctx.get("company_name",""),
                    placeholder="e.g. Turabit Technologies")
                industry = st.selectbox("Industry", [
                    "Technology / SaaS","Finance / Banking","Healthcare",
                    "Manufacturing","Retail / E-Commerce","Legal Services",
                    "Marketing / Media","Logistics / Supply Chain","Education","Other"])
            with c2:
                company_size = st.selectbox("Company Size", [
                    "1-10 employees","11-50 employees","51-200 employees",
                    "201-500 employees","500+ employees"], index=2)
                region = st.selectbox("Region", [
                    "India","United States","United Kingdom","UAE / Middle East",
                    "Canada","Australia","Europe","Other"])

        with st.container(border=True):
            st.markdown("**📂 Select Document**")
            c3, c4 = st.columns(2)
            with c3:
                selected_dept = st.selectbox("Department", [d["department"] for d in depts])
            dept_data = next((d for d in depts if d["department"] == selected_dept), None)
            with c4:
                selected_doc_type = st.selectbox("Document Type", dept_data["doc_types"] if dept_data else [])

        st.write("")
        if st.button("Continue →", type="primary", use_container_width=True):
            if not company_name.strip():
                st.error("Please enter your company name.")
            else:
                with st.spinner("Loading sections…"):
                    safe = selected_doc_type.replace("/","%2F").replace("(","%28").replace(")","%29")
                    data = api_get(f"/sections/{safe}")
                if data:
                    st.session_state.company_ctx       = {"company_name": company_name.strip(),
                                                          "industry": industry,
                                                          "company_size": company_size,
                                                          "region": region}
                    st.session_state.selected_dept     = selected_dept
                    st.session_state.selected_dept_id  = dept_data["doc_id"]
                    st.session_state.selected_doc_type = selected_doc_type
                    st.session_state.doc_sec_id        = data["doc_sec_id"]
                    seen, deduped = set(), []
                    for s in data["doc_sec"]:
                        if s not in seen:
                            seen.add(s); deduped.append(s)
                    st.session_state.sections = deduped
                    st.session_state.update({
                        "section_questions": {}, "section_answers": {},
                        "section_contents":  {}, "full_document": "",
                        "gen_id": None, "docx_bytes_cache": None, "_answer_drafts": {},
                        "_gen_failed_q_sections": set(), "_gen_failed_doc_sections": set(),
                        "gen_questions_running": False, "gen_doc_running": False,
                    })
                    st.session_state.step = 2
                    st.rerun()

    # ── Step 2: Generate Questions — live, no blank flash ──────────────────────
    elif st.session_state.step == 2:
        sections = st.session_state.sections
        total    = len(sections)
        q_map    = st.session_state.section_questions
        failed_q = st.session_state.get("_gen_failed_q_sections", set())
        done     = len(q_map)

        st.markdown("## ❓ Generate Questions")
        st.caption(f"{st.session_state.selected_doc_type} · {total} sections")
        st.divider()

        # ── Always render these placeholders first so the page is never blank ──
        progress_ph = st.empty()
        status_ph   = st.empty()
        grid_ph     = st.empty()

        # Immediately show current state (before any generation starts)
        progress_ph.progress(
            done / total if total else 0,
            text=f"{done} / {total} sections generated"
                 + (f"  ·  {len(failed_q)} skipped" if failed_q else ""),
        )
        _render_section_grid(grid_ph, sections, set(q_map.keys()), failed_q)

        # ── Live generation loop — runs in same script execution, no rerun ─────
        if st.session_state.gen_questions_running and done < total:
            for sec in sections:
                if sec in q_map or sec in failed_q:
                    continue
                status_ph.info(f"⏳ Generating questions for: **{sec}**")
                res = api_post("/questions/generate", {
                    "doc_sec_id":      st.session_state.doc_sec_id,
                    "doc_id":          st.session_state.selected_dept_id,
                    "section_name":    sec,
                    "doc_type":        st.session_state.selected_doc_type,
                    "department":      st.session_state.selected_dept,
                    "company_context": st.session_state.company_ctx,
                })
                if res:
                    q_map[sec] = {
                        "sec_id":       res["sec_id"],
                        "questions":    res.get("questions", []),
                        "section_type": res.get("section_type", "text"),
                    }
                else:
                    failed_q.add(sec)
                    st.session_state._gen_failed_q_sections = failed_q

                done = len(q_map)
                # Update progress bar and grid in-place — no blank flash
                progress_ph.progress(
                    done / total,
                    text=f"{done} / {total} sections generated"
                         + (f"  ·  {len(failed_q)} skipped" if failed_q else ""),
                )
                _render_section_grid(grid_ph, sections, set(q_map.keys()), failed_q)

            # Generation complete — clear running flag and do ONE final rerun
            st.session_state.gen_questions_running = False
            status_ph.success("✅ All questions generated!")
            _time_mod.sleep(0.4)
            st.rerun()

        # ── Action buttons — only visible when not generating ─────────────────
        effective_total = total - len(failed_q)
        st.write("")

        if done < effective_total and not st.session_state.gen_questions_running:
            if st.button("⚡ Generate Questions for All Sections",
                         type="primary", use_container_width=True, key="gen_q_btn"):
                st.session_state.gen_questions_running  = True
                st.session_state._gen_failed_q_sections = set()
                st.rerun()
            if failed_q and st.button("🔄 Retry Failed Sections",
                                      use_container_width=True, key="retry_q_btn"):
                st.session_state._gen_failed_q_sections = set()
                st.session_state.gen_questions_running  = True
                st.rerun()
        elif done > 0 and not st.session_state.gen_questions_running:
            status_ph.success(f"✅ All {done} sections ready!")
            if st.button("Start Answering →", type="primary",
                         use_container_width=True, key="goto_answers"):
                st.session_state.step = 3
                st.rerun()

    # ── Step 3: Answers + Document Generation ─────────────────────────────────
    elif st.session_state.step == 3:
        sections   = st.session_state.sections
        ans_map    = st.session_state.section_answers
        q_map      = st.session_state.section_questions
        unanswered = [s for s in sections if s not in ans_map]

        # ── Sub-step 3b — all answers collected, generate document sections ────
        if not unanswered:
            contents   = st.session_state.section_contents
            failed_doc = st.session_state.get("_gen_failed_doc_sections", set())
            done_doc   = len(contents)
            total_sec  = len(sections)

            st.markdown("## ✅ All Sections Answered")
            st.divider()

            # ── Always render placeholders first so page is never blank ────────
            progress_ph = st.empty()
            status_ph   = st.empty()
            grid_ph     = st.empty()

            progress_ph.progress(
                done_doc / total_sec if total_sec else 0,
                text=f"{done_doc} / {total_sec} sections drafted"
                     + (f"  ·  {len(failed_doc)} skipped" if failed_doc else ""),
            )
            _render_section_grid(grid_ph, sections, set(contents.keys()), failed_doc)

            # ── Live generation loop — same execution, no blank flash ──────────
            if st.session_state.gen_doc_running and done_doc < total_sec:
                for sec in sections:
                    if sec in contents or sec in failed_doc:
                        continue
                    status_ph.info(f"⏳ Drafting section: **{sec}**")
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
                        contents[sec] = res["content"]
                    else:
                        failed_doc.add(sec)
                        st.session_state._gen_failed_doc_sections = failed_doc

                    done_doc = len(contents)
                    # Update in-place — no blank flash
                    progress_ph.progress(
                        done_doc / total_sec,
                        text=f"{done_doc} / {total_sec} sections drafted"
                             + (f"  ·  {len(failed_doc)} skipped" if failed_doc else ""),
                    )
                    _render_section_grid(grid_ph, sections, set(contents.keys()), failed_doc)

                # Done — clear flag, one final rerun
                st.session_state.gen_doc_running = False
                status_ph.success("✅ Full draft ready!")
                _time_mod.sleep(0.4)
                st.rerun()

            # ── Action buttons ─────────────────────────────────────────────────
            effective_total = total_sec - len(failed_doc)
            st.write("")

            if done_doc < effective_total and not st.session_state.gen_doc_running:
                if st.button("⚡ Generate Document", type="primary",
                             use_container_width=True, key="gen_doc_btn"):
                    st.session_state.gen_doc_running          = True
                    st.session_state._gen_failed_doc_sections = set()
                    st.rerun()
                if failed_doc and st.button("🔄 Retry Failed Sections",
                                            use_container_width=True, key="retry_doc_btn"):
                    st.session_state._gen_failed_doc_sections = set()
                    st.session_state.gen_doc_running          = True
                    st.rerun()
            elif done_doc > 0 and not st.session_state.gen_doc_running:
                status_ph.success("✅ Full Draft Ready!")
                if st.button("Finalize and Save →", type="primary",
                             use_container_width=True, key="finalize_btn"):
                    doc_lines = []
                    for sec in sections:
                        c = contents.get(sec, "").strip()
                        if c:
                            doc_lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
                    full_doc = "\n".join(doc_lines).strip()
                    ids      = [q_map.get(s, {}).get("sec_id") for s in sections if q_map.get(s, {}).get("sec_id")]
                    save_res = api_post("/document/save", {
                        "doc_id":          st.session_state.selected_dept_id,
                        "doc_sec_id":      st.session_state.doc_sec_id,
                        "sec_id":          ids[-1] if ids else 0,
                        "gen_doc_sec_dec": list(contents.values()),
                        "gen_doc_full":    full_doc,
                    })
                    st.session_state.gen_id           = save_res.get("gen_id", 0) if save_res else 0
                    st.session_state.full_document    = full_doc
                    st.session_state.docx_bytes_cache = None
                    st.session_state.step = 4
                    st.rerun()

        # ── Sub-step 3a — collect answers one section at a time ───────────────
        else:
            current   = unanswered[0]
            done_cnt  = len(ans_map)
            q_data    = q_map.get(current, {})
            questions = q_data.get("questions", [])
            sec_id    = q_data.get("sec_id")

            st.markdown(f"## ✏️  {current}")
            st.progress(done_cnt / len(sections) if sections else 0,
                        text=f"{done_cnt} / {len(sections)} answered")
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
                    cur_val = (st.session_state._answer_drafts[current][i]
                               if i < len(st.session_state._answer_drafts[current]) else "")
                    a = st.text_area(f"Q{i+1}: {q}", value=cur_val, key=f"draft_{current}_{i}",
                                     height=85, placeholder="Your answer (or leave blank)…")
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

    # ── Step 4: Review & Edit ──────────────────────────────────────────────────
    elif st.session_state.step == 4:
        active   = st.session_state.sections
        contents = st.session_state.section_contents
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
            cur      = contents.get(sel) or ""
            sec_type = st.session_state.section_questions.get(sel, {}).get("section_type", "text")
            icon     = {"table":"📊","flowchart":"🔀","raci":"👥","signature":"✍️","text":"✏️"}.get(sec_type,"✏️")
            st.markdown(f"**{icon} {sel}**")
            with st.expander("Current Content", expanded=True):
                if cur:
                    st.text(cur)
                else:
                    st.caption("(empty)")

            st.write("")
            instr = st.text_area("AI Edit Instruction",
                                 placeholder="e.g. Make more formal · Add detail · Shorten",
                                 height=60, key="edit_instr", label_visibility="collapsed")
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("🤖 Apply AI Edit", type="primary", use_container_width=True):
                    if not instr.strip():
                        st.warning("Enter an instruction.")
                    else:
                        with st.spinner("Editing…"):
                            res = api_post("/section/edit", {
                                "gen_id":           st.session_state.gen_id or 0,
                                "sec_id":           st.session_state.section_questions.get(sel, {}).get("sec_id", 0),
                                "section_name":     sel,
                                "doc_type":         st.session_state.selected_doc_type,
                                "current_content":  cur,
                                "edit_instruction": instr,
                            }, timeout=120)
                        if res:
                            contents[sel] = res["updated_content"]
                            st.session_state.docx_bytes_cache = None
                            rebuild_doc()
                            st.success("✅ Updated!")
                            st.rerun()
            with ec2:
                manual = st.text_area("Manual Edit", value=cur, height=180,
                                      key=f"manual_{sel}", label_visibility="collapsed")
                if st.button("💾 Save Manual", use_container_width=True, key=f"save_{sel}"):
                    contents[sel] = manual
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
            c1.metric("Document",   doc_type)
            c2.metric("Department", st.session_state.selected_dept)
            c3.metric("Company",    ctx.get("company_name","--"))
            c4, c5 = st.columns(2)
            c4.metric("Sections", len(active))
            c5.metric("Words",    f"~{len(full_doc.split())}")

        st.divider()
        col_n, col_d = st.columns(2)

        with col_n:
            with st.container(border=True):
                st.markdown("**📓 Publish to Notion**")
                st.caption("Send to your Notion workspace.")
                if st.button("🚀 Publish to Notion", type="primary", use_container_width=True):
                    with st.spinner("Publishing…"):
                        res = api_post("/document/publish", {
                            "gen_id":          st.session_state.gen_id or 0,
                            "doc_type":        doc_type,
                            "department":      st.session_state.selected_dept,
                            "gen_doc_full":    full_doc,
                            "company_context": ctx,
                        })
                    if res:
                        url = res.get("notion_url","")
                        st.success(f"✅ Published! Version {res.get('version','')}.")
                        if url:
                            st.link_button("🔗 Open in Notion", url, use_container_width=True)

        with col_d:
            with st.container(border=True):
                st.markdown("**📥 Download**")
                safe = doc_type.replace(" ","_").replace("/","-").replace("(","").replace(")","")
                if DOCX_AVAILABLE:
                    if (st.session_state.get("docx_bytes_cache") is None
                            or st.session_state.get("docx_cache_doc") != doc_type):
                        try:
                            st.session_state.docx_bytes_cache = build_docx(
                                doc_type=doc_type, department=st.session_state.selected_dept,
                                company_name=ctx.get("company_name","Company"),
                                industry=ctx.get("industry",""), region=ctx.get("region",""),
                                sections=[{"name": s, "content": contents.get(s,"")}
                                          for s in active if contents.get(s)])
                            st.session_state.docx_cache_doc = doc_type
                        except Exception as e:
                            st.error(f"DOCX error: {e}")
                            st.session_state.docx_bytes_cache = None
                    if st.session_state.get("docx_bytes_cache"):
                        st.download_button(
                            "⬇️ Download .docx", data=st.session_state.docx_bytes_cache,
                            file_name=f"{safe}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True, type="primary")
                else:
                    st.warning("docx_builder.py not found.")
                st.download_button("⬇️ Download .txt", data=full_doc,
                                   file_name=f"{safe}.txt", mime="text/plain",
                                   use_container_width=True)

        st.divider()
        with st.expander("📄 Preview Full Document", expanded=True):
            with st.container(border=True):
                st.markdown(full_doc)
        st.write("")
        if st.button("➕ Create Another Document", type="primary", use_container_width=True):
            saved_ctx   = st.session_state.company_ctx
            saved_depts = st.session_state.departments
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            init_session()
            st.session_state.update({"company_ctx": saved_ctx, "departments": saved_depts, "step": 1})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB: Tickets
# ══════════════════════════════════════════════════════════════════════════════

elif active_tab == "agent":
    st.markdown("## 🎫 Tickets")
    st.caption(
        "Knowledge-gap tickets — say **\"create a ticket\"** or **\"raise a ticket\"** "
        "in 💬 CiteRAG to log a missing answer."
    )
    st.divider()

    col_mem, col_tix = st.columns([1, 2], gap="large")

    with col_mem:
        st.markdown("#### 🧠 Session Memory")
        mem = st.session_state.agent_memory
        if mem:
            for icon, key in [("👤","user_name"),("🏭","industry"),("🎯","last_intent"),("📄","last_doc")]:
                val = mem.get(key,"")
                if val:
                    st.markdown(f"{icon} `{str(val)[:30]}`")
        else:
            st.caption("No memory yet — ask a question in 💬 CiteRAG to populate this.")

        st.divider()
        st.markdown("#### ⚙️ Context Hints")
        st.caption("Pre-fill memory so the agent knows who you are.")
        _hn = st.text_input("Your name",       value=mem.get("user_name",""), placeholder="e.g. Rahul",           key="ag_hint_name")
        _hi = st.text_input("Industry / dept", value=mem.get("industry",""),  placeholder="e.g. Technology / HR", key="ag_hint_industry")
        if st.button("💾 Save Hints", key="ag_save_hints", use_container_width=True):
            st.session_state.agent_memory.update({"user_name": _hn.strip(), "industry": _hi.strip()})
            res_sync = api_post("/agent/memory", {
                "session_id": st.session_state.rag_active_chat,
                "memory":     {"user_name": _hn.strip(), "industry": _hi.strip()},
            })
            if res_sync:
                st.success("Hints saved and synced to backend!")
            else:
                err = st.session_state.pop("_last_api_error","Unknown error")
                st.error(f"Sync failed: {err}")
            st.rerun()

    with col_tix:
        th1, th2, th3 = st.columns([2, 1, 1])
        with th1:
            st.markdown("#### 🎫 Knowledge-Gap Tickets")
        with th2:
            tix_filter = st.selectbox("Status filter", ["All","Open","In Progress","Resolved"],
                                      key="ag_tix_filter", label_visibility="collapsed")
        with th3:
            if st.button("↺ Refresh", key="ag_refresh_tix", use_container_width=True):
                st.session_state.agent_tickets_loaded = False

        if not st.session_state.agent_tickets_loaded:
            with st.spinner("Loading tickets from Notion…"):
                td = api_get("/agent/tickets")
            if td and "error" not in td:
                st.session_state.agent_tickets        = td.get("tickets", [])
                st.session_state.agent_tickets_loaded = True
            else:
                err_msg = (td.get("error","Unknown") if td else "Backend unreachable")
                st.info(f"ℹ️ Tickets endpoint: `{err_msg}`\n\nTickets appear here once the backend is deployed.")

        all_tix  = st.session_state.agent_tickets
        show_tix = all_tix if tix_filter == "All" else [t for t in all_tix if t.get("status","") == tix_filter]

        if not all_tix:
            st.info(
                "No tickets yet.\n\n"
                "Ask a question in **💬 CiteRAG**, then say:\n"
                "**\"create a ticket\"** · **\"raise a ticket\"** · **\"open a case\"**\n\n"
                "Duplicate questions are automatically detected before creating a new ticket."
            )
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total",       len(all_tix))
            m2.metric("Open",        sum(1 for t in all_tix if t.get("status") == "Open"))
            m3.metric("In Progress", sum(1 for t in all_tix if t.get("status") == "In Progress"))
            m4.metric("Resolved",    sum(1 for t in all_tix if t.get("status") == "Resolved"))
            st.write("")

            if not show_tix:
                st.info(f"No tickets with status '{tix_filter}'.")
            else:
                for t in show_tix:
                    ts   = t.get("status",   "Open")
                    tp   = t.get("priority", "Medium")
                    tq   = t.get("question", "—")
                    tid  = t.get("ticket_id","—")
                    tdt  = t.get("created_time","")[:10]
                    turl = t.get("url","")
                    tsum = t.get("summary","")
                    tsrc = t.get("attempted_sources",[])

                    status_icon   = {"Open":"🔴","In Progress":"🟡","Resolved":"🟢"}.get(ts,"⚪")
                    priority_icon = {"High":"🔥","Medium":"⚡","Low":"❄️"}.get(tp,"❄️")

                    with st.container(border=True):
                        ca, cb = st.columns([4, 1])
                        with ca:
                            st.markdown(f"{status_icon} **{ts}**  ·  {priority_icon} {tp}  ·  `#{tid}`  ·  {tdt}")
                            st.markdown(f"**{tq[:110]}{'…' if len(tq)>110 else ''}**")
                            if tsum:
                                st.caption(tsum[:200])
                        with cb:
                            if turl:
                                st.link_button("Notion →", turl, use_container_width=True)

                        if tsrc:
                            with st.expander("📎 Attempted sources"):
                                for s in tsrc:
                                    st.markdown(f"- {s}")

                        _opts   = ["Open","In Progress","Resolved"]
                        _curidx = _opts.index(ts) if ts in _opts else 0
                        _wkey   = t.get("page_id", tid).replace("-","")[:16]

                        col_sel, col_btn = st.columns([2, 1])
                        with col_sel:
                            new_s = st.selectbox("Update status", _opts, index=_curidx,
                                                 key=f"tix_sel_{_wkey}", label_visibility="collapsed")
                        with col_btn:
                            if new_s != ts:
                                if st.button("✅ Update", key=f"tix_upd_{_wkey}",
                                             use_container_width=True, type="primary"):
                                    ur = api_post("/agent/tickets/update", {"ticket_id": tid, "status": new_s})
                                    if ur and "error" not in ur:
                                        st.success(f"Ticket #{tid} → {new_s}")
                                        st.session_state.agent_tickets_loaded = False
                                        st.rerun()
                                    else:
                                        st.error(ur.get("error","Update failed") if ur else "Backend unreachable")