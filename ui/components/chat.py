import streamlit as st
import time as _time_mod
import uuid as _uuid
import base64 as _b64
import requests
import json
import streamlit.components.v1 as _components
import logging

_log = logging.getLogger("docforge.frontend")
API_URL = "http://localhost:8000/api"

def set_rag_prefill(q):
    st.session_state._prefill_q = q

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

def render_chat():
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
        _log.info("💬 [CiteRAG] User submitted question | session=%s | q=%r", active_id, question)
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
                _log.info("[CiteRAG] POST /rag/ask session=%s top_k=15 q=%r", active_id, question[:80])
                with requests.post(
                    f"{API_URL}/rag/ask",
                    json={"question": question, "session_id": active_id, "top_k": 15, "stream": True},
                    timeout=120, stream=True,
                ) as resp:
                    resp.raise_for_status()
                    _log.info("[CiteRAG] streaming started HTTP 200")

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
                                    _rd = res_box["result"]
                                    _log.info("[CiteRAG] stream done tool=%s confidence=%s citations=%d",
                                              _rd.get("tool_used","?"), _rd.get("confidence","?"),
                                              len(_rd.get("citations") or []))

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
                _log.error("[CiteRAG] HTTP error — %s", err)
                stream_placeholder.error(f"**Security Alert:** {err}")
                res = None
            except Exception as e:
                _log.error("[CiteRAG] connection error — %s", e)
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
            _log.info("[CiteRAG] answer stored — tool=%s confidence=%s citations=%d chunks=%d followups=%d",
                      ai_msg["tool_used"], ai_msg["confidence"],
                      len(ai_msg["citations"]), len(st.session_state._last_chunks),
                      len(ai_msg["followups"]))
        else:
            err = st.session_state.pop("_last_api_error", "Could not reach the RAG service.")
            _log.warning("[CiteRAG] no result — error: %s", err)
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
