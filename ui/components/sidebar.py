import streamlit as st
import uuid as _uuid
import time as _time_mod
import logging

_log = logging.getLogger("docforge.frontend")

_TAB_MAP = {
    "💬 CiteRAG": "ask",
    "⚡ DocForge": "generate",
    "📚 Library":  "library",
    "📊 RAGAS":    "ragas",
    "🎫 Tickets":  "agent",
}

def render_sidebar():
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
                _log.info("[CiteRAG] New chat created id=%s", _cn)
                st.rerun()

            st.caption("Recent")
            _sorted = sorted(st.session_state.rag_chats.items(), key=lambda x: x[1].get("created", 0), reverse=True)
            for _cid, _chat in _sorted:
                _active = _cid == st.session_state.rag_active_chat
                _title  = _chat["title"][:22] + ("…" if len(_chat["title"]) > 22 else "")
                _msgs   = len([m for m in _chat["messages"] if m["role"] == "user"])
                _label  = f"{'💬' if _msgs else '🆕'}  {_title}"

                if st.session_state.get(f"renaming_{_cid}"):
                    _new = st.text_input("New chat name", value=_chat["title"], key=f"rename_input_{_cid}",
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
                                _log.info("[CiteRAG] Chat deleted id=%s msgs=%d", _cid, len(_chat["messages"]))
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
    return active_tab
