import streamlit as st
import json
import time as _time_mod
from ui.services.api_client import api_get, api_post
import logging
_log = logging.getLogger("frontend")


def render_library():
        ch, cb = st.columns([4, 1])
        with ch:
            st.markdown("### 📚 Document Library")
        with cb:
            if st.button("↺ Refresh", use_container_width=True):
                _log.info("[Library] refresh button clicked")
                st.session_state["_library_data"] = None

        if st.session_state.get("_library_data") is None:
            with st.spinner("Loading from Notion…"):
                lib = api_get("/library/notion")
            st.session_state["_library_data"] = lib or {}
            _log.info("[Library] loaded %d docs", len((lib or {}).get("documents", [])))

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


