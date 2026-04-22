import streamlit as st
from ui.services.api_client import api_get, api_post

def render_tickets():
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
