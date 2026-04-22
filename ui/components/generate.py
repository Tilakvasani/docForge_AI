import streamlit as st
import time as _time_mod
from ui.services.api_client import api_get, api_post
from ui.utils.session import init_session

try:
    from docx_builder import build_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

def _render_section_grid(ph, sections, done_set, failed_set):
    """Helper for Step 2 & 3: Render pill grid."""
    html = ['<div style="display:flex;flex-wrap:wrap;gap:6px;">']
    for s in sections:
        if s in failed_set:
            color, bg, icon = "#842029", "#f8d7da", "❌"
        elif s in done_set:
            color, bg, icon = "#0f5132", "#d1e7dd", "✅"
        else:
            color, bg, icon = "#495057", "#e9ecef", "⏳"
        html.append(f'<span style="background:{bg};color:{color};padding:4px 10px;border-radius:12px;font-size:13px;">{icon} {s}</span>')
    html.append('</div>')
    ph.markdown("".join(html), unsafe_allow_html=True)

def render_generate():
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

    elif st.session_state.step == 2:
        sections = st.session_state.sections
        total    = len(sections)
        q_map    = st.session_state.section_questions
        failed_q = st.session_state.get("_gen_failed_q_sections", set())
        done     = len(q_map)

        st.markdown("## ❓ Generate Questions")
        st.caption(f"{st.session_state.selected_doc_type} · {total} sections")
        st.divider()

        progress_ph = st.empty()
        status_ph   = st.empty()
        grid_ph     = st.empty()

        progress_ph.progress(
            done / total if total else 0,
            text=f"{done} / {total} sections generated"
                 + (f"  ·  {len(failed_q)} skipped" if failed_q else ""),
        )
        _render_section_grid(grid_ph, sections, set(q_map.keys()), failed_q)

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
                progress_ph.progress(
                    done / total,
                    text=f"{done} / {total} sections generated"
                         + (f"  ·  {len(failed_q)} skipped" if failed_q else ""),
                )
                _render_section_grid(grid_ph, sections, set(q_map.keys()), failed_q)

            st.session_state.gen_questions_running = False
            status_ph.success("✅ All questions generated!")
            _time_mod.sleep(0.4)
            st.rerun()

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

    elif st.session_state.step == 3:
        sections   = st.session_state.sections
        ans_map    = st.session_state.section_answers
        q_map      = st.session_state.section_questions
        unanswered = [s for s in sections if s not in ans_map]

        if not unanswered:
            contents   = st.session_state.section_contents
            failed_doc = st.session_state.get("_gen_failed_doc_sections", set())
            done_doc   = len(contents)
            total_sec  = len(sections)

            st.markdown("## ✅ All Sections Answered")
            st.divider()

            progress_ph = st.empty()
            status_ph   = st.empty()
            grid_ph     = st.empty()

            progress_ph.progress(
                done_doc / total_sec if total_sec else 0,
                text=f"{done_doc} / {total_sec} sections drafted"
                     + (f"  ·  {len(failed_doc)} skipped" if failed_doc else ""),
            )
            _render_section_grid(grid_ph, sections, set(contents.keys()), failed_doc)

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
                    progress_ph.progress(
                        done_doc / total_sec,
                        text=f"{done_doc} / {total_sec} sections drafted"
                             + (f"  ·  {len(failed_doc)} skipped" if failed_doc else ""),
                    )
                    _render_section_grid(grid_ph, sections, set(contents.keys()), failed_doc)

                st.session_state.gen_doc_running = False
                status_ph.success("✅ Full draft ready!")
                _time_mod.sleep(0.4)
                st.rerun()

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
            sel = st.radio("Section", active, label_visibility="collapsed", key="sec_radio")

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
