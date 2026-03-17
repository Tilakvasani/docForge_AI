"""
DocForge AI — streamlit_app.py  v5.0
Steps: 1 Select → 2 Questions → 3 Answer → 4 Review & Edit → 5 Export
Step 4 (Generate Content) is removed — generation runs inside Step 3's button.
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

st.set_page_config(page_title="DocForge AI", page_icon="📄", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
  padding:1.5rem 2rem;border-radius:12px;color:white;margin-bottom:1.5rem}
.step-badge{background:#667eea;color:white;padding:3px 12px;border-radius:20px;
  font-size:13px;font-weight:600;display:inline-block;margin-bottom:.5rem}
.s-done{color:#22c55e;font-size:13px;padding:2px 0}
.s-skip{color:#f59e0b;font-size:13px;padding:2px 0}
.s-pend{color:#94a3b8;font-size:13px;padding:2px 0}
.lib-card{background:#1e293b;border:1px solid #334155;border-radius:8px;
  padding:.75rem 1rem;margin-bottom:.5rem}
</style>
""", unsafe_allow_html=True)


# ─── API helpers ──────────────────────────────────────────────────────────────

def api_get(ep):
    try:
        r = httpx.get(f"{API_URL}{ep}", timeout=30)
        r.raise_for_status(); return r.json()
    except Exception as e:
        st.error(f"API Error: {e}"); return None

def api_post(ep, data, timeout=120):
    try:
        r = httpx.post(f"{API_URL}{ep}", json=data, timeout=timeout)
        r.raise_for_status(); return r.json()
    except Exception as e:
        st.error(f"API Error: {e}"); return None


# ─── Session init ─────────────────────────────────────────────────────────────

def init_session():
    defaults = dict(
        step=1, company_ctx={}, departments=[],
        selected_dept=None, selected_dept_id=None,
        selected_doc_type=None, doc_sec_id=None, sections=[],
        section_questions={}, section_answers={},
        skipped_sections=set(), section_contents={},
        sec_ids_ordered=[], gen_id=None, full_document="",
        active_tab="generate",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📄 DocForge AI")
    st.markdown("---")

    tab = st.radio("", ["📝 Generate", "📚 Library"],
                   label_visibility="collapsed", key="main_tab")
    st.session_state.active_tab = "library" if "Library" in tab else "generate"
    st.markdown("---")

    if st.session_state.active_tab == "generate":
        steps = [
            (1, "Select Document"),
            (2, "Generate Questions"),
            (3, "Answer Questions"),
            (4, "Review & Edit"),
            (5, "Export"),
        ]
        cur = st.session_state.step
        for n, lbl in steps:
            if   n < cur:  ic, co, w = "✅", "#22c55e", "400"
            elif n == cur: ic, co, w = "▶️", "#667eea", "700"
            else:          ic, co, w = "⭕", "#94a3b8", "400"
            st.markdown(
                f'<div style="color:{co};font-weight:{w};padding:3px 0">'
                f'{ic} Step {n}: {lbl}</div>', unsafe_allow_html=True)
        st.markdown("---")

        ctx = st.session_state.company_ctx
        if ctx:
            st.markdown("**🏢 Company**")
            st.caption(ctx.get("company_name", "—"))
            st.caption(f"{ctx.get('industry','—')} · {ctx.get('region','—')}")
        if st.session_state.selected_doc_type:
            st.markdown("**📄 Document**")
            st.caption(st.session_state.selected_dept or "")
            st.caption(st.session_state.selected_doc_type or "")
        if st.session_state.sections:
            done  = len(st.session_state.section_contents)
            skip  = len(st.session_state.skipped_sections)
            total = len(st.session_state.sections)
            active = max(total - skip, 1)
            st.markdown(f"**Sections: {done}/{active}**")
            if skip: st.caption(f"⚠️ {skip} skipped")
        st.markdown("---")
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header"><h2 style="margin:0">📄 DocForge AI</h2>'
    '<p style="margin:4px 0 0;opacity:.85">AI-Powered Enterprise Document Generator</p>'
    '</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LIBRARY TAB
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.active_tab == "library":
    st.markdown("## 📚 Document Library")
    st.markdown("All documents published to your Notion database.")
    st.markdown("---")

    if st.button("🔄 Refresh Library", type="primary"):
        st.session_state["_library_data"] = None

    if "_library_data" not in st.session_state or st.session_state["_library_data"] is None:
        with st.spinner("Loading from Notion..."):
            lib = api_get("/library/notion")
        st.session_state["_library_data"] = lib

    lib = st.session_state.get("_library_data")
    if not lib:
        st.info("Could not load library. Make sure the backend is running.")
    elif lib.get("total", 0) == 0:
        st.info("No documents published yet.")
    else:
        docs = lib["documents"]
        dept_filter = st.selectbox("Filter by Department",
            ["All"] + sorted({d["department"] for d in docs if d["department"]}))
        search = st.text_input("🔍 Search by title", placeholder="Type to filter...")

        filtered = [
            d for d in docs
            if (dept_filter == "All" or d["department"] == dept_filter)
            and (not search or search.lower() in d["title"].lower())
        ]
        st.markdown(f"**{len(filtered)} documents**")
        for doc in filtered:
            with st.container():
                st.markdown('<div class="lib-card">', unsafe_allow_html=True)
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{doc['title']}**")
                    st.caption(f"{doc['doc_type']} · {doc['industry']}")
                with c2:
                    st.caption(f"🏢 {doc['department']}")
                    st.caption(f"📅 {doc['created_at']}")
                with c3:
                    status_color = {
                        "Generated": "🟢", "Draft": "🟡",
                        "Reviewed": "🔵", "Archived": "⚫"
                    }.get(doc["status"], "⚪")
                    st.caption(f"{status_color} {doc['status']}")
                    if doc.get("notion_url"):
                        st.link_button("Open →", doc["notion_url"])
                st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE TAB
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.active_tab == "generate":

    # ─── Step 1: Select Document ───────────────────────────────────────────────

    if st.session_state.step == 1:
        st.markdown('<span class="step-badge">Step 1 of 5</span>', unsafe_allow_html=True)
        st.markdown("## 🚀 Get Started")
        st.markdown("Enter your company details and choose the document to generate.")
        st.markdown("---")

        if not st.session_state.departments:
            with st.spinner("Loading document catalog..."):
                data = api_get("/departments")
                if data: st.session_state.departments = data["departments"]

        depts = st.session_state.departments
        if not depts:
            st.error("❌ Backend not reachable. Run: `uvicorn backend.main:app --reload`")
            st.stop()

        st.markdown("### 🏢 Company Info")
        c1, c2 = st.columns(2)
        with c1:
            company_name = st.text_input(
                "Company Name *",
                value=st.session_state.company_ctx.get("company_name", ""),
                placeholder="e.g. Turabit Technologies")
            industry = st.selectbox("Industry", [
                "Technology / SaaS", "Finance / Banking", "Healthcare",
                "Manufacturing", "Retail / E-Commerce", "Legal Services",
                "Marketing / Media", "Logistics / Supply Chain", "Education", "Other"])
        with c2:
            company_size = st.selectbox("Company Size", [
                "1-10 employees", "11-50 employees", "51-200 employees",
                "201-500 employees", "500+ employees"], index=2)
            region = st.selectbox("Region", [
                "India", "United States", "United Kingdom", "UAE / Middle East",
                "Canada", "Australia", "Europe", "Other"])

        st.markdown("---")
        st.markdown("### 📂 Select Document")
        dept_names = [d["department"] for d in depts]
        c3, c4 = st.columns(2)
        with c3:
            selected_dept = st.selectbox("Department", dept_names)
        dept_data = next((d for d in depts if d["department"] == selected_dept), None)
        doc_types = dept_data["doc_types"] if dept_data else []
        with c4:
            selected_doc_type = st.selectbox("Document Type", doc_types)

        st.markdown("---")
        if st.button("Load Sections & Continue →", type="primary", use_container_width=True):
            if not company_name.strip():
                st.error("Please enter your company name.")
            else:
                with st.spinner("Loading document sections..."):
                    safe = (selected_doc_type.replace("/", "%2F")
                            .replace("(", "%28").replace(")", "%29"))
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
                            seen.add(s); deduped.append(s)
                    st.session_state.sections = deduped
                    st.session_state.step = 2
                    st.rerun()


    # ─── Step 2: Generate Questions ───────────────────────────────────────────

    elif st.session_state.step == 2:
        st.markdown('<span class="step-badge">Step 2 of 5</span>', unsafe_allow_html=True)
        st.markdown("## ❓ Generate Questions")
        st.markdown(
            f"**{st.session_state.selected_doc_type}** · "
            f"{len(st.session_state.sections)} sections")
        st.markdown("*Smart count: 0–3 questions per section based on section type.*")
        st.markdown("---")

        sections = st.session_state.sections
        total    = len(sections)

        grid_slot    = st.empty()
        counter_slot = st.empty()

        def render_grid():
            generated = st.session_state.section_questions
            cols = st.columns(3)
            for i, s in enumerate(sections):
                if s in generated:
                    sec_type = generated[s].get("section_type", "text")
                    badge = {"table": " 📊", "flowchart": " 🔀", "raci": " 👥",
                             "signature": " ✍️"}.get(sec_type, "")
                    cols[i % 3].markdown(
                        f'<div class="s-done">✅ {s[:33]}{badge}</div>',
                        unsafe_allow_html=True)
                else:
                    cols[i % 3].markdown(
                        f'<div class="s-pend">⭕ {s[:35]}</div>',
                        unsafe_allow_html=True)

        def render_counter():
            done = len(st.session_state.section_questions)
            counter_slot.markdown(f"**{done} / {total} ready**")

        with grid_slot.container():
            render_grid()
        render_counter()

        st.markdown("---")

        done = len(st.session_state.section_questions)
        if done < total:
            if st.button("🤖 Generate Questions for All Sections", type="primary",
                         use_container_width=True):
                bar = st.progress(0); status = st.empty()
                for i, sec in enumerate(sections):
                    generated = st.session_state.section_questions
                    if sec in generated:
                        bar.progress((i + 1) / total); continue
                    status.markdown(f"⏳ **{sec}**...")
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
                            "questions":    res["questions"],
                            "section_type": res.get("section_type", "text"),
                        }
                    with grid_slot.container():
                        render_grid()
                    render_counter()
                    bar.progress((i + 1) / total)
                bar.empty(); status.markdown("✅ Done!"); st.rerun()

        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("← Back"):
                st.session_state.step = 1; st.rerun()
        with c2:
            if len(st.session_state.section_questions) == total:
                if st.button("Start Answering →", type="primary", use_container_width=True):
                    st.session_state.step = 3; st.rerun()


    # ─── Step 3: Answer Questions + Generate Document ─────────────────────────

    elif st.session_state.step == 3:
        sections = st.session_state.sections
        ans_map  = st.session_state.section_answers
        skipped  = st.session_state.skipped_sections
        q_map    = st.session_state.section_questions

        unanswered = [s for s in sections if s not in ans_map and s not in skipped]

        # ── All sections answered — show Generate button ───────────────────────
        if not unanswered:
            st.markdown('<span class="step-badge">Step 3 of 5</span>', unsafe_allow_html=True)
            st.markdown("## ✅ All Sections Done!")
            st.markdown(f"**{len(ans_map)} answered · {len(skipped)} skipped**")
            if skipped:
                st.warning(f"⚠️ Skipped: **{', '.join(skipped)}**")
            st.markdown("---")

            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("← Back"):
                    st.session_state.step = 2; st.rerun()
            with c2:
                if st.button("🤖 Generate Document →", type="primary", use_container_width=True):
                    active = [s for s in sections if s not in skipped]
                    total  = len(active)

                    st.markdown("---")
                    bar    = st.progress(0)
                    status = st.empty()
                    ids    = []

                    for i, sec in enumerate(active):
                        # Skip already-generated sections
                        if sec in st.session_state.section_contents:
                            ids.append(q_map.get(sec, {}).get("sec_id", 0))
                            bar.progress((i + 1) / total)
                            continue

                        status.markdown(f"✍️ Writing **{sec}**...")
                        q_data = q_map.get(sec, {})
                        sec_id = q_data.get("sec_id")

                        res = api_post("/section/generate", {
                            "sec_id":          sec_id,
                            "doc_sec_id":      st.session_state.doc_sec_id,
                            "doc_id":          st.session_state.selected_dept_id,
                            "section_name":    sec,
                            "doc_type":        st.session_state.selected_doc_type,
                            "department":      st.session_state.selected_dept,
                            "company_context": st.session_state.company_ctx,
                            "num_sections":    total,
                        }, timeout=120)

                        if res:
                            st.session_state.section_contents[sec] = res["content"]
                            ids.append(sec_id)

                        bar.progress((i + 1) / total)

                    st.session_state.sec_ids_ordered = ids
                    status.markdown("✅ All sections written!")

                    # Assemble full document
                    lines = []
                    for sec in active:
                        c = st.session_state.section_contents.get(sec, "").strip()
                        if c:
                            lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
                    full_doc = "\n".join(lines).strip()

                    # Save to DB
                    pri_sec  = ids[-1] if ids else 0
                    save_res = api_post("/document/save", {
                        "doc_id":          st.session_state.selected_dept_id,
                        "doc_sec_id":      st.session_state.doc_sec_id,
                        "sec_id":          pri_sec,
                        "gen_doc_sec_dec": list(st.session_state.section_contents.values()),
                        "gen_doc_full":    full_doc,
                    })
                    st.session_state.gen_id        = save_res.get("gen_id", 0) if save_res else 0
                    st.session_state.full_document = full_doc

                    # Jump straight to Review & Edit
                    st.session_state.step = 4
                    st.rerun()

        # ── Still answering sections ───────────────────────────────────────────
        else:
            current  = unanswered[0]
            done_cnt = len(ans_map) + len(skipped)
            total    = len(sections)

            st.markdown('<span class="step-badge">Step 3 of 5</span>', unsafe_allow_html=True)
            st.markdown("## ✍️ Answer Questions")
            st.markdown(f"**{done_cnt} / {total} done** · {len(unanswered)} remaining")
            st.progress(done_cnt / total)
            st.markdown("---")

            q_data    = q_map.get(current, {})
            questions = q_data.get("questions", [])
            sec_id    = q_data.get("sec_id")
            sec_type  = q_data.get("section_type", "text")

            # Section type badge
            type_labels = {
                "table":     "📊 This section will be a **data table**. Your answers populate the rows.",
                "flowchart": "🔀 This section will be a **process flowchart**. Describe the steps.",
                "raci":      "👥 This section will be a **RACI matrix**. Describe the roles involved.",
                "signature": "✍️ This section is a **sign-off block**. Auto-generated.",
                "text":      "",
            }
            type_msg = type_labels.get(sec_type, "")

            st.markdown(f"### 📌 {current}")
            if type_msg:
                st.info(type_msg)

            user_answers = []
            if not questions:
                st.info("No questions needed — content will be auto-generated professionally.")
            else:
                st.caption("Leave blank = auto-fill with professional content")
                for i, q in enumerate(questions):
                    a = st.text_area(
                        f"**Q{i+1}:** {q}",
                        key=f"ans_{current}_{i}",
                        height=90,
                        placeholder="Your answer (or leave blank for auto-fill)...")
                    user_answers.append(a)

            st.markdown("---")
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("⏭️ Skip", use_container_width=True):
                    if sec_id:
                        api_post("/answers/save", {
                            "sec_id":      sec_id,
                            "doc_sec_id":  st.session_state.doc_sec_id,
                            "doc_id":      st.session_state.selected_dept_id,
                            "section_name": current,
                            "questions":   questions,
                            "answers":     ["not answered"] * max(len(questions), 1),
                        })
                    st.session_state.skipped_sections.add(current)
                    st.rerun()
            with c2:
                if st.button("Save & Next →", type="primary", use_container_width=True):
                    filled = [a.strip() if a.strip() else "not answered" for a in user_answers]
                    if sec_id:
                        api_post("/answers/save", {
                            "sec_id":      sec_id,
                            "doc_sec_id":  st.session_state.doc_sec_id,
                            "doc_id":      st.session_state.selected_dept_id,
                            "section_name": current,
                            "questions":   questions,
                            "answers":     filled or ["not answered"],
                        })
                    st.session_state.section_answers[current] = filled
                    st.rerun()

            # Progress summary at bottom
            if ans_map or skipped:
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    if ans_map:
                        st.markdown("**✅ Answered**")
                        for s in sections:
                            if s in ans_map:
                                st.markdown(
                                    f'<div class="s-done">✅ {s}</div>',
                                    unsafe_allow_html=True)
                with c2:
                    if skipped:
                        st.markdown("**⏭️ Skipped**")
                        for s in skipped:
                            st.markdown(
                                f'<div class="s-skip">⏭️ {s}</div>',
                                unsafe_allow_html=True)


    # ─── Step 4: Review & Edit ────────────────────────────────────────────────

    elif st.session_state.step == 4:
        st.markdown('<span class="step-badge">Step 4 of 5</span>', unsafe_allow_html=True)
        st.markdown("## 🔍 Review & Edit")
        st.markdown("---")

        skipped  = st.session_state.skipped_sections
        active   = [s for s in st.session_state.sections if s not in skipped]
        contents = st.session_state.section_contents

        def rebuild_doc():
            lines = []
            for sec in active:
                c = contents.get(sec, "").strip()
                if c:
                    lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
            st.session_state.full_document = "\n".join(lines).strip()

        cl, cr = st.columns([1, 2])
        with cl:
            st.markdown("### 📋 Sections")
            sel = st.radio("", active, label_visibility="collapsed", key="sec_radio")

        with cr:
            cur      = contents.get(sel, "")
            sec_type = st.session_state.section_questions.get(sel, {}).get("section_type", "text")

            type_icons = {
                "table": "📊", "flowchart": "🔀",
                "raci": "👥", "signature": "✍️", "text": "✏️"
            }
            st.markdown(f"### {type_icons.get(sec_type, '✏️')} {sel}")

            with st.expander("📄 Current Content", expanded=True):
                st.text(cur or "(no content)")

            st.markdown("---")
            instr = st.text_area(
                "🤖 AI edit instruction",
                placeholder="e.g. Make more formal · Add detail · Shorten · Legal tone",
                height=65, key="edit_instr")

            c1, c2 = st.columns(2)
            with c1:
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
                            rebuild_doc()
                            st.success("✅ Updated!")
                            st.rerun()
            with c2:
                manual = st.text_area(
                    "📝 Manual edit:",
                    value=cur, height=200,
                    key=f"manual_{sel}")
                if st.button("💾 Save Manual", use_container_width=True, key=f"save_{sel}"):
                    st.session_state.section_contents[sel] = manual
                    rebuild_doc()
                    st.success("✅ Saved!")
                    st.rerun()

        st.markdown("---")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("← Back"):
                st.session_state.step = 3; st.rerun()
        with c2:
            if st.button("Export →", type="primary", use_container_width=True):
                st.session_state.step = 5; st.rerun()


    # ─── Step 5: Export ───────────────────────────────────────────────────────

    elif st.session_state.step == 5:
        st.markdown('<span class="step-badge">Step 5 of 5</span>', unsafe_allow_html=True)
        st.markdown("## 💾 Export")
        st.markdown("---")

        ctx      = st.session_state.company_ctx
        doc_type = st.session_state.selected_doc_type
        full_doc = st.session_state.full_document
        skipped  = st.session_state.skipped_sections
        active   = [s for s in st.session_state.sections if s not in skipped]
        contents = st.session_state.section_contents

        if not full_doc:
            st.error("No document found — go back to Step 3.")
            if st.button("← Step 3"):
                st.session_state.step = 3; st.rerun()
            st.stop()

        st.success(f"✅ **{doc_type}** ready for export!")
        st.markdown(f"""| | |
|---|---|
| Document | `{doc_type}` |
| Department | `{st.session_state.selected_dept}` |
| Company | `{ctx.get('company_name','—')}` |
| Industry | `{ctx.get('industry','—')}` |
| Sections | `{len(active)}` active · `{len(skipped)}` skipped |
""")
        st.markdown("---")

        # ── Publish to Notion ──────────────────────────────────────────────────
        st.markdown("### 📓 Publish to Notion")
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
                st.success("✅ Published to Notion!")
                if url:
                    st.link_button("🔗 Open in Notion", url)

        st.markdown("---")

        # ── Downloads ──────────────────────────────────────────────────────────
        st.markdown("### 📥 Download")
        safe = (doc_type.replace(" ", "_").replace("/", "-")
                        .replace("(", "").replace(")", ""))

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**📄 Word Document (.docx)**")
            if not DOCX_AVAILABLE:
                st.warning("docx_builder.py not found.")
            else:
                if ("docx_bytes_cache" not in st.session_state or
                        st.session_state.get("docx_cache_doc") != doc_type):
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
                        st.error(f"DOCX build error: {e}")
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

        with c2:
            st.markdown("**📃 Plain Text (.txt)**")
            st.download_button(
                "⬇️ Download .txt",
                data=full_doc,
                file_name=f"{safe}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### 📄 Document Preview")
        with st.expander("View Full Document", expanded=False):
            st.text(full_doc)

        st.markdown("---")
        if st.button("➕ Create Another Document", type="primary", use_container_width=True):
            saved_ctx   = st.session_state.company_ctx
            saved_depts = st.session_state.departments
            for k in list(st.session_state.keys()): del st.session_state[k]
            init_session()
            st.session_state["company_ctx"]  = saved_ctx
            st.session_state["departments"]  = saved_depts
            st.session_state["step"]         = 1
            st.rerun()