"""
DocForge AI — streamlit_app.py  v3.0
Fixes applied:
  ✅ Industry-standard document lengths (calibrated per doc type)
  ✅ Plain text storage — no markdown anywhere
  ✅ Plain text in Notion pages
  ✅ Download: .docx (Word) + .txt (plain text)
  ✅ Library tab shows all Notion docs with live links
  ✅ Notion: Industry + Doc Type filled (no Tags)
  ✅ Smart questions 0-3 per section
  ✅ Empty answer → "not answered"
  ✅ Skip section → excluded from doc
  ✅ Steps 1+2 merged
  ✅ gen_doc saves plain text full doc
"""
import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import httpx
import json

# Direct DOCX builder — no backend roundtrip needed
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
.s-done{color:#22c55e;font-size:13px}
.s-skip{color:#f59e0b;font-size:13px}
.s-pend{color:#94a3b8;font-size:13px}
.lib-card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
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

    # Tab switch
    tab = st.radio("", ["📝 Generate", "📚 Library"],
                   label_visibility="collapsed",
                   key="main_tab")
    st.session_state.active_tab = "library" if "Library" in tab else "generate"

    st.markdown("---")

    if st.session_state.active_tab == "generate":
        steps = [(1,"Select Document"),(2,"Generate Questions"),
                 (3,"Answer Questions"),(4,"Generate Content"),
                 (5,"Review & Edit"),(6,"Export")]
        cur = st.session_state.step
        for n, lbl in steps:
            if   n < cur:  ic,co,w = "✅","#22c55e","400"
            elif n == cur: ic,co,w = "▶️","#667eea","700"
            else:          ic,co,w = "⭕","#94a3b8","400"
            st.markdown(f'<div style="color:{co};font-weight:{w};padding:3px 0">'
                        f'{ic} Step {n}: {lbl}</div>', unsafe_allow_html=True)
        st.markdown("---")
        ctx = st.session_state.company_ctx
        if ctx:
            st.markdown("**🏢 Company**")
            st.caption(ctx.get("company_name","—"))
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
            st.progress(done / active)
        st.markdown("---")
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h2 style="margin:0">📄 DocForge AI</h2>'
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
        st.info("No documents published yet. Generate and publish a document to see it here.")
    else:
        docs = lib["documents"]
        st.markdown(f"**{lib['total']} documents found**")
        st.markdown("")

        # Filter controls
        fc1, fc2 = st.columns(2)
        all_depts = sorted(set(d.get("department","") for d in docs if d.get("department")))
        with fc1:
            dept_filter = st.selectbox("Filter by Department",
                                        ["All"] + all_depts, key="lib_dept_filter")
        with fc2:
            search = st.text_input("Search by title", placeholder="e.g. Employee Offer...",
                                    key="lib_search")

        filtered = docs
        if dept_filter != "All":
            filtered = [d for d in filtered if d.get("department") == dept_filter]
        if search:
            filtered = [d for d in filtered if search.lower() in d.get("title","").lower()]

        st.markdown(f"*Showing {len(filtered)} of {len(docs)}*")
        st.markdown("")

        for doc in filtered:
            with st.container():
                c1, c2 = st.columns([4, 1])
                with c1:
                    title    = doc.get("title", "Untitled")
                    dept     = doc.get("department", "")
                    doc_type = doc.get("doc_type", "")
                    industry = doc.get("industry", "")
                    status   = doc.get("status", "")
                    created  = doc.get("created_at", "")
                    url      = doc.get("notion_url", "")

                    dept_badge = f"**{dept}**" if dept else ""
                    st.markdown(f"#### {title}")
                    meta_parts = [p for p in [dept_badge, doc_type, industry, created] if p]
                    st.markdown(" · ".join(meta_parts))
                    if status:
                        color = {"Generated":"🟢","Reviewed":"🔵","Draft":"🟡","Archived":"⚫"}.get(status,"⚪")
                        st.markdown(f"{color} {status}")
                with c2:
                    if doc.get("notion_url"):
                        st.link_button("🔗 Open in Notion", url, use_container_width=True)

                st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #e2e8f0'>",
                            unsafe_allow_html=True)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE TAB — STEP WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Step 1: Company Info + Select Document (MERGED) ─────────────────────────

if st.session_state.step == 1:
    st.markdown('<span class="step-badge">Step 1 of 6</span>', unsafe_allow_html=True)
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
        company_name = st.text_input("Company Name *",
            value=st.session_state.company_ctx.get("company_name",""),
            placeholder="e.g. Turabit Technologies")
        industry = st.selectbox("Industry", [
            "Technology / SaaS","Finance / Banking","Healthcare","Manufacturing",
            "Retail / E-Commerce","Legal Services","Marketing / Media",
            "Logistics / Supply Chain","Education","Other"])
    with c2:
        company_size = st.selectbox("Company Size",[
            "1-10 employees","11-50 employees","51-200 employees",
            "201-500 employees","500+ employees"], index=2)
        region = st.selectbox("Region",[
            "India","United States","United Kingdom","UAE / Middle East",
            "Canada","Australia","Europe","Other"])

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
                safe = (selected_doc_type.replace("/","%2F")
                        .replace("(","%28").replace(")","%29"))
                data = api_get(f"/sections/{safe}")
            if data:
                st.session_state.company_ctx = {
                    "company_name": company_name.strip(), "industry": industry,
                    "company_size": company_size, "region": region}
                st.session_state.selected_dept     = selected_dept
                st.session_state.selected_dept_id  = dept_data["doc_id"]
                st.session_state.selected_doc_type = selected_doc_type
                st.session_state.doc_sec_id        = data["doc_sec_id"]
                st.session_state.sections          = data["doc_sec"]
                st.session_state.step = 2
                st.rerun()


# ─── Step 2: Generate Questions ───────────────────────────────────────────────

elif st.session_state.step == 2:
    st.markdown('<span class="step-badge">Step 2 of 6</span>', unsafe_allow_html=True)
    st.markdown(f"## ❓ Generate Questions")
    st.markdown(f"**{st.session_state.selected_doc_type}** · {len(st.session_state.sections)} sections")
    st.markdown("*Smart count: 0–3 questions per section based on what's needed.*")
    st.markdown("---")

    sections  = st.session_state.sections
    generated = st.session_state.section_questions
    total     = len(sections)
    done      = len(generated)

    cols = st.columns(3)
    for i, s in enumerate(sections):
        cls = "s-done" if s in generated else "s-pend"
        ic  = "✅" if s in generated else "⭕"
        cols[i % 3].markdown(f'<div class="{cls}">{ic} {s[:38]}</div>',
                             unsafe_allow_html=True)
    st.markdown("---")
    st.progress(done / total)
    st.markdown(f"**{done} / {total} ready**")

    if done < total:
        if st.button("🤖 Generate Questions for All Sections", type="primary",
                     use_container_width=True):
            bar = st.progress(0); status = st.empty()
            for i, sec in enumerate(sections):
                if sec in generated:
                    bar.progress((i+1)/total); continue
                status.markdown(f"⏳ **{sec}**...")
                res = api_post("/questions/generate", {
                    "doc_sec_id":      st.session_state.doc_sec_id,
                    "doc_id":          st.session_state.selected_dept_id,
                    "section_name":    sec,
                    "doc_type":        st.session_state.selected_doc_type,
                    "department":      st.session_state.selected_dept,
                    "company_context": st.session_state.company_ctx})
                if res:
                    st.session_state.section_questions[sec] = {
                        "sec_id": res["sec_id"], "questions": res["questions"]}
                bar.progress((i+1)/total)
            status.markdown("✅ Done!"); st.rerun()

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("← Back"): st.session_state.step = 1; st.rerun()
    with c2:
        if done == total:
            if st.button("Start Answering →", type="primary", use_container_width=True):
                st.session_state.step = 3; st.rerun()


# ─── Step 3: Answer Questions ─────────────────────────────────────────────────

elif st.session_state.step == 3:
    sections  = st.session_state.sections
    ans_map   = st.session_state.section_answers
    skipped   = st.session_state.skipped_sections
    q_map     = st.session_state.section_questions

    unanswered = [s for s in sections if s not in ans_map and s not in skipped]

    if not unanswered:
        st.markdown('<span class="step-badge">Step 3 of 6</span>', unsafe_allow_html=True)
        st.markdown("## ✅ All Sections Done!")
        st.markdown(f"**{len(ans_map)} answered · {len(skipped)} skipped**")
        if skipped:
            st.warning(f"⚠️ Skipped sections removed from document: **{', '.join(skipped)}**")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("← Back"): st.session_state.step = 2; st.rerun()
        with c2:
            if st.button("Generate Document →", type="primary", use_container_width=True):
                st.session_state.step = 4; st.rerun()
    else:
        current  = unanswered[0]
        done_cnt = len(ans_map) + len(skipped)
        total    = len(sections)

        st.markdown('<span class="step-badge">Step 3 of 6</span>', unsafe_allow_html=True)
        st.markdown("## ✍️ Answer Questions")
        st.progress(done_cnt / total)
        st.markdown(f"**{done_cnt} / {total} done** · {len(unanswered)} remaining")
        st.markdown("---")
        st.markdown(f"### 📌 {current}")

        q_data    = q_map.get(current, {})
        questions = q_data.get("questions", [])
        sec_id    = q_data.get("sec_id")

        user_answers = []
        if not questions:
            st.info("No questions for this section — content will be auto-generated.")
        else:
            st.markdown("*Leave blank = auto-fill with professional content*")
            for i, q in enumerate(questions):
                a = st.text_area(f"**Q{i+1}:** {q}", key=f"a_{current}_{i}",
                                  height=72, placeholder="Your answer (or leave blank)...")
                user_answers.append(a)

        st.markdown("---")
        c1, c2 = st.columns([1, 3])

        with c1:
            if st.button("⏭️ Skip", use_container_width=True):
                if sec_id:
                    api_post("/answers/save", {
                        "sec_id": sec_id, "doc_sec_id": st.session_state.doc_sec_id,
                        "doc_id": st.session_state.selected_dept_id,
                        "section_name": current, "questions": questions,
                        "answers": ["__skipped__"] * len(questions)})
                st.session_state.skipped_sections.add(current); st.rerun()

        with c2:
            if st.button("Save & Next →", type="primary", use_container_width=True):
                filled = [a.strip() if a.strip() else "not answered" for a in user_answers]
                if sec_id:
                    api_post("/answers/save", {
                        "sec_id": sec_id, "doc_sec_id": st.session_state.doc_sec_id,
                        "doc_id": st.session_state.selected_dept_id,
                        "section_name": current, "questions": questions, "answers": filled})
                st.session_state.section_answers[current] = filled; st.rerun()

        if ans_map or skipped:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if ans_map:
                    st.markdown("**✅ Answered**")
                    for s in sections:
                        if s in ans_map:
                            st.markdown(f'<div class="s-done">✅ {s}</div>',
                                        unsafe_allow_html=True)
            with c2:
                if skipped:
                    st.markdown("**⏭️ Skipped (removed)**")
                    for s in skipped:
                        st.markdown(f'<div class="s-skip">⏭️ {s}</div>',
                                    unsafe_allow_html=True)


# ─── Step 4: Generate Content + Assemble ──────────────────────────────────────

elif st.session_state.step == 4:
    st.markdown('<span class="step-badge">Step 4 of 6</span>', unsafe_allow_html=True)
    st.markdown("## ⚙️ Generate Document Content")
    st.markdown("---")

    sections  = st.session_state.sections
    skipped   = st.session_state.skipped_sections
    active    = [s for s in sections if s not in skipped]
    q_map     = st.session_state.section_questions
    contents  = st.session_state.section_contents
    total     = len(active)
    done      = len(contents)

    st.progress(done / max(total, 1))
    st.markdown(f"**{done} / {total} sections written** · {len(skipped)} excluded")

    if done < total:
        if st.button("🤖 Write All Sections", type="primary", use_container_width=True):
            bar = st.progress(0); status = st.empty(); ids = []

            for i, sec in enumerate(active):
                if sec in contents:
                    ids.append(q_map.get(sec, {}).get("sec_id", 0))
                    bar.progress((i+1)/total); continue
                status.markdown(f"✍️ **{sec}**...")
                q_data = q_map.get(sec, {}); sec_id = q_data.get("sec_id")

                res = api_post("/section/generate", {
                    "sec_id":          sec_id,
                    "doc_sec_id":      st.session_state.doc_sec_id,
                    "doc_id":          st.session_state.selected_dept_id,
                    "section_name":    sec,
                    "doc_type":        st.session_state.selected_doc_type,
                    "department":      st.session_state.selected_dept,
                    "company_context": st.session_state.company_ctx,
                    "num_sections":    total,       # pass total for length calibration
                }, timeout=120)

                if res:
                    st.session_state.section_contents[sec] = res["content"]
                    ids.append(sec_id)
                bar.progress((i+1)/total)

            st.session_state.sec_ids_ordered = ids
            status.markdown("✅ Done!"); st.rerun()

    # Assemble plain text full document
    contents = st.session_state.section_contents
    done     = len(contents)

    if done == total and total > 0 and not st.session_state.full_document:
        st.markdown("---")
        if st.button("🔧 Assemble Full Document", type="primary", use_container_width=True):
            with st.spinner("Assembling..."):
                ctx = st.session_state.company_ctx
                doc_type = st.session_state.selected_doc_type
                lines = []

                for sec in active:
                    content = contents.get(sec, "").strip()
                    if not content: continue
                    lines += [sec.upper(), "-" * len(sec), "", content, "", ""]

                full_doc = "\n".join(lines).strip()

                # Save to gen_doc
                pri_sec = (st.session_state.sec_ids_ordered[-1]
                           if st.session_state.sec_ids_ordered else 0)
                res = api_post("/document/save", {
                    "doc_id":          st.session_state.selected_dept_id,
                    "doc_sec_id":      st.session_state.doc_sec_id,
                    "sec_id":          pri_sec,
                    "gen_doc_sec_dec": list(contents.values()),
                    "gen_doc_full":    full_doc,
                })
                st.session_state.gen_id       = res.get("gen_id", 0) if res else 0
                st.session_state.full_document = full_doc
            st.rerun()

    if st.session_state.full_document:
        st.success(f"✅ Document assembled! **gen_id: {st.session_state.gen_id}**")
        with st.expander("👁️ Preview", expanded=False):
            st.text(st.session_state.full_document[:1500] + "\n\n...(truncated)")
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("← Back"): st.session_state.step = 3; st.rerun()
        with c2:
            if st.button("Review & Edit →", type="primary", use_container_width=True):
                st.session_state.step = 5; st.rerun()


# ─── Step 5: Review & Edit ────────────────────────────────────────────────────

elif st.session_state.step == 5:
    st.markdown('<span class="step-badge">Step 5 of 6</span>', unsafe_allow_html=True)
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
        sel = st.radio("", active, label_visibility="collapsed")
    with cr:
        st.markdown(f"### ✏️ {sel}")
        cur = contents.get(sel, "")

        with st.expander("📄 Current Content", expanded=True):
            st.text(cur or "(no content)")

        st.markdown("---")
        instr = st.text_area("🤖 AI edit instruction",
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
                            "sec_id":           st.session_state.section_questions.get(sel,{}).get("sec_id",0),
                            "section_name":     sel,
                            "current_content":  cur,
                            "edit_instruction": instr}, timeout=120)
                    if res:
                        st.session_state.section_contents[sel] = res["updated_content"]
                        rebuild_doc(); st.success("✅ Updated!"); st.rerun()
        with c2:
            manual = st.text_area("📝 Manual edit:", value=cur, height=200, key="manual_txt")
            if st.button("💾 Save Manual", use_container_width=True):
                st.session_state.section_contents[sel] = manual
                rebuild_doc(); st.success("Saved!"); st.rerun()

    st.markdown("---")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("← Back"): st.session_state.step = 4; st.rerun()
    with c2:
        if st.button("Export →", type="primary", use_container_width=True):
            st.session_state.step = 6; st.rerun()


# ─── Step 6: Export ───────────────────────────────────────────────────────────

elif st.session_state.step == 6:
    st.markdown('<span class="step-badge">Step 6 of 6</span>', unsafe_allow_html=True)
    st.markdown("## 💾 Export")
    st.markdown("---")

    ctx      = st.session_state.company_ctx
    doc_type = st.session_state.selected_doc_type
    full_doc = st.session_state.full_document
    skipped  = st.session_state.skipped_sections
    active   = [s for s in st.session_state.sections if s not in skipped]
    contents = st.session_state.section_contents

    if not full_doc:
        st.error("No document found — go back to Step 4 to assemble.")
        if st.button("← Step 4"): st.session_state.step = 4; st.rerun()
        st.stop()

    st.success(f"✅ **{doc_type}** ready!")
    st.markdown(f"""| | |
|---|---|
| Document | `{doc_type}` |
| Department | `{st.session_state.selected_dept}` |
| Company | `{ctx.get('company_name','—')}` |
| Industry | `{ctx.get('industry','—')}` |
| Active Sections | `{len(active)}` (skipped: `{len(skipped)}`) |
| gen_id | `{st.session_state.gen_id}` |
""")

    st.markdown("---")

    # ── Publish to Notion ──────────────────────────────────────────────────────
    st.markdown("### 📓 Publish to Notion")
    if st.button("🚀 Publish to Notion", type="primary", use_container_width=True):
        with st.spinner("Publishing..."):
            res = api_post("/document/publish", {
                "gen_id":          st.session_state.gen_id or 0,
                "doc_type":        doc_type,
                "department":      st.session_state.selected_dept,
                "gen_doc_full":    full_doc,         # plain text
                "company_context": ctx})
        if res:
            url = res.get("notion_url","")
            st.success("✅ Published to Notion!")
            if url:
                st.markdown(f"[🔗 Open in Notion]({url})")

    st.markdown("---")

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("### 📥 Download")
    safe = (doc_type.replace(" ","_").replace("/","-")
                    .replace("(","").replace(")",""))

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Word Document (.docx)**")
        if not DOCX_AVAILABLE:
            st.warning("docx_builder.py not found in project root.")
        elif st.button("⬇️ Generate & Download .docx", use_container_width=True, type="primary"):
            with st.spinner("Building Word document..."):
                sections_data = [
                    {"name": sec, "content": contents.get(sec, "")}
                    for sec in active if contents.get(sec)
                ]
                try:
                    docx_bytes = build_docx(
                        doc_type=doc_type,
                        department=st.session_state.selected_dept,
                        company_name=ctx.get("company_name", "Company"),
                        industry=ctx.get("industry", ""),
                        region=ctx.get("region", ""),
                        sections=sections_data,
                    )
                    st.download_button(
                        "📄 Click to download .docx",
                        data=docx_bytes,
                        file_name=f"{safe}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"DOCX error: {e}")

    with c2:
        st.markdown("**Plain Text (.txt)**")
        st.download_button(
            "⬇️ Download .txt",
            data=full_doc,                      # already plain text
            file_name=f"{safe}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("---")

    # ── Preview ────────────────────────────────────────────────────────────────
    st.markdown("### 📄 Document Preview")
    with st.expander("View Full Document (plain text)", expanded=False):
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