import streamlit as st
import json
import time as _time_mod
from ui.services.api_client import api_get, api_post
import logging
_log = logging.getLogger("frontend")

try:
    from docx_builder import build_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

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


def render_ragas():
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


