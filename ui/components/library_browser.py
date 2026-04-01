import streamlit as st
import httpx

API_URL = "http://localhost:8000/api"

def render_library_browser():
    try:
        response = httpx.get(f"{API_URL}/library", timeout=15)
        if response.status_code != 200:
            st.error(f"Failed to load library: {response.text}")
            return

        data = response.json()
        docs = data.get("documents", [])
        total = data.get("total", 0)

        col1, col2 = st.columns(2)
        doc_types = ["All"] + sorted(set(d["doc_type"] for d in docs if d.get("doc_type")))
        statuses  = ["All", "Generated", "Draft", "Reviewed", "Archived"]

        with col1:
            ftype = st.selectbox("Type", doc_types)
        with col2:
            fstatus = st.selectbox("Status", statuses)

        filtered = docs
        if ftype   != "All": filtered = [d for d in filtered if d.get("doc_type") == ftype]
        if fstatus != "All": filtered = [d for d in filtered if d.get("status") == fstatus]

        st.markdown(f'<div class="count-line">Showing <b>{len(filtered)}</b> of <b>{total}</b> documents</div>', unsafe_allow_html=True)

        if not filtered:
            st.markdown("""
            <div class="empty">
                <div class="empty-icon">◫</div>
                <div class="empty-text">No documents yet.</div>
            </div>
            """, unsafe_allow_html=True)
            return

        st.markdown('<div class="lib-grid">', unsafe_allow_html=True)
        for doc in filtered:
            status = doc.get("status", "Generated")
            badge_cls = {"Generated": "badge-gen", "Draft": "badge-draft"}.get(status, "badge-rev")
            tags = doc.get("tags", [])
            chips = "".join([f'<span class="chip chip-grey">{t}</span>' for t in tags[:4]])
            wc = doc.get("word_count", 0)
            created = doc.get("created_at", "")[:10] or "—"
            notion_url = doc.get("notion_url", "")
            notion_btn = f'<a href="{notion_url}" target="_blank" class="notion-link">◫ Notion</a>' if notion_url else ""

            st.markdown(f"""
            <div class="lcard">
                <div>
                    <div class="lcard-title">{doc.get('title','Untitled')}</div>
                    <div class="lcard-meta">{doc.get('doc_type','')} · {doc.get('industry','')} · {created} · by {doc.get('created_by','—')}</div>
                    <div class="lcard-chips">{chips}</div>
                </div>
                <div class="lcard-right">
                    <span class="badge {badge_cls}">{status}</span>
                    <div class="lcard-stat">
                        <div class="lcard-stat-n">{wc:,}</div>
                        <div class="lcard-stat-l">words</div>
                    </div>
                    {notion_btn}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    except httpx.ConnectError:
        st.error("Cannot connect to backend. Run: uvicorn backend.main:app --reload")
    except Exception as e:
        st.error(f"Error: {e}")