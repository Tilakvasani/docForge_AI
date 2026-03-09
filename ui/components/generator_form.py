import streamlit as st
import httpx

API_URL = "http://localhost:8000/api"

def render_generator_form():
    st.markdown('<div class="panel"><div class="panel-label">Document Configuration</div>', unsafe_allow_html=True)

    with st.form("doc_form"):
        title = st.text_input("Document Title", placeholder="e.g. Master Service Agreement v2.0")

        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Industry", value="SaaS", disabled=True)
        with col2:
            doc_type = st.selectbox("Document Type", [
                "NDA", "Privacy Policy", "Terms of Service", "Employment Contract",
                "SLA", "Business Proposal", "Technical Spec", "Project Charter",
                "Risk Assessment", "Compliance Report", "Invoice Template",
                "Partnership Agreement"
            ])

        description = st.text_area("Context", placeholder="Describe the purpose, parties involved, or specific clauses needed...", height=90)

        col3, col4 = st.columns(2)
        with col3:
            tags = st.text_input("Tags", placeholder="gdpr, b2b, enterprise")
        with col4:
            created_by = st.text_input("Author", value="admin")

        submitted = st.form_submit_button("◈  Generate Document")

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if not title:
            st.error("Please enter a document title.")
            return

        with st.spinner("Generating..."):
            try:
                response = httpx.post(f"{API_URL}/generate", json={
                    "title": title,
                    "industry": "SaaS",
                    "doc_type": doc_type,
                    "description": description,
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    "created_by": created_by
                }, timeout=60)

                if response.status_code == 200:
                    doc = response.json()
                    st.session_state["last_doc"] = doc
                    st.session_state["published"] = False
                    st.success("Document generated successfully.")
                else:
                    st.error(f"Generation failed: {response.text}")
                    return
            except Exception as e:
                st.error(f"Error: {e}")
                return

    if st.session_state.get("last_doc"):
        doc = st.session_state["last_doc"]
        tags_list = doc.get("tags", [])
        chips = "".join([f'<span class="chip chip-grey">{t}</span>' for t in tags_list])

        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-head">
                <div class="doc-title-text">{doc.get('title','')}</div>
                <div class="doc-chips">
                    <span class="chip chip-rose">{doc.get('doc_type','')}</span>
                    <span class="chip chip-amber">{doc.get('industry','')}</span>
                    <span class="chip chip-teal">v{doc.get('version','1.0')}</span>
                </div>
            </div>
            <div class="doc-body">{doc.get('content','')}</div>
            <div class="doc-footer">
                <div class="doc-chips">{chips}</div>
                <span class="pub-note">by {doc.get('created_by','admin')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not st.session_state.get("published"):
            if st.button("◫  Publish to Notion"):
                with st.spinner("Publishing..."):
                    try:
                        pub = httpx.post(f"{API_URL}/publish", json=doc, timeout=30)
                        if pub.status_code == 200:
                            notion_url = pub.json().get("notion_url", "")
                            st.session_state["published"] = True
                            st.session_state["notion_url"] = notion_url
                            st.success("Published to Notion.")
                            st.rerun()
                        else:
                            st.error(f"Publish failed: {pub.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            notion_url = st.session_state.get("notion_url", "#")
            st.markdown(f'<a href="{notion_url}" target="_blank" class="notion-link">◫ View in Notion →</a>', unsafe_allow_html=True)
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("◈  New Document"):
                st.session_state.pop("last_doc", None)
                st.session_state.pop("published", None)
                st.session_state.pop("notion_url", None)
                st.rerun()