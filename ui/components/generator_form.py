import streamlit as st
import httpx

API_URL = "http://localhost:8000/api"

def render_generator_form():
    st.header("📄 Generate Document")

    with st.form("doc_form"):
        title = st.text_input("Document Title", placeholder="e.g. Data Retention Policy")

        col1, col2 = st.columns(2)
        with col1:
            industry = st.selectbox("Industry", [
                "telecom", "saas", "healthcare", "finance", "retail"
            ])
        with col2:
            doc_type = st.selectbox("Document Type", [
                "sop", "policy", "proposal", "sow", "incident_report",
                "faq", "business_case", "security_policy", "kpi_report", "runbook"
            ])

        description = st.text_area("Description (optional)", placeholder="Brief context...")
        tags = st.text_input("Tags (comma separated)", placeholder="e.g. compliance, security")
        created_by = st.text_input("Created By", value="admin")

        submitted = st.form_submit_button("🚀 Generate Document")

    if submitted:
        if not title:
            st.error("Please enter a document title!")
            return

        with st.spinner("Generating document..."):
            try:
                response = httpx.post(f"{API_URL}/generate", json={
                    "title": title,
                    "industry": industry,
                    "doc_type": doc_type,
                    "description": description,
                    "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    "created_by": created_by
                }, timeout=60)

                if response.status_code == 200:
                    doc = response.json()
                    st.success("✅ Document generated successfully!")
                    st.session_state["last_doc"] = doc

                    st.subheader(doc["title"])
                    st.markdown(f"**Industry:** {doc['industry']} | **Type:** {doc['doc_type']} | **Version:** {doc['version']}")
                    st.markdown("---")
                    st.markdown(doc["content"])

                    if st.button("📓 Publish to Notion"):
                        pub = httpx.post(f"{API_URL}/publish", json=doc, timeout=30)
                        if pub.status_code == 200:
                            st.success(f"Published! {pub.json().get('notion_url', '')}")
                        else:
                            st.error("Publish failed!")
                else:
                    st.error(f"Generation failed: {response.text}")

            except Exception as e:
                st.error(f"Error: {e}")
