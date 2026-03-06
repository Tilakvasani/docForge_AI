import streamlit as st
import httpx

API_URL = "http://localhost:8000/api"

def render_library_browser():
    st.header("📚 Document Library")

    try:
        response = httpx.get(f"{API_URL}/library", timeout=10)
        if response.status_code != 200:
            st.error("Failed to load library")
            return

        data = response.json()
        docs = data.get("documents", [])

        st.markdown(f"**Total Documents:** {data.get('total', 0)}")
        st.markdown("---")

        if not docs:
            st.info("No documents generated yet. Go to Generator tab!")
            return

        col1, col2 = st.columns(2)
        with col1:
            filter_industry = st.selectbox("Filter by Industry", ["All", "telecom", "saas", "healthcare", "finance", "retail"])
        with col2:
            filter_type = st.selectbox("Filter by Type", ["All", "sop", "policy", "proposal", "sow",
                                                           "incident_report", "faq", "business_case",
                                                           "security_policy", "kpi_report", "runbook"])

        filtered = docs
        if filter_industry != "All":
            filtered = [d for d in filtered if d["industry"] == filter_industry]
        if filter_type != "All":
            filtered = [d for d in filtered if d["doc_type"] == filter_type]

        st.markdown(f"**Showing:** {len(filtered)} documents")

        for doc in filtered:
            with st.expander(f"📄 {doc['title']} | {doc['industry']} | {doc['doc_type']}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Version", doc.get("version", "v1.0"))
                col2.metric("Created By", doc.get("created_by", "admin"))
                col3.metric("Published", "✅" if doc.get("published") else "❌")

                st.markdown(doc["content"][:500] + "...")

                if not doc.get("published"):
                    if st.button(f"📓 Publish to Notion", key=doc["doc_id"]):
                        pub = httpx.post(f"{API_URL}/publish", json=doc, timeout=30)
                        if pub.status_code == 200:
                            st.success("Published to Notion!")
                        else:
                            st.error("Failed to publish!")
                else:
                    st.markdown(f"[View in Notion]({doc.get('notion_url', '#')})")

    except Exception as e:
        st.error(f"Error loading library: {e}")
