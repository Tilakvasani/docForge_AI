import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from ui.components.generator_form import render_generator_form
from ui.components.library_browser import render_library_browser

st.set_page_config(
    page_title="AI Doc Generator",
    page_icon="📄",
    layout="wide"
)
st.title("📄 AI Document Generator")
st.markdown("Generate industry-ready documents powered by LangChain + Groq")
st.markdown("---")
tab1, tab2 = st.tabs(["🚀 Generate Document", "📚 Document Library"])
with tab1:
    render_generator_form()
with tab2:
    render_library_browser()