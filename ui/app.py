import streamlit as st
import logging
import sys
from pathlib import Path

# Add project root to sys.path so 'ui' and 'backend' imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.logger import _setup_logging
_setup_logging()

_log = logging.getLogger("frontend")

st.set_page_config(
    page_title="DocForge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ui.utils.session import init_session
from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat
from ui.components.generate import render_generate
from ui.components.library import render_library
from ui.components.ragas import render_ragas
from ui.components.tickets import render_tickets

def main():
    init_session()
    
    active_tab = render_sidebar()
    
    if active_tab == "ask":
        render_chat()
    elif active_tab == "generate":
        render_generate()
    elif active_tab == "library":
        render_library()
    elif active_tab == "ragas":
        render_ragas()
    elif active_tab == "agent":
        render_tickets()

if __name__ == "__main__":
    main()
