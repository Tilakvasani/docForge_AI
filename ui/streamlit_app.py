import streamlit as st

st.set_page_config(
    page_title="DocForge AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from ui.components.generator_form import render_generator_form
from ui.components.library_browser import render_library_browser

# ── Minimal CSS — only safe overrides ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

[data-testid="stAppViewContainer"] { background: #0a0a0a; }
[data-testid="stHeader"]           { display: none; }
[data-testid="stSidebar"]          { display: none; }
.block-container                   { padding-top: 2rem; padding-bottom: 2rem; }
footer                             { display: none; }

/* Buttons */
.stButton > button {
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #242424;
    border-color: #d4a64a;
    color: #d4a64a;
}
.stButton > button[kind="primary"] {
    background: #d4a64a;
    color: #0a0a0a;
    border: none;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background: #e0b55a;
    color: #0a0a0a;
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
}
.stSelectbox > div > div {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    color: #e0e0e0;
}

/* Labels */
label { color: #666 !important; font-size: 0.8rem !important; }

/* Metric */
[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #1e1e1e;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #d4a64a; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #555; font-size: 0.75rem !important; }

/* Divider */
hr { border-color: #1e1e1e !important; margin: 1.5rem 0; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "generate"


# ══════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "landing":

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hero ──────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:

        # Badge
        st.markdown(
            '<p style="color:#d4a64a; font-size:0.75rem; letter-spacing:0.12em; '
            'font-family:monospace; margin-bottom:0.5rem;">● POWERED BY LLAMA 3.3 · 70B</p>',
            unsafe_allow_html=True
        )

        # Title
        st.markdown(
            '<h1 style="font-size:3.5rem; font-weight:700; color:#f0f0f0; '
            'line-height:1.1; margin-bottom:1rem; font-family:Inter,sans-serif;">'
            'Generate enterprise<br>'
            '<span style="color:#d4a64a;">docs in seconds</span>'
            '</h1>',
            unsafe_allow_html=True
        )

        # Subtitle
        st.markdown(
            '<p style="color:#555; font-size:1rem; line-height:1.7; '
            'margin-bottom:2rem; max-width:480px;">'
            'Answer 7 guided questions per section. DocForge AI writes '
            'production-ready SaaS documents — saved to PostgreSQL, published to Notion.'
            '</p>',
            unsafe_allow_html=True
        )

        # ── Stats row ─────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Departments", "10")
        with m2: st.metric("Doc Types", "12")
        with m3: st.metric("Sections", "7")
        with m4: st.metric("Generate", "~60s")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Feature cards ─────────────────────────────────────────
        f1, f2, f3 = st.columns(3)

        with f1:
            st.markdown("""
            <div style="background:#141414; border:1px solid #1e1e1e; border-radius:10px; padding:1.2rem;">
                <div style="font-size:1.4rem; margin-bottom:0.6rem;">◈</div>
                <div style="color:#c0c0c0; font-weight:500; font-size:0.875rem; margin-bottom:0.4rem;">Guided by AI</div>
                <div style="color:#444; font-size:0.78rem; line-height:1.6;">Section-by-section Q&A tailored to your department and doc type</div>
            </div>
            """, unsafe_allow_html=True)

        with f2:
            st.markdown("""
            <div style="background:#141414; border:1px solid #1e1e1e; border-radius:10px; padding:1.2rem;">
                <div style="font-size:1.4rem; margin-bottom:0.6rem;">◫</div>
                <div style="color:#c0c0c0; font-weight:500; font-size:0.875rem; margin-bottom:0.4rem;">Notion Native</div>
                <div style="color:#444; font-size:0.78rem; line-height:1.6;">One-click publish with full metadata, tags, version and department</div>
            </div>
            """, unsafe_allow_html=True)

        with f3:
            st.markdown("""
            <div style="background:#141414; border:1px solid #1e1e1e; border-radius:10px; padding:1.2rem;">
                <div style="font-size:1.4rem; margin-bottom:0.6rem;">◉</div>
                <div style="color:#c0c0c0; font-weight:500; font-size:0.875rem; margin-bottom:0.4rem;">PostgreSQL Backed</div>
                <div style="color:#444; font-size:0.78rem; line-height:1.6;">All 14 Q&A answers saved flat with complete document record</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # ── CTA Button ────────────────────────────────────────────
        b1, b2, b3 = st.columns([1.5, 2, 1.5])
        with b2:
            if st.button("⚡  Start Generating", type="primary", use_container_width=True):
                st.session_state.page = "app"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Doc type tags ─────────────────────────────────────────
        st.markdown(
            '<p style="color:#2a2a2a; font-size:0.72rem; text-align:center; '
            'font-family:monospace; letter-spacing:0.08em;">'
            'NDA · Privacy Policy · SOP · SLA · PRD · Technical Spec · '
            'Incident Report · Security Policy · Customer Onboarding · Business Proposal · Employment Contract'
            '</p>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════
else:

    # ── Top bar ───────────────────────────────────────────────────
    nav1, nav2, nav3 = st.columns([2, 3, 2])
    with nav1:
        st.markdown(
            '<p style="color:#f0f0f0; font-size:1.1rem; font-weight:600; '
            'font-family:Inter,sans-serif; padding-top:0.4rem;">'
            '⚡ DocForge <span style="color:#d4a64a;">AI</span></p>',
            unsafe_allow_html=True
        )
    with nav3:
        st.markdown(
            '<p style="color:#333; font-size:0.65rem; font-family:monospace; '
            'text-align:right; padding-top:0.6rem; letter-spacing:0.1em;">'
            'SAAS · ENTERPRISE DOCS · v2.0</p>',
            unsafe_allow_html=True
        )

    st.divider()

    # ── Tab switcher ──────────────────────────────────────────────
    t1, t2, t3 = st.columns([2, 1, 2])
    with t2:
        tab_col1, tab_col2 = st.columns(2)
        with tab_col1:
            gen_type = "primary" if st.session_state.active_tab == "generate" else "secondary"
            if st.button("◈ Generate", use_container_width=True, type=gen_type):
                st.session_state.active_tab = "generate"
                st.rerun()
        with tab_col2:
            lib_type = "primary" if st.session_state.active_tab == "library" else "secondary"
            if st.button("◫ Library", use_container_width=True, type=lib_type):
                st.session_state.active_tab = "library"
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Content ───────────────────────────────────────────────────
    left, center, right = st.columns([1, 4, 1])
    with center:
        if st.session_state.active_tab == "generate":
            render_generator_form()
        else:
            render_library_browser()

    st.divider()

    # ── Back button ───────────────────────────────────────────────
    b1, b2, b3 = st.columns([3, 1, 3])
    with b2:
        if st.button("← Home", use_container_width=True):
            st.session_state.page = "landing"
            st.rerun()