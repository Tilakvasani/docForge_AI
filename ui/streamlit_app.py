"""
DocForge AI — streamlit_app.py  v7.0
Premium dark UI with amber/orange glow aesthetic.
Bug fixes:
  - Answer textarea values stored in session state via on_change → never disappear
  - docx cache invalidated on edit
  - library loads/caches correctly
  - section_questions missing key handled gracefully
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

st.set_page_config(page_title="DocForge AI", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Base — Light theme ── */
[data-testid="stAppViewContainer"] {
    background: #f5f6fa;
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8f9fc 100%);
    border-right: 1px solid #e8eaf0;
}
[data-testid="stSidebar"] * { color: #374151 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem; }

/* ── Sidebar brand ── */
.sb-brand {
    padding: 1.2rem 0 0.5rem;
    border-bottom: 1px solid #e8eaf0;
    margin-bottom: 1rem;
}
.sb-brand-name {
    font-size: 1.2rem; font-weight: 800;
    background: linear-gradient(135deg, #1f2937, #ea580c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.sb-brand-sub {
    font-size: 0.65rem; color: #9ca3af !important;
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px;
}

/* ── Sidebar steps ── */
.sb-step {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 8px;
    font-size: 0.8rem; margin-bottom: 3px; cursor: default;
}
.sb-done   { color: #ea580c !important; }
.sb-active { background: rgba(234,88,12,0.08); border: 1px solid rgba(234,88,12,0.2);
             color: #ea580c !important; font-weight: 700; }
.sb-pend   { color: #d1d5db !important; }
.sb-dot    { width: 20px; height: 20px; border-radius: 50%; display: flex;
             align-items: center; justify-content: center; font-size: 0.7rem;
             font-weight: 800; flex-shrink: 0; }
.sb-dot-done   { background: rgba(234,88,12,0.12); color: #ea580c !important; }
.sb-dot-active { background: linear-gradient(135deg,#ea580c,#f97316);
                 color: white !important; box-shadow: 0 2px 8px rgba(234,88,12,0.35); }
.sb-dot-pend   { background: #f3f4f6; color: #d1d5db !important; }

/* ── Sidebar info block ── */
.sb-info {
    background: #f8f9fc;
    border: 1px solid #e8eaf0;
    border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
}
.sb-info-lbl { font-size: 0.65rem; color: #ea580c !important; font-weight: 700;
               letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 3px; }
.sb-info-val { font-size: 0.82rem; color: #111827 !important; }
.sb-info-sub { font-size: 0.72rem; color: #6b7280 !important; margin-top: 1px; }

/* ── Header ── */
.df-header {
    background: linear-gradient(135deg, #fff7ed 0%, #ffffff 100%);
    border: 1px solid #fed7aa;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.df-header::before {
    content: '';
    position: absolute; top: -50%; left: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(234,88,12,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.df-hrow { display: flex; align-items: center; gap: 1rem; }
.df-icon { font-size: 2.4rem; filter: drop-shadow(0 0 12px rgba(255,107,0,0.6)); }
.df-title { font-size: 1.8rem; font-weight: 900; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #1f2937 0%, #ea580c 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.df-sub { font-size: 0.8rem; color: #6b7280; margin-top: 2px; }
.df-ver {
    margin-left: auto;
    background: rgba(234,88,12,0.08);
    border: 1px solid rgba(234,88,12,0.25);
    border-radius: 999px; padding: 4px 14px;
    font-size: 0.72rem; font-weight: 700;
    color: #ea580c; letter-spacing: 0.05em;
}

/* ── Step pill ── */
.step-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 999px; padding: 6px 18px;
    font-size: 0.78rem; font-weight: 700;
    color: #ea580c; margin-bottom: 1.25rem;
    letter-spacing: 0.03em;
}

/* ── Card ── */
.df-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.df-card:hover { border-color: #fed7aa; }
.df-card-glow::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #f97316, transparent);
    border-radius: 14px 14px 0 0;
}
.df-card-title {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
    color: #ea580c; text-transform: uppercase; margin-bottom: 1rem;
}

/* ── Stats ── */
.stat-row { display: grid; gap: 1rem; margin-bottom: 1.25rem; }
.stat-box {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px; padding: 1rem 1.25rem;
    text-align: center; transition: all 0.2s;
}
.stat-box:hover {
    border-color: #fed7aa;
    background: #fff7ed;
}
.stat-num  { font-size: 2rem; font-weight: 900; color: #ea580c;
             line-height: 1; }
.stat-lbl  { font-size: 0.7rem; color: #9ca3af; margin-top: 4px;
             text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Section grid ── */
.sec-done {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 8px; padding: 7px 10px; margin-bottom: 5px;
    font-size: 0.78rem; color: #ea580c;
    display: flex; align-items: center; gap: 6px;
}
.sec-pend {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px; padding: 7px 10px; margin-bottom: 5px;
    font-size: 0.78rem; color: #9ca3af;
    display: flex; align-items: center; gap: 6px;
}
.sec-answered {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 7px 10px; margin-bottom: 5px;
    font-size: 0.78rem; color: #4ade80;
    display: flex; align-items: center; gap: 6px;
}

/* ── Type badges ── */
.tbadge {
    display: inline-block; border-radius: 5px;
    padding: 1px 7px; font-size: 0.68rem; font-weight: 700;
    margin-left: 4px; vertical-align: middle;
}
.tb-table     { background: rgba(59,130,246,0.15); color: #60a5fa; }
.tb-flowchart { background: rgba(34,197,94,0.12);  color: #4ade80; }
.tb-raci      { background: rgba(167,139,250,0.15); color: #a78bfa; }
.tb-signature { background: rgba(244,114,182,0.12); color: #f472b6; }

/* ── Content box ── */
.content-box {
    background: #f8f9fc;
    border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1rem 1.1rem;
    font-family: 'Monaco','Menlo','Courier New',monospace !important;
    font-size: 0.76rem; color: #374151; line-height: 1.65;
    max-height: 600px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
}

/* ── Answer display ── */
.answer-chip {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;
}
.answer-q { font-size: 0.72rem; color: #ea580c; font-weight: 700;
            margin-bottom: 3px; }
.answer-a { font-size: 0.82rem; color: #374151; line-height: 1.5; }

/* ── Library ── */
.lib-card {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 8px;
    transition: all 0.2s;
}
.lib-card:hover {
    border-color: #fed7aa;
    background: #fff7ed;
}
.lib-title { font-size: 0.9rem; font-weight: 600; color: #111827; }
.lib-meta  { font-size: 0.73rem; color: #6b7280; margin-top: 3px; }

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(90deg,#ea580c,#f97316) !important; }

/* ── Fix black progress bar track ── */
.stProgress > div {
    background: #fee2e2 !important;
    border-radius: 999px !important;
    height: 6px !important;
}
.stProgress > div > div {
    border-radius: 999px !important;
    height: 6px !important;
}

/* ── Fix black selectbox ── */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #111827 !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.12) !important;
}
/* Dropdown popup container */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background: #ffffff !important;
    color: #111827 !important;
}
[data-baseweb="popover"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.1) !important;
    overflow: hidden !important;
}
[data-baseweb="menu"] {
    background: #ffffff !important;
}
[data-baseweb="option"],
[role="option"] {
    background: #ffffff !important;
    color: #111827 !important;
}
[data-baseweb="option"]:hover,
[role="option"]:hover {
    background: #fff7ed !important;
    color: #ea580c !important;
    cursor: pointer;
}
[aria-selected="true"],
[aria-selected="true"][data-baseweb="option"],
[aria-selected="true"][role="option"] {
    background: #fff7ed !important;
    color: #ea580c !important;
    font-weight: 600 !important;
}
/* Streamlit selectbox list container */
.stSelectbox [data-baseweb="select"] span {
    color: #111827 !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    color: #111827 !important;
}
/* Selectbox search input cursor */
[data-baseweb="select"] input {
    caret-color: #ea580c !important;
    color: #111827 !important;
}
/* Placeholder inside selectbox */
[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] {
    color: #9ca3af !important;
}

/* ── Fix radio buttons — show text, hide native circle ── */
[data-testid="stRadio"] > div {
    gap: 4px !important;
    flex-direction: column !important;
}
/* Hide blank label ONLY inside radio widgets */
[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 0 !important;
}
[data-testid="stRadio"] label {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    color: #374151 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stRadio"] label:hover {
    border-color: #f97316 !important;
    background: #fff7ed !important;
    color: #ea580c !important;
}
[data-testid="stRadio"] label[data-checked="true"] {
    background: #fff7ed !important;
    border-color: #ea580c !important;
    color: #ea580c !important;
    font-weight: 600 !important;
}
/* Hide native radio circle, show text */
[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
[data-testid="stRadio"] label p {
    color: #111827 !important;
    font-size: 0.82rem !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}
/* Sidebar radio — keep normal */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: flex !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent !important;
    border: none !important;
    padding: 4px 8px !important;
}

/* ── Fix download button ── */
[data-testid="stDownloadButton"] button {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #374151 !important;
    border-radius: 10px !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: #f97316 !important;
    color: #ea580c !important;
    background: #fff7ed !important;
}

/* ── Fix expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #374151 !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff6b00, #ff4500) !important;
    border: none !important; color: white !important;
    font-weight: 700 !important; border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(255,107,0,0.3) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #ff8c00, #ff6b00) !important;
    box-shadow: 0 6px 20px rgba(255,107,0,0.45) !important;
    transform: translateY(-1px);
}
.stButton > button {
    border-radius: 10px !important;
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #374151 !important;
}
.stButton > button:hover {
    border-color: #f97316 !important;
    color: #ea580c !important;
    background: #fff7ed !important;
}

/* ── Inputs ── */
.stTextInput input, 
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #111827 !important;
    font-size: 0.85rem !important;
    caret-color: #000000 !important;
    cursor: text !important;
}
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}
/* Fix placeholder text in selectbox */
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] span,
[data-baseweb="select"] input::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.12) !important;
    caret-color: #000000 !important;   /* ✅ BLACK cursor */
    cursor: text !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    color: #111827 !important;
    caret-color: #ea580c !important;
    cursor: pointer !important;
}
/* Selectbox placeholder text */
.stSelectbox > div > div > div[data-baseweb="select"] span[aria-hidden="true"],
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="placeholder"] {
    color: #9ca3af !important;
}

label { color: #374151 !important; font-size: 0.78rem !important;
        font-weight: 600 !important; letter-spacing: 0.02em !important; }
/* Ensure all widget labels are visible */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #374151 !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* ── Alerts ── */
.stSuccess { background: #f0fdf4 !important;
             border-color: #bbf7d0 !important; color: #15803d !important; }
.stError   { background: #fef2f2 !important;
             border-color: #fecaca !important; color: #dc2626 !important; }
.stInfo    { background: #fff7ed !important;
             border-color: #fed7aa !important; color: #ea580c !important; }
.stInfo p, .stInfo div, [data-testid="stNotification"] p {
    color: #ea580c !important;
    font-weight: 500 !important;
}
.stWarning { background: #fffbeb !important;
             border-color: #fde68a !important; color: #d97706 !important; }
.stWarning p, .stWarning div { color: #d97706 !important; font-weight: 500 !important; }
.stSuccess p, .stSuccess div { color: #15803d !important; font-weight: 500 !important; }
.stError p, .stError div { color: #dc2626 !important; font-weight: 500 !important; }
hr { border-color: #e5e7eb !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #f8f9fc !important;
    border-radius: 8px !important;
    color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api_get(ep):
    try:
        r = httpx.get(f"{API_URL}{ep}", timeout=30)
        r.raise_for_status(); return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def api_post(ep, data, timeout=120):
    try:
        r = httpx.post(f"{API_URL}{ep}", json=data, timeout=timeout)
        r.raise_for_status(); return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

TYPE_ICON  = {"table":"📊","flowchart":"🔀","raci":"👥","signature":"✍️","text":"✏️"}
TYPE_CLS   = {"table":"tb-table","flowchart":"tb-flowchart","raci":"tb-raci","signature":"tb-signature"}

def tbadge(sec_type):
    cls = TYPE_CLS.get(sec_type, "")
    if not cls: return ""
    return f'<span class="tbadge {cls}">{TYPE_ICON.get(sec_type,"")} {sec_type}</span>'

def stat_box(num, lbl):
    return (f'<div class="stat-box"><div class="stat-num">{num}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>')


# ─── Session ──────────────────────────────────────────────────────────────────

def init_session():
    defaults = dict(
        step=1, company_ctx={}, departments=[],
        selected_dept=None, selected_dept_id=None,
        selected_doc_type=None, doc_sec_id=None, sections=[],
        section_questions={}, section_answers={},
        section_contents={},
        sec_ids_ordered=[], gen_id=None, full_document="",
        active_tab="generate",
        docx_bytes_cache=None, docx_cache_doc=None,
        _library_data=None,
        # BUG FIX: store answer widget values here so they persist across reruns
        _answer_drafts={},
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-name">⚡ DocForge AI</div>
        <div class="sb-brand-sub">Enterprise Document Generator</div>
    </div>""", unsafe_allow_html=True)

    tab = st.radio("", ["⚡ Generate", "📚 Library"],
                   label_visibility="collapsed", key="main_tab")
    st.session_state.active_tab = "library" if "Library" in tab else "generate"

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.session_state.active_tab == "generate":
        steps = [(1,"Setup"),(2,"Questions"),(3,"Answers"),(4,"Review"),(5,"Export")]
        cur   = st.session_state.step
        for n, lbl in steps:
            if n < cur:
                dot_cls, step_cls, dot_txt = "sb-dot-done",  "sb-done",   "✓"
            elif n == cur:
                dot_cls, step_cls, dot_txt = "sb-dot-active","sb-active", str(n)
            else:
                dot_cls, step_cls, dot_txt = "sb-dot-pend",  "sb-pend",   str(n)
            st.markdown(
                f'<div class="sb-step {step_cls}">'
                f'<div class="sb-dot {dot_cls}">{dot_txt}</div>'
                f'<span>Step {n} — {lbl}</span></div>',
                unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        ctx = st.session_state.company_ctx
        if ctx.get("company_name"):
            st.markdown(
                f'<div class="sb-info">'
                f'<div class="sb-info-lbl">🏢 Company</div>'
                f'<div class="sb-info-val">{ctx["company_name"]}</div>'
                f'<div class="sb-info-sub">{ctx.get("industry","—")} · {ctx.get("region","—")}</div>'
                f'</div>', unsafe_allow_html=True)

        if st.session_state.selected_doc_type:
            st.markdown(
                f'<div class="sb-info">'
                f'<div class="sb-info-lbl">📄 Document</div>'
                f'<div class="sb-info-val">{st.session_state.selected_doc_type}</div>'
                f'<div class="sb-info-sub">{st.session_state.selected_dept or ""}</div>'
                f'</div>', unsafe_allow_html=True)

        if st.session_state.sections:
            done_n   = len(st.session_state.section_contents)
            total_n  = len(st.session_state.sections)
            active_n = total_n
            pct = int(done_n / active_n * 100)
            sb_progress_slot = st.empty()
            sb_progress_slot.markdown(
                f'<div class="sb-info">'
                f'<div class="sb-info-lbl">📊 Progress</div>'
                f'<div class="sb-info-val">{done_n}/{active_n} sections · {pct}%</div>'
                f'<div style="background:#fee2e2;border-radius:999px;height:4px;margin-top:6px">'
                f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);height:4px;'
                f'border-radius:999px;width:{pct}%"></div></div>'
                f'</div>', unsafe_allow_html=True)
            st.session_state._sb_progress_slot = sb_progress_slot

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("↺  Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="df-header">
    <div class="df-hrow">
        <div class="df-icon">⚡</div>
        <div>
            <div class="df-title">DocForge AI</div>
            <div class="df-sub">AI-Powered Enterprise Document Generator</div>
        </div>
        <div class="df-ver">GPT-4.1-mini</div>
    </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "library":
    ch, cb = st.columns([4,1])
    with ch: st.markdown("<h3 style='color:rgb(234 88 12)'>📚 Document Library</h3>",unsafe_allow_html=True)
    with cb:
        if st.button("↺ Refresh", use_container_width=True):
            st.session_state["_library_data"] = None

    if st.session_state.get("_library_data") is None:
        with st.spinner("Loading from Notion..."):
            lib = api_get("/library/notion")
        st.session_state["_library_data"] = lib or {}

    lib  = st.session_state.get("_library_data", {})
    docs = lib.get("documents", []) if isinstance(lib, dict) else []

    if not docs:
        st.markdown('<div class="df-card" style="text-align:center;padding:3rem">'
                    '<div style="font-size:2rem;margin-bottom:8px">📭</div>'
                    '<div style="color:#3a3a5a">No documents yet. Generate your first one!</div>'
                    '</div>', unsafe_allow_html=True)
    else:
        f1, f2 = st.columns([2,3])
        with f1:
            dept_filter = st.selectbox("Dept", ["All"]+sorted({d.get("department","") for d in docs if d.get("department")}), label_visibility="collapsed")
        with f2:
            search = st.text_input("Search", placeholder="🔍 Search documents...", label_visibility="collapsed")

        filtered = [d for d in docs
            if (dept_filter=="All" or d.get("department")==dept_filter)
            and (not search or search.lower() in d.get("title","").lower())]

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(stat_box(len(docs),"Total"), unsafe_allow_html=True)
        with c2: st.markdown(stat_box(len({d.get("department") for d in docs}),"Departments"), unsafe_allow_html=True)
        with c3: st.markdown(stat_box(len(filtered),"Showing"), unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        for doc in filtered:
            sc = {"Generated":"#ff6b00","Draft":"#f59e0b","Reviewed":"#60a5fa","Archived":"#3a3a5a"}.get(doc.get("status",""),"#3a3a5a")
            a,b,c = st.columns([4,2,1])
            with a:
                title = doc.get("title", "—")
                doc_type = doc.get("doc_type", "—")
                industry = doc.get("industry", "—")

                version_html = ""
                if doc.get("version"):
                    version_html = (
                    f'&nbsp;<span style="background:rgba(234,88,12,0.1);'
                    f'color:#ea580c;border:1px solid rgba(234,88,12,0.25);'
                    f'border-radius:999px;padding:1px 8px;font-size:0.68rem;'
                    f'font-weight:700;letter-spacing:0.04em">'
                    f'v{int(doc["version"])}</span>'
                    )

                st.markdown(
                f'<div class="lib-card">'
                f'<div class="lib-title">{title}{version_html}</div>'
                f'<div class="lib-meta">{doc_type} · {industry}</div>'
                f'</div>',
                unsafe_allow_html=True
                )
            with b:
                st.markdown(f'<div class="lib-card"><div class="lib-meta">🏢 {doc.get("department","—")}</div>'
                            f'<div class="lib-meta">📅 {doc.get("created_at","—")}</div>'
                            f'<div class="lib-meta" style="color:{sc}">● {doc.get("status","—")}</div></div>',
                            unsafe_allow_html=True)
            with c:
                if doc.get("notion_url"):
                    st.link_button("Open →", doc["notion_url"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERATE
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "generate":

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    if st.session_state.step == 1:
        st.markdown('<div class="step-pill">⚡ Step 1 of 5 — Setup</div>', unsafe_allow_html=True)

        if not st.session_state.departments:
            with st.spinner("Loading catalog..."):
                data = api_get("/departments")
                if data: st.session_state.departments = data["departments"]

        depts = st.session_state.departments
        if not depts:
            st.error("❌ Backend not reachable — run: `uvicorn backend.main:app --reload`")
            st.stop()

        st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">🏢 Company Information</div>',
                    unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            company_name = st.text_input("Company Name",
                value=st.session_state.company_ctx.get("company_name",""),
                placeholder="e.g. Turabit Technologies")
            industry = st.selectbox("Industry",
                ["Technology / SaaS","Finance / Banking","Healthcare","Manufacturing",
                "Retail / E-Commerce","Legal Services","Marketing / Media",
                "Logistics / Supply Chain","Education","Other"],
                index=0)
        with c2:
            company_size = st.selectbox("Company Size",[
                "1-10 employees","11-50 employees","51-200 employees",
                "201-500 employees","500+ employees"], index=2)
            region = st.selectbox("Region",[
                "India","United States","United Kingdom","UAE / Middle East",
                "Canada","Australia","Europe","Other"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">📂 Select Document</div>',
                    unsafe_allow_html=True)
        c3,c4 = st.columns(2)
        with c3:
            selected_dept = st.selectbox("Department", [d["department"] for d in depts])
        dept_data = next((d for d in depts if d["department"]==selected_dept), None)
        with c4:
            selected_doc_type = st.selectbox("Document Type", dept_data["doc_types"] if dept_data else [])
        st.markdown(f'<div style="margin-top:6px;font-size:0.75rem;color:#ff6b00;">'
                    f'▸ {selected_dept} → {selected_doc_type}</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Continue →", type="primary", use_container_width=True):
            if not company_name.strip():
                st.error("Please enter your company name.")
            else:
                with st.spinner("Loading sections..."):
                    safe = selected_doc_type.replace("/","%2F").replace("(","%28").replace(")","%29")
                    data = api_get(f"/sections/{safe}")
                if data:
                    st.session_state.company_ctx    = {"company_name": company_name.strip(),
                        "industry": industry, "company_size": company_size, "region": region}
                    st.session_state.selected_dept      = selected_dept
                    st.session_state.selected_dept_id   = dept_data["doc_id"]
                    st.session_state.selected_doc_type  = selected_doc_type
                    st.session_state.doc_sec_id         = data["doc_sec_id"]
                    seen, deduped = set(), []
                    for s in data["doc_sec"]:
                        if s not in seen: seen.add(s); deduped.append(s)
                    st.session_state.sections           = deduped
                    st.session_state.section_questions  = {}
                    st.session_state.section_answers    = {}
                    st.session_state.section_contents   = {}
                    st.session_state.full_document      = ""
                    st.session_state.gen_id             = None
                    st.session_state.docx_bytes_cache   = None
                    st.session_state._answer_drafts     = {}
                    st.session_state.step               = 2
                    st.rerun()

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    elif st.session_state.step == 2:
        sections = st.session_state.sections
        total    = len(sections)

        st.markdown('<div class="step-pill">❓ Step 2 of 5 — Generate Questions</div>',
                    unsafe_allow_html=True)

        # Slot 1: stats + section grid (declared first = renders above everything)
        grid_slot   = st.empty()
        # Slot 2: counter + status + progress bar (renders between grid and button)
        status_slot = st.empty()

        def render_grid():
            """Re-render stats + section grid."""
            live_q = st.session_state.section_questions
            done   = len(live_q)
            pct    = int(done / total * 100) if total else 0
            rows   = []
            rows.append(
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1rem">' +
                stat_box(total, "Sections") +
                stat_box(done, "Ready") +
                stat_box(str(pct) + "%", "Complete") +
                '</div>'
            )
            rows.append('<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px">')
            for s in sections:
                if s in live_q:
                    st_type = live_q[s].get("section_type", "text")
                    rows.append(
                        f'<div class="sec-done">✓ <span style="flex:1;overflow:hidden;'
                        f'text-overflow:ellipsis;white-space:nowrap">{s[:28]}</span>'
                        f'{tbadge(st_type)}</div>')
                else:
                    rows.append(f'<div class="sec-pend">○ {s[:32]}</div>')
            rows.append('</div>')
            grid_slot.markdown("".join(rows), unsafe_allow_html=True)

        def render_status(done, pct, status_text=""):
            """Re-render counter + status + progress bar above the button."""
            status_part = (
                f'<span style="color:#f97316;font-size:0.78rem">⏳ {status_text}</span>'
                if status_text else ""
            )
            html = (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'margin-top:10px;margin-bottom:6px">' +
                f'<span style="color:#ea580c;font-size:0.82rem;font-weight:700">'
                f'{done} of {total} sections ready</span>' +
                status_part + '</div>' +
                f'<div style="background:#fee2e2;border-radius:999px;height:6px;margin-bottom:10px">' +
                f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);'
                f'height:6px;border-radius:999px;width:{pct}%"></div></div>'
            )
            status_slot.markdown(html, unsafe_allow_html=True)

        render_grid()

        if len(st.session_state.section_questions) < total:
            if st.button("⚡ Generate Questions for All Sections", type="primary", use_container_width=True):
                for i, sec in enumerate(sections):
                    live_done = len(st.session_state.section_questions)
                    live_pct  = int(live_done / total * 100)

                    if sec in st.session_state.section_questions:
                        render_grid()
                        render_status(live_done, live_pct)
                        continue

                    render_status(live_done, live_pct, status_text=sec)

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
                            "questions":    res.get("questions", []),
                            "section_type": res.get("section_type", "text"),
                        }
                    render_grid()
                    new_done = len(st.session_state.section_questions)
                    render_status(new_done, int(new_done/total*100))

                status_slot.empty()
                st.rerun()

        if len(st.session_state.section_questions) == total:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("Start Answering →", type="primary", use_container_width=True):
                st.session_state.step = 3; st.rerun()

    # ── STEP 3 ────────────────────────────────────────────────────────────────
    elif st.session_state.step == 3:
        sections   = st.session_state.sections
        ans_map    = st.session_state.section_answers
        q_map      = st.session_state.section_questions
        unanswered = [s for s in sections if s not in ans_map]

        # ── All done → generate ───────────────────────────────────────────────
        if not unanswered:
            st.markdown('<div class="step-pill">✅ Step 3 of 5 — Ready to Generate</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1: st.markdown(stat_box(len(ans_map),"Answered"), unsafe_allow_html=True)
            with c2: st.markdown(stat_box(len(sections),"Will Generate"), unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("⚡ Generate Document", type="primary", use_container_width=True):
                active = sections
                total  = len(active)
                gen_status = st.empty()
                ids = []

                for i, sec in enumerate(active):
                    if sec in st.session_state.section_contents:
                        ids.append(q_map.get(sec, {}).get("sec_id", 0))
                        _pct = int((i + 1) / total * 100)
                        gen_status.markdown(
                            f'<div style="background:#fee2e2;border-radius:999px;height:6px">'
                            f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);'
                            f'height:6px;border-radius:999px;width:{_pct}%"></div></div>',
                            unsafe_allow_html=True)
                        continue
                    gen_status.markdown(
                        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">'
                        f'<span style="color:#ea580c;font-size:0.78rem;font-weight:600">{len(st.session_state.section_contents)} of {total} sections written</span>'
                        f'<span style="color:#f97316;font-size:0.78rem">✍️ Writing: {sec}</span>'
                        f'</div>'
                        f'<div style="background:#fee2e2;border-radius:999px;height:6px">'
                        f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);height:6px;border-radius:999px;width:{int(len(st.session_state.section_contents)/total*100)}%"></div></div>',
                        unsafe_allow_html=True)
                    q_data = q_map.get(sec, {})
                    res = api_post("/section/generate", {
                        "sec_id":          q_data.get("sec_id"),
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
                        ids.append(q_data.get("sec_id"))
                    _pct = int((i + 1) / total * 100)
                    _done = len(st.session_state.section_contents)
                    gen_status.markdown(
                        f'<div style="background:#fee2e2;border-radius:999px;height:6px">'
                        f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);'
                        f'height:6px;border-radius:999px;width:{_pct}%"></div></div>',
                        unsafe_allow_html=True)
                    # Update sidebar progress live
                    _slot = st.session_state.get("_sb_progress_slot")
                    if _slot:
                        _pct2 = int(_done / total * 100)
                        _slot.markdown(
                            f'<div class="sb-info">'
                            f'<div class="sb-info-lbl">📊 Progress</div>'
                            f'<div class="sb-info-val">{_done}/{total} sections · {_pct2}%</div>'
                            f'<div style="background:#fee2e2;border-radius:999px;height:4px;margin-top:6px">'
                            f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);height:4px;'
                            f'border-radius:999px;width:{_pct2}%"></div></div>'
                            f'</div>', unsafe_allow_html=True)

                st.session_state.sec_ids_ordered = ids
                doc_lines = []
                for sec in active:
                    c = st.session_state.section_contents.get(sec, "").strip()
                    if c:
                        doc_lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
                full_doc = "\n".join(doc_lines).strip()

                save_res = api_post("/document/save", {
                    "doc_id":          st.session_state.selected_dept_id,
                    "doc_sec_id":      st.session_state.doc_sec_id,
                    "sec_id":          ids[-1] if ids else 0,
                    "gen_doc_sec_dec": list(st.session_state.section_contents.values()),
                    "gen_doc_full":    full_doc,
                })
                st.session_state.gen_id           = save_res.get("gen_id", 0) if save_res else 0
                st.session_state.full_document    = full_doc
                st.session_state.docx_bytes_cache = None
                st.session_state.step             = 4
                st.rerun()

        # ── Still answering ───────────────────────────────────────────────────
        else:
            current   = unanswered[0]
            done_cnt  = len(ans_map)
            total     = len(sections)
            q_data    = q_map.get(current, {})
            questions = q_data.get("questions", [])
            sec_id    = q_data.get("sec_id")
            sec_type  = q_data.get("section_type", "text")

            st.markdown('<div class="step-pill">✍️ Step 3 of 5 — Answer Questions</div>',
                unsafe_allow_html=True)

            # Progress
            pct3 = int(done_cnt / total * 100) if total else 0
            st.markdown(
                f'<div style="background:#fee2e2;border-radius:999px;height:6px;margin-bottom:6px">'
                f'<div style="background:linear-gradient(90deg,#ea580c,#f97316);height:6px;'
                f'border-radius:999px;width:{pct3}%;transition:width 0.3s"></div></div>'
                f'<div style="color:#ea580c;font-size:0.75rem;margin-bottom:1rem">'
                f'{done_cnt} of {total} complete · {len(unanswered)} remaining</div>',
                unsafe_allow_html=True)

            # Section type hint
            hints = {
                "table":     ("📊","Data table — your answers will populate the rows.","#eff6ff"),
                "flowchart": ("🔀","Process flowchart — describe the steps and decisions.","#f0fdf4"),
                "raci":      ("👥","RACI matrix — list the roles involved.","#faf5ff"),
                "signature": ("✍️","Sign-off block — auto-generated, no input needed.","#fdf2f8"),
                "text":      ("✏️","",""),
            }
            icon, hint, bg = hints.get(sec_type, ("✏️","",""))

            st.markdown(
                f'<div class="df-card" style="background:{bg or "#ffffff"};margin-bottom:12px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:{"0.5rem" if hint else "0"}">'
                f'<span style="font-size:1rem;font-weight:700;color:#111827">{icon} {current}</span>'
                f'{tbadge(sec_type)}</div>'
                + (f'<div style="font-size:0.82rem;color:#6b7280;line-height:1.5">{hint}</div>' if hint else "")
                + '</div>',
                unsafe_allow_html=True)

            # ── ANSWER WIDGETS ──────────────────────────────────────────────
            # BUG FIX: We store draft answers in session state keyed by section.
            # This means values persist across reruns — they only clear when
            # we explicitly save or skip.
            if "_answer_drafts" not in st.session_state:
                st.session_state._answer_drafts = {}
            if current not in st.session_state._answer_drafts:
                st.session_state._answer_drafts[current] = [""] * len(questions)

            user_answers = []
            if not questions:
                st.markdown(
                    '<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;'
                    'padding:10px 16px;color:#ea580c;font-size:0.82rem">'
                    '✨ No questions needed — this section is auto-generated professionally.</div>'
                    '<div style="height:12px"></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="font-size:0.8rem;color:#6b7280;margin-bottom:10px">'
                    'Leave blank to auto-fill with professional content</div>',
                    unsafe_allow_html=True)
                for i, q in enumerate(questions):
                    # Use session state value as default — persists across rerun
                    current_val = st.session_state._answer_drafts[current][i] if i < len(st.session_state._answer_drafts[current]) else ""
                    a = st.text_area(
                        f"Q{i+1}: {q}",
                        value=current_val,
                        key=f"draft_{current}_{i}",
                        height=85,
                        placeholder="Your answer (or leave blank for auto-fill)...")
                    # Store immediately in draft so it survives reruns
                    if i < len(st.session_state._answer_drafts[current]):
                        st.session_state._answer_drafts[current][i] = a
                    user_answers.append(a)

            b1, = [st.columns(1)[0]]
            with b1:
                if st.button("Save & Next →", type="primary", use_container_width=True):
                    filled = [a.strip() if a.strip() else "not answered" for a in user_answers]
                    if sec_id:
                        api_post("/answers/save",{
                            "sec_id":sec_id,
                            "doc_sec_id":st.session_state.doc_sec_id,
                            "doc_id":st.session_state.selected_dept_id,
                            "section_name":current,
                            "questions":questions,
                            "answers":filled or ["not answered"],
                        })
                    # Save to permanent answers map
                    st.session_state.section_answers[current] = filled
                    # Clear draft (no longer needed)
                    st.session_state._answer_drafts.pop(current, None)
                    st.rerun()

            # ── Answered/Skipped summary ──────────────────────────────────
            if ans_map:
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                if ans_map:
                    st.markdown('<div style="font-size:0.68rem;color:#ff6b00;font-weight:700;'
                                'letter-spacing:0.08em;margin-bottom:6px">ANSWERED</div>',
                                unsafe_allow_html=True)
                    for s in sections:
                        if s in ans_map:
                            st.markdown(f'<div class="sec-answered">✓ <strong>{s}</strong></div>',
                                        unsafe_allow_html=True)


    # ── STEP 4 ────────────────────────────────────────────────────────────────
    elif st.session_state.step == 4:
        active   = st.session_state.sections
        contents = st.session_state.section_contents

        st.markdown('<div class="step-pill">🔍 Step 4 of 5 — Review & Edit</div>',
                    unsafe_allow_html=True)

        def rebuild_doc():
            lines = []
            for sec in active:
                c = contents.get(sec, "").strip()
                if c:
                    lines += [sec.upper(), "-" * len(sec), "", c, "", ""]
            st.session_state.full_document = "\n".join(lines).strip()

        left, right = st.columns([1, 2])

        with left:
            st.markdown('<div style="font-size:0.68rem;color:#ea580c;font-weight:700;' +
                        'letter-spacing:0.08em;margin-bottom:8px">SECTIONS</div>',
                        unsafe_allow_html=True)
            sel = st.radio("", active, label_visibility="collapsed", key="sec_radio")

        with right:
            cur      = contents.get(sel, "")
            sec_type = st.session_state.section_questions.get(sel, {}).get("section_type", "text")
            icon     = TYPE_ICON.get(sec_type, "✏️")

            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.75rem">' +
                f'<span style="font-size:0.95rem;font-weight:700;color:#111827">{icon} {sel}</span>' +
                f'{tbadge(sec_type)}</div>',
                unsafe_allow_html=True)

            with st.expander("📄 Current Content", expanded=True):
                if "```mermaid" in (cur or ""):
                    st.markdown(cur)
                elif not cur:
                    st.markdown('<div style="color:#9ca3af;font-style:italic;padding:1rem">(empty)</div>',
                                unsafe_allow_html=True)
                else:
                    _lines = cur.split("\n")
                    _html = '<div style="font-family:Georgia,serif;font-size:0.88rem;color:#1f2937;line-height:1.8;padding:1.2rem 1.5rem;background:#fafafa;border-radius:10px;border:1px solid #e5e7eb;">'
                    for _line in _lines:
                        if not _line.strip():
                            _html += '<div style="height:6px"></div>'
                        else:
                            _html += f'<p style="margin:0 0 4px 0">{_line}</p>'
                    _html += '</div>'
                    st.markdown(_html, unsafe_allow_html=True)

            instr = st.text_area("AI Edit Instruction",
                placeholder="e.g. Make more formal · Add detail · Shorten · Change tone",
                height=60, key="edit_instr", label_visibility="collapsed")

            ec1, ec2 = st.columns(2)
            with ec1:
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
                            st.session_state.docx_bytes_cache = None
                            rebuild_doc()
                            st.success("✅ Updated!")
                            st.rerun()
            with ec2:
                manual = st.text_area("Manual Edit", value=cur, height=180,
                                      key=f"manual_{sel}", label_visibility="collapsed")
                if st.button("💾 Save Manual", use_container_width=True, key=f"save_{sel}"):
                    st.session_state.section_contents[sel] = manual
                    st.session_state.docx_bytes_cache = None
                    rebuild_doc()
                    st.success("✅ Saved!")
                    st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Export →", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()

    # ── STEP 5 ────────────────────────────────────────────────────────────────
    elif st.session_state.step == 5:
        ctx      = st.session_state.company_ctx
        doc_type = st.session_state.selected_doc_type
        full_doc = st.session_state.full_document
        active   = st.session_state.sections
        contents = st.session_state.section_contents

        st.markdown('<div class="step-pill">💾 Step 5 of 5 — Export</div>', unsafe_allow_html=True)

        if not full_doc:
            st.markdown('''
            <div class="df-card" style="text-align:center;padding:3rem 2rem">
                <div style="font-size:3rem;margin-bottom:1rem">📄</div>
                <div style="font-size:1.2rem;font-weight:700;color:#1f2937;margin-bottom:0.5rem">No document found</div>
                <div style="font-size:0.85rem;color:#6b7280;margin-bottom:1.5rem">
                    Go back to the home page and start a new document.
                </div>
            </div>''', unsafe_allow_html=True)
            if st.button("🏠 Go to Home Page", type="primary", use_container_width=False):
                for k in list(st.session_state.keys()): del st.session_state[k]
                init_session()
                st.rerun()
            st.stop()

        # Summary card
        st.markdown(f"""
        <div class="df-card df-card-glow">
            <div class="df-card-title">📄 Document Ready</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:4px">
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Document</div>
                     <div style="font-size:0.9rem;color:#111827;font-weight:600">{doc_type}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Department</div>
                     <div style="font-size:0.9rem;color:#111827;font-weight:600">{st.session_state.selected_dept}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Company</div>
                     <div style="font-size:0.9rem;color:#111827;font-weight:600">{ctx.get("company_name","—")}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Industry</div>
                     <div style="font-size:0.9rem;color:#374151">{ctx.get("industry","—")}</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Sections</div>
                     <div style="font-size:0.9rem;color:#374151">{len(active)} sections</div></div>
                <div><div style="font-size:0.68rem;color:#ea580c;font-weight:700;margin-bottom:3px;text-transform:uppercase;letter-spacing:0.05em">Words</div>
                     <div style="font-size:0.9rem;color:#374151">~{len(full_doc.split())} words</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

        col_n, col_d = st.columns(2)

        with col_n:
            st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">📓 Publish to Notion</div>',
                        unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.78rem;color:#3a3a5a;margin-bottom:10px">'
                        'Publish to your Notion workspace database.</div>', unsafe_allow_html=True)
            if st.button("🚀 Publish to Notion", type="primary", use_container_width=True):
                with st.spinner("Publishing..."):
                    res = api_post("/document/publish",{
                        "gen_id":          st.session_state.gen_id or 0,
                        "doc_type":        doc_type,
                        "department":      st.session_state.selected_dept,
                        "gen_doc_full":    full_doc,
                        "company_context": ctx,
                    })
                if res:
                    url = res.get("notion_url","")
                    st.success("✅ Published to Notion!")
                    if url: st.link_button("🔗 Open in Notion", url, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="df-card df-card-glow"><div class="df-card-title">📥 Download</div>',
                        unsafe_allow_html=True)
            safe = doc_type.replace(" ","_").replace("/","-").replace("(","").replace(")","")

            if DOCX_AVAILABLE:
                if (st.session_state.get("docx_bytes_cache") is None or
                        st.session_state.get("docx_cache_doc") != doc_type):
                    try:
                        sections_data = [{"name":sec,"content":contents.get(sec,"")}
                                         for sec in active if contents.get(sec)]
                        st.session_state.docx_bytes_cache = build_docx(
                            doc_type=doc_type, department=st.session_state.selected_dept,
                            company_name=ctx.get("company_name","Company"),
                            industry=ctx.get("industry",""), region=ctx.get("region",""),
                            sections=sections_data)
                        st.session_state.docx_cache_doc = doc_type
                    except Exception as e:
                        st.error(f"DOCX error: {e}")
                        st.session_state.docx_bytes_cache = None

                if st.session_state.get("docx_bytes_cache"):
                    st.download_button("⬇️ Download .docx",
                        data=st.session_state.docx_bytes_cache,
                        file_name=f"{safe}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True, type="primary")
            else:
                st.warning("docx_builder.py not found.")

            st.download_button("⬇️ Download .txt", data=full_doc,
                file_name=f"{safe}.txt", mime="text/plain", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📄 Preview Full Document", expanded=False):
            # Render as nicely formatted HTML — section titles as headings
            import re as _re
            def _render_doc(text):
                lines = text.split("\n")
                html = '<div style="font-family:Georgia,serif;font-size:0.88rem;color:#1f2937;line-height:1.8;padding:1.5rem 2rem;background:#fff;border-radius:10px;border:1px solid #e5e7eb;">' 
                i = 0
                while i < len(lines):
                    line = lines[i]
                    # Section heading: ALL CAPS line followed by dashes
                    if i+1 < len(lines) and _re.match(r'^-{3,}$', lines[i+1].strip()) and line.strip() == line.strip().upper() and line.strip():
                        html += f'<h3 style="font-size:1rem;font-weight:700;color:#ea580c;border-bottom:2px solid #fed7aa;padding-bottom:4px;margin:1.5rem 0 0.5rem;">{line.strip().title()}</h3>'
                        i += 2  # skip the dashes line
                        continue
                    # Dash separator line — skip
                    elif _re.match(r'^-{3,}$', line.strip()):
                        i += 1
                        continue
                    # Empty line — small spacer
                    elif not line.strip():
                        html += '<div style="height:4px"></div>'
                    # Normal text
                    else:
                        html += f'<p style="margin:0 0 4px 0">{line}</p>'
                    i += 1
                html += '</div>'
                return html
            st.markdown(_render_doc(full_doc), unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("➕ Create Another Document", type="primary", use_container_width=True):
            saved_ctx   = st.session_state.company_ctx
            saved_depts = st.session_state.departments
            for k in list(st.session_state.keys()): del st.session_state[k]
            init_session()
            st.session_state["company_ctx"]  = saved_ctx
            st.session_state["departments"]  = saved_depts
            st.session_state["step"]         = 1
            st.rerun()