import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="DocForge AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Space+Grotesk:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:       #0f0e0c;
    --surface:  #1a1814;
    --surface2: #211f1a;
    --border:   #2e2b24;
    --amber:    #f5a623;
    --amber2:   #ffcc70;
    --teal:     #00d4aa;
    --rose:     #ff6b6b;
    --text:     #f0ede6;
    --muted:    #7a7568;
    --card:     #161410;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 100% 60% at 50% -20%, rgba(245,166,35,0.08) 0%, transparent 70%),
        var(--bg) !important;
}

[data-testid="stHeader"], #MainMenu, footer { display: none !important; }
.stDeployButton { display: none !important; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── TOP NAV ── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 3rem;
    border-bottom: 1px solid var(--border);
    background: rgba(15,14,12,0.9);
    backdrop-filter: blur(20px);
    position: sticky;
    top: 0;
    z-index: 100;
}

.logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 900;
    color: var(--text);
    letter-spacing: -0.02em;
}

.logo em {
    font-style: normal;
    color: var(--amber);
}

.nav-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(245,166,35,0.1);
    border: 1px solid rgba(245,166,35,0.2);
    border-radius: 100px;
    padding: 6px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--amber);
    letter-spacing: 0.08em;
}

.nav-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--teal);
    animation: blink 2s ease-in-out infinite;
}

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── HERO ── */
.hero {
    padding: 5rem 3rem 3rem;
    max-width: 900px;
}

.hero-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.hero-h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(3rem, 6vw, 5.5rem) !important;
    font-weight: 900 !important;
    line-height: 1.0 !important;
    letter-spacing: -0.03em !important;
    color: var(--text) !important;
    margin-bottom: 1.5rem !important;
}

.hero-h1 .hl { color: var(--amber); }

.hero-desc {
    font-size: 1.05rem;
    color: var(--muted);
    line-height: 1.7;
    max-width: 520px;
    font-weight: 300;
    margin-bottom: 2.5rem;
}

.hero-stats {
    display: flex;
    gap: 3rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
}

.stat { display: flex; flex-direction: column; gap: 4px; }

.stat-n {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}

.stat-l {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── CONTENT AREA ── */
.content-wrap {
    padding: 0 3rem 3rem;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 2rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.25s !important;
}

.stTabs [aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
}

.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── INPUTS ── */
.stTextInput input,
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: var(--amber) !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.12) !important;
    outline: none !important;
}

.stTextInput input:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}

.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

label,
.stTextInput label,
.stTextArea label,
.stSelectbox label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 6px !important;
}

/* ── BUTTONS ── */
.stFormSubmitButton button,
.stButton button {
    background: var(--amber) !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 14px 28px !important;
    width: 100% !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}

.stFormSubmitButton button:hover,
.stButton button:hover {
    background: var(--amber2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(245,166,35,0.3) !important;
}

/* ── FORM PANEL ── */
.panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.panel-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
}

.panel-label::before {
    content: '◈';
    color: var(--amber);
    font-size: 0.8rem;
}

/* ── DOC OUTPUT ── */
.doc-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-top: 1.5rem;
    overflow: hidden;
}

.doc-card-head {
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--surface);
}

.doc-title-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
}

.doc-chips { display: flex; gap: 8px; flex-wrap: wrap; }

.chip {
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    font-weight: 700;
    text-transform: uppercase;
}

.chip-amber { background: rgba(245,166,35,0.15); color: var(--amber); border: 1px solid rgba(245,166,35,0.25); }
.chip-teal  { background: rgba(0,212,170,0.12); color: var(--teal); border: 1px solid rgba(0,212,170,0.2); }
.chip-rose  { background: rgba(255,107,107,0.12); color: var(--rose); border: 1px solid rgba(255,107,107,0.2); }
.chip-grey  { background: rgba(122,117,104,0.15); color: var(--muted); border: 1px solid var(--border); }

.doc-body {
    padding: 2rem;
    font-size: 0.88rem;
    line-height: 1.85;
    color: #c8c4bb;
    max-height: 520px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.doc-body::-webkit-scrollbar { width: 3px; }
.doc-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.doc-footer {
    padding: 1rem 2rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.notion-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    color: var(--teal);
    text-decoration: none;
    padding: 7px 14px;
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 6px;
    background: rgba(0,212,170,0.06);
    transition: all 0.2s;
}

.notion-link:hover {
    background: rgba(0,212,170,0.14);
    border-color: var(--teal);
}

/* ── LIBRARY CARDS ── */
.lib-grid { display: flex; flex-direction: column; gap: 12px; }

.lcard {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    transition: all 0.2s;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 1rem;
    align-items: center;
}

.lcard:hover {
    border-color: rgba(245,166,35,0.3);
    background: var(--surface);
    transform: translateX(4px);
}

.lcard-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
}

.lcard-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}

.lcard-chips { display: flex; gap: 6px; flex-wrap: wrap; }

.lcard-right { display: flex; flex-direction: column; align-items: flex-end; gap: 10px; }

.lcard-stat {
    text-align: right;
}

.lcard-stat-n {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}

.lcard-stat-l {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── STATUS BADGES ── */
.badge {
    padding: 4px 10px;
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
}
.badge-gen   { background: rgba(0,212,170,0.12); color: var(--teal); border: 1px solid rgba(0,212,170,0.2); }
.badge-draft { background: rgba(245,166,35,0.1); color: var(--amber); border: 1px solid rgba(245,166,35,0.2); }
.badge-rev   { background: rgba(122,117,104,0.15); color: var(--muted); border: 1px solid var(--border); }

/* ── ALERTS ── */
.stSuccess > div {
    background: rgba(0,212,170,0.08) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    border-radius: 8px !important;
    color: var(--teal) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

.stError > div {
    background: rgba(255,107,107,0.08) !important;
    border: 1px solid rgba(255,107,107,0.2) !important;
    border-radius: 8px !important;
    color: var(--rose) !important;
}

/* ── MISC ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

.empty {
    text-align: center;
    padding: 5rem 2rem;
    color: var(--muted);
}

.empty-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.3; }

.empty-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: var(--muted);
    font-style: italic;
}

.filter-row { margin-bottom: 1.5rem; }

.count-line {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    margin-bottom: 1.5rem;
}

.count-line b { color: var(--amber); }

.pub-note {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# Nav
st.markdown("""
<div class="topnav">
    <div class="logo">Doc<em>Forge</em></div>
    <div class="nav-pill"><div class="nav-dot"></div> LangChain · Groq · Notion</div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◈ AI Document Generator · SaaS Edition</div>
    <h1 class="hero-h1">Generate docs<br>that <span class="hl">mean business.</span></h1>
    <p class="hero-desc">
        Production-ready SaaS legal, technical, and business documents — 
        crafted by AI, published to Notion in one click.
    </p>
    <div class="hero-stats">
        <div class="stat">
            <span class="stat-n">12</span>
            <span class="stat-l">Doc Types</span>
        </div>
        <div class="stat">
            <span class="stat-n">∞</span>
            <span class="stat-l">Generations</span>
        </div>
        <div class="stat">
            <span class="stat-n">1-click</span>
            <span class="stat-l">Notion Publish</span>
        </div>
    </div>
</div>
<div class="content-wrap">
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["  ◈ GENERATE  ", "  ◫ LIBRARY  "])

with tab1:
    from ui.components.generator_form import render_generator_form
    render_generator_form()

with tab2:
    from ui.components.library_browser import render_library_browser
    render_library_browser()

st.markdown('</div>', unsafe_allow_html=True)