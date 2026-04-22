import streamlit as st

def init_session():
    """Initialize all default session state variables."""
    defaults = dict(
        step=1, company_ctx={}, departments=[],
        selected_dept=None, selected_dept_id=None,
        selected_doc_type=None, doc_sec_id=None, sections=[],
        section_questions={}, section_answers={},
        section_contents={}, gen_id=None, full_document="",
        main_tab="💬 CiteRAG",
        rag_chats={}, rag_active_chat=None,
        docx_bytes_cache=None, docx_cache_doc=None,
        _library_data=None, _answer_drafts={},
        _last_chunks=[], _last_not_found=False,
        _last_ragas_scores=None, _ragas_history=[],
        agent_tickets=[], agent_tickets_loaded=False,
        agent_memory={},
        _gen_failed_q_sections=set(),
        _gen_failed_doc_sections=set(),
        gen_questions_running=False,
        gen_doc_running=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
