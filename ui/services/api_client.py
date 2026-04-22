import os
import logging
import httpx
import streamlit as st

API_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/") + "/api"
_log = logging.getLogger("docforge.frontend")

@st.cache_data(ttl=60, show_spinner=False)
def api_get_cached(ep: str) -> dict | None:
    """Cached version of API GET for idempotent fetches."""
    return _api_get_internal(ep)

def api_get(ep: str) -> dict | None:
    """Uncached API GET."""
    return _api_get_internal(ep)

def _api_get_internal(ep: str) -> dict | None:
    _log.info("→ GET  %s%s", API_URL, ep)
    try:
        r = httpx.get(f"{API_URL}{ep}", timeout=30)
        r.raise_for_status()
        _log.info("← 200  GET  %-40s  (%d bytes)", ep, len(r.content))
        return r.json()
    except httpx.HTTPStatusError as e:
        _log.error("← %s  GET  %s  — %s", e.response.status_code, ep, e.response.text[:200])
        st.error(f"API {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        _log.error("← ERR  GET  %s  — %s", ep, e)
        st.error(f"Connection error: {e}")
    return None

def api_post(ep: str, data: dict, timeout: int = 120) -> dict | None:
    _log.info("→ POST %s%s  keys=%s  timeout=%ds", API_URL, ep, list(data.keys()), timeout)
    try:
        r = httpx.post(f"{API_URL}{ep}", json=data, timeout=timeout)
        r.raise_for_status()
        _log.info("← 200  POST %-40s  (%d bytes)", ep, len(r.content))
        return r.json()
    except httpx.HTTPStatusError as e:
        try:
            msg = e.response.json().get("detail", e.response.text[:200])
        except Exception:
            msg = e.response.text[:200]
        _log.error("← %s  POST %s  — %s", e.response.status_code, ep, msg)
        st.session_state._last_api_error = msg
        return None
    except Exception as e:
        _log.error("← ERR  POST %s  — %s", ep, e)
        st.session_state._last_api_error = f"Connection error: {e}"
        return None
