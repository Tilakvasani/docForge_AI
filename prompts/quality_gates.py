from typing import Tuple

REQUIRED_SECTIONS = {
    "sop": ["purpose", "scope", "procedure", "responsibilities"],
    "policy": ["purpose", "scope", "definitions", "exceptions"],
    "proposal": ["overview", "objectives", "timeline", "budget"],
    "sow": ["scope", "deliverables", "timeline", "payment"],
    "incident_report": ["summary", "impact", "root cause", "resolution"],
    "faq": ["question", "answer"],
    "business_case": ["problem", "solution", "benefits", "cost"],
    "security_policy": ["scope", "definitions", "requirements", "exceptions"],
    "kpi_report": ["overview", "metrics", "analysis", "recommendations"],
    "runbook": ["purpose", "prerequisites", "steps", "troubleshooting"],
}

MIN_WORD_COUNT = 150

def check_quality(content: str, doc_type: str) -> Tuple[bool, str]:
    """Check if generated document meets quality standards."""
    content_lower = content.lower()

    word_count = len(content.split())
    if word_count < MIN_WORD_COUNT:
        return False, f"Too short: {word_count} words (minimum {MIN_WORD_COUNT})"

    required = REQUIRED_SECTIONS.get(doc_type, [])
    missing = [s for s in required if s not in content_lower]

    if missing:
        return False, f"Missing required sections: {', '.join(missing)}"

    return True, "Quality check passed"
