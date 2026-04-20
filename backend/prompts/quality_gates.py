"""
Quality gate validation for generated document content.

Checks that a generated section meets minimum length and contains
required structural keywords for known document types. The gate is
applied after LLM generation; failures are logged and surfaced to the
frontend as warnings without blocking the workflow.
"""

from typing import Tuple


DOC_TYPE_NORMALISE: dict[str, str] = {
    "standard operating procedure": "sop",
    "sop":                          "sop",
    "policy":                       "policy",
    "hr policy":                    "policy",
    "proposal":                     "proposal",
    "business proposal":            "proposal",
    "statement of work":            "sow",
    "sow":                          "sow",
    "incident report":              "incident_report",
    "faq":                          "faq",
    "frequently asked questions":   "faq",
    "business case":                "business_case",
    "security policy":              "security_policy",
    "kpi report":                   "kpi_report",
    "runbook":                      "runbook",
    "run book":                     "runbook",
}

REQUIRED_SECTIONS = {
    "sop":             ["purpose", "scope", "procedure", "responsibilities"],
    "policy":          ["purpose", "scope", "definitions", "exceptions"],
    "proposal":        ["overview", "objectives", "timeline", "budget"],
    "sow":             ["scope", "deliverables", "timeline", "payment"],
    "incident_report": ["summary", "impact", "root cause", "resolution"],
    "faq":             ["question", "answer"],
    "business_case":   ["problem", "solution", "benefits", "cost"],
    "security_policy": ["scope", "definitions", "requirements", "exceptions"],
    "kpi_report":      ["overview", "metrics", "analysis", "recommendations"],
    "runbook":         ["purpose", "prerequisites", "steps", "troubleshooting"],
}

MIN_WORD_COUNT = 150


def normalise_doc_type(doc_type: str) -> str:
    """
    Convert a display name or raw slug to the canonical `REQUIRED_SECTIONS` key.

    For example, "Standard Operating Procedure" → "sop".
    Unrecognized strings are returned lowercased and stripped.

    Args:
        doc_type: Raw document type string from the request or database.

    Returns:
        The normalized slug key used for quality gate lookup.
    """
    return DOC_TYPE_NORMALISE.get(doc_type.lower().strip(), doc_type.lower().strip())


def check_quality(content: str, doc_type: str) -> Tuple[bool, str]:
    """
    Validate that generated content meets minimum quality standards.

    Two checks are applied in order:
        1. Word count must meet or exceed `MIN_WORD_COUNT`.
        2. For recognized document types, all required section keywords
           must appear somewhere in the content (case-insensitive).

    Args:
        content:  The generated document text to evaluate.
        doc_type: The document type string (display name or slug).

    Returns:
        A tuple of `(passed: bool, note: str)` where `note` describes
        the failure reason if `passed` is `False`, or "Quality check passed"
        if the content meets all criteria.
    """
    content_lower = content.lower()

    word_count = len(content.split())
    if word_count < MIN_WORD_COUNT:
        return False, f"Too short: {word_count} words (minimum {MIN_WORD_COUNT})"

    slug     = normalise_doc_type(doc_type)
    required = REQUIRED_SECTIONS.get(slug, [])
    missing  = [s for s in required if s not in content_lower]

    if missing:
        return False, f"Missing required sections: {', '.join(missing)}"

    return True, "Quality check passed"
