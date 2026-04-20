"""
document_utils.py — Shared document processing utilities
=========================================================

Provides text normalisation, word-count targets, and section routing
used throughout the document generation pipeline.

Exports:
    markdown_to_plain_text(md)    — Strip markdown while preserving pipe tables
                                    and mermaid diagram blocks.
    DOC_WORD_TARGETS              — Industry-standard total word counts per doc type.
    get_words_per_section(...)    — Distribute the total word budget across sections.
    SECTIONS_NEEDING_TABLES       — Regex patterns for sections that must include a table.
"""
import re
from typing import Dict, List


def markdown_to_plain_text(md: str) -> str:
    """
    Strip markdown formatting from a single line of text.

    Preserves lines containing tables, mermaid syntax, or diagram declarations
    to maintain document structure during processing.

    Args:
        md: A single line of markdown text.

    Returns:
        The line with inline markdown stripped, or the line unmodified if it
        matches one of the preservation guards above.
    """
    if '|' in md:
        return md.rstrip()

    stripped = md.strip()
    if stripped.startswith('```mermaid') or stripped == '```':
        return md.rstrip()

    if stripped.startswith('flowchart') or stripped.startswith('graph '):
        return md.rstrip()

    if '-->' in md or '--->' in md:
        return md.rstrip()

    t = md

    # Strip HTML tags
    t = re.sub(r'<[^>]+>', '', t)

    # Strip markdown links [text](url) → text
    t = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', t)

    # Strip single backtick inline code
    t = re.sub(r'`([^`]+)`', r'\1', t)

    # Strip bold/italic
    t = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', t)
    t = re.sub(r'\*\*(.+?)\*\*',     r'\1', t)
    t = re.sub(r'\*(.+?)\*',         r'\1', t)
    t = re.sub(r'__(.+?)__',         r'\1', t)
    t = re.sub(r'_(.+?)_',           r'\1', t)

    # Strip heading markers
    t = re.sub(r'^#{1,6}\s+', '', t, flags=re.MULTILINE)

    # Strip horizontal rules
    t = re.sub(r'^---+\s*$',    '', t, flags=re.MULTILINE)
    t = re.sub(r'^\*\*\*+\s*$', '', t, flags=re.MULTILINE)

    # Normalise bullets
    t = re.sub(r'^\s*[-*+]\s+', '  - ', t, flags=re.MULTILINE)
    t = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', t, flags=re.MULTILINE)

    # Collapse excessive blank lines
    t = re.sub(r'\n{3,}', '\n\n', t)

    return '\n'.join(line.rstrip() for line in t.split('\n')).strip()


# ─── Sections that need tables ────────────────────────────────────────────────

SECTIONS_NEEDING_TABLES: List[str] = [
    # Commission Report
    "commission report|commission",
    "commission report|earnings",
    "commission report|sales performance",
    "commission report|payout",
    "commission report|rep",

    # Budget / Finance reports
    "budget report|budget",
    "budget report|expense",
    "budget report|forecast",
    "cost analysis|cost breakdown",
    "cost analysis|comparison",
    "financial statement|summary",
    "expense reimbursement|expenses",
    "tax filing|tax",

    # Sales docs
    "quotation document|pricing",
    "quotation document|line item",
    "sales proposal|pricing",
    "sales proposal|cost",
    "deal summary|metrics",
    "deal summary|financial",

    # HR
    "performance review|rating",
    "performance review|competency",
    "performance review|goals",
    "employee handbook|leave",
    "employee handbook|benefits",

    # Procurement
    "bid evaluation|bid",
    "bid evaluation|vendor",
    "vendor evaluation|criteria",
    "vendor evaluation|score",
    "supplier risk|risk",
    "purchase order|item",
    "purchase requisition|item",
    "inventory report|inventory",
    "delivery acceptance|item",

    # Operations
    "risk assessment|risk",
    "risk assessment|impact",
    "quality control|checklist",
    "quality control|criteria",
    "supplier evaluation|criteria",

    # IT
    "software license|license",
    "it asset|asset",
    "incident report|timeline",
    "system maintenance|schedule",

    # Product
    "competitive analysis|comparison",
    "competitive analysis|feature",
    "product roadmap|milestone",
    "product roadmap|timeline",
    "release notes|changelog",

    # Customer Support
    "sla agreement|metrics",
    "sla agreement|response time",
    "support ticket|summary",
    "customer feedback|score",
]


# ─── Industry-Standard Total Word Counts ──────────────────────────────────────

DOC_WORD_TARGETS: Dict[str, int] = {
    # HR
    "Employee Offer Letter": 350,
    "Employment Contract": 900,
    "Employee Handbook": 2500,
    "Performance Review Report": 600,
    "Leave Approval Letter": 180,
    "Disciplinary Notice": 350,
    "Internship Agreement": 600,
    "Exit Clearance Form": 250,
    "Job Description Document": 500,
    "Training Completion Certificate": 160,

    # Finance
    "Invoice": 200,
    "Purchase Order": 220,
    "Expense Reimbursement Form": 200,
    "Budget Report": 700,
    "Payment Receipt": 150,
    "Vendor Payment Approval": 250,
    "Financial Statement Summary": 600,
    "Tax Filing Summary": 500,
    "Cost Analysis Report": 700,
    "Refund Authorization Form": 200,

    # Legal
    "Non-Disclosure Agreement (NDA)": 1200,
    "Service Agreement": 1400,
    "Partnership Agreement": 1600,
    "Terms of Service": 2000,
    "Privacy Policy": 1800,
    "Vendor Contract": 1200,
    "Licensing Agreement": 1100,
    "Legal Notice Letter": 350,
    "Compliance Certification": 400,
    "Intellectual Property Assignment": 800,

    # Sales
    "Sales Proposal": 800,
    "Sales Contract": 900,
    "Quotation Document": 280,
    "Sales Agreement": 700,
    "Deal Summary Report": 500,
    "Commission Report": 450,
    "Customer Onboarding Document": 600,
    "Discount Approval Form": 200,
    "Lead Qualification Report": 400,
    "Renewal Agreement": 600,

    # Marketing
    "Marketing Campaign Plan": 900,
    "Content Strategy Document": 800,
    "Social Media Plan": 650,
    "Brand Guidelines": 1200,
    "Market Research Report": 1000,
    "Press Release": 450,
    "SEO Strategy Report": 750,
    "Advertising Brief": 500,
    "Email Campaign Plan": 600,
    "Influencer Agreement": 650,

    # IT
    "IT Access Request Form": 220,
    "Incident Report": 500,
    "System Maintenance Report": 380,
    "Software Installation Request": 200,
    "Data Backup Policy": 650,
    "Security Incident Report": 550,
    "IT Asset Allocation Form": 220,
    "Network Access Agreement": 500,
    "Software License Report": 380,
    "System Upgrade Proposal": 600,

    # Operations
    "Standard Operating Procedure (SOP)": 1000,
    "Operations Report": 600,
    "Process Improvement Plan": 650,
    "Risk Assessment Report": 700,
    "Inventory Report": 380,
    "Production Plan": 600,
    "Logistics Plan": 600,
    "Supplier Evaluation Report": 500,
    "Quality Control Checklist": 380,
    "Business Continuity Plan": 900,

    # Customer Support
    "Support Ticket Report": 380,
    "Customer Complaint Report": 450,
    "Customer Feedback Report": 500,
    "SLA Agreement": 800,
    "Support Resolution Report": 380,
    "Customer Escalation Report": 380,
    "Service Improvement Plan": 600,
    "Customer Onboarding Guide": 650,
    "FAQ Document": 600,
    "Support Training Manual": 900,

    # Procurement
    "Vendor Registration Form": 350,
    "Vendor Evaluation Report": 500,
    "Purchase Requisition": 220,
    "Vendor Contract": 1000,
    "Procurement Plan": 650,
    "Bid Evaluation Report": 600,
    "Supplier Risk Assessment": 600,
    "Contract Renewal Notice": 260,
    "Delivery Acceptance Report": 260,
    "Procurement Compliance Checklist": 320,

    # Product Management
    "Product Requirements Document (PRD)": 1200,
    "Product Roadmap": 650,
    "Feature Specification": 600,
    "Release Notes": 380,
    "Product Launch Plan": 800,
    "Competitive Analysis Report": 700,
    "Product Strategy Document": 900,
    "User Persona Document": 500,
    "Product Feedback Report": 500,
    "Product Change Request": 380,
}

DEFAULT_TOTAL_WORDS = 500


def get_words_per_section(doc_type: str, num_sections: int) -> int:
    """Return target words per section."""
    total = DOC_WORD_TARGETS.get(doc_type, DEFAULT_TOTAL_WORDS)
    per   = total // max(num_sections, 1)
    return max(20, min(320, per))