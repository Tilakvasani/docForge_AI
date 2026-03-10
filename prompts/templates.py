"""
DocForge AI — prompts/templates.py
════════════════════════════════════════════════════════════════
Built EXACTLY from the Derived Universal Document Template in
the 100-document analysis report.

REAL frequencies from the document:
  Document Metadata          100%  ← AUTO-GENERATED, no user questions
  Version Control             98%  ← AUTO-GENERATED, no user questions
  Confidentiality Notice      96%  ← AUTO-GENERATED, no user questions
  Executive Summary           92%  ← user answers 2 questions
  Scope and Applicability     88%  ← user answers 2 questions
  Roles & Responsibilities    85%  ← user answers 2 questions
  Definitions & Terminology   82%  ← user answers 2 questions
  Primary Content            ~100% ← user answers 2 questions (MORPHS per doc type)
  Metrics & Performance     stated ← user answers 2 questions (only where logical)
  Exceptions & Limitations  stated ← user answers 2 questions (only where logical)
  References & Dependencies stated ← user answers 2 questions (only where logical)
  Approvals & Sign-off      stated ← user answers 2 questions (always last)

NOTE: 78%, 74%, 71%, 89% were NEVER in the document. Removed.
NOTE: "Incident Timeline", "Root Cause", "User Stories" etc are NOT separate
      sections — they are variants of "Primary Content" with different questions.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  PRIMARY CONTENT QUESTIONS — one section, morphs per doc type
#  This is Group III "Substantive Core" from the Derived Universal Template
# ─────────────────────────────────────────────────────────────────────────────
PRIMARY_CONTENT_QUESTIONS = {
    "SOP": [
        ("pc_q1",
         "Describe the step-by-step procedure this SOP defines. What happens first through to completion?",
         "e.g. 1. Receive request → 2. Verify credentials → 3. Execute task → 4. Log outcome → 5. Notify stakeholders"),
        ("pc_q2",
         "What tools, systems, or checklists are used during this procedure?",
         "e.g. Jira for ticket tracking, Confluence for runbooks, AWS Console for execution"),
    ],
    "SLA": [
        ("pc_q1",
         "What specific services are being committed to, and what are the uptime and response time guarantees?",
         "e.g. Cloud hosting: 99.9% uptime · P1 incident response: 20 min · Scheduled maintenance: Sunday 02:00–04:00 UTC"),
        ("pc_q2",
         "How are service credits calculated when targets are missed, and what is the escalation path?",
         "e.g. Each 0.1% below 99.9% = 10% monthly credit · Escalation: Support → Engineering Lead → CTO"),
    ],
    "Terms of Service": [
        ("pc_q1",
         "What are the core user rights, restrictions, and obligations this agreement establishes?",
         "e.g. Users may not resell access · Company retains all IP · 30-day written termination notice required"),
        ("pc_q2",
         "What is the governing law, dispute resolution method, and jurisdiction for this agreement?",
         "e.g. Governed by laws of Delaware, USA · Disputes via AAA arbitration · Venue: San Francisco, CA"),
    ],
    "Privacy Policy": [
        ("pc_q1",
         "What personal data is collected, for what purpose, and how is it processed, stored, and shared?",
         "e.g. Email, usage data, payment info · Used for service delivery and analytics · Shared with Stripe, AWS, Google Analytics"),
        ("pc_q2",
         "What rights do data subjects have (access, deletion, portability) and how can they exercise them?",
         "e.g. Right to access/delete via privacy@company.com · 30-day response SLA · GDPR and CCPA compliant"),
    ],
    "Employment Contract": [
        ("pc_q1",
         "What are the key employment terms: role, start date, compensation, working hours, and IP ownership?",
         "e.g. Senior Engineer · 1 April 2026 · $120,000/year + equity · 40hrs/week · All IP belongs to employer"),
        ("pc_q2",
         "What are the termination conditions, notice period, and post-employment restrictions (non-compete/NDA)?",
         "e.g. Either party: 30 days notice · Cause: immediate · 12-month non-compete in same vertical"),
    ],
    "NDA": [
        ("pc_q1",
         "What information is classified as confidential under this agreement, and what are the permitted uses?",
         "e.g. All technical, financial, and business info shared during partnership · Permitted: due diligence only"),
        ("pc_q2",
         "What are the obligations of the receiving party, the term duration, and what is explicitly excluded from confidentiality?",
         "e.g. 3-year term · Obligations: no disclosure, need-to-know basis · Excluded: already public, independently developed"),
    ],
    "Product Requirement Document": [
        ("pc_q1",
         "What specific user problem are you solving, who is the target user, and what are the must-have features for this release?",
         "e.g. HR managers spend 40hrs/week on manual docs · Target: 500 enterprise HR teams · Must-have: AI generation, Notion publish"),
        ("pc_q2",
         "What are the key technical and business assumptions this product is built on that must be validated?",
         "e.g. Assumes GDPR compliance needed · LLaMA 3.3 accuracy > 90% · 500+ docs/month volume justified"),
    ],
    "Technical Specification": [
        ("pc_q1",
         "What is the system architecture, key components, and what technical constraints or standards must be met?",
         "e.g. FastAPI backend · PostgreSQL 14 · React frontend · Must support 10,000 concurrent users · REST + WebSocket"),
        ("pc_q2",
         "What are the integration points, external dependencies, and API contracts this system must fulfill?",
         "e.g. Integrates with Notion API, Groq API, Stripe · OAuth 2.0 auth · JSON:API spec · <200ms response SLA"),
    ],
    "Incident Report": [
        ("pc_q1",
         "Describe what happened, when it was detected, what systems were affected, and provide a timeline of key events.",
         "e.g. DB breach detected 14 Mar 03:47 UTC · Affected: PII table ~12,000 records · 04:30 isolated · 06:00 patched"),
        ("pc_q2",
         "What was the confirmed root cause, what containment actions were taken, and what remediation is planned?",
         "e.g. Root cause: unpatched SQL injection · Contained: DB isolated, credentials rotated · Remediation: WAF + pen test Q2"),
    ],
    "Security Policy": [
        ("pc_q1",
         "What are the core security controls this policy mandates? Cover access control, encryption, patching, and incident response.",
         "e.g. MFA required all systems · AES-256 at rest · TLS 1.3 in transit · Critical patches within 24hrs · IR plan tested quarterly"),
        ("pc_q2",
         "What are the consequences of non-compliance, the exception/waiver process, and the audit mechanism?",
         "e.g. Violation: disciplinary action up to termination · Waiver: CISO approval + risk acceptance · Annual SOC 2 audit"),
    ],
    "Customer Onboarding Guide": [
        ("pc_q1",
         "What are the onboarding phases and milestones the customer must complete to reach their first meaningful outcome?",
         "e.g. Week 1: Account setup + data import · Week 2: First workflow live · Week 3: Team training · Week 4: Go-live sign-off"),
        ("pc_q2",
         "What support resources, training materials, and communication touchpoints are provided during onboarding?",
         "e.g. Dedicated CSM for 90 days · Video library · Weekly check-in calls · Slack support channel · Knowledge base"),
    ],
    "Business Proposal": [
        ("pc_q1",
         "Describe the proposed solution, its key components, and why it is the best approach to the identified problem.",
         "e.g. AI-powered DocForge platform · Replaces 40hrs/week manual drafting · LLaMA 3.3 70B · ROI in 6 months"),
        ("pc_q2",
         "What is the investment required, the project timeline with key milestones, and the projected ROI?",
         "e.g. Setup: $45,000 · Annual: $18,000 · Phase 1 MVP: 8 weeks · Phase 2 full deploy: 16 weeks · $240K annual savings"),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION DEFINITIONS
#  Only 4 logical sections shown to user + Approvals.
#  Group I (Metadata, Version Control, Confidentiality) = AUTO-GENERATED.
# ─────────────────────────────────────────────────────────────────────────────

# Sections that appear in every document (Group II from the analysis)
SECTION_EXECUTIVE_SUMMARY = {
    "name": "Executive Summary / Purpose",
    "icon": "◎",
    "freq": "92%",
    "questions": [
        ("es_q1",
         "What is the primary business problem, gap, or regulatory requirement this document addresses — and what is the expected outcome?",
         "e.g. GDPR compliance gap from Q1 audit → full data protection compliance by Q3 2026"),
        ("es_q2",
         "Who is the intended audience and what specific action should they take immediately after reading this?",
         "e.g. All department heads → enforce data handling procedures by April 1"),
    ]
}

SECTION_SCOPE = {
    "name": "Scope and Applicability",
    "icon": "◐",
    "freq": "88%",
    "questions": [
        ("sc_q1",
         "Who and what does this document apply to? (people, systems, departments, geographies)",
         "e.g. All full-time employees, contractors, and third-party vendors operating in EU regions"),
        ("sc_q2",
         "What is deliberately excluded from scope?",
         "e.g. Does not apply to legacy on-premise systems pre-2020 or US-based operations"),
    ]
}

SECTION_ROLES = {
    "name": "Roles and Responsibilities",
    "icon": "◑",
    "freq": "85%",
    "questions": [
        ("rr_q1",
         "Who is ultimately accountable for this document's outcomes, and which roles are directly responsible for executing it?",
         "e.g. Accountable: CISO · Responsible: Engineering Lead, Data Protection Officer"),
        ("rr_q2",
         "Which teams must be consulted before key decisions, and who only needs to be informed after actions are taken?",
         "e.g. Consulted: Legal, Compliance · Informed: All employees, Board of Directors"),
    ]
}

SECTION_DEFINITIONS = {
    "name": "Definitions and Terminology",
    "icon": "◒",
    "freq": "82%",
    "questions": [
        ("df_q1",
         "List the 5–8 key terms or acronyms that must be formally defined to prevent misinterpretation.",
         "e.g. PII, Data Controller, Data Processor, Consent, Breach Notification Window, Data Subject"),
        ("df_q2",
         "Are there any terms that carry a different meaning in this specific department vs general business usage?",
         "e.g. 'User' in Engineering = automated system account; in Marketing = human customer"),
    ]
}

# Primary Content — always present, questions come from PRIMARY_CONTENT_QUESTIONS
SECTION_PRIMARY_CONTENT = {
    "name": "Primary Content",   # label overridden per doc type in get_sections_for_doc_type
    "icon": "◓",
    "freq": "~100%",
    # questions injected dynamically per doc type
}

# Group IV sections — included only where logically relevant
SECTION_METRICS = {
    "name": "Metrics and Performance Standards",
    "icon": "●",
    "freq": "stated",
    "questions": [
        ("mt_q1",
         "What are the specific, measurable KPIs or service level targets that define success for this document?",
         "e.g. 99.9% uptime · Response < 200ms · Breach notification within 72hrs · 100% employee policy sign-off"),
        ("mt_q2",
         "What penalties, corrective actions, or escalation steps apply when performance falls below targets?",
         "e.g. SLA breach → 10% service credit · Policy violation → formal HR warning · P1 breach → exec escalation"),
    ]
}

SECTION_EXCEPTIONS = {
    "name": "Exceptions and Limitations",
    "icon": "◔",
    "freq": "stated",
    "questions": [
        ("ex_q1",
         "Under what specific conditions or pre-approved scenarios is deviation from this document permitted?",
         "e.g. Emergency outage: CTO written approval · Pre-approved regulatory audit access · Disaster recovery mode"),
        ("ex_q2",
         "What external events beyond organizational control suspend or modify obligations in this document?",
         "e.g. Government-mandated shutdown, regulatory framework change, third-party vendor insolvency"),
    ]
}

SECTION_REFERENCES = {
    "name": "References and Dependencies",
    "icon": "◕",
    "freq": "stated",
    "questions": [
        ("rf_q1",
         "Which regulatory frameworks, international standards, or parent policies does this document reference?",
         "e.g. GDPR Article 32, NIST SP 800-53, ISO 27001:2022, company Information Security Policy v3.1"),
        ("rf_q2",
         "Which internal documents, systems, or processes must stay synchronized with this document?",
         "e.g. HR Handbook v3.2, Employee Portal API, Data Retention Policy — review triggered within 30 days of change"),
    ]
}

SECTION_APPROVALS = {
    "name": "Approvals and Sign-off",
    "icon": "◉",
    "freq": "stated",
    "questions": [
        ("ap_q1",
         "Who are the required approvers by name and role, and what is the sign-off workflow and signature method?",
         "e.g. Legal Counsel → CISO → CEO · Sequential approval · DocuSign electronic signature"),
        ("ap_q2",
         "How often is this document reviewed, and what specific events trigger an immediate unscheduled revision?",
         "e.g. Annual review every March · Triggers: regulatory change, data breach, M&A activity, major product launch"),
    ]
}


# ─────────────────────────────────────────────────────────────────────────────
#  DOC TYPE → SECTION LIST
#  Exactly which sections (from the 4 groups) apply to each doc type.
#  Group I (Metadata, Version Control, Confidentiality) is ALWAYS auto-generated.
# ─────────────────────────────────────────────────────────────────────────────
DOC_TYPE_SECTION_MAP = {
    # ── Legal / contractual docs ── core = Terms and Conditions ────────────────
    "Terms of Service": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,
        SECTION_DEFINITIONS,       # "User", "Service", "Governing Law"
        SECTION_PRIMARY_CONTENT,   # user rights, restrictions, governing law
        SECTION_EXCEPTIONS,        # liability limits, disclaimers
        SECTION_REFERENCES,        # GDPR, CalOPPA, jurisdiction
        SECTION_APPROVALS,
    ],
    "Privacy Policy": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,
        SECTION_DEFINITIONS,       # PII, Data Controller, Data Processor
        SECTION_ROLES,             # DPO, data owners, processors
        SECTION_PRIMARY_CONTENT,   # data collection, use, sharing, deletion
        SECTION_EXCEPTIONS,        # cookies, third-party links
        SECTION_REFERENCES,        # GDPR, CCPA, CalOPPA
        SECTION_APPROVALS,
    ],
    "Employment Contract": [
        SECTION_EXECUTIVE_SUMMARY, # role summary, start date, compensation
        SECTION_SCOPE,             # jurisdiction, legal entity
        SECTION_DEFINITIONS,       # Employer, Employee, Confidential Info
        SECTION_ROLES,             # reporting structure, duties
        SECTION_PRIMARY_CONTENT,   # salary, hours, IP, termination
        SECTION_REFERENCES,        # Employment law, HR Handbook
        SECTION_APPROVALS,
    ],
    "NDA": [
        SECTION_EXECUTIVE_SUMMARY, # parties, relationship, purpose
        SECTION_SCOPE,             # what info is covered, geography, term
        SECTION_DEFINITIONS,       # Confidential Information, Receiving/Disclosing Party
        SECTION_PRIMARY_CONTENT,   # obligations, permitted use, duration
        SECTION_EXCEPTIONS,        # public domain, court-ordered disclosure
        SECTION_REFERENCES,        # governing law, jurisdiction
        SECTION_APPROVALS,
    ],
    # ── Operational / procedural docs ──────────────────────────────────────────
    "SOP": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,
        SECTION_DEFINITIONS,
        SECTION_ROLES,             # who executes each step
        SECTION_PRIMARY_CONTENT,   # numbered step-by-step procedure
        SECTION_EXCEPTIONS,        # deviation approval
        SECTION_REFERENCES,        # related SOPs, tools, standards
        SECTION_APPROVALS,
    ],
    "SLA": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,             # which services are covered
        SECTION_DEFINITIONS,       # Uptime, Incident, Service Credit, Rider
        SECTION_ROLES,             # provider vs client obligations
        SECTION_PRIMARY_CONTENT,   # service commitments, uptime targets, credits
        SECTION_METRICS,           # uptime %, response times, credit calculation
        SECTION_EXCEPTIONS,        # force majeure, scheduled downtime
        SECTION_REFERENCES,
        SECTION_APPROVALS,
    ],
    # ── Product / technical docs ────────────────────────────────────────────────
    "Product Requirement Document": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,             # in/out of this release
        SECTION_DEFINITIONS,
        SECTION_ROLES,             # PM, Eng, Design, QA ownership
        SECTION_PRIMARY_CONTENT,   # user stories, must-have features, assumptions
        SECTION_METRICS,           # success KPIs, OKRs, acceptance criteria
        SECTION_REFERENCES,
        SECTION_APPROVALS,
    ],
    "Technical Specification": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,
        SECTION_DEFINITIONS,       # APIs, data models, acronyms
        SECTION_ROLES,             # architects, devs, QA
        SECTION_PRIMARY_CONTENT,   # architecture, APIs, DB schema, constraints
        SECTION_METRICS,           # latency, throughput, uptime targets
        SECTION_EXCEPTIONS,        # known constraints, out-of-scope features
        SECTION_REFERENCES,        # parent PRD, third-party APIs, standards
        SECTION_APPROVALS,
    ],
    # ── Incident / security docs ────────────────────────────────────────────────
    "Incident Report": [
        SECTION_EXECUTIVE_SUMMARY, # brief: what, when, impact summary
        SECTION_SCOPE,             # systems and data impacted
        SECTION_ROLES,             # incident commander, responders, comms lead
        SECTION_PRIMARY_CONTENT,   # timeline + root cause + containment + remediation
        SECTION_METRICS,           # MTTR, MTTD, SLA breach impact
        SECTION_EXCEPTIONS,        # still under investigation items
        SECTION_REFERENCES,        # security policy, DR plan, regulators notified
        SECTION_APPROVALS,         # post-incident sign-off
    ],
    "Security Policy": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,
        SECTION_DEFINITIONS,       # CIA Triad, PII, threat actor, zero-day
        SECTION_ROLES,             # CISO, security team, all employees
        SECTION_PRIMARY_CONTENT,   # access control, encryption, patch mgmt, IR
        SECTION_METRICS,           # scan frequency, patch SLA, audit cadence
        SECTION_EXCEPTIONS,        # waiver process, approved exceptions
        SECTION_REFERENCES,        # NIST, ISO 27001, SOC 2, GDPR
        SECTION_APPROVALS,
    ],
    # ── Customer / commercial docs ──────────────────────────────────────────────
    "Customer Onboarding Guide": [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,             # which plans/products covered
        SECTION_DEFINITIONS,       # time-to-value, DAU, churn, CSM
        SECTION_ROLES,             # CSM, customer champion, support team
        SECTION_PRIMARY_CONTENT,   # phases, milestones, support resources
        SECTION_METRICS,           # 30/60/90 day KPIs, churn triggers
        SECTION_EXCEPTIONS,        # what support is excluded
        SECTION_REFERENCES,        # product docs, SLA, support portal
        SECTION_APPROVALS,
    ],
    "Business Proposal": [
        SECTION_EXECUTIVE_SUMMARY, # problem, opportunity, why now
        SECTION_SCOPE,             # what is included in this proposal
        SECTION_ROLES,             # account team, client stakeholders
        SECTION_PRIMARY_CONTENT,   # solution description + ROI + timeline + cost
        SECTION_METRICS,           # ROI targets, success KPIs
        SECTION_EXCEPTIONS,        # what is NOT included, assumptions
        SECTION_REFERENCES,        # case studies, integrations
        SECTION_APPROVALS,         # proposal validity, sign-off
    ],
}

# Primary Content section name per doc type (displayed in UI)
PRIMARY_CONTENT_LABEL = {
    "SOP":                          "Procedures and Operational Details",
    "SLA":                          "Service Commitments and Terms",
    "Terms of Service":             "Terms, Rights, and Obligations",
    "Privacy Policy":               "Data Processing and User Rights",
    "Employment Contract":          "Employment Terms and Conditions",
    "NDA":                          "Confidentiality Obligations and Terms",
    "Product Requirement Document": "Problem Statement and Requirements",
    "Technical Specification":      "Architecture and Technical Details",
    "Incident Report":              "Incident Timeline and Root Cause",
    "Security Policy":              "Security Controls and Procedures",
    "Customer Onboarding Guide":    "Onboarding Journey and Milestones",
    "Business Proposal":            "Proposed Solution and Investment",
}


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — used by generator_form.py and generator.py
# ─────────────────────────────────────────────────────────────────────────────
def get_sections_for_doc_type(doc_type: str) -> list:
    """
    Returns ordered list of section dicts for the Streamlit form.
    Each dict: { name, icon, freq, questions: [(key, label, placeholder)] }
    Primary Content questions are injected dynamically.
    """
    template = DOC_TYPE_SECTION_MAP.get(doc_type, [
        SECTION_EXECUTIVE_SUMMARY,
        SECTION_SCOPE,
        SECTION_ROLES,
        SECTION_PRIMARY_CONTENT,
        SECTION_APPROVALS,
    ])

    result = []
    for sec in template:
        entry = dict(sec)  # copy so we don't mutate the original

        if entry["name"] == "Primary Content":
            # Inject doc-type-specific label and questions
            entry["name"]      = PRIMARY_CONTENT_LABEL.get(doc_type, "Primary Content")
            entry["questions"] = PRIMARY_CONTENT_QUESTIONS.get(doc_type, [
                ("pc_q1", "Describe the main content, procedure, or terms of this document.", ""),
                ("pc_q2", "What tools or systems are involved?", ""),
            ])

        result.append(entry)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — Document Generation Prompt
# ─────────────────────────────────────────────────────────────────────────────
PHASE_2_PROMPT = """You are an enterprise documentation assistant.

The system has analyzed 100 enterprise documents across multiple departments and document types. \
From this analysis we identified the following common structural sections:
- Document Metadata · Executive Summary / Purpose · Scope and Applicability
- Definitions and Terminology · Roles and Responsibilities · Primary Content / Procedures / Terms
- Metrics and Performance Standards · Exceptions and Limitations · References · Approvals
- Version Control · Confidentiality Notice

Input provided by the system:
- Industry:       {industry}
- Department:     {department}
- Document Type:  {document_type}

PHASE 2 — GENERATE DOCUMENT

User answers from the guided form:
{answers_block}

════════════════════════════════════════════
DOCUMENT GENERATION RULES
════════════════════════════════════════════

1. ALWAYS start the document with these three auto-generated blocks (no user input needed):

   ┌─ DOCUMENT METADATA ──────────────────────────────────────┐
   │  Title · Reference ID · Department · Author · Status     │
   │  Security Classification · Effective Date · Review Date  │
   └──────────────────────────────────────────────────────────┘

   ┌─ VERSION CONTROL TABLE ──────────────────────────────────┐
   │  Version │ Date │ Author │ Description of Change         │
   └──────────────────────────────────────────────────────────┘

   ┌─ CONFIDENTIALITY NOTICE ─────────────────────────────────┐
   │  Standard proprietary and IP protection statement        │
   └──────────────────────────────────────────────────────────┘

2. Then generate sections using the user answers above as headings — in the order provided.
3. Write in professional enterprise tone — formal, clear, authoritative.
4. EXPAND user answers into detailed, structured content. Do NOT repeat answers verbatim.
5. For Roles and Responsibilities: always use RACI format (Responsible · Accountable · Consulted · Informed).
6. Include governance clauses where relevant: Force Majeure, Severability, Governing Law.
7. Ensure natural logical flow between sections.
8. Do NOT hallucinate — only expand logically from the user's provided answers.
9. Target length:
   - Standard (SOP, NDA, Employment Contract, Incident Report): 700–1000 words
   - Complex (SLA, PRD, Security Policy, Technical Spec, Privacy Policy): 1000–1500 words
10. Output must be ready for real business use without further editing.

Generate the complete enterprise document now:"""


def get_prompt_template(doc_type: str = None) -> str:
    return PHASE_2_PROMPT


def build_answers_block(answers: dict, sections: list) -> str:
    """Formats Q&A answers into the prompt's {answers_block} variable."""
    lines = []
    for sec in sections:
        lines.append(f"\n── {sec['name'].upper()} ──")
        for (key, label, _ph) in sec.get("questions", []):
            val = answers.get(key, "").strip() or "Not provided"
            lines.append(f"Q: {label}")
            lines.append(f"A: {val}\n")
    return "\n".join(lines)