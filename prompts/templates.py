"""
DocForge AI — prompts/docforge_prompts.py
════════════════════════════════════════════════════════════════
Enhanced prompt system for 100+ industry-standard documents.
Covers: question generation, full document generation, section
editing/enhancement, and structural metadata per document type.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  DOCUMENT STRUCTURAL METADATA
#  Defines which documents need tables, flowcharts, RACI, signature blocks, etc.
#  All document types and departments come from your database — this metadata
#  layer tells the LLM HOW to render each one.
# ─────────────────────────────────────────────────────────────────────────────

DOC_STRUCTURE_METADATA = {
    # ── HR ───────────────────────────────────────────────────────────────────
    "Employee Offer Letter": {
        "has_table": True,           # Compensation & benefits table
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Compensation breakdown (Base, Bonus, Equity, Benefits)",
        "tone": "formal_warm",
        "doc_purpose": "Official employment offer to a candidate",
    },
    "Employment Contract": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Key terms summary (Role, Start Date, Salary, Notice Period, Location)",
        "tone": "legal_formal",
        "doc_purpose": "Legally binding employment agreement",
    },
    "Employee Handbook": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Leave entitlement table (Leave Type, Days Per Year, Carry Forward)",
        "flowchart_hint": "Grievance escalation process (Employee → Manager → HR → Senior Leadership)",
        "tone": "professional_friendly",
        "doc_purpose": "Company-wide policies and employee guidelines reference",
    },
    "Performance Review Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "KPI scorecard (Objective, Target, Actual, Score, Weight)",
        "raci_hint": "Performance review process RACI (Reviewer, HR, Manager, Employee)",
        "tone": "objective_professional",
        "doc_purpose": "Structured employee performance evaluation",
    },
    "Leave Approval Letter": {
        "has_table": False,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "tone": "formal_brief",
        "doc_purpose": "Official approval of employee leave request",
    },
    "Disciplinary Notice": {
        "has_table": False,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "flowchart_hint": "Disciplinary procedure flow (Verbal Warning → Written Warning → Final Warning → Termination)",
        "tone": "stern_formal",
        "doc_purpose": "Formal notice of disciplinary action",
    },
    "Internship Agreement": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Internship terms (Duration, Stipend, Working Hours, Department, Supervisor)",
        "tone": "formal_warm",
        "doc_purpose": "Internship terms and conditions agreement",
    },
    "Exit Clearance Form": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Department clearance checklist (Department, Item, Status, Cleared By, Date)",
        "flowchart_hint": "Exit clearance process (IT → Finance → HR → Manager → Facilities)",
        "tone": "procedural_formal",
        "doc_purpose": "Employee exit process and asset return tracking",
    },
    "Job Description Document": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": False,
        "table_hint": "Competency matrix (Skill, Level Required: Beginner/Intermediate/Expert)",
        "tone": "professional_engaging",
        "doc_purpose": "Detailed role requirements for recruitment",
    },
    "Training Completion Certificate": {
        "has_table": False,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "tone": "formal_celebratory",
        "doc_purpose": "Official recognition of training completion",
    },

    # ── FINANCE ──────────────────────────────────────────────────────────────
    "Invoice": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Line items (Description, Qty, Unit Price, Tax %, Total)",
        "tone": "professional_brief",
        "doc_purpose": "Formal billing document for goods or services rendered",
    },
    "Purchase Order": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Order items (Item Code, Description, Qty, Unit Cost, Total Cost)",
        "tone": "formal_precise",
        "doc_purpose": "Authorised order for goods or services from a vendor",
    },
    "Expense Reimbursement Form": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Expense items (Date, Category, Description, Amount, Receipt Attached)",
        "tone": "formal_procedural",
        "doc_purpose": "Employee expense claim for reimbursement",
    },
    "Budget Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Budget vs Actuals (Category, Budgeted Amount, Actual Spend, Variance, % Variance)",
        "tone": "analytical_formal",
        "doc_purpose": "Financial budget performance analysis",
    },
    "Cost Analysis Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Cost breakdown (Cost Component, Current Cost, Projected Cost, Savings Opportunity)",
        "tone": "analytical_formal",
        "doc_purpose": "Detailed cost analysis with optimization recommendations",
    },

    # ── LEGAL ─────────────────────────────────────────────────────────────────
    "Non-Disclosure Agreement (NDA)": {
        "has_table": False,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "tone": "legal_formal",
        "doc_purpose": "Legally binding confidentiality agreement between parties",
    },
    "Service Agreement": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Service scope and SLA table (Service, Description, SLA, Penalty for Breach)",
        "tone": "legal_formal",
        "doc_purpose": "Agreement defining service delivery terms and conditions",
    },
    "Partnership Agreement": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Profit/loss sharing schedule (Partner, Contribution %, Profit Share %, Loss Share %)",
        "tone": "legal_formal",
        "doc_purpose": "Formal partnership terms between two or more entities",
    },
    "Terms of Service": {
        "has_table": False,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": False,
        "tone": "legal_formal",
        "doc_purpose": "User-facing legal terms governing product or service use",
    },
    "Privacy Policy": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": False,
        "table_hint": "Data collection summary (Data Type, Purpose, Retention Period, Third-Party Shared)",
        "tone": "legal_clear",
        "doc_purpose": "Data privacy practices and user rights disclosure",
    },

    # ── SALES ─────────────────────────────────────────────────────────────────
    "Sales Proposal": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Pricing tiers (Package, Features Included, Price, Recommended For)",
        "flowchart_hint": "Proposed implementation timeline (Discovery → Onboarding → Go-Live → Support)",
        "tone": "persuasive_professional",
        "doc_purpose": "Compelling sales proposal to win a prospect's business",
    },
    "Quotation Document": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Quote line items (Product/Service, Qty, Unit Price, Discount, Net Total)",
        "tone": "formal_precise",
        "doc_purpose": "Formal price quotation for a client",
    },
    "Commission Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Commission summary (Sales Rep, Total Sales, Rate %, Commission Earned, YTD)",
        "tone": "analytical_formal",
        "doc_purpose": "Sales commission calculation and summary report",
    },
    "Customer Onboarding Document": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Onboarding milestones (Phase, Task, Owner, Due Date, Status)",
        "flowchart_hint": "Onboarding journey (Contract Signed → Kickoff → Setup → Training → Go-Live)",
        "raci_hint": "Onboarding RACI (Account Manager, Customer Success, IT, Client POC)",
        "tone": "professional_welcoming",
        "doc_purpose": "Structured customer onboarding plan and welcome guide",
    },

    # ── MARKETING ─────────────────────────────────────────────────────────────
    "Marketing Campaign Plan": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Campaign budget allocation (Channel, Budget, Expected Reach, KPI, Owner)",
        "flowchart_hint": "Campaign execution timeline (Planning → Creative → Launch → Optimize → Report)",
        "raci_hint": "Campaign RACI (Marketing Manager, Designer, Copywriter, Digital Analyst, Approver)",
        "tone": "strategic_professional",
        "doc_purpose": "End-to-end marketing campaign planning document",
    },
    "Market Research Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Competitive analysis matrix (Competitor, Market Share, Strengths, Weaknesses, Pricing)",
        "tone": "analytical_formal",
        "doc_purpose": "Structured market and competitive landscape research",
    },
    "Press Release": {
        "has_table": False,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "tone": "journalistic_formal",
        "doc_purpose": "Official public announcement for media distribution",
    },
    "Brand Guidelines": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": False,
        "table_hint": "Color palette (Color Name, HEX, RGB, CMYK, Usage Context)",
        "tone": "authoritative_creative",
        "doc_purpose": "Brand identity standards and usage guidelines",
    },

    # ── IT ────────────────────────────────────────────────────────────────────
    "IT Access Request Form": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Access permissions requested (System/Application, Access Level, Business Justification)",
        "flowchart_hint": "Access approval workflow (Request → Manager Approval → IT Review → Provisioning → Notification)",
        "tone": "procedural_formal",
        "doc_purpose": "Formal request for system or application access",
    },
    "Security Incident Report": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Incident impact summary (Affected System, Data Exposed, Users Affected, Severity Level)",
        "flowchart_hint": "Incident response timeline (Detection → Containment → Eradication → Recovery → Lessons Learned)",
        "raci_hint": "Incident response RACI (CISO, IT Security, Affected Team Lead, Legal, Communications)",
        "tone": "technical_formal",
        "doc_purpose": "Security incident documentation and response report",
    },
    "System Maintenance Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Maintenance log (System, Task Performed, Start Time, End Time, Status, Technician)",
        "tone": "technical_formal",
        "doc_purpose": "Record of system maintenance activities and outcomes",
    },
    "System Upgrade Proposal": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Cost-benefit analysis (Current State Cost, Upgrade Cost, Annual Savings, ROI, Payback Period)",
        "flowchart_hint": "Upgrade implementation plan (Assessment → Procurement → Testing → Rollout → Validation)",
        "raci_hint": "Upgrade project RACI (IT Manager, Project Lead, Finance, Business Owner, Vendor)",
        "tone": "technical_persuasive",
        "doc_purpose": "Business case and technical plan for system upgrade",
    },

    # ── OPERATIONS ────────────────────────────────────────────────────────────
    "Standard Operating Procedure (SOP)": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Materials/tools required (Item, Specification, Quantity, Purpose)",
        "flowchart_hint": "Step-by-step procedure flow with decision points",
        "raci_hint": "Procedure execution RACI (Operator, Supervisor, Quality, Safety Officer)",
        "tone": "procedural_precise",
        "doc_purpose": "Standardised instructions for repeatable operational tasks",
    },
    "Risk Assessment Report": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Risk register (Risk ID, Description, Likelihood (1-5), Impact (1-5), Risk Score, Mitigation, Owner)",
        "flowchart_hint": "Risk escalation matrix (Low → Monitor, Medium → Action Plan, High → Executive Review, Critical → Immediate Response)",
        "tone": "analytical_formal",
        "doc_purpose": "Structured risk identification, assessment, and mitigation plan",
    },
    "Business Continuity Plan": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Critical business functions (Function, RTO, RPO, Backup System, Owner)",
        "flowchart_hint": "Business continuity activation flow (Incident → Assessment → Activation → Recovery → Resumption → Review)",
        "raci_hint": "BCP execution RACI (BCP Lead, IT, Operations, HR, Communications, Executive Sponsor)",
        "tone": "strategic_precise",
        "doc_purpose": "Plan to maintain business operations during and after a disruption",
    },
    "Process Improvement Plan": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Improvement initiatives (Initiative, Current State KPI, Target KPI, Owner, Timeline, Investment)",
        "flowchart_hint": "PDCA or DMAIC improvement cycle",
        "raci_hint": "Improvement project RACI",
        "tone": "strategic_analytical",
        "doc_purpose": "Structured plan to optimise and improve a business process",
    },

    # ── CUSTOMER SUPPORT ──────────────────────────────────────────────────────
    "SLA Agreement": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "SLA commitments (Service, Priority Level, Response Time, Resolution Time, Penalty)",
        "flowchart_hint": "Ticket escalation path (L1 Support → L2 Technical → L3 Engineering → Account Manager)",
        "tone": "legal_precise",
        "doc_purpose": "Service level commitments and accountability framework",
    },
    "Customer Escalation Report": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Escalation timeline (Date/Time, Action Taken, Owner, Customer Response, Status)",
        "flowchart_hint": "Escalation resolution flow (Escalation Received → Root Cause Analysis → Resolution Plan → Client Communication → Closure)",
        "tone": "empathetic_professional",
        "doc_purpose": "Documented account of a customer escalation and resolution",
    },
    "Support Training Manual": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Training modules (Module, Topic, Duration, Assessment Method, Pass Mark)",
        "flowchart_hint": "Support ticket handling flow (Receive → Categorise → Diagnose → Resolve → Close → Follow-up)",
        "tone": "instructional_clear",
        "doc_purpose": "Training guide for customer support agents",
    },

    # ── PROCUREMENT ───────────────────────────────────────────────────────────
    "Vendor Evaluation Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Vendor scorecard (Vendor Name, Quality Score, Price Score, Delivery Score, Support Score, Total Score, Recommendation)",
        "tone": "analytical_formal",
        "doc_purpose": "Objective assessment of vendor performance or selection",
    },
    "Bid Evaluation Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Bid comparison matrix (Bidder, Technical Score, Financial Score, Experience Score, Weighted Total, Rank)",
        "raci_hint": "Evaluation panel RACI (Lead Evaluator, Technical Expert, Finance, Procurement Head, Approver)",
        "tone": "analytical_formal",
        "doc_purpose": "Structured evaluation of submitted bids for procurement",
    },
    "Procurement Compliance Checklist": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Compliance items (Requirement, Policy Reference, Status: Compliant/Non-Compliant/N/A, Evidence, Remarks)",
        "tone": "procedural_formal",
        "doc_purpose": "Audit-ready procurement compliance verification",
    },

    # ── PRODUCT MANAGEMENT ────────────────────────────────────────────────────
    "Product Requirements Document (PRD)": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": True,
        "has_signature_block": True,
        "table_hint": "Feature requirements (Feature ID, Feature Name, Priority: Must/Should/Could, Acceptance Criteria, Owner)",
        "flowchart_hint": "Product development lifecycle (Discovery → Design → Development → QA → Launch → Feedback)",
        "raci_hint": "PRD stakeholder RACI (Product Manager, Engineering Lead, Design, QA, Business Stakeholder)",
        "tone": "strategic_technical",
        "doc_purpose": "Comprehensive product feature requirements for engineering",
    },
    "Product Roadmap": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Roadmap timeline (Initiative, Q1, Q2, Q3, Q4, Status, Owner)",
        "flowchart_hint": "Release pipeline (Backlog → Planning → Development → Release → Post-Launch Review)",
        "tone": "strategic_visual",
        "doc_purpose": "Strategic product timeline and initiative prioritisation",
    },
    "Competitive Analysis Report": {
        "has_table": True,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Competitive comparison matrix (Feature/Attribute, Our Product, Competitor A, Competitor B, Competitor C)",
        "tone": "analytical_strategic",
        "doc_purpose": "In-depth analysis of competitive landscape for product positioning",
    },
    "Feature Specification": {
        "has_table": True,
        "has_flowchart": True,
        "has_raci": False,
        "has_signature_block": True,
        "table_hint": "Acceptance criteria (Scenario, Given, When, Then, Priority)",
        "flowchart_hint": "Feature user flow (Entry Point → User Action → System Response → Output/Next State)",
        "tone": "technical_precise",
        "doc_purpose": "Detailed specification for a product feature for development",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_doc_metadata(doc_type: str) -> dict:
    """Fetch structural metadata for a doc type. Returns safe defaults if not found."""
    return DOC_STRUCTURE_METADATA.get(doc_type, {
        "has_table": False,
        "has_flowchart": False,
        "has_raci": False,
        "has_signature_block": True,
        "tone": "professional_formal",
        "doc_purpose": "Professional business document",
    })


def build_structure_instructions(meta: dict) -> str:
    """Build dynamic rendering instructions based on doc metadata."""
    instructions = []

    if meta.get("has_table"):
        hint = meta.get("table_hint", "")
        instructions.append(
            f"- TABLE REQUIRED: Include a professional markdown table. Suggested content: {hint}. "
            f"Use bold headers, align columns clearly, and populate with realistic placeholder data."
        )

    if meta.get("has_flowchart"):
        hint = meta.get("flowchart_hint", "")
        instructions.append(
            f"- FLOWCHART REQUIRED: Insert a Mermaid.js flowchart diagram using ```mermaid code block. "
            f"Suggested flow: {hint}. Use TD (top-down) direction. Label each step clearly. "
            f"Use decision diamonds ({{...}}) where the process branches."
        )

    if meta.get("has_raci"):
        hint = meta.get("raci_hint", "")
        instructions.append(
            f"- RACI MATRIX REQUIRED: Include a RACI responsibility table. Suggested roles: {hint}. "
            f"Columns: Activity | Responsible | Accountable | Consulted | Informed. "
            f"Cover at least 6-8 key activities for this document type."
        )

    if meta.get("has_signature_block"):
        instructions.append(
            "- SIGNATURE BLOCK REQUIRED: End the document with a formal Approvals & Sign-off section "
            "containing a markdown table with columns: Role | Name | Signature | Date. "
            "Include at least 3 relevant approver roles for this document type."
        )

    if not instructions:
        instructions.append(
            "- This document type does not require tables, flowcharts, or RACI matrices. "
            "Focus on clear, structured prose with well-organised sections."
        )

    return "\n".join(instructions)


def build_answers_block(answers: dict, sections: list) -> str:
    """Format user answers by section for injection into the generation prompt."""
    lines = []
    for sec in sections:
        lines.append(f"\n[SECTION: {sec['name']}]")
        for (key, label, _) in sec.get("questions", []):
            val = answers.get(key, "Not provided")
            lines.append(f"  Q: {label}")
            lines.append(f"  A: {val}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1 — QUESTION GENERATION PROMPT
#  Called when sections are fetched from DB and sent to LLM to generate
#  contextual questions section by section. Questions saved back to DB.
# ─────────────────────────────────────────────────────────────────────────────

PHASE_1_QUESTION_GEN_PROMPT = """You are DocForge AI — an expert enterprise documentation specialist.

Your task: Generate smart, targeted questions for ONE SECTION of a professional business document.
These questions will be shown to the user in the UI. Their answers will be used to generate a
complete industry-standard document.

DOCUMENT TYPE: {doc_type}
DEPARTMENT: {department}
INDUSTRY: {industry}
COMPANY SIZE: {company_size}

CURRENT SECTION: {section_name}
SECTION PURPOSE: {section_purpose}

REQUIREMENTS:
1. Generate exactly {question_count} questions for this section.
2. Questions must be specific to {doc_type} — not generic.
3. Ask for concrete details: names, dates, numbers, percentages, policies, procedures.
4. Use plain, professional English. No jargon.
5. Each question should unlock a different piece of information needed for this section.
6. Order questions from most important to least important.
7. Do NOT ask for information already covered in previous sections: {completed_sections}

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:
{{
  "section": "{section_name}",
  "questions": [
    {{
      "key": "unique_snake_case_key",
      "label": "Clear question text shown to user",
      "input_type": "text|textarea|date|number|select",
      "placeholder": "Example answer to guide the user",
      "required": true
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2 — FULL DOCUMENT GENERATION PROMPT
#  Called after ALL section answers are collected from the user.
#  All Q&A data + user inputs are combined and sent to LLM in one call.
# ─────────────────────────────────────────────────────────────────────────────

PHASE_2_DOCUMENT_GEN_PROMPT = """You are DocForge AI — an enterprise documentation specialist with 20+ years of experience
creating industry-standard business documents across all departments and industries.

════════════════════════════════════════════════════════════
DOCUMENT BRIEF
════════════════════════════════════════════════════════════
Document Type   : {doc_type}
Department      : {department}
Industry        : {industry}
Company Name    : {company_name}
Company Size    : {company_size}
Document Purpose: {doc_purpose}
Tone            : {tone}

════════════════════════════════════════════════════════════
USER-PROVIDED INFORMATION (Answers by Section)
════════════════════════════════════════════════════════════
{answers_block}

════════════════════════════════════════════════════════════
DOCUMENT SECTIONS TO GENERATE (from database)
════════════════════════════════════════════════════════════
{section_list}

════════════════════════════════════════════════════════════
STRUCTURAL REQUIREMENTS FOR THIS DOCUMENT TYPE
════════════════════════════════════════════════════════════
{structure_instructions}

════════════════════════════════════════════════════════════
GENERATION RULES — FOLLOW EXACTLY
════════════════════════════════════════════════════════════
FORMATTING:
1. Start with a professional document header:
   - Company name and logo placeholder: [COMPANY LOGO]
   - Document title (bold, centered)
   - Document reference number: DOC-{department_code}-[AUTO]
   - Version, Date, Prepared By, Approved By
   - Horizontal rule (---)

2. Use ## for each section heading (exactly matching the section names from the database).
3. Write 200–350 words per section — expand user answers into professional prose.
   Do NOT copy user answers verbatim. Transform them into polished business language.
4. Maintain consistent terminology throughout (e.g., always use the same job title,
   company name, product name as provided).

CONTENT QUALITY:
5. Write as a subject matter expert, not as a template filler.
6. Include specific details from user answers — dates, names, numbers, percentages.
7. For any field marked "Not provided", use a realistic professional placeholder in [brackets].
8. Legal and compliance language should be precise and unambiguous.
9. Each section must flow logically into the next.

TABLES & DIAGRAMS (only include what STRUCTURE REQUIREMENTS specifies):
10. Tables: Use markdown table format with bold column headers. Include realistic data rows.
11. Flowcharts: Use Mermaid.js ```mermaid code block. Direction: TD. Min 5 nodes.
12. RACI: Full matrix covering all key activities for this document type.
13. All tables must be complete — no empty cells, use "N/A" or "TBD" where appropriate.

SIGN-OFF:
14. Always end with an ## Approvals & Sign-off section with a signature table if required.
15. The final line should be: *This document is confidential and intended solely for the
    named recipients. Unauthorised distribution is prohibited.*

Generate the complete, professional {doc_type} now:"""


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3A — SECTION ENHANCEMENT PROMPT
#  Called when user clicks "Enhance" on a single section in the editor.
#  Only that section's content is sent to the LLM.
# ─────────────────────────────────────────────────────────────────────────────

PHASE_3A_SECTION_ENHANCE_PROMPT = """You are DocForge AI — an enterprise documentation specialist.

Your task: ENHANCE a single section of an existing business document.

DOCUMENT TYPE: {doc_type}
DEPARTMENT: {department}
SECTION NAME: {section_name}

CURRENT SECTION CONTENT:
\"\"\"
{current_section_content}
\"\"\"

ENHANCEMENT INSTRUCTIONS:
1. Preserve all factual information (names, dates, numbers, percentages).
2. Improve professional tone and readability.
3. Expand thin content to 200–350 words with industry-appropriate language.
4. Add structure if missing: sub-headings, bullet points, or numbered lists where appropriate.
5. Strengthen opening and closing sentences.
6. Remove redundant phrases and tighten language.
7. If this section should contain a table based on its context, add one.
8. Do NOT change the meaning or intent of any statement.
9. Do NOT add new facts that were not in the original.

OUTPUT: Return ONLY the enhanced section content (starting from the section heading ##).
Do not include any preamble, explanation, or surrounding context."""


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3B — SECTION EDIT PROMPT
#  Called when user makes a specific edit instruction via text input.
# ─────────────────────────────────────────────────────────────────────────────

PHASE_3B_SECTION_EDIT_PROMPT = """You are DocForge AI — an enterprise documentation specialist.

Your task: EDIT a single section of a business document based on specific user instructions.

DOCUMENT TYPE: {doc_type}
DEPARTMENT: {department}
SECTION NAME: {section_name}

CURRENT SECTION CONTENT:
\"\"\"
{current_section_content}
\"\"\"

USER'S EDIT INSTRUCTION:
\"{edit_instruction}\"

EDITING RULES:
1. Apply the edit instruction precisely and completely.
2. Preserve all content that the instruction does NOT ask to change.
3. Maintain the professional tone and document style.
4. Keep section length appropriate (150–350 words unless instruction specifies otherwise).
5. If the instruction is ambiguous, apply the most reasonable professional interpretation.

OUTPUT: Return ONLY the edited section content (starting from the section heading ##).
Do not include any preamble, explanation, or note about what was changed."""


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 4 — FULL DOCUMENT RE-GENERATION (for credits/re-run)
#  Same as Phase 2 but triggered by user requesting a full regeneration
#  after edits. Passes current edited sections as context.
# ─────────────────────────────────────────────────────────────────────────────

PHASE_4_REGEN_PROMPT = """You are DocForge AI — an enterprise documentation specialist.

The user has requested a full document regeneration based on their edited sections.
Regenerate the complete document using the edited content as the authoritative source of information.

DOCUMENT TYPE: {doc_type}
DEPARTMENT: {department}
INDUSTRY: {industry}
COMPANY NAME: {company_name}

CURRENT EDITED SECTIONS (use these as the source of truth):
{edited_sections_block}

SECTIONS TO REGENERATE:
{section_list}

STRUCTURAL REQUIREMENTS:
{structure_instructions}

Apply all the same formatting, table, flowchart, and sign-off rules as the original generation.
Improve consistency, tone, and flow across sections. Do NOT invent new facts.

Generate the complete regenerated {doc_type} now:"""


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT BUILDER — Main entry point called by your backend
# ─────────────────────────────────────────────────────────────────────────────

def build_question_prompt(doc_type: str, department: str, industry: str,
                          company_size: str, section_name: str,
                          section_purpose: str, completed_sections: list,
                          question_count: int = 5) -> str:
    return PHASE_1_QUESTION_GEN_PROMPT.format(
        doc_type=doc_type,
        department=department,
        industry=industry,
        company_size=company_size,
        section_name=section_name,
        section_purpose=section_purpose,
        completed_sections=", ".join(completed_sections) if completed_sections else "None",
        question_count=question_count,
    )


def build_document_prompt(doc_type: str, department: str, industry: str,
                          company_name: str, company_size: str,
                          answers: dict, sections: list) -> str:
    meta = get_doc_metadata(doc_type)
    answers_block = build_answers_block(answers, sections)
    section_list = "\n".join([f"  {i+1}. {s['name']}" for i, s in enumerate(sections)])
    structure_instructions = build_structure_instructions(meta)
    department_code = department.upper().replace(" ", "_")[:6]

    return PHASE_2_DOCUMENT_GEN_PROMPT.format(
        doc_type=doc_type,
        department=department,
        industry=industry,
        company_name=company_name,
        company_size=company_size,
        doc_purpose=meta.get("doc_purpose", "Professional business document"),
        tone=meta.get("tone", "professional_formal"),
        answers_block=answers_block,
        section_list=section_list,
        structure_instructions=structure_instructions,
        department_code=department_code,
    )


def build_enhance_prompt(doc_type: str, department: str,
                         section_name: str, current_content: str) -> str:
    return PHASE_3A_SECTION_ENHANCE_PROMPT.format(
        doc_type=doc_type,
        department=department,
        section_name=section_name,
        current_section_content=current_content,
    )


def build_edit_prompt(doc_type: str, department: str, section_name: str,
                      current_content: str, edit_instruction: str) -> str:
    return PHASE_3B_SECTION_EDIT_PROMPT.format(
        doc_type=doc_type,
        department=department,
        section_name=section_name,
        current_section_content=current_content,
        edit_instruction=edit_instruction,
    )


def build_regen_prompt(doc_type: str, department: str, industry: str,
                       company_name: str, edited_sections: dict,
                       sections: list) -> str:
    meta = get_doc_metadata(doc_type)
    edited_block = "\n\n".join(
        [f"[{name}]\n{content}" for name, content in edited_sections.items()]
    )
    section_list = "\n".join([f"  {i+1}. {s['name']}" for i, s in enumerate(sections)])
    structure_instructions = build_structure_instructions(meta)

    return PHASE_4_REGEN_PROMPT.format(
        doc_type=doc_type,
        department=department,
        industry=industry,
        company_name=company_name,
        edited_sections_block=edited_block,
        section_list=section_list,
        structure_instructions=structure_instructions,
    )