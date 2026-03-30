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
3. Ensure questions extract MECE (Mutually Exclusive, Collectively Exhaustive) data.
4. Ask for concrete details: quantities, roles, timelines, dates, numbers, percentages.
5. Use plain, professional English. No jargon.
6. Each question should unlock a different piece of information needed for this section.
7. Order questions from most important to least important.
8. Do NOT ask for information already covered in previous sections: {completed_sections}

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
11. Flowcharts: Use Mermaid.js ```mermaid code block. Direction: TD. Min 5 nodes. Ensure valid syntax (no special characters in node names without quotes).
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
10. CRITICAL: Maintain original markdown formatting (bolding, lists, links) unless inherently changed.

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
6. CRITICAL: Maintain original markdown formatting (bolding, lists, tables) unless the edit instruction specifically requires changing them.

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
#  QUESTION GENERATION PROMPTS  (moved from generator.py)
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.prompts import PromptTemplate

# ── Text ──────────────────────────────────────────────────────────────────────

TEXT_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "company_size", "region"],
    template="""You are an expert enterprise documentation specialist.

Decide how many questions (0, 1, 2, or 3) are needed to fill this section, then write exactly that many.

Document Type : {doc_type}
Department    : {department}
Section       : {section_name}
Company       : {company_name} | Industry: {industry} | Size: {company_size} | Region: {region}

Decision rules:
- 0 questions: Purely structural section — intro boilerplate, version history, disclaimer.
  Respond with exactly: NONE
- 1 question: One concrete detail unlocks the whole section (e.g. effective date, policy owner)
- 2 questions: Two distinct pieces of context needed
- 3 questions: Complex section needing multiple specifics (maximum — do not exceed 3)

Quality rules:
- Ask for SPECIFIC data: names, dates, numbers, percentages, policy details
- Do NOT ask generic questions like "describe the company" or "what is the purpose"
- Each question must unlock a DIFFERENT piece of information
- Questions must be directly relevant to writing the {section_name} of a {doc_type}

Output: CRITICAL: Output ONLY the raw questions, one per line. No numbering, no bullet points, no extra text. Do not write "Here are the questions:".
If 0 questions: respond NONE

Respond now:"""
)


# ── Table ─────────────────────────────────────────────────────────────────────

TABLE_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "table_hint"],
    template="""You are an expert enterprise documentation specialist.

This section will be rendered as a DATA TABLE. Write 1–3 questions to collect the exact
row data that should appear in the table.

Document Type : {doc_type}
Department    : {department}
Section       : {section_name}
Company       : {company_name} | Industry: {industry}
Table hint    : {table_hint}

Rules:
- Ask for the ACTUAL DATA ROWS, not descriptions or explanations
- Be specific about the format you expect
  Good: "List each expense item with: date, category, description, and amount (one item per line)"
  Bad:  "What expenses were incurred?"
- If the table has a natural primary key (employee name, vendor name, product), ask for it explicitly
- Maximum 3 questions
- CRITICAL: Output ONLY the raw questions, one per line. No numbering, no bullet points. Do not write "Here are the questions:".

Respond now:"""
)


# ── Flowchart ─────────────────────────────────────────────────────────────────

FLOWCHART_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "flowchart_hint"],
    template="""You are an expert enterprise documentation specialist.

This section will be rendered as a PROCESS FLOWCHART (Mermaid diagram).
Write 1–3 questions to collect the process steps and decision points.

Document Type  : {doc_type}
Department     : {department}
Section        : {section_name}
Company        : {company_name} | Industry: {industry}
Flowchart hint : {flowchart_hint}

Rules:
- Ask about the SEQUENCE OF STEPS in the process
- Ask about DECISION POINTS (yes/no branches, approvals, conditions)
- Ask about the ROLES or SYSTEMS involved at each step
- Maximum 3 questions
- CRITICAL: Output ONLY the raw questions, one per line. No numbering, no bullet points. Do not write "Here are the questions:".

Good example questions for a process flow:
  "List the sequential steps in this process from start to finish (e.g. Step 1: Submit request, Step 2: Manager review...)"
  "At which steps does the process branch based on a yes/no decision? What are the two outcomes?"
  "Which team or role is responsible for each step?"

Respond now:"""
)


# ── RACI ──────────────────────────────────────────────────────────────────────

RACI_QUESTIONS_PROMPT = PromptTemplate(
    input_variables=["section_name", "doc_type", "department",
                     "company_name", "industry", "raci_hint"],
    template="""You are an expert enterprise documentation specialist.

This section will be rendered as a RACI RESPONSIBILITY MATRIX TABLE.
Write 1–2 questions to collect role and activity information.

Document Type : {doc_type}
Department    : {department}
Section       : {section_name}
Company       : {company_name} | Industry: {industry}
RACI hint     : {raci_hint}

Rules:
- Question 1: Ask for the list of ROLES or JOB TITLES involved in this process
- Question 2 (optional): Ask for the key ACTIVITIES or TASKS to include in the matrix
- Maximum 2 questions
- CRITICAL: Output ONLY the raw questions, one per line. No numbering, no bullet points. Do not write "Here are the questions:".

Respond now:"""
)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION CONTENT GENERATION PROMPTS  (moved from generator.py)
# ─────────────────────────────────────────────────────────────────────────────

# ── Text ──────────────────────────────────────────────────────────────────────

SECTION_TEXT_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "company_size", "region", "qa_block", "target_words"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}

User-provided information:
{qa_block}

STRICT RULES:
1. Write EXACTLY {target_words} words — hard limit, do NOT exceed
2. PLAIN TEXT ONLY — zero markdown, no asterisks (*), no # symbols, no backticks
3. Paragraphs separated by one blank line
4. Lists: use "1. Item" or "- Item" syntax only
5. If an answer is "not answered" — write realistic, industry-appropriate placeholder content
6. Do NOT include the section heading in your output
7. Professional {department} department tone throughout
8. Do NOT begin with "This section..." or "In this section..."

Write now:"""
)


# ── Table ─────────────────────────────────────────────────────────────────────

SECTION_TABLE_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "region", "qa_block", "table_hint"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}. This section MUST contain a data table.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}
Table guidance: {table_hint}

User-provided data:
{qa_block}

OUTPUT FORMAT — follow EXACTLY:
1. One professional sentence introducing the table (plain text, no markdown)
2. A blank line
3. A properly formatted pipe table using the user's data:

Column1 | Column2 | Column3
------- | ------- | -------
value1  | value2  | value3
value2  | value2  | value3

STRICT RULES:
- Column names must be industry-standard for {section_name} in a {doc_type}
- Use the user's data to populate rows; if data is missing, use realistic placeholders in [brackets]
- Minimum 3 data rows, maximum 10 rows
- NO markdown outside the table — no **, no ##, no backticks
- The intro sentence goes BEFORE the table, never inside a cell
- Do NOT include the section heading in output

Write now:"""
)


# ── Flowchart ─────────────────────────────────────────────────────────────────

SECTION_FLOWCHART_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "region", "qa_block", "flowchart_hint"],
    template="""You are a professional enterprise documentation writer and process designer.

Write the "{section_name}" section of a {doc_type}. This section MUST contain a Mermaid flowchart.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}
Process hint: {flowchart_hint}

User-provided process information:
{qa_block}

OUTPUT FORMAT — follow EXACTLY:
1. One professional sentence describing the process (plain text, no markdown)
2. A blank line
3. A Mermaid flowchart using this EXACT format:

```mermaid
flowchart TD
    A[Start: Step Name] --> B[Step Name]
    B --> C{{Decision Point?}}
    C -->|Yes| D[Step if Yes]
    C -->|No| E[Step if No]
    D --> F[Next Step]
    E --> F
    F --> G([End])
```

STRICT RULES:
- Use TD direction (top-down)
- Minimum 6 nodes, maximum 12 nodes
- Use [Rectangle] for regular steps
- Use {{Diamond}} for decision/approval steps (Yes/No branches)
- Use ([Rounded]) for Start and End nodes
- Label all arrow branches on decision nodes with |Yes| or |No| or relevant label
- Node text must be SHORT — max 5 words per node
- Use ACTUAL steps from the user's answers; if no data, use standard steps for {section_name} of a {doc_type}
- Close the mermaid block with ``` on its own line
- NO other markdown outside the mermaid block — no **, no ##
- Do NOT include the section heading in output

Write now:"""
)


# ── RACI ──────────────────────────────────────────────────────────────────────

SECTION_RACI_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "section_name", "company_name",
                     "industry", "region", "qa_block", "raci_hint"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type}. This section MUST contain a RACI matrix.

Company: {company_name} | Dept: {department} | Industry: {industry} | Region: {region}
RACI guidance: {raci_hint}

User-provided role information:
{qa_block}

OUTPUT FORMAT — follow EXACTLY:
1. One professional sentence about accountability for this process (plain text, no markdown)
2. A blank line
3. A RACI table in this EXACT pipe format:

Activity | [Role 1] | [Role 2] | [Role 3] | [Role 4]
-------- | -------- | -------- | -------- | --------
Activity Name | R | A | C | I
Activity Name | C | R | A | I
Activity Name | I | C | R | A

RACI KEY (add this after the table):
R = Responsible | A = Accountable | C = Consulted | I = Informed

STRICT RULES:
- Replace [Role N] with actual role names from user's answers, or use standard roles for {doc_type}
- Minimum 6 activities, maximum 10 activities
- Every row must have exactly one R and exactly one A
- Activities must be specific to {section_name} of a {doc_type} — not generic
- Do NOT use markdown outside the table — no **, no ##, no backticks
- Do NOT include the section heading in output

Write now:"""
)


# ── Signature ─────────────────────────────────────────────────────────────────

SECTION_SIGNATURE_PROMPT = PromptTemplate(
    input_variables=["doc_type", "department", "company_name", "section_name"],
    template="""You are a professional enterprise documentation writer.

Write the "{section_name}" section of a {doc_type} for {company_name} ({department} department).

This section is a formal approval and sign-off block.

OUTPUT FORMAT — follow EXACTLY:
1. One sentence stating this document requires the following authorised signatures (plain text)
2. A blank line
3. A pipe-format signature table:

Role | Name | Signature | Date
---- | ---- | --------- | ----
[Relevant Role 1] | __________________ | __________________ | __________
[Relevant Role 2] | __________________ | __________________ | __________
[Relevant Role 3] | __________________ | __________________ | __________

STRICT RULES:
- Use 3–5 role rows appropriate for a {doc_type} in the {department} department
- Role names must be specific and relevant (e.g., "HR Manager", "Chief People Officer", "Employee")
- Name, Signature, Date fields must be blank lines (__________) for manual completion
- NO other markdown — no **, no ##, no backticks
- Do NOT include the section heading in output

Write now:"""
)


# ─────────────────────────────────────────────────────────────────────────────
#  EDIT PROMPT  (moved from generator.py)
# ─────────────────────────────────────────────────────────────────────────────

EDIT_PROMPT = PromptTemplate(
    input_variables=["section_name", "section_type", "current_content", "edit_instruction"],
    template="""Professional enterprise document editor.

Section      : {section_name}
Section Type : {section_type}

Current Content:
{current_content}

Edit Instruction: {edit_instruction}

OUTPUT RULES based on section type:
- text      → PLAIN TEXT ONLY, no markdown, no asterisks, no # symbols
- table     → Keep pipe-format table intact; plain text intro sentence only
- flowchart → Keep the ```mermaid ... ``` block intact; update steps if instructed
- raci      → Keep pipe-format RACI table intact; update roles/activities if instructed
- signature → Keep pipe-format signature table intact

Apply the edit instruction to the content above and return ONLY the updated content.
Do not add explanations, preambles, or notes about what changed."""
)

# ─────────────────────────────────────────────────────────────────────────────
#  CiteRAG PROMPTS  (used by rag_service.py)
#  Import in rag_service.py with:
#    from backend.services.rag.prompts import _RAG_PROMPTS as _P
# ─────────────────────────────────────────────────────────────────────────────

_RAG_PROMPTS: dict = {}

_RAG_PROMPTS["ANSWER"] = """\
You are CiteRAG — a precise legal and business document analyst for turabit.
Use ONLY the context below. Do NOT use outside knowledge.
If the answer is not in the context, say exactly:
"I could not find information about this in the available documents."

{history}

Context:
{context}

Question: {question}

CRITICAL RULES:
1. Start with FINAL ANSWER — a direct YES/NO or 1-sentence verdict
2. Then provide supporting analysis with specific document names and sections
3. Never use vague labels — always give ACTUAL content (numbers, names, dates, conditions)
4. Cross-check ALL documents, not just the first match
5. Always flag: undefined terms (reasonable, promptly, material breach, good faith)
6. Always flag: missing standard clauses (indemnity, liability cap, force majeure, dispute resolution)
7. For YES/NO questions — still provide evidence and flag risks even if answer is YES
8. Never stop at 1 sentence — always provide structured analysis

OUTPUT FORMAT by question type:

FINAL ANSWER
[YES/NO or direct answer — 1-2 sentences]

Single fact → exact value + document reference
What/How/Why → 2-5 sentences with specific document citations
List → bullet points with document references per item
Analysis → use CONTRADICTIONS / INCONSISTENCIES / GAPS / AMBIGUITIES sections
Comparison → per-document breakdown then SUMMARY

Answer:"""

_RAG_PROMPTS["COMPARE"] = """\
You are CiteRAG — a senior document analyst for turabit.
Compare the two documents on the question below. Follow all 4 steps.

STEP 1 — SCOPE CHECK:
Do the retrieved documents actually match what the question asks?
- If the documents are NOT the type asked for:
  → Explicitly state: "The documents retrieved are [X] and [Y], not [asked type]."
  → Then proceed to analyze what IS available.

STEP 2 — PER-DOCUMENT FINDINGS:
For each document answer the question using ONLY its content.
State what is present, what is absent, and what is ambiguous.

STEP 3 — GAP & RISK (if a clause or section is missing):
Identify what is missing, the legal/operational risk, and severity.

STEP 4 — COMPARISON INSIGHT:
State expected best practice vs actual finding, with a fix.

Question: {question}

Content from {doc_a}:
{content_a}

Content from {doc_b}:
{content_b}

Respond in this EXACT format:

FINAL ANSWER
[1-2 sentences. Direct answer. If comparison not possible, state why explicitly.]

DOCUMENT A -- {doc_a}
[Findings: specific facts, numbers, dates. State explicitly if clause is missing.]

DOCUMENT B -- {doc_b}
[Findings: specific facts, numbers, dates. State explicitly if clause is missing.]

COMPARISON TABLE
| Aspect | {doc_a} | {doc_b} |
|---|---|---|
| [Key aspect 1] | [finding] | [finding] |
| [Key aspect 2] | [finding] | [finding] |
| [Key aspect 3] | [finding] | [finding] |

GAP IDENTIFIED:
What: [what is missing or problematic]
Where: [document and section]
Risk:
- [specific legal impact]
- [specific legal impact]
Severity: [🔴 HIGH / 🟡 MEDIUM / 🟢 LOW]
Severity Reason: [1 sentence why this severity]

KEY DIFFERENCE:
[state the actual difference, or "No substantive difference" if same]

SYSTEMIC ISSUE (if applicable):
[If operational docs used instead of formal legal agreements, state it]

COMPARISON INSIGHT:
Expected: [best practice]
Actual: [what was found]
Fix: [concrete recommendation]

SUMMARY: [2-3 sentences covering scope issues, main findings, and recommended action.]"""

_RAG_PROMPTS["HYDE"] = """\
Write a brief factual description (2-3 sentences) about this business topic: {question}"""

_RAG_PROMPTS["SUMMARY"] = """\
You are CiteRAG — a professional document analyst for turabit.
Write a structured, scannable summary. Use ONLY the context below.

Context:
{context}

Topic/Question: {question}

Output format — follow EXACTLY:

SUMMARY
[One sentence: what this document/policy covers and its purpose.]

KEY FUNCTIONS

**1. [Function Name]**
[1-2 sentences. Real facts: names, numbers, conditions, timelines. No vague labels.]

**2. [Function Name]**
[1-2 sentences. Real facts only.]

**3. [Function Name]**
[1-2 sentences. Real facts only.]

(Continue up to 8 functions maximum)

CONCLUSION
[1 sentence. What this document/policy achieves overall.]

RULES:
- Under 220 words total
- No bullet points inside sections
- No intro or outro phrases
- Every section must contain real content — skip if not in context
- Short, structured, scannable — not a paragraph essay

Summary:"""

_RAG_PROMPTS["EXPAND"] = """\
Rewrite this question in 3 different ways using different words and synonyms that mean the same thing.
Keep each version short (under 15 words). Return only the 3 versions, one per line, no numbering.

Question: {question}"""

_RAG_PROMPTS["REWRITE"] = """\
You are a query understanding assistant for a company legal document system at turabit.
The user asked: "{question}"

Your job:
1. Understand what the user REALLY wants
2. Rewrite it as a clear, precise question that will find the right document content
3. Identify the intent type

REWRITING RULES:
- Fix typos and informal language
- Expand abbreviations (HR → Human Resources, IP → Intellectual Property)
- Make vague questions specific
- For legal/contract questions, always include: contract agreement clause legal terms
- For abstract questions, rewrite to find concrete document content

MANDATORY REWRITES (use these exact patterns):
- "do notice periods create any conflicts or risks" → "termination notice period 30 days 60 days conflict risk contracts agreements"
- "does the document follow a logical structure" → "document structure sections headings organization format layout contracts"
- "is there a hierarchy between related agreements" → "master agreement MSA parent child precedence governance supersedes framework"
- "is there a clause hierarchy or precedence rule" → "agreement precedence rule supersedes clause hierarchy governing order MSA"
- "are key terms properly defined" → "undefined key terms material breach reasonable period promptly force majeure good faith definitions contracts"
- "are definitions used consistently" → "consistent definitions key terms material breach reasonable promptly undefined contracts legal agreements"
- "are enforcement mechanisms strong enough" → "enforcement mechanisms penalties financial liability audit compliance contracts legal"
- "are roles and responsibilities clearly defined" → "roles responsibilities RACI accountability defined contracts agreements vendor employment"
- "does this agreement align with industry best practices" → "industry best practices indemnity liability force majeure dispute resolution standards contracts"
- "does the agreement scale well for future changes" → "amendment modification renewal scalability future changes contracts agreements"
- "are there any one-sided or unfair clauses" → "one-sided unfair clauses liability cap indemnity termination fees compensation contracts"
- "does the contract expose one party to excessive liability" → "excessive liability cap indemnity limitation damages force majeure contracts"
- "is there a fair exit mechanism" → "exit mechanism termination notice period fees severance post-termination obligations contracts"
- "are tax responsibilities clearly assigned" → "tax responsibilities GST TDS withholding income tax contracts vendor employment assignment"
- "are penalties or late fees properly defined" → "penalties late fees payment terms interest rate defined contracts invoices"
- "are termination rights clearly defined" → "termination rights notice period grounds conditions both parties contracts"

INTENT CLASSIFICATION RULES — read carefully:
- GENERAL → code generation, math problems, creative writing, general world knowledge, tech tutorials — anything NOT about turabit company documents
- GREETING → hello, hi, thanks, bye, who are you
- ANALYSIS → review/audit/gaps/contradictions/issues/risks INSIDE the turabit documents
- COMPARE → compare two specific turabit documents against each other
- SUMMARY → summarize a specific turabit document or policy
- YESNO → yes/no question about document content
- SPECIFIC → specific fact lookup inside documents
- LIST → list items from documents
- EXPLAIN → explain something from the documents
- SEARCH → general search inside documents

EXAMPLES — GENERAL intent:
- "write fibonacci in java" → GENERAL
- "write me a python function to reverse a string" → GENERAL
- "explain how neural networks work" → GENERAL
- "what is the capital of france" → GENERAL
- "write me an email to my manager asking for leave" → GENERAL
- "calculate compound interest formula" → GENERAL
- "tell me a joke" → GENERAL
- "what is docker" → GENERAL
- "how do i use git rebase" → GENERAL
- "write a poem about rain" → GENERAL

EXAMPLES — DOCUMENT intent (anything else):
- "what is the notice period in our NDA?" → SPECIFIC
- "are there any conflicting clauses?" → ANALYSIS
- "compare SOW vs employment contract" → COMPARE
- "summarize the vendor agreement" → SUMMARY

Reply in this exact format:
REWRITTEN: [the clear precise question]
INTENT: [one of: GREETING, GENERAL, COMPARE, FULL_DOC, SUMMARY, LIST, YESNO, SPECIFIC, EXPLAIN, ANALYSIS, SEARCH]"""

_RAG_PROMPTS["ANALYSIS"] = """\
You are CiteRAG — a senior legal and business document analyst for turabit.
Analyze the provided documents and answer the question precisely.

CRITICAL DEFINITIONS — apply strictly:

CONTRADICTION: Two statements that CANNOT both be true simultaneously.
  Real example: Doc A says 30-day notice period AND Doc B says 60 days.
  NOT a contradiction: vague wording, different terminology, missing info.

INCONSISTENCY: Same concept, different wording — not logically conflicting.
GAP: A standard clause or section that is completely missing.
AMBIGUITY: Wording that is unclear or interpretable in multiple ways.

Document content:
{context}

Question: {question}

FORMAT — include ONLY sections with actual findings:

FINAL ANSWER
[1-2 sentences. Direct YES/NO or overall verdict answering the question.]

## CONTRADICTIONS
[If none: **No true contradictions found.**]

## INCONSISTENCIES
[Skip if none]

## GAPS
[Skip if none]

## AMBIGUITIES
[Skip if none]

For EACH finding:
- **What:** [specific issue — quote exact wording from document]
  **Where:** [document name] > [section name]
  **Risk:** [concrete legal or operational impact]
  **Severity:** 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
  **Severity Reason:** [1 sentence explaining why this severity level]
  **Fix:** [concrete, actionable recommendation]

## CONCLUSION
[2-3 sentences. Overall assessment: how serious? What is the priority action?]

RULES:
- Always write FINAL ANSWER first before any section headers
- FINAL ANSWER: YES/NO for yes/no questions, or a direct verdict
- Only report what is in the documents — no hallucination
- Be specific: quote exact wording, name exact sections and documents
- Cross-check ALL documents, not just the first match
- Flag undefined terms (reasonable, promptly, material breach) as AMBIGUITIES
- Flag missing standard clauses (indemnity, liability cap, force majeure) as GAPS

Analysis:"""

_RAG_PROMPTS["GENERAL_SYSTEM"] = """\
You are a helpful AI assistant — like ChatGPT, but also integrated with a company document system.
For this query, NO document search is needed. Answer directly from your knowledge.

RULES:
- CODE: write clean, working, commented code inside markdown code blocks with the correct language tag (e.g. ```java, ```python)
- MATH: show step-by-step working clearly
- EXPLANATIONS: be clear, use examples, keep it concise
- CREATIVE WRITING: be creative and engaging (poems, jokes, stories, emails)
- GENERAL KNOWLEDGE: be accurate and concise
- Always use markdown formatting when it improves readability
- Be conversational and friendly
- Never say "I can only answer document questions" — answer everything the user asks"""