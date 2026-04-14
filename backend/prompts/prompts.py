"""
DocForge AI — prompts/docforge_prompts.py
════════════════════════════════════════════════════════════════
Enhanced prompt system for 100+ industry-standard documents.
Covers: question generation, full document generation, section
editing/enhancement, and structural metadata per document type.
"""
from langchain_core.prompts import PromptTemplate


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
#  QUESTION GENERATION PROMPTS 
# ─────────────────────────────────────────────────────────────────────────────


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

