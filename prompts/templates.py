def get_prompt_template(doc_type: str) -> str:
    """Return the appropriate prompt template for each document type"""

    templates = {
        "sop": """You are an expert technical writer creating industry-ready Standard Operating Procedures.

Create a detailed SOP document with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The SOP MUST include these sections:
1. Purpose
2. Scope
3. Definitions
4. Responsibilities
5. Procedure (step-by-step)
6. Quality Control
7. References

Write in a professional, clear, and formal tone suitable for {industry} industry.
Use numbered steps and clear headings. Minimum 400 words.""",

        "policy": """You are an expert policy writer creating industry-ready policy documents.

Create a detailed Policy document with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The Policy MUST include these sections:
1. Purpose
2. Scope
3. Definitions
4. Policy Statement
5. Exceptions
6. Compliance & Enforcement
7. Review Schedule

Write in formal, authoritative language appropriate for {industry} industry. Minimum 400 words.""",

        "proposal": """You are an expert business writer creating professional proposals.

Create a detailed Proposal document with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The Proposal MUST include these sections:
1. Executive Summary
2. Problem Statement
3. Proposed Solution / Objectives
4. Methodology
5. Timeline
6. Budget Overview
7. Expected Outcomes

Write persuasively and professionally for {industry} industry. Minimum 400 words.""",

        "sow": """You are an expert contract writer creating Statements of Work.

Create a detailed SOW document with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The SOW MUST include these sections:
1. Project Overview
2. Scope of Work
3. Deliverables
4. Timeline & Milestones
5. Payment Terms
6. Assumptions & Dependencies
7. Change Management

Write in precise, contractual language for {industry} industry. Minimum 400 words.""",

        "incident_report": """You are an expert incident manager creating formal incident reports.

Create a detailed Incident Report with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The Incident Report MUST include these sections:
1. Incident Summary
2. Impact Assessment
3. Timeline of Events
4. Root Cause Analysis
5. Resolution Steps
6. Preventive Measures
7. Lessons Learned

Write factually and analytically for {industry} industry. Minimum 400 words.""",

        "faq": """You are an expert documentation writer creating comprehensive FAQ documents.

Create a detailed FAQ document with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

Include at least 10 relevant Question and Answer pairs covering:
- General questions
- Technical questions
- Process questions
- Troubleshooting questions

Format as Q: [question] followed by A: [detailed answer].
Write clearly and helpfully for {industry} industry. Minimum 400 words.""",

        "business_case": """You are an expert business analyst creating business case documents.

Create a detailed Business Case with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The Business Case MUST include these sections:
1. Executive Summary
2. Problem Statement
3. Proposed Solution
4. Cost-Benefit Analysis
5. Risk Assessment
6. Expected Benefits / ROI
7. Recommendation

Write analytically and persuasively for {industry} industry. Minimum 400 words.""",

        "security_policy": """You are an expert cybersecurity writer creating security policy documents.

Create a detailed Security Policy with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The Security Policy MUST include these sections:
1. Purpose & Scope
2. Definitions
3. Security Requirements
4. Access Control
5. Exceptions & Exemptions
6. Compliance & Penalties
7. Review & Update Schedule

Write authoritatively for {industry} industry following best security practices. Minimum 400 words.""",

        "kpi_report": """You are an expert analyst creating KPI and performance reports.

Create a detailed KPI Report with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The KPI Report MUST include these sections:
1. Executive Overview
2. Key Metrics Summary
3. Performance Analysis
4. Trends & Insights
5. Areas of Concern
6. Recommendations
7. Next Steps

Write analytically with clear metrics language for {industry} industry. Minimum 400 words.""",

        "runbook": """You are an expert DevOps engineer creating operational runbooks.

Create a detailed Runbook with the following details:
- Title: {title}
- Industry: {industry}
- Context: {description}

The Runbook MUST include these sections:
1. Purpose
2. Prerequisites & Requirements
3. Step-by-Step Procedures
4. Verification Steps
5. Troubleshooting Guide
6. Rollback Procedure
7. Contact & Escalation

Write technically and precisely for {industry} industry operations teams. Minimum 400 words.""",
    }

    return templates.get(doc_type, templates["sop"])
