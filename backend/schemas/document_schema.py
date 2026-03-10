from pydantic import BaseModel
from typing import Optional, List, Dict
from enum import Enum
from datetime import datetime


class Department(str, Enum):
    HR               = "Human Resources (HR)"
    LEGAL            = "Legal"
    FINANCE          = "Finance / Accounting"
    SALES            = "Sales"
    MARKETING        = "Marketing"
    ENGINEERING      = "Engineering / Development"
    PRODUCT          = "Product Management"
    OPERATIONS       = "Operations"
    CUSTOMER_SUPPORT = "Customer Support"
    COMPLIANCE       = "Compliance / Risk Management"


class DocType(str, Enum):
    TERMS_OF_SERVICE        = "Terms of Service"
    EMPLOYMENT_CONTRACT     = "Employment Contract"
    PRIVACY_POLICY          = "Privacy Policy"
    SOP                     = "SOP"
    SLA                     = "SLA"
    PRD                     = "Product Requirement Document"
    TECHNICAL_SPECIFICATION = "Technical Specification"
    INCIDENT_REPORT         = "Incident Report"
    SECURITY_POLICY         = "Security Policy"
    CUSTOMER_ONBOARDING     = "Customer Onboarding Guide"
    BUSINESS_PROPOSAL       = "Business Proposal"
    NDA                     = "NDA"


class DocumentRequest(BaseModel):
    title:           str
    industry:        str
    department:      Optional[str] = None
    doc_type:        str
    description:     Optional[str] = None
    tags:            Optional[List[str]] = None
    created_by:      Optional[str] = None
    # Flexible dict — keys depend on doc type (es_q1, sc_q1, pc_q1, etc.)
    section_answers: Optional[Dict[str, str]] = None


class DocumentResponse(BaseModel):
    doc_id:          str
    title:           str
    industry:        str
    department:      Optional[str] = None
    doc_type:        str
    content:         str
    tags:            Optional[List[str]] = None
    created_by:      Optional[str] = None
    created_at:      datetime
    version:         str = "1.0"
    notion_url:      Optional[str] = None
    published:       bool = False
    section_answers: Optional[Dict[str, str]] = None