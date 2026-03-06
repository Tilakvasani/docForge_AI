from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime

class Industry(str, Enum):
    TELECOM = "telecom"
    SAAS = "saas"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"

class DocType(str, Enum):
    SOP = "sop"
    POLICY = "policy"
    PROPOSAL = "proposal"
    SOW = "sow"
    INCIDENT_REPORT = "incident_report"
    FAQ = "faq"
    BUSINESS_CASE = "business_case"
    SECURITY_POLICY = "security_policy"
    KPI_REPORT = "kpi_report"
    RUNBOOK = "runbook"

class DocumentRequest(BaseModel):
    title: str = Field(..., description="Document title")
    industry: Industry = Field(..., description="Target industry")
    doc_type: DocType = Field(..., description="Type of document")
    description: Optional[str] = Field(None, description="Brief context or description")
    tags: Optional[List[str]] = Field(default=[], description="Tags for the document")
    created_by: Optional[str] = Field(default="admin", description="Creator name")

class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    industry: str
    doc_type: str
    content: str
    tags: List[str]
    created_by: str
    created_at: datetime
    version: str = "v1.0"
    notion_url: Optional[str] = None
    published: bool = False
