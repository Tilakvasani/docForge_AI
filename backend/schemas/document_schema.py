"""
Pydantic request schemas for the DocForge document generation pipeline.

These models are used as FastAPI request bodies across the document
generation, question-answer, and Notion publishing endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class GenerateQuestionsRequest(BaseModel):
    """
    Request to generate contextual questions for a document section.

    The LLM uses these fields to produce section-specific questions that
    are then presented to the user to gather company-specific details.
    """

    doc_sec_id:      int
    doc_id:          int
    section_name:    str
    doc_type:        str
    department:      str
    company_context: Optional[Dict[str, str]] = Field(default_factory=dict)


class SaveAnswersRequest(BaseModel):
    """Request to persist user-provided answers for a document section."""

    sec_id:       int
    doc_sec_id:   int
    doc_id:       int
    section_name: str
    questions:    List[str]
    answers:      List[str]


class GenerateSectionRequest(BaseModel):
    """
    Request to generate content for a single document section via the LLM.

    `num_sections` is used to calculate a per-section word target so that
    the total document stays within industry-standard length bounds.
    """

    sec_id:          int
    doc_sec_id:      int
    doc_id:          int
    section_name:    str
    doc_type:        str
    department:      str
    company_context: Optional[Dict[str, str]] = Field(default_factory=dict)
    num_sections:    Optional[int] = 10


class EditSectionRequest(BaseModel):
    """Request to apply a user-provided edit instruction to existing section content."""

    gen_id:           int
    sec_id:           int
    section_name:     str
    doc_type:         Optional[str] = None
    current_content:  str
    edit_instruction: str


class NotionPublishRequest(BaseModel):
    """
    Request to publish a fully assembled document to a Notion database.

    `gen_doc_full` contains the complete markdown/plain-text document body.
    The version number is auto-calculated by the service layer.
    """

    gen_id:          int
    doc_type:        str
    department:      str
    gen_doc_full:    str
    company_context: Optional[Dict[str, str]] = Field(default_factory=dict)
