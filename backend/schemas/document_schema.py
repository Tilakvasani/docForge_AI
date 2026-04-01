from pydantic import BaseModel
from typing import Optional, List, Dict


class GenerateQuestionsRequest(BaseModel):
    doc_sec_id: int
    doc_id: int
    section_name: str
    doc_type: str
    department: str
    company_context: Optional[Dict[str, str]] = {}


class SaveAnswersRequest(BaseModel):
    sec_id: int
    doc_sec_id: int
    doc_id: int
    section_name: str
    questions: List[str]
    answers: List[str]


class GenerateSectionRequest(BaseModel):
    sec_id: int
    doc_sec_id: int
    doc_id: int
    section_name: str
    doc_type: str
    department: str
    company_context: Optional[Dict[str, str]] = {}
    num_sections: Optional[int] = 10


class EditSectionRequest(BaseModel):
    gen_id: int
    sec_id: int
    section_name: str
    doc_type: Optional[str] = None
    current_content: str
    edit_instruction: str


class NotionPublishRequest(BaseModel):
    gen_id: int
    doc_type: str
    department: str
    gen_doc_full: str
    company_context: Optional[Dict[str, str]] = {}