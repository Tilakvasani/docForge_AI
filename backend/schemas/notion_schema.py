from pydantic import BaseModel, Field
from typing import Optional, List

class NotionPublishRequest(BaseModel):
    doc_id: str
    title: str
    industry: str
    doc_type: str
    content: str
    tags: List[str] = Field(default_factory=list)
    created_by: str = "admin"
    version: str = "v1.0"
    template_id: Optional[str] = None
