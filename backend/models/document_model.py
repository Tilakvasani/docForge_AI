from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid

@dataclass
class DocumentModel:
    title: str
    industry: str
    doc_type: str
    content: str
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: List[str] = field(default_factory=list)
    created_by: str = "admin"
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "v1.0"
    notion_url: Optional[str] = None
    published: bool = False

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "industry": self.industry,
            "doc_type": self.doc_type,
            "content": self.content,
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "notion_url": self.notion_url,
            "published": self.published
        }
