"""
Pydantic schema for the Notion publish request used in the agent layer.

Note: The primary publish schema used by the DocForge generation pipeline
is `NotionPublishRequest` in `backend/schemas/document_schema.py`.
This schema is retained for agent-layer integrations that require a
different field structure (doc_id, title, tags, template_id).
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class NotionPublishRequest(BaseModel):
    """
    Alternative Notion publish request schema for agent-layer workflows.

    Used when publishing documents that require explicit title, industry,
    tag, and template metadata rather than the generation-pipeline context.
    """

    doc_id:      str
    title:       str
    industry:    str
    doc_type:    str
    content:     str
    tags:        List[str]    = Field(default_factory=list)
    created_by:  str          = "admin"
    version:     str          = "v1.0"
    template_id: Optional[str] = None
