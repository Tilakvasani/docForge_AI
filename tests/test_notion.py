import pytest
from backend.schemas.notion_schema import NotionPublishRequest

def test_notion_schema_valid():
    req = NotionPublishRequest(
        doc_id="123",
        title="Test Policy",
        industry="saas",
        doc_type="policy",
        content="Sample content",
        tags=["test"],
        created_by="admin"
    )
    assert req.title == "Test Policy"
    assert req.version == "v1.0"

def test_notion_schema_defaults():
    req = NotionPublishRequest(
        doc_id="456",
        title="Test Doc",
        industry="telecom",
        doc_type="sop",
        content="Content here"
    )
    assert req.tags == []
    assert req.created_by == "admin"
    assert req.template_id is None
