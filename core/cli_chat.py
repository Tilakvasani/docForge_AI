from typing import List, Tuple
from mcp.types import Prompt, PromptMessage
from core.chat import Chat
from core.claude import Claude
from mcp_client import MCPClient

SYSTEM_PROMPT = """You are a smart CRM and document assistant with full access to HubSpot CRM
and a document library through your tools.

## HubSpot CRM — What you can do
CONTACTS : search, get, create, update, delete, list, merge
COMPANIES: search, get, create, update, get_company_contacts
DEALS    : search, get, create, update, move stage, delete, set collaborators
TICKETS  : search, get, create, update, escalate
TASKS    : create, complete, list
CALLS    : log calls with outcome and notes
NOTES    : add notes to contacts or deals
ASSOCIATE: link contacts↔deals, contacts↔companies, deals↔companies, contacts↔tickets

## Context Resources (read-only, auto-injected when needed)
- hubspot://pipelines          — all deal pipeline stages with IDs
- hubspot://ticket-pipelines   — all ticket pipeline stages with IDs
- hubspot://owners             — all team members with owner IDs
- hubspot://contact-properties — all valid contact property names
- hubspot://deal-properties    — all valid deal property names

## Slash Commands (user triggers)
/find_contact <email or name>
/deal_report [stage]
/create_contact_flow <email>
/contact_summary <contact_id>
/pipeline_overview
/log_call_flow <name or email>
/ticket_triage [priority]

## Document Library
Reference documents with @filename — e.g. @report.pdf

## Rules
1. Always confirm before creating, updating or deleting any HubSpot record.
2. Always check hubspot://pipelines or hubspot://owners resource before using IDs.
3. Present CRM data in clean readable tables or formatted summaries.
4. If an operation fails, explain the error clearly and suggest a fix.
5. Never expose the raw API token or credentials.
"""


class CliChat(Chat):
    def __init__(self, doc_client: MCPClient, clients: dict[str, MCPClient], claude_service: Claude):
        super().__init__(clients=clients, claude_service=claude_service)
        self.doc_client = doc_client

    async def list_prompts(self) -> list[Prompt]:
        all_prompts = []
        for client in self.clients.values():
            try:
                all_prompts.extend(await client.list_prompts())
            except Exception:
                pass
        return all_prompts

    async def list_docs_ids(self) -> list[str]:
        try:
            return await self.doc_client.read_resource("docs://documents")
        except Exception:
            return []

    async def get_doc_content(self, doc_id: str) -> str:
        try:
            return await self.doc_client.read_resource(f"docs://documents/{doc_id}")
        except Exception:
            return ""

    async def get_prompt(self, command: str, arg: str) -> list[PromptMessage]:
        for client in self.clients.values():
            for key in [{"identifier": arg}, {"contact_id": arg}, {"email": arg},
                        {"doc_id": arg}, {"stage_filter": arg},
                        {"contact_identifier": arg}, {"priority": arg}]:
                try:
                    result = await client.get_prompt(command, key)
                    if result:
                        return result
                except Exception:
                    continue
        return []

    async def _extract_resources(self, query: str) -> str:
        mentions = [w[1:] for w in query.split() if w.startswith("@")]
        doc_ids = await self.list_docs_ids()
        mentioned = [(d, await self.get_doc_content(d)) for d in doc_ids if d in mentions]
        return "".join(f'\n<document id="{d}">\n{c}\n</document>\n' for d, c in mentioned)

    async def _process_command(self, query: str) -> bool:
        if not query.startswith("/"):
            return False
        words = query.split()
        command = words[0].lstrip("/")
        arg = words[1] if len(words) > 1 else ""
        messages = await self.get_prompt(command, arg)
        if messages:
            self.messages += convert_prompt_messages_to_message_params(messages)
            return True
        return False

    async def _process_query(self, query: str):
        if await self._process_command(query):
            return
        resources = await self._extract_resources(query)
        prompt = f"""The user has a question or request:
<query>
{query}
</query>
{f"<context>{resources}</context>" if resources else ""}
Answer directly. If you need to act on HubSpot data, use your tools.
"""
        self.messages.append({"role": "user", "content": prompt})


def convert_prompt_message_to_message_param(pm: PromptMessage) -> dict:
    role = "user" if pm.role == "user" else "assistant"
    content = pm.content
    ctype = content.get("type") if isinstance(content, dict) else getattr(content, "type", None)
    if ctype == "text":
        text = content.get("text", "") if isinstance(content, dict) else getattr(content, "text", "")
        return {"role": role, "content": text}
    if isinstance(content, list):
        parts = []
        for item in content:
            t = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            if t == "text":
                parts.append(item.get("text", "") if isinstance(item, dict) else getattr(item, "text", ""))
        if parts:
            return {"role": role, "content": "\n".join(parts)}
    return {"role": role, "content": ""}


def convert_prompt_messages_to_message_params(messages: List[PromptMessage]) -> List[dict]:
    return [convert_prompt_message_to_message_param(m) for m in messages]
