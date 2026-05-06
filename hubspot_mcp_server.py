"""
HubSpot MCP Server — Tools + Resources + Prompts
=================================================
TOOLS     (model-controlled) : AI calls these to act on HubSpot CRM
RESOURCES (app-controlled)   : read-only HubSpot data injected as context
PROMPTS   (user-controlled)  : slash commands e.g. /find_contact, /deal_report
"""
import os, json, httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP 

load_dotenv()
mcp = FastMCP("HubSpotMCP", log_level="ERROR")
BASE = "https://api.hubapi.com"

# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _h():
    t = os.getenv("HUBSPOT_TOKEN", "")
    if not t:
        raise ValueError("HUBSPOT_TOKEN missing in .env")
    return {"Authorization": f"Bearer {t}", "Content-Type": "application/json"}

def _get(path, params=None):
    r = httpx.get(f"{BASE}{path}", headers=_h(), params=params, timeout=15)
    r.raise_for_status(); return r.json()

def _post(path, body):
    r = httpx.post(f"{BASE}{path}", headers=_h(), json=body, timeout=15)
    r.raise_for_status(); return r.json()

def _patch(path, body):
    r = httpx.patch(f"{BASE}{path}", headers=_h(), json=body, timeout=15)
    r.raise_for_status(); return r.json()

def _delete(path):
    r = httpx.delete(f"{BASE}{path}", headers=_h(), timeout=15)
    r.raise_for_status(); return {"status": "deleted"}

def ok(d): return json.dumps(d, indent=2)

# =============================================================================
# TOOLS  — model-controlled, AI calls these autonomously
# =============================================================================

# ── Contacts ──────────────────────────────────────────────────────────────────
@mcp.tool()
def get_contact(contact_id: str) -> str:
    """Get a HubSpot contact by ID. Returns all CRM properties."""
    return ok(_get(f"/crm/v3/objects/contacts/{contact_id}",
                   {"properties": "firstname,lastname,email,phone,company,hs_lead_status,createdate,lastmodifieddate"}))

@mcp.tool()
def search_contacts(query: str, limit: int = 10) -> str:
    """Search HubSpot contacts by email, name or phone number."""
    return ok(_post("/crm/v3/objects/contacts/search", {
        "query": query, "limit": limit,
        "properties": ["firstname","lastname","email","phone","company","hs_lead_status"]}))

@mcp.tool()
def create_contact(email: str, firstname: str = "", lastname: str = "",
                   phone: str = "", company: str = "") -> str:
    """Create a new contact in HubSpot CRM."""
    props = {"email": email}
    if firstname: props["firstname"] = firstname
    if lastname:  props["lastname"]  = lastname
    if phone:     props["phone"]     = phone
    if company:   props["company"]   = company
    return ok(_post("/crm/v3/objects/contacts", {"properties": props}))

@mcp.tool()
def update_contact(contact_id: str, properties: dict) -> str:
    """Update any properties of a HubSpot contact by ID.
    Example: {"email": "new@mail.com", "phone": "+1234567890"}"""
    return ok(_patch(f"/crm/v3/objects/contacts/{contact_id}", {"properties": properties}))

@mcp.tool()
def delete_contact(contact_id: str) -> str:
    """Archive (soft-delete) a HubSpot contact by ID."""
    return ok(_delete(f"/crm/v3/objects/contacts/{contact_id}"))

@mcp.tool()
def list_contacts(limit: int = 20, after: str = "") -> str:
    """List HubSpot contacts with pagination. Pass 'after' cursor for next page."""
    params = {"limit": limit, "properties": "firstname,lastname,email,phone,company"}
    if after: params["after"] = after
    return ok(_get("/crm/v3/objects/contacts", params))

@mcp.tool()
def merge_contacts(primary_id: str, secondary_id: str) -> str:
    """Merge two duplicate contacts. secondary_id gets merged into primary_id."""
    return ok(_post("/crm/v3/objects/contacts/merge",
                    {"primaryObjectId": primary_id, "objectIdToMerge": secondary_id}))

# ── Companies ─────────────────────────────────────────────────────────────────
@mcp.tool()
def get_company(company_id: str) -> str:
    """Get a HubSpot company by ID."""
    return ok(_get(f"/crm/v3/objects/companies/{company_id}",
                   {"properties": "name,domain,industry,city,phone,annualrevenue,numberofemployees"}))

@mcp.tool()
def search_companies(query: str, limit: int = 10) -> str:
    """Search HubSpot companies by name or domain."""
    return ok(_post("/crm/v3/objects/companies/search", {
        "query": query, "limit": limit,
        "properties": ["name","domain","industry","city","phone","annualrevenue"]}))

@mcp.tool()
def create_company(name: str, domain: str = "", industry: str = "",
                   phone: str = "", city: str = "") -> str:
    """Create a new company record in HubSpot."""
    props = {"name": name}
    if domain:   props["domain"]   = domain
    if industry: props["industry"] = industry
    if phone:    props["phone"]    = phone
    if city:     props["city"]     = city
    return ok(_post("/crm/v3/objects/companies", {"properties": props}))

@mcp.tool()
def update_company(company_id: str, properties: dict) -> str:
    """Update any properties of a HubSpot company by ID."""
    return ok(_patch(f"/crm/v3/objects/companies/{company_id}", {"properties": properties}))

@mcp.tool()
def get_company_contacts(company_id: str) -> str:
    """Get all contacts associated with a HubSpot company."""
    return ok(_get(f"/crm/v3/objects/companies/{company_id}/associations/contacts"))

# ── Deals ─────────────────────────────────────────────────────────────────────
@mcp.tool()
def get_deal(deal_id: str) -> str:
    """Get a HubSpot deal by ID including stage, amount and close date."""
    return ok(_get(f"/crm/v3/objects/deals/{deal_id}",
                   {"properties": "dealname,amount,dealstage,closedate,hubspot_owner_id,pipeline"}))

@mcp.tool()
def search_deals(query: str, limit: int = 10) -> str:
    """Search HubSpot deals by deal name."""
    return ok(_post("/crm/v3/objects/deals/search", {
        "query": query, "limit": limit,
        "properties": ["dealname","amount","dealstage","closedate","pipeline","hubspot_owner_id"]}))

@mcp.tool()
def list_deals(limit: int = 20, after: str = "") -> str:
    """List all HubSpot deals with pagination."""
    params = {"limit": limit, "properties": "dealname,amount,dealstage,closedate,pipeline"}
    if after: params["after"] = after
    return ok(_get("/crm/v3/objects/deals", params))

@mcp.tool()
def create_deal(dealname: str, pipeline: str, dealstage: str,
                amount: float = 0.0, closedate: str = "") -> str:
    """Create a new HubSpot deal.
    pipeline: pipeline ID — use hubspot://pipelines resource to get IDs
    dealstage: stage ID — use hubspot://pipelines resource to get stage IDs
    closedate: YYYY-MM-DD format"""
    props: dict = {"dealname": dealname, "pipeline": pipeline, "dealstage": dealstage}
    if amount:    props["amount"]    = str(amount)
    if closedate: props["closedate"] = closedate
    return ok(_post("/crm/v3/objects/deals", {"properties": props}))

@mcp.tool()
def update_deal(deal_id: str, properties: dict) -> str:
    """Update any deal properties — stage, amount, close date, owner etc."""
    return ok(_patch(f"/crm/v3/objects/deals/{deal_id}", {"properties": properties}))

@mcp.tool()
def move_deal_stage(deal_id: str, dealstage: str) -> str:
    """Move a deal to a new pipeline stage by stage ID."""
    return ok(_patch(f"/crm/v3/objects/deals/{deal_id}", {"properties": {"dealstage": dealstage}}))

@mcp.tool()
def delete_deal(deal_id: str) -> str:
    """Archive a HubSpot deal by ID."""
    return ok(_delete(f"/crm/v3/objects/deals/{deal_id}"))

@mcp.tool()
def set_deal_collaborators(deal_id: str, owner_ids: list) -> str:
    """Set team collaborators on a HubSpot deal (July 2025 feature).
    owner_ids: list of HubSpot owner ID strings"""
    return ok(_patch(f"/crm/v3/objects/deals/{deal_id}",
                     {"properties": {"hs_deal_collaborators": ",".join(owner_ids)}}))

# ── Tickets ───────────────────────────────────────────────────────────────────
@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Get a HubSpot support ticket by ID."""
    return ok(_get(f"/crm/v3/objects/tickets/{ticket_id}",
                   {"properties": "subject,content,hs_ticket_priority,hs_pipeline_stage,hubspot_owner_id"}))

@mcp.tool()
def search_tickets(query: str, limit: int = 10) -> str:
    """Search HubSpot support tickets by subject or content."""
    return ok(_post("/crm/v3/objects/tickets/search", {
        "query": query, "limit": limit,
        "properties": ["subject","content","hs_ticket_priority","hs_pipeline_stage","hubspot_owner_id"]}))

@mcp.tool()
def create_ticket(subject: str, content: str = "",
                  hs_ticket_priority: str = "MEDIUM",
                  hs_pipeline_stage: str = "1") -> str:
    """Create a HubSpot support ticket.
    hs_ticket_priority: LOW | MEDIUM | HIGH | URGENT
    hs_pipeline_stage: use hubspot://ticket-pipelines resource for stage IDs"""
    return ok(_post("/crm/v3/objects/tickets", {"properties": {
        "subject": subject, "content": content,
        "hs_ticket_priority": hs_ticket_priority,
        "hs_pipeline_stage": hs_pipeline_stage}}))

@mcp.tool()
def update_ticket(ticket_id: str, properties: dict) -> str:
    """Update a HubSpot ticket — priority, status, owner or any property."""
    return ok(_patch(f"/crm/v3/objects/tickets/{ticket_id}", {"properties": properties}))

@mcp.tool()
def escalate_ticket(ticket_id: str, owner_id: str = "") -> str:
    """Escalate a ticket to URGENT priority and optionally reassign to a new owner."""
    props: dict = {"hs_ticket_priority": "URGENT"}
    if owner_id: props["hubspot_owner_id"] = owner_id
    return ok(_patch(f"/crm/v3/objects/tickets/{ticket_id}", {"properties": props}))

# ── Engagements ───────────────────────────────────────────────────────────────
@mcp.tool()
def create_note(body_text: str, contact_id: str = "", deal_id: str = "") -> str:
    """Log a note on a HubSpot contact or deal."""
    import time
    props = {"hs_note_body": body_text, "hs_timestamp": str(int(time.time() * 1000))}
    result = _post("/crm/v3/objects/notes", {"properties": props})
    note_id = result.get("id", "")
    if contact_id and note_id:
        try:
            _post(f"/crm/v4/objects/notes/{note_id}/associations/contacts/{contact_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 202}])
        except Exception: pass
    if deal_id and note_id:
        try:
            _post(f"/crm/v4/objects/notes/{note_id}/associations/deals/{deal_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 214}])
        except Exception: pass
    return ok(result)

@mcp.tool()
def create_task(subject: str, body: str = "", due_date: str = "",
                owner_id: str = "", contact_id: str = "") -> str:
    """Create a follow-up task in HubSpot.
    due_date: YYYY-MM-DD format
    owner_id: use hubspot://owners resource to get owner IDs"""
    props: dict = {"hs_task_subject": subject, "hs_task_body": body, "hs_task_status": "NOT_STARTED"}
    if due_date:  props["hs_timestamp"] = f"{due_date}T00:00:00.000Z"
    if owner_id:  props["hubspot_owner_id"] = owner_id
    result = _post("/crm/v3/objects/tasks", {"properties": props})
    task_id = result.get("id", "")
    if contact_id and task_id:
        try:
            _post(f"/crm/v4/objects/tasks/{task_id}/associations/contacts/{contact_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 204}])
        except Exception: pass
    return ok(result)

@mcp.tool()
def complete_task(task_id: str) -> str:
    """Mark a HubSpot task as complete."""
    return ok(_patch(f"/crm/v3/objects/tasks/{task_id}",
                     {"properties": {"hs_task_status": "COMPLETED"}}))

@mcp.tool()
def list_tasks(owner_id: str = "", limit: int = 20) -> str:
    """List HubSpot tasks, optionally filtered by owner ID."""
    params: dict = {"limit": limit, "properties": "hs_task_subject,hs_task_status,hs_timestamp,hubspot_owner_id"}
    if owner_id: params["filters"] = f"hubspot_owner_id={owner_id}"
    return ok(_get("/crm/v3/objects/tasks", params))

@mcp.tool()
def log_call(contact_id: str, call_notes: str, duration_seconds: int = 0,
             call_outcome: str = "CONNECTED") -> str:
    """Log a phone call on a HubSpot contact.
    call_outcome: CONNECTED | LEFT_VOICEMAIL | NO_ANSWER | WRONG_NUMBER"""
    import time
    props = {
        "hs_call_body": call_notes,
        "hs_call_duration": str(duration_seconds * 1000),
        "hs_call_disposition": call_outcome,
        "hs_call_status": "COMPLETED",
        "hs_timestamp": str(int(time.time() * 1000)),
    }
    result = _post("/crm/v3/objects/calls", {"properties": props})
    call_id = result.get("id", "")
    if call_id:
        try:
            _post(f"/crm/v4/objects/calls/{call_id}/associations/contacts/{contact_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 194}])
        except Exception: pass
    return ok(result)

@mcp.tool()
def associate_records(from_object: str, from_id: str, to_object: str, to_id: str) -> str:
    """Link any two HubSpot CRM records together.
    Supported: contacts↔deals, contacts↔companies, deals↔companies, contacts↔tickets, deals↔tickets"""
    assoc_map = {
        ("contacts","deals"):     [{"associationCategory":"HUBSPOT_DEFINED","associationTypeId":4}],
        ("contacts","companies"): [{"associationCategory":"HUBSPOT_DEFINED","associationTypeId":279}],
        ("deals","companies"):    [{"associationCategory":"HUBSPOT_DEFINED","associationTypeId":342}],
        ("contacts","tickets"):   [{"associationCategory":"HUBSPOT_DEFINED","associationTypeId":16}],
        ("deals","tickets"):      [{"associationCategory":"HUBSPOT_DEFINED","associationTypeId":28}],
    }
    key = (from_object.lower(), to_object.lower())
    types = assoc_map.get(key, [{"associationCategory":"HUBSPOT_DEFINED","associationTypeId":1}])
    result = _post(f"/crm/v4/objects/{from_object}/{from_id}/associations/{to_object}/{to_id}", types)
    return ok({"status": "associated", "result": result})

@mcp.tool()
def list_owners(limit: int = 100) -> str:
    """List all HubSpot users (owners) with their IDs and email addresses."""
    return ok(_get("/crm/v3/owners", {"limit": limit}))

# =============================================================================
# RESOURCES  — app-controlled, injected as read-only context
# =============================================================================

@mcp.resource("hubspot://pipelines")
def get_pipelines() -> str:
    """All HubSpot deal pipelines with stage names and IDs.
    Reference this before creating or moving deals."""
    data = _get("/crm/v3/pipelines/deals")
    out = []
    for p in data.get("results", []):
        out.append({
            "pipeline_id": p.get("id"), "pipeline_name": p.get("label"),
            "stages": [{"stage_id": s.get("id"), "stage_name": s.get("label"),
                        "probability": s.get("metadata", {}).get("probability")}
                       for s in p.get("stages", [])]})
    return json.dumps(out, indent=2)

@mcp.resource("hubspot://ticket-pipelines")
def get_ticket_pipelines() -> str:
    """All HubSpot ticket pipelines with stage IDs.
    Reference this before creating or updating tickets."""
    data = _get("/crm/v3/pipelines/tickets")
    out = []
    for p in data.get("results", []):
        out.append({
            "pipeline_id": p.get("id"), "pipeline_name": p.get("label"),
            "stages": [{"stage_id": s.get("id"), "stage_name": s.get("label")}
                       for s in p.get("stages", [])]})
    return json.dumps(out, indent=2)

@mcp.resource("hubspot://owners")
def get_owners_resource() -> str:
    """All HubSpot owners (users) with IDs, emails and names.
    Reference this before assigning deals, tasks or tickets."""
    data = _get("/crm/v3/owners", {"limit": 100})
    out = [{"owner_id": o.get("id"), "email": o.get("email"),
            "name": f"{o.get('firstName','')} {o.get('lastName','')}".strip()}
           for o in data.get("results", [])]
    return json.dumps(out, indent=2)

@mcp.resource("hubspot://contact-properties")
def get_contact_properties() -> str:
    """All valid HubSpot contact property names and types.
    Reference this before creating or updating contacts."""
    data = _get("/crm/v3/properties/contacts")
    out = [{"name": p.get("name"), "label": p.get("label"),
            "type": p.get("type"), "field_type": p.get("fieldType")}
           for p in data.get("results", []) if not p.get("hidden", False)]
    return json.dumps(out, indent=2)

@mcp.resource("hubspot://deal-properties")
def get_deal_properties() -> str:
    """All valid HubSpot deal property names and types.
    Reference this before creating or updating deals."""
    data = _get("/crm/v3/properties/deals")
    out = [{"name": p.get("name"), "label": p.get("label"), "type": p.get("type")}
           for p in data.get("results", []) if not p.get("hidden", False)]
    return json.dumps(out, indent=2)

# =============================================================================
# PROMPTS  — user-controlled slash commands
# =============================================================================

@mcp.prompt()
def find_contact(identifier: str) -> str:
    """/find_contact <email or name> — Search and display a contact summary."""
    return f"""
Search HubSpot for a contact matching: "{identifier}"
Use the search_contacts tool to find them.
Present the result as:
─────────────────────────────────────
👤  Name:    [firstname lastname]
📧  Email:   [email]
📞  Phone:   [phone]
🏢  Company: [company]
🆔  ID:      [id]
📊  Status:  [hs_lead_status]
─────────────────────────────────────
If not found say: "No contact found for '{identifier}' in HubSpot."
"""

@mcp.prompt()
def deal_report(stage_filter: str = "") -> str:
    """/deal_report [stage] — Show all deals, optionally filtered by stage."""
    filt = f'Filter deals in stage: "{stage_filter}"' if stage_filter else "Show all open deals"
    return f"""
{filt}
Use search_deals or list_deals to retrieve them.
Present as a table:
──────────────────────────────────────────────────────
Deal Name         | Amount    | Stage        | Close
──────────────────────────────────────────────────────
[dealname]        | $[amount] | [stage]      | [date]
──────────────────────────────────────────────────────
At the bottom include:
  Total deals: X
  Total pipeline value: $X
"""

@mcp.prompt()
def create_contact_flow(email: str) -> str:
    """/create_contact_flow <email> — Guided flow to create a new HubSpot contact."""
    return f"""
The user wants to create a HubSpot contact with email: {email}
First use search_contacts to check if they already exist.
If they exist: show their details and ask if user wants to update instead.
If not: ask for first name, last name, phone, company one at a time.
Then use create_contact to create them.
Confirm: "✅ Contact created! ID: [id]"
"""

@mcp.prompt()
def contact_summary(contact_id: str) -> str:
    """/contact_summary <contact_id> — Full CRM summary for a contact."""
    return f"""
Pull a full CRM summary for contact ID: {contact_id}
Step 1: get_contact for basic info
Step 2: search_deals to find their deals (search by their company name)
Step 3: search_tickets to find any open tickets

Present as:
═══════════════════════════════════════
  CONTACT SUMMARY — HubSpot CRM
═══════════════════════════════════════
👤 [Full Name]
📧 [email]  📞 [phone]
🏢 [company]

💰 DEALS ([count])
  • [deal name] — $[amount] — [stage]

🎫 TICKETS ([count])
  • [subject] — [priority] — [status]

📋 Suggest a follow-up task if no recent activity.
═══════════════════════════════════════
"""

@mcp.prompt()
def pipeline_overview() -> str:
    """/pipeline_overview — Full overview of the HubSpot sales pipeline by stage."""
    return """
Get a full pipeline overview.
Step 1: Use hubspot://pipelines resource to get all stages
Step 2: Use search_deals for deals in each stage
Present:
═══════════════════════════════════════
  PIPELINE OVERVIEW — HubSpot CRM
═══════════════════════════════════════
Stage              | Deals | Total Value
───────────────────|───────|────────────
[stage name]       |  [n]  | $[value]
TOTAL: $[grand total]
═══════════════════════════════════════
"""

@mcp.prompt()
def log_call_flow(contact_identifier: str) -> str:
    """/log_call_flow <name or email> — Guided flow to log a call on a contact."""
    return f"""
User wants to log a call for: "{contact_identifier}"
Step 1: search_contacts to find their contact ID
Step 2: Ask - how long was the call? (minutes)
Step 3: Ask - outcome? (Connected / Left voicemail / No answer)
Step 4: Ask - any notes or action items?
Step 5: Use log_call to record it
Step 6: Ask if they want a follow-up task. If yes, use create_task.
Confirm: "✅ Call logged on [contact name]'s record."
"""

@mcp.prompt()
def ticket_triage(priority: str = "HIGH") -> str:
    """/ticket_triage [priority] — List and triage open tickets by priority."""
    return f"""
Find all open HubSpot tickets with priority: {priority}
Use search_tickets to find them.
For each ticket present:
  🎫 [subject] | Priority: [priority] | Owner: [owner] | Stage: [stage]
Then ask: "Would you like to escalate, reassign, or update any of these tickets?"
If yes, use update_ticket or escalate_ticket accordingly.
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")

# =============================================================================
# MISSING TOOLS — completing all 62
# =============================================================================

# ── Contacts (missing) ────────────────────────────────────────────────────────

@mcp.tool()
def batch_create_contacts(contacts: list) -> str:
    """Create multiple contacts at once (up to 100).
    contacts: list of dicts, each with at minimum {"email": "..."} plus any other properties."""
    inputs = [{"properties": c} for c in contacts]
    return ok(_post("/crm/v3/objects/contacts/batch/create", {"inputs": inputs}))

@mcp.tool()
def batch_update_contacts(updates: list) -> str:
    """Update multiple contacts at once.
    updates: list of dicts like [{"id": "123", "properties": {"phone": "+1..."}}]"""
    inputs = [{"id": u["id"], "properties": u["properties"]} for u in updates]
    return ok(_post("/crm/v3/objects/contacts/batch/update", {"inputs": inputs}))

@mcp.tool()
def get_contact_associations(contact_id: str, to_object: str = "deals") -> str:
    """Get all records associated with a contact.
    to_object: deals | companies | tickets | notes | tasks | calls"""
    return ok(_get(f"/crm/v3/objects/contacts/{contact_id}/associations/{to_object}"))

# ── Companies (missing) ───────────────────────────────────────────────────────

@mcp.tool()
def list_companies(limit: int = 20, after: str = "") -> str:
    """List all HubSpot companies with pagination."""
    params = {"limit": limit, "properties": "name,domain,industry,city,phone,annualrevenue"}
    if after: params["after"] = after
    return ok(_get("/crm/v3/objects/companies", params))

@mcp.tool()
def delete_company(company_id: str) -> str:
    """Archive (soft-delete) a HubSpot company by ID."""
    return ok(_delete(f"/crm/v3/objects/companies/{company_id}"))

@mcp.tool()
def get_company_deals(company_id: str) -> str:
    """Get all deals associated with a HubSpot company."""
    return ok(_get(f"/crm/v3/objects/companies/{company_id}/associations/deals"))

# ── Deals (missing) ───────────────────────────────────────────────────────────

@mcp.tool()
def get_deal_line_items(deal_id: str) -> str:
    """Get all product line items attached to a HubSpot deal."""
    assoc = _get(f"/crm/v3/objects/deals/{deal_id}/associations/line_items")
    ids = [r["id"] for r in assoc.get("results", [])]
    if not ids:
        return ok({"results": [], "message": "No line items on this deal."})
    results = []
    for lid in ids:
        results.append(_get(f"/crm/v3/objects/line_items/{lid}",
                            {"properties": "name,quantity,price,amount,hs_product_id"}))
    return ok({"results": results})

# ── Tickets (missing) ─────────────────────────────────────────────────────────

@mcp.tool()
def list_tickets(limit: int = 20, after: str = "") -> str:
    """List all HubSpot support tickets with pagination."""
    params = {"limit": limit,
              "properties": "subject,content,hs_ticket_priority,hs_pipeline_stage,hubspot_owner_id"}
    if after: params["after"] = after
    return ok(_get("/crm/v3/objects/tickets", params))

@mcp.tool()
def delete_ticket(ticket_id: str) -> str:
    """Archive (soft-delete) a HubSpot ticket by ID."""
    return ok(_delete(f"/crm/v3/objects/tickets/{ticket_id}"))

# ── Engagements (missing) ─────────────────────────────────────────────────────

@mcp.tool()
def log_email(contact_id: str, subject: str, body: str, direction: str = "EMAIL") -> str:
    """Log an outbound email on a HubSpot contact record.
    direction: EMAIL (default)"""
    import time
    props = {
        "hs_email_subject": subject,
        "hs_email_text": body,
        "hs_email_direction": direction,
        "hs_email_status": "SENT",
        "hs_timestamp": str(int(time.time() * 1000)),
    }
    result = _post("/crm/v3/objects/emails", {"properties": props})
    email_id = result.get("id", "")
    if email_id:
        try:
            _post(f"/crm/v4/objects/emails/{email_id}/associations/contacts/{contact_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 198}])
        except Exception: pass
    return ok(result)

@mcp.tool()
def create_meeting(title: str, body: str = "", start_time: str = "",
                   end_time: str = "", contact_id: str = "") -> str:
    """Log a meeting on a HubSpot contact.
    start_time / end_time: ISO format e.g. 2025-06-01T10:00:00.000Z"""
    import time
    props: dict = {
        "hs_meeting_title": title,
        "hs_meeting_body": body,
        "hs_timestamp": str(int(time.time() * 1000)),
    }
    if start_time: props["hs_meeting_start_time"] = start_time
    if end_time:   props["hs_meeting_end_time"]   = end_time
    result = _post("/crm/v3/objects/meetings", {"properties": props})
    meeting_id = result.get("id", "")
    if contact_id and meeting_id:
        try:
            _post(f"/crm/v4/objects/meetings/{meeting_id}/associations/contacts/{contact_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 200}])
        except Exception: pass
    return ok(result)

@mcp.tool()
def get_activity_timeline(contact_id: str, limit: int = 20) -> str:
    """Get the full engagement activity timeline for a HubSpot contact.
    Includes calls, emails, notes, meetings and tasks."""
    results = {}
    for obj in ["notes", "calls", "emails", "meetings", "tasks"]:
        try:
            assoc = _get(f"/crm/v3/objects/contacts/{contact_id}/associations/{obj}")
            ids = [r["id"] for r in assoc.get("results", [])[:5]]
            items = []
            for oid in ids:
                prop_map = {
                    "notes":    "hs_note_body,hs_timestamp",
                    "calls":    "hs_call_body,hs_call_duration,hs_call_disposition,hs_timestamp",
                    "emails":   "hs_email_subject,hs_email_text,hs_timestamp",
                    "meetings": "hs_meeting_title,hs_meeting_start_time,hs_timestamp",
                    "tasks":    "hs_task_subject,hs_task_status,hs_timestamp",
                }
                items.append(_get(f"/crm/v3/objects/{obj}/{oid}",
                                  {"properties": prop_map[obj]}))
            results[obj] = items
        except Exception:
            results[obj] = []
    return ok(results)

@mcp.tool()
def send_marketing_email(contact_id: str, email_id: str) -> str:
    """Trigger a HubSpot marketing email to a specific contact.
    email_id: the ID of the marketing email to send (from list_campaigns)"""
    return ok(_post(f"/marketing/v3/transactional/single-email/send", {
        "emailId": int(email_id),
        "message": {"to": contact_id},
    }))

# ── Commerce Hub ──────────────────────────────────────────────────────────────

@mcp.tool()
def list_products(limit: int = 20, after: str = "") -> str:
    """List all products in the HubSpot product library."""
    params = {"limit": limit, "properties": "name,description,price,hs_sku,hs_product_type"}
    if after: params["after"] = after
    return ok(_get("/crm/v3/objects/products", params))

@mcp.tool()
def create_product(name: str, price: float, description: str = "", sku: str = "") -> str:
    """Create a new product in the HubSpot product library."""
    props: dict = {"name": name, "price": str(price)}
    if description: props["description"] = description
    if sku:         props["hs_sku"]      = sku
    return ok(_post("/crm/v3/objects/products", {"properties": props}))

@mcp.tool()
def create_line_item(deal_id: str, product_id: str, quantity: int = 1, price: float = 0.0) -> str:
    """Add a product line item to a HubSpot deal.
    product_id: from list_products"""
    props: dict = {
        "hs_product_id": product_id,
        "quantity":      str(quantity),
    }
    if price: props["price"] = str(price)
    result = _post("/crm/v3/objects/line_items", {"properties": props})
    line_item_id = result.get("id", "")
    if line_item_id:
        try:
            _post(f"/crm/v4/objects/line_items/{line_item_id}/associations/deals/{deal_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 20}])
        except Exception: pass
    return ok(result)

@mcp.tool()
def list_quotes(limit: int = 20, after: str = "") -> str:
    """List all HubSpot CPQ quotes."""
    params = {"limit": limit,
              "properties": "hs_title,hs_status,hs_expiration_date,hs_quote_amount"}
    if after: params["after"] = after
    return ok(_get("/crm/v3/objects/quotes", params))

@mcp.tool()
def create_quote(title: str, deal_id: str, expiration_date: str = "") -> str:
    """Create a new HubSpot sales quote linked to a deal.
    expiration_date: YYYY-MM-DD format"""
    props: dict = {"hs_title": title, "hs_status": "DRAFT"}
    if expiration_date: props["hs_expiration_date"] = expiration_date
    result = _post("/crm/v3/objects/quotes", {"properties": props})
    quote_id = result.get("id", "")
    if quote_id:
        try:
            _post(f"/crm/v4/objects/quotes/{quote_id}/associations/deals/{deal_id}",
                  [{"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": 64}])
        except Exception: pass
    return ok(result)

@mcp.tool()
def update_quote_status(quote_id: str, status: str) -> str:
    """Update the status of a HubSpot quote.
    status: DRAFT | PENDING_APPROVAL | APPROVED | REJECTED | PUBLISHED"""
    return ok(_patch(f"/crm/v3/objects/quotes/{quote_id}",
                     {"properties": {"hs_status": status}}))

@mcp.tool()
def list_payments(limit: int = 20, after: str = "") -> str:
    """List HubSpot payment records."""
    params = {"limit": limit,
              "properties": "hs_payment_status,hs_payment_amount,hs_currency_code,createdate"}
    if after: params["after"] = after
    return ok(_get("/crm/v3/objects/commerce_payments", params))

@mcp.tool()
def get_payment(payment_id: str) -> str:
    """Get details of a specific HubSpot payment by ID."""
    return ok(_get(f"/crm/v3/objects/commerce_payments/{payment_id}",
                   {"properties": "hs_payment_status,hs_payment_amount,hs_currency_code,createdate"}))

# ── Marketing Hub ─────────────────────────────────────────────────────────────

@mcp.tool()
def list_contact_lists(limit: int = 20) -> str:
    """List all HubSpot contact lists (static and dynamic/active)."""
    return ok(_get("/contacts/v1/lists", {"count": limit}))

@mcp.tool()
def create_contact_list(name: str, dynamic: bool = False) -> str:
    """Create a new HubSpot contact list.
    dynamic=True creates an active (smart) list, False creates a static list."""
    return ok(_post("/contacts/v1/lists",
                    {"name": name, "dynamic": dynamic, "filters": []}))

@mcp.tool()
def add_contacts_to_list(list_id: str, contact_ids: list) -> str:
    """Add contacts to a HubSpot static list.
    contact_ids: list of contact ID strings"""
    return ok(_post(f"/contacts/v1/lists/{list_id}/add", {"vids": contact_ids}))

@mcp.tool()
def list_campaigns(limit: int = 20) -> str:
    """List all HubSpot marketing campaigns (upgraded API — July 2025)."""
    return ok(_get("/marketing/v3/campaigns", {"limit": limit}))

@mcp.tool()
def create_campaign(name: str, start_date: str = "", end_date: str = "") -> str:
    """Create a new HubSpot marketing campaign.
    start_date / end_date: YYYY-MM-DD format"""
    body: dict = {"name": name}
    if start_date: body["startDate"] = start_date
    if end_date:   body["endDate"]   = end_date
    return ok(_post("/marketing/v3/campaigns", body))

@mcp.tool()
def list_forms(limit: int = 20) -> str:
    """List all HubSpot forms."""
    return ok(_get("/marketing/v3/forms", {"limit": limit}))

@mcp.tool()
def get_form_submissions(form_id: str, limit: int = 20) -> str:
    """Get all submissions for a specific HubSpot form."""
    return ok(_get(f"/form-integrations/v1/submissions/forms/{form_id}",
                   {"limit": limit}))

# ── Meta / Utility (missing) ──────────────────────────────────────────────────

@mcp.tool()
def get_owner_by_email(email: str) -> str:
    """Find a HubSpot owner (user) by their email address."""
    return ok(_get("/crm/v3/owners", {"email": email, "limit": 1}))

@mcp.tool()
def list_properties(object_type: str = "contacts") -> str:
    """List all available properties for a CRM object type.
    object_type: contacts | deals | companies | tickets | products | quotes"""
    return ok(_get(f"/crm/v3/properties/{object_type}"))

@mcp.tool()
def create_property(object_type: str, name: str, label: str,
                    field_type: str = "text", group_name: str = "contactinformation") -> str:
    """Create a custom property on any HubSpot CRM object.
    object_type: contacts | deals | companies | tickets
    field_type: text | textarea | number | date | select | checkbox | booleancheckbox
    group_name: contactinformation | dealinformation | companyinformation"""
    body = {
        "name":      name,
        "label":     label,
        "type":      "string" if field_type in ("text","textarea") else field_type,
        "fieldType": field_type,
        "groupName": group_name,
    }
    return ok(_post(f"/crm/v3/properties/{object_type}", body))
