import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style
from core.cli_chat import CliChat

BANNER = """
╔══════════════════════════════════════════════════════════╗
║           HubSpot CRM Agent  ·  Powered by Azure AI     ║
╠══════════════════════════════════════════════════════════╣
║  Connections:                                            ║
║    📄 Document MCP Server  ✅ connected                  ║
║    🟠 HubSpot MCP Server   {hs_status}                   ║
╠══════════════════════════════════════════════════════════╣
║  Slash commands:                                         ║
║    /find_contact <email or name>                         ║
║    /deal_report [stage]                                  ║
║    /create_contact_flow <email>                          ║
║    /contact_summary <contact_id>                         ║
║    /pipeline_overview                                    ║
║    /log_call_flow <name or email>                        ║
║    /ticket_triage [priority]                             ║
║    /summarize <doc_id>  · /rewrite_as_markdown <doc_id>  ║
╠══════════════════════════════════════════════════════════╣
║  Reference docs with @filename  ·  Type 'exit' to quit  ║
╚══════════════════════════════════════════════════════════╝
"""

STYLE = Style.from_dict({
    "prompt":   "ansicyan bold",
    "":         "ansiwhite",
})


class CliApp:
    def __init__(self, chat: CliChat):
        self.chat = chat
        self.session: PromptSession = PromptSession(history=InMemoryHistory())
        self._hs_connected = False

    async def initialize(self):
        """Check connections and print banner."""
        # Check HubSpot connection
        try:
            hs_client = self.chat.clients.get("hubspot_client")
            if hs_client:
                tools = await hs_client.list_tools()
                self._hs_connected = len(tools) > 0
        except Exception:
            self._hs_connected = False

        hs_status = "✅ connected" if self._hs_connected else "❌ check HUBSPOT_TOKEN"
        # Pad to fixed width
        padding = " " * (14 - len(hs_status.replace("✅ connected","").replace("❌ check HUBSPOT_TOKEN","")))
        print(BANNER.format(hs_status=hs_status + padding))

    async def run(self):
        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.prompt("You ▶ ", style=STYLE)
                )
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 👋")
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye! 👋")
                break

            try:
                print("\nAssistant ▶ ", end="", flush=True)
                response = await self.chat.run(user_input)
                print(response)
                print()
            except Exception as e:
                print(f"\n[Error] {e}\n")
