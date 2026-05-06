import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.claude import Claude
from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# ── Azure OpenAI config ───────────────────────────────────────────────────────
azure_deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
azure_api_key     = os.getenv("AZURE_OPENAI_API_KEY", "")
azure_endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")

assert azure_deployment, "AZURE_OPENAI_DEPLOYMENT_NAME missing in .env"
assert azure_api_key,    "AZURE_OPENAI_API_KEY missing in .env"
assert azure_endpoint,   "AZURE_OPENAI_ENDPOINT missing in .env"

# ── HubSpot token check (warn but don't crash) ────────────────────────────────
if not os.getenv("HUBSPOT_TOKEN"):
    print("⚠️  WARNING: HUBSPOT_TOKEN not set in .env — HubSpot tools will be unavailable\n")


async def main():
    command = ("uv", ["run"]) if os.getenv("USE_UV", "0") == "1" else ("python", [])

    claude_service = Claude(model=azure_deployment)

    async with AsyncExitStack() as stack:
        # ── Document MCP server (always on) ──────────────────────────────────
        doc_args = command[1] + ["mcp_server.py"]
        doc_client = await stack.enter_async_context(
            MCPClient(command=command[0], args=doc_args)
        )
# 
        clients: dict[str, MCPClient] = {"doc_client": doc_client}

        # ── HubSpot MCP server (always on, graceful if token missing) ─────────
        hs_args = command[1] + ["hubspot_mcp_server.py"]
        try:
            hs_client = await stack.enter_async_context(
                MCPClient(command=command[0], args=hs_args)
            )
            clients["hubspot_client"] = hs_client
        except Exception as e:
            print(f"⚠️  HubSpot MCP server failed to start: {e}")

        # ── Any extra servers passed as CLI args ──────────────────────────────
        for i, script in enumerate(sys.argv[1:]):
            try:
                extra_args = command[1] + [script]
                extra_client = await stack.enter_async_context(
                    MCPClient(command=command[0], args=extra_args)
                )
                clients[f"extra_{i}"] = extra_client
            except Exception as e:
                print(f"⚠️  Could not start {script}: {e}")

        chat = CliChat(doc_client=doc_client, clients=clients, claude_service=claude_service)
        cli  = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
