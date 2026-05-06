from core.claude import Claude
from mcp_client import MCPClient
from core.tools import ToolManager


class Chat:
    def __init__(self, claude_service: Claude, clients: dict[str, MCPClient]):
        self.claude_service = claude_service
        self.clients = clients
        self.messages: list[dict] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(self, query: str) -> str:
        from core.cli_chat import SYSTEM_PROMPT
        await self._process_query(query)

        while True:
            response = self.claude_service.chat(
                messages=self.messages,
                system=SYSTEM_PROMPT,
                tools=await ToolManager.get_all_tools(self.clients),
            )
            self.claude_service.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                print(self.claude_service.text_from_message(response))
                tool_results = await ToolManager.execute_tool_requests(self.clients, response)
                self.claude_service.add_user_message(self.messages, tool_results)
            else:
                return self.claude_service.text_from_message(response)
