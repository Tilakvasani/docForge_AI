import os
import json
from dataclasses import dataclass, field
from typing import Any, List
from openai import AzureOpenAI


@dataclass
class TextBlock:
    """Mimics Anthropic's TextBlock so the rest of the codebase needs no changes."""
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    """Mimics Anthropic's ToolUseBlock so tools.py works without changes."""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


class AzureMessage:
    """
    Mimics Anthropic's Message object.
    Wraps Azure OpenAI response so chat.py and tools.py need no changes.
    """
    def __init__(self, content: List[Any], stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason
        # Raw OpenAI tool_call dicts needed to rebuild the assistant message
        # in the correct format when appending to conversation history.
        self._raw_tool_calls: list = []


class Claude:
    def __init__(self, model: str):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        # model = Azure deployment name
        self.model = model

    # ------------------------------------------------------------------
    # Message history helpers
    # ------------------------------------------------------------------

    def add_user_message(self, messages: list, message):
        if isinstance(message, list):
            # Tool results - convert Anthropic-style dicts to OpenAI format.
            # Anthropic: {"tool_use_id": ..., "type": "tool_result", "content": ..., "is_error": ...}
            # OpenAI:    {"role": "tool", "tool_call_id": ..., "content": ...}
            for tool_result in message:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_use_id"],
                    "content": str(tool_result["content"]),
                })
        elif isinstance(message, AzureMessage):
            text = self.text_from_message(message)
            messages.append({"role": "user", "content": text})
        else:
            messages.append({"role": "user", "content": str(message)})

    def add_assistant_message(self, messages: list, message):
        if isinstance(message, AzureMessage):
            if message._raw_tool_calls:
                # OpenAI requires the assistant message to include tool_calls
                # so subsequent tool results can be correlated by ID.
                text_content = self.text_from_message(message)
                messages.append({
                    "role": "assistant",
                    "content": text_content or None,
                    "tool_calls": message._raw_tool_calls,
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": self.text_from_message(message),
                })
        else:
            messages.append({"role": "assistant", "content": str(message)})

    def text_from_message(self, message: AzureMessage) -> str:
        return "\n".join(
            block.text for block in message.content if isinstance(block, TextBlock)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_tools(self, tools: list) -> list:
        """
        Convert Anthropic-style tool dicts to OpenAI function-calling format.
        Anthropic:  {"name":..., "description":..., "input_schema":{...}}
        OpenAI:     {"type":"function","function":{"name":...,"description":...,"parameters":{...}}}
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]

    # ------------------------------------------------------------------
    # Main chat call
    # ------------------------------------------------------------------

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,        # ignored - not supported by Azure OpenAI
        thinking_budget=1024,  # ignored
    ) -> AzureMessage:
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})

        openai_messages += messages

        params: dict = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": openai_messages,
            "temperature": temperature,
        }

        if stop_sequences:
            params["stop"] = stop_sequences

        if tools:
            params["tools"] = self._convert_tools(tools)
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)

        choice = response.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason

        # Build content blocks matching Anthropic's Message.content shape
        content_blocks: list = []

        if msg.content:
            content_blocks.append(TextBlock(type="text", text=msg.content))

        raw_tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                content_blocks.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                )
                raw_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })

        # Map OpenAI finish_reason -> Anthropic stop_reason so chat.py is unchanged
        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

        azure_msg = AzureMessage(content=content_blocks, stop_reason=stop_reason)
        azure_msg._raw_tool_calls = raw_tool_calls
        return azure_msg
