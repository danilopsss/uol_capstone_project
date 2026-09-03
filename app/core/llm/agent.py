import json
import logging
from collections.abc import Callable, Iterator

import ollama
from ollama import Message, Options, Tool

from .tools.skills import get_skill, list_skills

logger = logging.getLogger("OllamaAgent")

ToolFunction = Callable[..., object]


class OllamaAgent:
    def __init__(
        self,
        model: str,
        tools: list[Tool],
        tool_functions: dict[str, ToolFunction],
        max_steps: int = 10,
        temperature: float = 0.6,
    ) -> None:
        self.model: str = model
        self.tools: list[Tool] = tools
        self.tool_functions: dict[str, ToolFunction] = tool_functions
        self.max_steps: int = max_steps
        self._list_skills_done: bool = False
        self.temperature: float = temperature

    @property
    def _default_system_prompt(self) -> str:
        return """
            - You are a legal document simplifier.
            - Explain legal clauses and contracts in plain, easy-to-understand English.
            - Simplify jargon, define legal terms, and keep the original meaning intact.
            - Provide clear, concise explanations without adding or removing meaning.
        """

    def _skill_content(self) -> str:
        paths = list_skills()
        if not paths:
            raise RuntimeError("No skills available")
        return get_skill(paths[0])

    def run(self, prompt: str) -> str:
        logger.info("Running agent with model=%s", self.model)
        messages = self._initial_messages(prompt)
        for step in range(1, self.max_steps + 1):
            logger.info("Calling LLM (step %d/%d)", step, self.max_steps)
            response = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
                model=self.model,
                messages=messages,
                tools=self.tools,
                think="medium",
                options=Options(temperature=self.temperature),
            )
            assistant_message = response.message
            messages.append(assistant_message)
            tool_calls = list(assistant_message.tool_calls or [])
            if not tool_calls:
                logger.info("LLM returned final response")
                return self._clean(assistant_message.content or "")
            self._execute_tool_calls(tool_calls, messages)
        raise RuntimeError(f"Agent exceeded maximum steps ({self.max_steps})")

    def run_stream(self, prompt: str) -> Iterator[str]:
        """Run the agent, yielding streamed content tokens as they arrive."""
        logger.info("Running streaming agent with model=%s", self.model)
        messages = self._initial_messages(prompt)
        for step in range(1, self.max_steps + 1):
            logger.info("Streaming LLM (step %d/%d)", step, self.max_steps)
            stream = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
                model=self.model,
                messages=messages,
                tools=self.tools,
                think="medium",
                options=Options(temperature=self.temperature),
                stream=True,
            )
            content_parts: list[str] = []
            tool_calls: list[Message.ToolCall] = []
            for chunk in stream:
                message = chunk.message
                if content := message.content:
                    content_parts.append(content)
                    yield content
                if message.tool_calls:
                    tool_calls = list(message.tool_calls)
            if not tool_calls:
                logger.info("LLM returned final response")
                return
            assistant_message = Message(
                role="assistant",
                content="".join(content_parts),
                tool_calls=tool_calls,
            )
            messages.append(assistant_message)
            self._execute_tool_calls(tool_calls, messages)
        raise RuntimeError(f"Agent exceeded maximum steps ({self.max_steps})")

    def _initial_messages(self, prompt: str) -> list[Message]:
        return [
            Message(role="system", content=self._default_system_prompt),
            Message(role="user", content=prompt),
            Message(role="system", content=self._skill_content()),
        ]

    @staticmethod
    def _clean(text: str) -> str:
        return text.replace("\\n", "\n").strip()

    def _execute_tool_calls(
        self, tool_calls: list[Message.ToolCall], messages: list[Message]
    ) -> None:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = dict(tool_call.function.arguments)
            if function_name == "get_skill" and not self._list_skills_done:
                self._run_list_skills_first(messages)
            logger.info("[TOOL CALL] %s(%s)", function_name, json.dumps(arguments))
            result = self._run_tool(function_name, arguments)
            if function_name == "list_skills":
                self._list_skills_done = True
            messages.append(
                Message(
                    role="tool",
                    tool_name=function_name,
                    content=json.dumps(result),
                )
            )

    def _run_list_skills_first(self, messages: list[Message]) -> None:
        logger.info(
            "[TOOL CALL] list_skills called before get_skill (enforced ordering)"
        )
        result = self._run_tool("list_skills", {})
        self._list_skills_done = True
        messages.append(
            Message(
                role="tool",
                tool_name="list_skills",
                content=json.dumps(result),
            )
        )

    def _run_tool(self, name: str, arguments: dict[str, object]) -> object:
        func = self.tool_functions.get(name)
        if not func:
            return {"error": f"Unknown tool: {name}"}
        return func(**arguments)
