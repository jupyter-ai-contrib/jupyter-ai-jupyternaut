import os
from typing import Any, Callable

import aiosqlite
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, wrap_tool_call
from langchain.agents.middleware.file_search import FilesystemFileSearchMiddleware
from langchain.agents.middleware.shell_tool import ShellToolMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from .chat_models import ChatLiteLLM
from .prompt_template import (
    JUPYTERNAUT_SYSTEM_PROMPT_TEMPLATE,
    JupyternautSystemPromptArgs,
)
from .toolkits.notebook import toolkit as nb_toolkit

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")


class ToolMonitoringMiddleware(AgentMiddleware):
    def __init__(self, *, stream_message: BasePersona.stream_message):
        self.stream_message = stream_message

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        running_tool_msg = f"Running **{request.tool_call['name']}** with *{request.tool_call['args']}*"
        await self.stream_message(self._aiter(running_tool_msg))
        try:
            result = await handler(request)
            if hasattr(result, "content") and result.content != "null":
                completed_tool_msg = str(result.content)[:100]
            else:
                completed_tool_msg = "Done!"
            await self.stream_message(self._aiter(completed_tool_msg))
            return result
        except Exception as e:
            await self.stream_message(f"**{request.tool_call['name']}** failed: {e}")
            return ToolMessage(
                tool_call_id=request.tool_call["id"], status="error", content=f"{e}"
            )

    async def _aiter(self, message: str):
        yield message


class JupyternautPersona(BasePersona):
    """
    The Jupyternaut persona, the main persona provided by Jupyter AI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Jupyternaut",
            avatar_path="/api/jupyternaut/static/jupyternaut.svg",
            description="The standard agent provided by JupyterLab. Currently has no tools.",
            system_prompt="...",
        )

    async def get_memory_store(self):
        if not hasattr(self, "_memory_store"):
            conn = await aiosqlite.connect(MEMORY_STORE_PATH, check_same_thread=False)
            self._memory_store = AsyncSqliteSaver(conn)
        return self._memory_store

    def get_tools(self):
        tools = []
        tools += nb_toolkit

        return tools

    async def get_agent(self, model_id: str, model_args, system_prompt: str):
        model = ChatLiteLLM(**model_args, model_id=model_id, streaming=True)
        memory_store = await self.get_memory_store()

        @wrap_tool_call
        def handle_tool_errors(request, handler):
            """Handle tool execution errors with custom messages."""
            try:
                return handler(request)
            except Exception as e:
                # Return a custom error message to the model
                return ToolMessage(
                    content=f"Error calling tool: ({str(e)})",
                    tool_call_id=request.tool_call["id"],
                )

        if not hasattr(self, "search_tool"):
            self.search_tool = FilesystemFileSearchMiddleware(
                root_path=self.parent.root_dir
            )
        if not hasattr(self, "shell_tool"):
            self.shell_tool = ShellToolMiddleware(workspace_root=self.parent.root_dir)
        if not hasattr(self, "tool_call_handler"):
            self.tool_call_handler = ToolMonitoringMiddleware(
                stream_message=self.stream_message
            )

        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=self.get_tools(),
            middleware=[self.search_tool, self.shell_tool, self.tool_call_handler],
        )

    async def process_message(self, message: Message) -> None:
        if not hasattr(self, "config_manager"):
            self.send_message(
                "Jupyternaut requires the `jupyter_ai_jupyternaut` server extension package.\n\n",
                "Please make sure to first install that package in your environment & restart the server.",
            )
        if not self.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return

        model_id = self.config_manager.chat_model
        model_args = self.config_manager.chat_model_args
        system_prompt = self.get_system_prompt(model_id=model_id, message=message)
        agent = await self.get_agent(
            model_id=model_id, model_args=model_args, system_prompt=system_prompt
        )

        async def create_aiter():
            async for chunk, _ in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                {"configurable": {"thread_id": self.ychat.get_id()}},
                stream_mode="messages",
            ):
                if (
                    hasattr(chunk, "content")
                    and (content := chunk.content)
                    and content != "null"
                ):
                    yield content

        response_aiter = create_aiter()
        await self.stream_message(response_aiter)

    def get_system_prompt(
        self, model_id: str, message: Message
    ) -> list[dict[str, Any]]:
        """
        Returns the system prompt, including attachments as a string.
        """
        system_msg_args = JupyternautSystemPromptArgs(
            model_id=model_id,
            persona_name=self.name,
            context=self.process_attachments(message),
        ).model_dump()

        return JUPYTERNAUT_SYSTEM_PROMPT_TEMPLATE.render(**system_msg_args)

    def shutdown(self):
        if self._memory_store:
            self.parent.event_loop.create_task(self._memory_store.conn.close())
        super().shutdown()
