import os
from typing import Any, Callable

import aiosqlite
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
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
from .toolkits.jupyterlab import toolkit as jlab_toolkit

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")


def format_tool_args_compact(args_dict, threshold=25):
    """
    Create a more compact string representation of tool call args.
    Each key-value pair is on its own line for better readability.

    Args:
        args_dict (dict): Dictionary of tool arguments
        threshold (int): Maximum number of lines before truncation (default: 25)

    Returns:
        str: Formatted string representation of arguments
    """
    if not args_dict:
        return "{}"

    formatted_pairs = []

    for key, value in args_dict.items():
        value_str = str(value)
        lines = value_str.split('\n')

        if len(lines) <= threshold:
            if len(lines) == 1 and len(value_str) > 80:
                # Single long line - truncate
                truncated = value_str[:77] + "..."
                formatted_pairs.append(f"  {key}: {truncated}")
            else:
                # Add indentation for multi-line values
                if len(lines) > 1:
                    indented_value = '\n    '.join([''] + lines)
                    formatted_pairs.append(f"  {key}:{indented_value}")
                else:
                    formatted_pairs.append(f"  {key}: {value_str}")
        else:
            # Truncate and add summary
            truncated_lines = lines[:threshold]
            remaining_lines = len(lines) - threshold
            indented_value = '\n    '.join([''] + truncated_lines)
            formatted_pairs.append(f"  {key}:{indented_value}\n    [+{remaining_lines} more lines]")

    return "{\n" + ",\n".join(formatted_pairs) + "\n}"


class ToolMonitoringMiddleware(AgentMiddleware):
    def __init__(self, *, persona: BasePersona):
        self.stream_message = persona.stream_message
        self.log = persona.log

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        args = format_tool_args_compact(request.tool_call['args'])
        self.log.info(f"{request.tool_call['name']}({args})")
        
        try:
            result = await handler(request)
            self.log.info(f"{request.tool_call['name']} Done!")
            return result
        except Exception as e:
            self.log.info(f"{request.tool_call['name']} failed: {e}")
            return ToolMessage(
                tool_call_id=request.tool_call["id"], status="error", content=f"{e}"
            )


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
        tools = nb_toolkit
        tools += jlab_toolkit
        return nb_toolkit

    async def get_agent(self, model_id: str, model_args, system_prompt: str):
        model = ChatLiteLLM(**model_args, model_id=model_id, streaming=True)
        memory_store = await self.get_memory_store()

        if not hasattr(self, "search_tool"):
            self.search_tool = FilesystemFileSearchMiddleware(
                root_path=self.parent.root_dir
            )
        if not hasattr(self, "shell_tool"):
            self.shell_tool = ShellToolMiddleware(workspace_root=self.parent.root_dir)
        if not hasattr(self, "tool_call_handler"):
            self.tool_call_handler = ToolMonitoringMiddleware(
                persona=self
            )
        
        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=self.get_tools(), # notebook and jlab tools
            middleware=[self.shell_tool, self.tool_call_handler],
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
            async for token, metadata in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                {"configurable": {"thread_id": self.ychat.get_id()}},
                stream_mode="messages",
            ):
                node = metadata["langgraph_node"]
                content_blocks = token.content_blocks
                if (
                    node == "model" 
                    and content_blocks
                ):
                    if token.text:
                        yield token.text

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
