import os
from typing import Any

import aiosqlite
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message, TokenUsage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .chat_models import ChatLiteLLM
from .prompt_template import (
    JUPYTERNAUT_SYSTEM_PROMPT_TEMPLATE,
    JupyternautSystemPromptArgs,
)
from .toolkits.notebook import toolkit as nb_toolkit
from .toolkits.jupyterlab import toolkit as jlab_toolkit
from .toolkits.code_execution import toolkit as exec_toolkit

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")

JUPYTERNAUT_AVATAR_PATH = str(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../static", "jupyternaut.svg")
    )
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
            avatar_path=JUPYTERNAUT_AVATAR_PATH,
            description="The standard agent provided by JupyterLab. Currently has no tools.",
            system_prompt="...",
        )

    @property
    def yroom_manager(self):
        return self.parent.serverapp.web_app.settings["yroom_manager"]

    async def get_memory_store(self):
        if not hasattr(self, "_memory_store"):
            conn = await aiosqlite.connect(MEMORY_STORE_PATH, check_same_thread=False)
            self._memory_store = AsyncSqliteSaver(conn)
        return self._memory_store

    def get_tools(self):
        tools = nb_toolkit
        tools += jlab_toolkit
        tools += exec_toolkit
        return tools

    def _create_tool_error_handler(self):
        """Creates a tool error handler with access to instance attributes."""
        @wrap_tool_call
        async def handle_tool_errors(request, handler):
            """
            LangChain middleware that catches exceptions raised by tools & returns a
            `ToolMessage` object to allow the agent to resume execution.
            """
            try:
                return await handler(request)
            except Exception as e:
                # Log the exception
                self.log.exception("Tool call raised an exception.")

                # Return a custom error message to the model
                return ToolMessage(
                    content=f"Tool error: Please check your input and try again. ({str(e)})",
                    tool_call_id=request.tool_call["id"]
                )

        return handle_tool_errors

    async def get_agent(self, model_id: str, model_args, system_prompt: str):
        model = ChatLiteLLM(**model_args, model=model_id, streaming=True)
        memory_store = await self.get_memory_store()

        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=self.get_tools(),
            middleware=[self._create_tool_error_handler()],
        )

    async def process_message(self, message: Message) -> None:
        if not hasattr(self, "config_manager"):
            self.send_message(
                "Jupyternaut requires the `jupyter_ai_jupyternaut` server extension package.\n\n"
                "Please make sure to first install that package in your environment & restart the server."
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

        context = {
            "thread_id": self.ychat.get_id(),
            "username": message.sender
        }

        # Track token usage
        usage_data = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }

        async def create_aiter():
            nonlocal usage_data
            async for token, metadata in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                {"configurable": context},
                stream_mode="messages",
            ):
                node = metadata["langgraph_node"]
                content_blocks = token.content_blocks

                # Extract usage information if available
                if hasattr(token, "usage_metadata") and token.usage_metadata:
                    usage_metadata = token.usage_metadata
                    usage_data["input_tokens"] = usage_metadata.get("input_tokens", usage_data["input_tokens"])
                    usage_data["output_tokens"] = usage_metadata.get("output_tokens", usage_data["output_tokens"])
                    usage_data["total_tokens"] = usage_metadata.get("total_tokens", usage_data["total_tokens"])

                if node == "model" and content_blocks:
                    if token.text:
                        yield token.text

        response_aiter = create_aiter()

        # Stream the message
        await self.stream_message(response_aiter)

        # After streaming, add usage if we captured any
        if usage_data["total_tokens"] > 0:
            usage = TokenUsage(
                input_tokens=usage_data["input_tokens"],
                output_tokens=usage_data["output_tokens"],
                total_tokens=usage_data["total_tokens"],
                model=model_id
            )
            # Update the last message with usage information
            messages = self.ychat.get_messages()
            if messages:
                last_msg = messages[-1]
                if last_msg.sender == self.id and not last_msg.usage:
                    last_msg.usage = usage
                    self.ychat.update_message(last_msg)

    def get_system_prompt(
        self, model_id: str, message: Message
    ) -> list[dict[str, Any]]:
        """
        Returns the system prompt, including attachments as a string.
        """

        context = self.process_attachments(message) or ""
        context = f"User's username is '{message.sender}'\n\n" + context

        system_msg_args = JupyternautSystemPromptArgs(
            model_id=model_id,
            persona_name=self.name,
            context=context,
        ).model_dump()

        return JUPYTERNAUT_SYSTEM_PROMPT_TEMPLATE.render(**system_msg_args)

    def shutdown(self):
        if hasattr(self, "_memory_store"):
            self.parent.event_loop.create_task(self._memory_store.conn.close())
        super().shutdown()
