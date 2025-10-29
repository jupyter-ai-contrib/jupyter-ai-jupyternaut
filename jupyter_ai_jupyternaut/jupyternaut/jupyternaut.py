import os
import aiosqlite
from typing import Any, Optional

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.persona_manager import SYSTEM_USERNAME
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .chat_models import ChatLiteLLM
from .prompt_template import (
    JUPYTERNAUT_SYSTEM_PROMPT_TEMPLATE,
    JupyternautSystemPromptArgs,
)

from .toolkits.code_execution import toolkit as exec_toolkit
from .toolkits.filesystem import toolkit as fs_toolkit
from .toolkits.notebook import toolkit as nb_toolkit


MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")


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
        tools += fs_toolkit

        return tools

    async def get_agent(self, model_id: str, model_args, system_prompt: str):
        model = ChatLiteLLM(**model_args, model_id=model_id, streaming=True)
        memory_store = await self.get_memory_store()
        return create_agent(
            model, 
            system_prompt=system_prompt, 
            checkpointer=memory_store,
            tools=self.get_tools()
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
            model_id=model_id, 
            model_args=model_args, 
            system_prompt=system_prompt
        )

        async def create_aiter():
            async for chunk, metadata in agent.astream(
                {"messages": [{"role": "user", "content": message.body}]},
                {"configurable": {"thread_id": self.ychat.get_id()}},
                stream_mode="messages",
            ):
                if chunk.content:
                    yield chunk.content

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
