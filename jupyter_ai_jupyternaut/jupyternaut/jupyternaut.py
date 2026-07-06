import os
from typing import Any

from jupyter_ai_persona_manager import (
  BasePersona,
  McpServerHttp,
  McpServerStdio,
  PersonaDefaults,
)
from jupyter_core.paths import jupyter_data_dir
from jupyterlab_chat.models import Message
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
  Connection,
  StreamableHttpConnection,
  StdioConnection,
)

from .chat_models import ChatLiteLLM
from .prompt_template import (
    JUPYTERNAUT_SYSTEM_PROMPT_TEMPLATE,
    JupyternautSystemPromptArgs,
)
from .toolkits.notebook import toolkit as nb_toolkit
from .toolkits.jupyterlab import toolkit as jlab_toolkit
from .toolkits.code_execution import toolkit as exec_toolkit

# if the context window is too small it won't see all of the
# system prompt
DEFAULT_OLLAMA_NUM_CTX = 16384

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

    async def get_memory_store(self):
        """
        Returns the checkpointer used to persist Jupyternaut's conversation
        memory.

        If the optional ``langgraph-checkpoint-sqlite`` package is installed
        (e.g. via ``jupyter-ai-jupyternaut[persistence]``), conversation memory
        is persisted to a local SQLite database so that it survives server
        restarts. Otherwise, Jupyternaut falls back to an in-memory checkpointer
        that retains memory only for the lifetime of the server process.
        """
        if not hasattr(self, "_memory_store"):
            self._memory_store = await self._create_memory_store()
        return self._memory_store

    async def _create_memory_store(self):
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError:
            from langgraph.checkpoint.memory import InMemorySaver

            self.log.info(
                "`langgraph-checkpoint-sqlite` is not installed; Jupyternaut "
                "will use in-memory conversation memory that is not persisted "
                "across server restarts. Install "
                "`jupyter-ai-jupyternaut[persistence]` to enable persistent "
                "memory."
            )
            return InMemorySaver()

        conn = await aiosqlite.connect(MEMORY_STORE_PATH, check_same_thread=False)
        return AsyncSqliteSaver(conn)

    async def get_tools(self, skip_exec_toolkit: bool=False):
        tools = nb_toolkit
        tools += jlab_toolkit

        # Bash tool conflicts with internal openclaw tools
        if not skip_exec_toolkit:
            tools += exec_toolkit

        # Add MCP tools
        mcp_settings = self.get_mcp_settings()
        connections: dict[str, Connection] = {}
        for mcp in mcp_settings.mcp_servers:
            if isinstance(mcp, McpServerHttp):
                connection: StreamableHttpConnection = {
                    "transport": mcp.type,
                    "url": mcp.url,
                    "headers": mcp.headers
                }
                connections[mcp.name] = connection
            elif isinstance(mcp, McpServerStdio):
                connection: StdioConnection = {
                    "transport": "stdio",
                    "command": mcp.command,
                    "args": mcp.args,
                    "env": {var.name: var.value for var in mcp.env}
                }
                connections[mcp.name] = connection
        client = MultiServerMCPClient(connections)
        tools += await client.get_tools()

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
        def is_true_flexible(val: str) -> bool:
            return val.strip().lower() in ("true", "1", "yes", "y", "t")

        if (model_id.startswith("ollama/") or model_id.startswith("ollama_chat/")) \
            and "num_ctx" not in model_args:
                model_args['num_ctx'] = DEFAULT_OLLAMA_NUM_CTX
        model = ChatLiteLLM(**model_args, model=model_id, streaming=True)

        from langgraph.checkpoint.memory import InMemorySaver
        use_persistence = is_true_flexible(model_args.get(
            'persistence', "false" if "/openclaw" in model_id or "/hermes" in model_id else "true"
        ))

        # openclaw will fail if you pass it a bash tool as this conflicts with the
        # internal openclaw tool.  It will also fail if persistence is true and it
        # pulls in a bash tool from a previous run

        skip_exec_toolkit = "/openclaw" in model_id
        memory_store = (await self.get_memory_store()) if use_persistence else InMemorySaver()

        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=await self.get_tools(skip_exec_toolkit),
            middleware=[self._create_tool_error_handler()]
        )

    async def process_message(self, message: Message) -> None:
        if not hasattr(self, "config_manager"):
            self.send_message(
                "Jupyternaut requires the `jupyter_ai_jupyternaut` server extension package.\n\n",
                "Please make sure to first install that package in your environment & restart the server.",
            )
            return
        if not self.config_manager.chat_model:
            self.send_message(
                "No chat model is configured.\n\n"
                "You must set one first in the Jupyter AI settings, found in 'Settings > AI Settings' from the menu bar."
            )
            return

        try:
            model_id = (message.metadata or {}).get(
                "model_id", self.config_manager.chat_model
            )
            model_args = self.config_manager.chat_model_args | \
                (message.metadata or {}).get(
                    "model_args", {}
                )
            system_prompt = self.get_system_prompt(
                model_id=model_id,
                message=message
            )
            self.log.debug("%s %s", model_id, model_args)
            agent = await self.get_agent(
                model_id=model_id,
                model_args=model_args,
                system_prompt=system_prompt
            )
            context = {
                "thread_id": self.ychat.get_id(),
                "username": message.sender
            }

            async def create_aiter():
                stream = await agent.astream_events(
                    {"messages": [{"role": "user", "content": message.body}]},
                    {"configurable": context},
                    version="v3"
                )
                async for event in stream:
                    if event["method"] != "messages":
                        continue
                    data = event["params"]["data"][0]
                    if not isinstance(data, dict):
                        continue
                    if data.get("event") != "content-block-delta":
                        continue

                    block = data.get("delta") or {}
                    if block.get("type") == "text-delta":
                        yield block.get("text", "")
                    elif block.get("type") == "reasoning-delta":
                        yield block.get('reasoning', '')

            response_aiter = create_aiter()
            await self.stream_message(response_aiter)
        except Exception as e:
            self.log.exception("Error while processing message.")
            self.send_message(f"Error: {e}")

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

    async def shutdown(self):
        # Only the SQLite-backed checkpointer holds an open connection that needs
        # to be closed; the in-memory fallback does not.
        memory_store = getattr(self, "_memory_store", None)
        conn = getattr(memory_store, "conn", None)
        if conn is not None:
            self.parent.event_loop.create_task(conn.close())
        await super().shutdown()
