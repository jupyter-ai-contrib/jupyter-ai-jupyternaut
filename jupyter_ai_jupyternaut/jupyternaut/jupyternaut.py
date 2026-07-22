import os
from typing import Any

from jupyter_ai_persona_manager import (
  BasePersona,
  McpServerHttp,
  McpServerStdio,
  ModelConfiguration,
  ModelOption,
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

# Sentinel model ID meaning "use Jupyternaut's configured default model" (the
# `model_provider_id` config value, seeded from the `initial_language_model`
# traitlet). The picker itself has no explicit default option — its built-in
# "Default" row sends a `None` selection — so this is normally only reached via
# that `None`. It is still recognized here so an explicit selection of it (e.g.
# a persona that sets a non-None current) also resolves to the configured
# default rather than being treated as a literal LiteLLM model ID.
DEFAULT_MODEL_ID = "jupyternaut/default"


def _litellm_chat_models() -> list[str]:
    """
    Return the sorted list of chat model IDs known to LiteLLM (the same IDs
    accepted as ``model=`` by ``litellm.completion``), e.g. ``openai/gpt-4o``,
    ``anthropic/claude-3-5-haiku-latest``, ``ollama/llama3``.

    LiteLLM's ``model_cost`` map catalogs every model it knows about along with
    its ``mode``; we keep only the chat models. This is a large list (well over a
    thousand entries), which is intentional — the model picker exposes the full
    range of models LiteLLM can talk to, not just a curated subset.
    """
    try:
        import litellm
    except ImportError:
        return []

    models = {
        model_id
        for model_id, spec in litellm.model_cost.items()
        if isinstance(spec, dict) and spec.get("mode") == "chat"
        # `sample_spec` is a documentation placeholder, not a real model.
        and model_id != "sample_spec"
    }
    return sorted(models)

MEMORY_STORE_PATH = os.path.join(jupyter_data_dir(), "jupyter_ai", "memory.sqlite")

JUPYTERNAUT_AVATAR_PATH = str(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../static", "jupyternaut.svg")
    )
)

def _extract_message_text(message: Any) -> str:
    """
    Extract the assistant text from a message object yielded by the agent's
    streaming events (an ``AIMessage``/``AIMessageChunk``). Handles both a plain
    string ``content`` and the structured content-block list form (a list of
    dicts with ``type == "text"``), ignoring tool-call and reasoning blocks.

    Returns an empty string for anything without extractable text.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


class JupyternautPersona(BasePersona):
    """
    The Jupyternaut persona, the main persona provided by Jupyter AI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Advertise this persona's model options (custom models + the built-in
        # default) over the awareness channel as soon as the persona is created,
        # so the model picker is populated without a REST round-trip. Guarded
        # because the config manager is only bound when the server extension is
        # installed (see `process_message`).
        if hasattr(self, "config_manager"):
            self.sync_model_configuration()

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Jupyternaut",
            avatar_path=JUPYTERNAUT_AVATAR_PATH,
            description="The standard agent provided by JupyterLab. Currently has no tools.",
            system_prompt="...",
        )

    ################################################
    # persona-manager model API (issue #61)
    ################################################
    # Jupyternaut is a LiteLLM/LangChain persona, not an ACP agent, so it has no
    # long-lived backend session to reconfigure. Instead, the selected model is
    # resolved fresh for each message in `process_message` (see `_resolve_model`).
    # The model picker's options are the user's custom models (defined in the
    # settings view) plus a single built-in "default" entry.
    def _build_model_configuration(self) -> ModelConfiguration:
        """
        Build the `ModelConfiguration` advertised over awareness. The picker
        shows, in order:

        1. The user's custom models (defined in the settings view), so they sort
           to the top.
        2. Every chat model LiteLLM knows about, so the user can select any
           supported model directly without first defining a custom model.

        Custom-model IDs are excluded from the LiteLLM list (they are not LiteLLM
        model IDs), but a custom model's underlying LiteLLM model may still also
        appear in the list — that's fine, selecting either works.

        There is no explicit "default" option: the picker always renders a
        built-in "Default" row (selection = the persona's current value, which
        Jupyternaut leaves as `None`), and a `None` selection resolves to the
        configured default model at message time (see `_resolve_model`). Adding
        our own default option would just duplicate that row.

        Jupyternaut advertises no per-message model settings — custom-model
        parameters are defined in the settings view and stored in config, not
        exposed as picker controls — so `settings` is left empty.
        """
        custom_models = self.config_manager.custom_models
        options = [
            ModelOption(id=cm.id, name=cm.name, description=cm.description)
            for cm in custom_models
        ]
        # The full LiteLLM chat model catalog, so any supported model is
        # directly selectable.
        options.extend(
            ModelOption(id=model_id, name=model_id)
            for model_id in _litellm_chat_models()
        )
        # `current=None` means "use the persona's default", which resolves to
        # the configured default model at message time.
        return ModelConfiguration(current=None, options=options, settings=[])

    def sync_model_configuration(self) -> None:
        """
        Rebuild and re-publish the model configuration over awareness. Called on
        persona creation and whenever the config changes (e.g. the user adds,
        edits, or reorders custom models in the settings view), so the picker
        reflects the latest custom models live.
        """
        self.report_model_configuration(self._build_model_configuration())

    async def update_model(self, model_id: str) -> None:
        """
        Apply a user's model selection. Jupyternaut resolves the concrete
        LiteLLM model per message (its backend is stateless — a fresh agent is
        built for each message in `get_agent`), so there is nothing to switch
        eagerly here: `BasePersona.apply_model_spec` records the new current
        model on awareness and `process_message` resolves it via
        `_resolve_model`. An unknown ID (e.g. a stale custom-model ID) is
        tolerated and falls back to the default at resolution time.
        """

    async def update_model_settings(self, settings: dict[str, str | None]) -> None:
        """
        No-op: Jupyternaut advertises no per-message model settings. Custom-model
        parameters are edited in the settings view and stored in config, not
        selected per message, so this is never called with a non-empty mapping.
        """

    async def update_settings(self, settings: dict[str, str | None]) -> None:
        """
        No-op: Jupyternaut advertises no general settings (mode, effort, etc.).
        """

    def _resolve_model(self) -> tuple[str | None, dict[str, Any]]:
        """
        Resolve the persona's current model selection (from awareness, set by
        the user's per-message selection) into a concrete LiteLLM model ID and
        its parameters:

        - A custom-model ID resolves to that custom model's LiteLLM model ID and
          saved parameters (looked up from config by ID — the parameters never
          travel in message metadata).
        - The built-in default entry (`DEFAULT_MODEL_ID`) or `None` resolves to
          the configured default model (`model_provider_id`, seeded from the
          `initial_language_model` traitlet) and its saved `fields` parameters.
        - A stale custom-model ID (has the ``custom-`` prefix but isn't in
          config — e.g. the custom model was deleted) falls back to the
          configured default.
        - Any other ID is a LiteLLM model ID selected directly from the picker;
          it is used as-is, with any parameters saved for it under config
          `fields` (matching how the configured default's params are looked up).

        Returns `(None, {})` when no model can be resolved (the default was
        selected but none is configured), which `process_message` surfaces as a
        prompt to configure a model.
        """
        selected = self.get_model()
        if selected and selected != DEFAULT_MODEL_ID:
            custom_model = self.config_manager.get_custom_model(selected)
            if custom_model is not None:
                return custom_model.model_id, dict(custom_model.params)
            # A stale custom-model ID (deleted model) falls back to the default;
            # any other ID is a LiteLLM model picked directly, used as-is with
            # any parameters saved for it in config.
            if not selected.startswith("custom-"):
                params = self._read_config_fields().get(selected, {})
                return selected, dict(params)

        default_model_id = self.config_manager.chat_model
        return default_model_id, dict(self.config_manager.chat_model_args)

    def _read_config_fields(self) -> dict[str, dict[str, Any]]:
        """Return the per-model parameter dictionaries from config (`fields`)."""
        return self.config_manager.get_config().fields

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

    async def get_tools(self):
        tools = []
        tools += nb_toolkit
        tools += jlab_toolkit
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
        memory_store = await self.get_memory_store()
        from langgraph.checkpoint.memory import InMemorySaver
        memory_store = (await self.get_memory_store()) \
            if is_true_flexible(model_args.get('persistence', "true")) \
            else InMemorySaver()

        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=await self.get_tools(),
            middleware=[self._create_tool_error_handler()],
        )

    async def process_message(self, message: Message) -> None:
        if not hasattr(self, "config_manager"):
            self.send_message(
                "Jupyternaut requires the `jupyter_ai_jupyternaut` server extension package.\n\n",
                "Please make sure to first install that package in your environment & restart the server.",
            )
            return
        # Resolve the user's per-message model selection (applied to awareness by
        # `BasePersona.apply_model_spec` before this runs) into a concrete
        # LiteLLM model ID + params. A custom model resolves to its saved LiteLLM
        # ID/params; the built-in default resolves to the configured default.
        model_id, model_args = self._resolve_model()
        model_id = (
            message.metadata or {}
        ).get(
            "model_id", model_id
        )
        model_args = model_args | (
            message.metadata or {}
        ).get(
            "model_args", {}
        )

        if not model_id:
            self.send_message(
                "No chat model is configured.\n\n"
                "Select a custom model from the model picker, or open the "
                "Jupyternaut settings to define one. A default model can also be "
                "configured by the server administrator."
            )
            return

        try:
            system_prompt = self.get_system_prompt(model_id=model_id, message=message)
            agent = await self.get_agent(
                model_id=model_id, model_args=model_args, system_prompt=system_prompt
            )

            context = {
                "thread_id": self.ychat.get_id(),
                "username": message.sender
            }

            async def create_aiter():
                # Tracks the assistant text already yielded, so we can emit only
                # the new suffix from each `messages` event. Some
                # langchain/langgraph versions stream the assistant reply as a
                # growing snapshot (a full AIMessage per event) rather than
                # per-token content-block deltas; prefix-diffing turns either
                # shape into deltas and prevents re-emitting text.
                emitted = ""
                stream = await agent.astream_events(
                    {"messages": [{"role": "user", "content": message.body}]},
                    {"configurable": context},
                    version="v3"
                )
                async for event in stream:
                    if event["method"] != "messages":
                        continue
                    data = event["params"]["data"][0]

                    # Preferred shape: content-block-delta protocol dicts, which
                    # already carry per-token deltas.
                    if isinstance(data, dict):
                        if data.get("event") != "content-block-delta":
                            continue
                        block = data.get("delta") or {}
                        if block.get("type") == "text-delta":
                            yield block.get("text", "")
                        elif block.get("type") == "reasoning-delta":
                            yield block.get('reasoning', '')
                        continue

                    # Fallback shape: the `messages` event carries a message
                    # object (an `AIMessage`/`AIMessageChunk`) instead of a delta
                    # dict. Yield only the portion of its text not already
                    # emitted so the reply still streams to the chat.
                    text = _extract_message_text(data)
                    if text and text.startswith(emitted):
                        yield text[len(emitted):]
                        emitted = text
                    elif text:
                        # Not an extension of what we've emitted (e.g. a genuine
                        # per-chunk delta rather than a snapshot): emit as-is.
                        yield text
                        emitted += text

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
