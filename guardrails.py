from typing import Any
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

from defines import NEMO_BLOCK_PHRASES

# ── Helper: convert LangChain messages → NeMo dict format ────────
def _to_nemo_messages(messages: list) -> list[dict]:
    """
    Convert LangChain BaseMessage list to NeMo's dict format.
    Skips ToolMessage / FunctionMessage — NeMo does not parse tool JSON.
    Only passes the last 10 messages to keep NeMo context tight
    and avoid context-dilution attacks.
    """
    result = []
    for m in messages[-10:]:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            continue
        content = m.content if isinstance(m.content, str) else ""
        if content.strip():
            result.append({"role": role, "content": content})
    return result

def _is_blocked(nemo_result) -> tuple[bool, str]:
    """Return (blocked, text) from a NeMo invoke result."""
    text = (
        nemo_result.get("output", "")
        if isinstance(nemo_result, dict)
        else str(nemo_result)
    )
    blocked = any(p.lower() in text.lower() for p in NEMO_BLOCK_PHRASES)
    return blocked, text


# ================================================================
# THE CORRECT MIDDLEWARE CLASS FOR LANGCHAIN 1.2.x
# ================================================================

class NemoGuardrailMiddleware(AgentMiddleware):
    """
    LangChain 1.2.x AgentMiddleware that applies NeMo Guardrails.

    INPUT rails  → before_model (sync, reads state["messages"])
    OUTPUT rails → wrap_model_call (sync) + awrap_model_call (async)

    Provides both sync and async implementations to support both
    invoke()/stream() and ainvoke()/astream() agent execution patterns.

    Constructor arguments
    ---------------------
    nemo_rails : RunnableRails
        Shared instance, created once at startup in build_enterprise_graph().
    slog : Any
        Structured logger instance (e.g., structlog.get_logger()).
    agent_name : str
        Used for structured log fields only. No functional effect.
    fail_closed : bool
        True  → if NeMo itself errors, BLOCK the request (safe default for prod)
        False → if NeMo itself errors, LOG and CONTINUE (fail-open for dev)
    """

    def __init__(
        self,
        nemo_rails: RunnableRails,
        slog: Any,
        agent_name: str = "agent",
        fail_closed: bool = True,        
    ):
        self.nemo_rails  = nemo_rails
        self.agent_name  = agent_name
        self.fail_closed = fail_closed
        self.slog        = slog

    # ── INPUT RAIL ────────────────────────────────────────────────
    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """
        SYNC hook — runs before the LLM call.

        Reads state["messages"], runs NeMo INPUT rails.
        If blocked: returns {"jump_to": "end"} to halt the agent
        and appends a safe AIMessage so the caller gets a response.

        If NeMo itself errors:
          fail_closed=True  → treat as blocked (safe for production)
          fail_closed=False → log warning and continue (dev/testing)
        """
        messages = state.get("messages", [])
        nemo_msgs = _to_nemo_messages(messages)
        if not nemo_msgs:
            return None  # no user message yet, nothing to check

        try:
            # NeMo expects {"input": "text"} format, not messages array
            # For input rails, we pass the last user message
            last_user_msg = nemo_msgs[-1]["content"] if nemo_msgs else ""
            result = self.nemo_rails.invoke({"input": last_user_msg})
            blocked, text = _is_blocked(result)
            if blocked:
                self.slog.warning(
                    "nemo.input_rail_blocked",
                    agent=self.agent_name,
                    phrase=text[:100],
                )
                # Append the block message so the graph has a response to return.
                # jump_to="end" stops the agent from calling the LLM.
                return {
                    "messages": messages + [AIMessage(content=text)],
                    "jump_to":  "end",
                }
        except Exception as e:
            self.slog.warning(
                "nemo.input_rail_error",
                agent=self.agent_name,
                error=str(e),
            )
            if self.fail_closed:
                block_msg = (
                    "Request could not be processed at this time. "
                    "Please try again or contact support."
                )
                return {
                    "messages": messages + [AIMessage(content=block_msg)],
                    "jump_to":  "end",
                }

        return None  # no change — continue to LLM

    # ── OUTPUT RAIL (SYNC) ───────────────────────────────────────────
    def wrap_model_call(self, request: ModelRequest, handler):
        """
        SYNC hook — wraps the actual LLM call for synchronous execution.

        Calls the LLM via handler(request), then runs NeMo OUTPUT
        rails on the response. If an output rail fires, replaces the
        response with a safe message.

        This is the correct hook for output rails because after_model
        (sync) can READ the response but cannot REPLACE it.
        wrap_model_call can return a different AIMessage entirely.

        Returns:
            ModelResponse or AIMessage: The (possibly modified) response.
        """
        # Call the actual LLM
        response = handler(request)

        # Extract text from response for NeMo to evaluate
        try:
            # Handle ModelResponse or AIMessage return types
            if hasattr(response, "result"):
                # ModelResponse - extract first message
                resp_msg = response.result[0] if response.result else None
            else:
                # Direct AIMessage
                resp_msg = response

            if resp_msg is None:
                return response

            resp_content = getattr(resp_msg, "content", None)

            if isinstance(resp_content, list):
                # Handles multimodal / tool_use / thinking blocks
                resp_content = " ".join(
                    b.get("text", "")
                    if isinstance(b, dict) and b.get("type") == "text"
                    else (b if isinstance(b, str) else "")
                    for b in resp_content
                )

            if isinstance(resp_content, str) and resp_content.strip():
                # NeMo expects {"input": "text"} format for output rails
                check = self.nemo_rails.invoke({"input": resp_content})
                blocked, text = _is_blocked(check)
                if blocked:
                    self.slog.warning(
                        "nemo.output_rail_blocked",
                        agent=self.agent_name,
                    )
                    # Return AIMessage directly - will be auto-converted
                    return AIMessage(
                        content=(
                            "Response withheld — output safety rail triggered. "
                            "Please rephrase your request or contact your administrator."
                        )
                    )
        except Exception as e:
            self.slog.warning(
                "nemo.output_rail_error",
                agent=self.agent_name,
                error=str(e),
            )
            if self.fail_closed:
                return AIMessage(
                    content=(
                        "Response could not be verified at this time. "
                        "Please try again."
                    )
                )

        return response

    # ── OUTPUT RAIL (ASYNC) ───────────────────────────────────────────
    async def awrap_model_call(self, request: ModelRequest, handler):
        """
        ASYNC hook — wraps the actual LLM call for async execution.

        Same logic as wrap_model_call but for async contexts (astream, ainvoke).
        """
        # Call the actual LLM
        response = await handler(request)

        # Extract text from response for NeMo to evaluate
        try:
            # Handle ModelResponse or AIMessage return types
            if hasattr(response, "result"):
                # ModelResponse - extract first message
                resp_msg = response.result[0] if response.result else None
            else:
                # Direct AIMessage
                resp_msg = response

            if resp_msg is None:
                return response

            resp_content = getattr(resp_msg, "content", None)

            if isinstance(resp_content, list):
                # Handles multimodal / tool_use / thinking blocks
                resp_content = " ".join(
                    b.get("text", "")
                    if isinstance(b, dict) and b.get("type") == "text"
                    else (b if isinstance(b, str) else "")
                    for b in resp_content
                )

            if isinstance(resp_content, str) and resp_content.strip():
                # NeMo expects {"input": "text"} format for output rails
                check = self.nemo_rails.invoke({"input": resp_content})
                blocked, text = _is_blocked(check)
                if blocked:
                    self.slog.warning(
                        "nemo.output_rail_blocked",
                        agent=self.agent_name,
                    )
                    # Return AIMessage directly - will be auto-converted
                    return AIMessage(
                        content=(
                            "Response withheld — output safety rail triggered. "
                            "Please rephrase your request or contact your administrator."
                        )
                    )
        except Exception as e:
            self.slog.warning(
                "nemo.output_rail_error",
                agent=self.agent_name,
                error=str(e),
            )
            if self.fail_closed:
                return AIMessage(
                    content=(
                        "Response could not be verified at this time. "
                        "Please try again."
                    )
                )

        return response


# ================================================================
# FACTORY — build one middleware instance per agent
# ================================================================

def build_nemo_middleware(
    nemo_rails: RunnableRails,
    agent_name: str,
    slog: Any,
    fail_closed: bool = True,
) -> NemoGuardrailMiddleware:
    """
    Returns a NemoGuardrailMiddleware for a single agent.

    Create one instance per agent (not shared across agents)
    so each agent produces distinct log traces in LangSmith.
    The underlying nemo_rails RunnableRails IS shared (one config).

    Usage in build_specialist_agents():
        middleware = build_nemo_middleware(nemo_guardrails, "kb_agent")
        agent = create_agent(..., middleware=[middleware])
    """
    return NemoGuardrailMiddleware(
        nemo_rails=nemo_rails,
        agent_name=agent_name,
        fail_closed=fail_closed,
        slog=slog
    )
