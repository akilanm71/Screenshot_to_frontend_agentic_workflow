
import os
import time
import logging
from typing import Any

from langchain_ollama import ChatOllama

from backend.observability.langsmith_config import configure_langsmith
from backend.observability.metrics_collector import metrics_collector


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# LLM FACTORY
# ---------------------------------------------------------

def create_llm(
    agent_name: str,
    temperature: float = 0.1,
    num_ctx: int = 16384,
    num_predict: int = 8192,
    repeat_penalty: float = 1.1,
    repeat_last_n: int = 64,
    model: str | None = None,
) -> ChatOllama:
    def create_llm(
    agent_name: str,
    temperature: float = 0.1,
    num_ctx: int = 16384,
    num_predict: int = 8192,
    repeat_penalty: float = 1.1,
    repeat_last_n: int = 64,
    model: str | None = None,
) -> ChatOllama:
     """
    Create and configure an Ollama chat model.

    Args:
        agent_name: Name of the agent using the model.
        temperature: Model response randomness.
        num_ctx: Context window size.
        num_predict: Maximum generated tokens.
        repeat_penalty: Penalty applied to repeated tokens.
        repeat_last_n: Number of recent tokens considered for repetition.
        model: Optional model override.

    Returns:
        Configured ChatOllama instance.
    """

    configure_langsmith()

    selected_model = (
        model
        or os.getenv("OLLAMA_MODEL")
        or "qwen2.5vl:3b"
    )

    base_url = (
        os.getenv("OLLAMA_BASE_URL")
        or "http://localhost:11434"
    )

    logger.info(
        "Initializing LLM | Agent: %s | Model: %s | "
        "Context: %s | Max output: %s",
        agent_name,
        selected_model,
        num_ctx,
        num_predict,
    )

    return ChatOllama(
        model=selected_model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
    )


# ---------------------------------------------------------
# METRICS HELPER
# ---------------------------------------------------------

def _record_llm_metrics(
    agent_name: str,
    response: Any,
    workflow_id: str,
    duration_ms: float
) -> None:
    """
    Record LLM usage metrics without interrupting execution.
    """

    try:
        usage = getattr(response, "usage_metadata", None) or {}
        metadata = getattr(response, "response_metadata", None) or {}

        model_name = metadata.get("model") or metadata.get("model_name") or "unknown"

        metrics_collector.record_llm_metrics(
            agent_name=agent_name,
            model_name=model_name,
            usage=usage,
            metadata=metadata,
            workflow_id=workflow_id
        )

        logger.info(
            "LLM metrics recorded | Agent: %s | Duration: %.2f ms",
            agent_name,
            duration_ms
        )

    except Exception:
        logger.exception(
            "LLM metrics recording failed for agent: %s",
            agent_name
        )


def _record_agent_failure(
    agent_name: str,
    workflow_id: str,
    duration_ms: float,
    error: Exception
) -> None:
    """
    Record agent failure without masking the original exception.
    """

    try:
        metrics_collector.record_agent_metrics(
            agent_name=agent_name,
            duration_ms=duration_ms,
            status="failed",
            workflow_id=workflow_id,
            error=str(error)
        )

    except Exception:
        logger.exception(
            "Agent failure metrics recording failed for: %s",
            agent_name
        )


# ---------------------------------------------------------
# LLM INVOCATION
# ---------------------------------------------------------

def invoke_llm(
    llm: ChatOllama,
    prompt: Any,
    agent_name: str,
    workflow_id: str = "local_test"
):
    """
    Invoke the configured LLM with LangSmith tracing.

    Args:
        llm: ChatOllama instance.
        prompt: String, message list, or supported LangChain prompt.
        agent_name: Name of the calling agent.
        workflow_id: Workflow execution identifier.

    Returns:
        LangChain AIMessage response.
    """

    start_time = time.perf_counter()

    try:
        logger.info(
            "LLM invocation started | Agent: %s | Workflow: %s",
            agent_name,
            workflow_id
        )

        response = llm.invoke(
            prompt,
            config={
                "run_name": f"{agent_name}_llm_call",
                "tags": [
                    "screenshot-to-frontend",
                    agent_name
                ],
                "metadata": {
                    "agent_name": agent_name,
                    "workflow_id": workflow_id
                }
            }
        )

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        _record_llm_metrics(
            agent_name=agent_name,
            response=response,
            workflow_id=workflow_id,
            duration_ms=duration_ms
        )

        logger.info(
            "LLM invocation completed | Agent: %s | Duration: %.2f ms",
            agent_name,
            duration_ms
        )

        return response

    except Exception as exc:

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "LLM invocation failed | Agent: %s | Duration: %.2f ms",
            agent_name,
            duration_ms
        )

        _record_agent_failure(
            agent_name=agent_name,
            workflow_id=workflow_id,
            duration_ms=duration_ms,
            error=exc
        )

        raise
