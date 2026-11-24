
import logging
from typing import Any, Dict, Iterator, Optional

from backend.workflow.graph import compile_workflow_graph
from backend.workflow.state import FrontendWorkflowState


logger = logging.getLogger(__name__)


# =========================================================
# GLOBAL WORKFLOW INSTANCE
# =========================================================

_compiled_workflow = None


# =========================================================
# GET COMPILED WORKFLOW
# =========================================================

def get_workflow():
    """
    Return the compiled LangGraph application.

    The graph is compiled only once and reused.
    Its MemorySaver checkpointer remains available
    while this Python process is running.
    """

    global _compiled_workflow

    if _compiled_workflow is None:

        logger.info("Creating compiled workflow instance")

        _compiled_workflow = compile_workflow_graph()

        logger.info("Compiled workflow instance created")

    return _compiled_workflow


# =========================================================
# BUILD WORKFLOW CONFIGURATION
# =========================================================

def build_workflow_config(
    workflow_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build LangGraph runtime configuration.

    Args:
        workflow_id: Unique workflow execution identifier.
        config: Optional additional LangGraph configuration.

    Returns:
        Configuration containing thread_id and recursion_limit.
    """

    if not workflow_id:
        raise ValueError("workflow_id is required")

    runtime_config = dict(config or {})

    configurable = dict(
        runtime_config.get("configurable", {})
    )

    # Associate checkpoints with this workflow.
    configurable["thread_id"] = workflow_id

    runtime_config["configurable"] = configurable

    # Protect against excessive graph recursion.
    runtime_config.setdefault("recursion_limit", 50)

    return runtime_config


# =========================================================
# VALIDATE INITIAL WORKFLOW STATE
# =========================================================

def validate_initial_state(initial_state: dict) -> None:
    """
    Validate the minimum required workflow input.
    """

    if not isinstance(initial_state, dict):
        raise TypeError("initial_state must be a dictionary")

    workflow_id = initial_state.get("workflow_id")

    if not workflow_id:
        raise ValueError(
            "initial_state must contain workflow_id"
        )

    input_path = initial_state.get("input_path")

    if not input_path:
        raise ValueError(
            "initial_state must contain input_path"
        )


# =========================================================
# INVOKE WORKFLOW
# =========================================================

def invoke_workflow(
    initial_state: FrontendWorkflowState,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute the complete workflow synchronously.

    Returns the final workflow state.
    """

    validate_initial_state(initial_state)

    app = get_workflow()

    workflow_id = initial_state["workflow_id"]

    runtime_config = build_workflow_config(
        workflow_id,
        config,
    )

    logger.info(
        "Starting workflow execution | workflow_id=%s",
        workflow_id,
    )

    result = app.invoke(
        initial_state,
        config=runtime_config,
    )

    logger.info(
        "Workflow execution completed | workflow_id=%s",
        workflow_id,
    )

    return result


# =========================================================
# STREAM WORKFLOW
# =========================================================

def stream_workflow(
    initial_state: FrontendWorkflowState,
    config: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Execute workflow and stream node-by-node updates.

    Each yielded item contains:
        {
            "node": node_name,
            "output": node_output
        }
    """

    validate_initial_state(initial_state)

    app = get_workflow()

    workflow_id = initial_state["workflow_id"]

    runtime_config = build_workflow_config(
        workflow_id,
        config,
    )

    logger.info(
        "Starting streamed workflow | workflow_id=%s",
        workflow_id,
    )

    for event in app.stream(
        initial_state,
        config=runtime_config,
        stream_mode="updates",
    ):

        for node_name, node_output in event.items():

            yield {
                "node": node_name,
                "output": node_output,
            }

    logger.info(
        "Streamed workflow completed | workflow_id=%s",
        workflow_id,
    )


# =========================================================
# GET CHECKPOINTED WORKFLOW STATE
# =========================================================

def get_workflow_state(workflow_id: str):
    """
    Retrieve the latest checkpointed state for a workflow.

    The workflow must have been executed using the same
    compiled app instance and thread_id.
    """

    if not workflow_id:
        raise ValueError("workflow_id is required")

    app = get_workflow()

    config = {
        "configurable": {
            "thread_id": workflow_id,
        }
    }

    return app.get_state(config)


# =========================================================
# COMPATIBILITY ALIASES
# =========================================================

def run_workflow(
    initial_state: FrontendWorkflowState,
    config: Optional[Dict[str, Any]] = None,
):
    """
    Backward-compatible alias for invoke_workflow().
    """

    return invoke_workflow(
        initial_state,
        config=config,
    )


def create_workflow():
    """
    Return the cached compiled workflow.
    """

    return get_workflow()


# =========================================================
# MODULE TEST
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    print("\nInitializing frontend automation workflow...")

    app = get_workflow()

    print("\nWorkflow compiled successfully!")
    print("Checkpointing: Enabled")
    print("Checkpointer: MemorySaver")

    try:
        print("\nWorkflow graph structure:")
        print(app.get_graph().draw_ascii())

    except Exception as exc:
        print(
            "\nGraph compiled, but ASCII rendering "
            f"is unavailable: {exc}"
        )
