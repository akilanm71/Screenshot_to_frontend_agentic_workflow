
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.workflow.state import FrontendWorkflowState

from backend.workflow.nodes import (
    classify_input_node,
    vision_analysis_node,
    pdf_extraction_node,
    planning_node,
    code_generation_node,
    build_validation_node,
    browser_execution_node,
    evaluation_node,
    refinement_node,
    finalize_node,
)

from backend.workflow.routing import workflow_error_node

from backend.workflow.edges import (
    input_router_edge,
    pdf_extraction_edge,
    vision_analysis_edge,
    agent_error_edge,
    build_validation_edge,
    browser_execution_edge,
    evaluation_edge,
    refinement_edge,
)


logger = logging.getLogger(__name__)


# =========================================================
# WORKFLOW NODE NAMES
# =========================================================

NODE_CLASSIFY_INPUT = "classify_input"
NODE_VISION_ANALYSIS = "vision_analysis"
NODE_PDF_EXTRACTION = "pdf_extraction"
NODE_PLANNING = "planning"
NODE_CODE_GENERATION = "code_generation"
NODE_BUILD_VALIDATION = "build_validation"
NODE_BROWSER_EXECUTION = "browser_execution"
NODE_EVALUATION = "evaluation"
NODE_REFINEMENT = "refinement"
NODE_FINALIZE = "finalize"
NODE_WORKFLOW_ERROR = "workflow_error"


# =========================================================
# GRAPH BUILDER
# =========================================================

def build_workflow_graph():
    """
    Construct the LangGraph workflow.

    Registers all agent nodes, configures conditional
    routing, and returns an uncompiled StateGraph.
    """

    logger.info("Initializing frontend automation workflow")

    workflow = StateGraph(FrontendWorkflowState)

    # -----------------------------------------------------
    # 1. REGISTER AGENT NODES
    # -----------------------------------------------------

    workflow.add_node(NODE_CLASSIFY_INPUT, classify_input_node)
    workflow.add_node(NODE_VISION_ANALYSIS, vision_analysis_node)
    workflow.add_node(NODE_PDF_EXTRACTION, pdf_extraction_node)
    workflow.add_node(NODE_PLANNING, planning_node)
    workflow.add_node(NODE_CODE_GENERATION, code_generation_node)
    workflow.add_node(NODE_BUILD_VALIDATION, build_validation_node)
    workflow.add_node(NODE_BROWSER_EXECUTION, browser_execution_node)
    workflow.add_node(NODE_EVALUATION, evaluation_node)
    workflow.add_node(NODE_REFINEMENT, refinement_node)
    workflow.add_node(NODE_FINALIZE, finalize_node)
    workflow.add_node(NODE_WORKFLOW_ERROR, workflow_error_node)

    # -----------------------------------------------------
    # 2. ENTRY POINT
    # -----------------------------------------------------

    workflow.add_edge(START, NODE_CLASSIFY_INPUT)

    # -----------------------------------------------------
    # 3. INPUT ROUTING
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_CLASSIFY_INPUT,
        input_router_edge,
        {
            "vision_analysis": NODE_VISION_ANALYSIS,
            "pdf_extraction": NODE_PDF_EXTRACTION,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 4. VISION ROUTING
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_VISION_ANALYSIS,
        vision_analysis_edge,
        {
            "planning": NODE_PLANNING,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 5. PDF ROUTING
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_PDF_EXTRACTION,
        pdf_extraction_edge,
        {
            "planning": NODE_PLANNING,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 6. PLANNING -> CODE GENERATION
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_PLANNING,
        agent_error_edge,
        {
            "continue": NODE_CODE_GENERATION,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 7. CODE GENERATION -> BUILD VALIDATION
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_CODE_GENERATION,
        agent_error_edge,
        {
            "continue": NODE_BUILD_VALIDATION,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 8. BUILD VALIDATION ROUTING
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_BUILD_VALIDATION,
        build_validation_edge,
        {
            "browser_execution": NODE_BROWSER_EXECUTION,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 9. BROWSER EXECUTION ROUTING
    # -----------------------------------------------------

    workflow.add_conditional_edges(
    NODE_BROWSER_EXECUTION,
    browser_execution_edge,
    {
        "evaluation": NODE_EVALUATION,
        "refinement": NODE_REFINEMENT,
        "workflow_error": NODE_WORKFLOW_ERROR,
    },
)

    # -----------------------------------------------------
    # 10. EVALUATION QUALITY GATE
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_EVALUATION,
        evaluation_edge,
        {
            "finalize": NODE_FINALIZE,
            "refinement": NODE_REFINEMENT,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 11. REFINEMENT LOOP
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        NODE_REFINEMENT,
        refinement_edge,
        {
            "build_validation": NODE_BUILD_VALIDATION,
            "workflow_error": NODE_WORKFLOW_ERROR,
        },
    )

    # -----------------------------------------------------
    # 12. TERMINAL EDGES
    # -----------------------------------------------------

    workflow.add_edge(NODE_FINALIZE, END)
    workflow.add_edge(NODE_WORKFLOW_ERROR, END)

    logger.info("Workflow graph constructed successfully")

    return workflow


# =========================================================
# COMPILE WORKFLOW WITH CHECKPOINT SUPPORT
# =========================================================

def compile_workflow_graph(checkpointer=None):
    """
    Build and compile the workflow with checkpoint support.

    Args:
        checkpointer:
            Optional LangGraph checkpointer.
            Defaults to MemorySaver.

    Returns:
        Compiled LangGraph application.

    Note:
        MemorySaver stores checkpoints in memory.
        Checkpoints are lost when the Python process exits.
    """

    logger.info("Compiling frontend automation workflow")

    workflow = build_workflow_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

        logger.info(
            "Using in-memory workflow checkpointing"
        )

    compiled_workflow = workflow.compile(
        checkpointer=checkpointer
    )

    logger.info(
        "Frontend automation workflow compiled successfully"
    )

    return compiled_workflow


# =========================================================
# PUBLIC GRAPH FACTORY
# =========================================================

def create_workflow(checkpointer=None):
    """
    Public factory used by workflow.py.

    Allows callers to provide a custom checkpointer.
    """

    return compile_workflow_graph(
        checkpointer=checkpointer
    )


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

    app = create_workflow()

    print("\nWorkflow graph compiled successfully!")
    print("Checkpoint support: Enabled")
    print("Checkpointer: MemorySaver")

    try:
        print("\nGraph structure:")
        print(app.get_graph().draw_ascii())

    except Exception as exc:
        print(
            "\nGraph compiled, but ASCII rendering "
            f"is unavailable: {exc}"
        )
