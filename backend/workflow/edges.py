
from typing import Literal

from backend.workflow.state import FrontendWorkflowState

from backend.workflow.routing import (
    route_input,
    route_after_pdf_extraction,
    route_after_vision,
    route_after_build,
    route_after_browser,
    route_after_evaluation,
    route_after_refinement,
    route_after_node,
)


# =========================================================
# TYPE DEFINITIONS
# =========================================================

InputRoute = Literal[
    "vision_analysis",
    "pdf_extraction",
    "workflow_error",
]

ProcessingRoute = Literal[
    "planning",
    "workflow_error",
]

BuildRoute = Literal[
    "browser_execution",
    "workflow_error",
]

BrowserRoute = Literal[
    "evaluation",
    "refinement",
    "workflow_error",
]

EvaluationRoute = Literal[
    "finalize",
    "refinement",
    "workflow_error",
]

RefinementRoute = Literal[
    "build_validation",
    "workflow_error",
]

GenericRoute = Literal[
    "continue",
    "workflow_error",
]


# =========================================================
# 1. INPUT CONDITIONAL EDGE
# =========================================================

def input_router_edge(
    state: FrontendWorkflowState,
) -> InputRoute:
    """Route classified input to the correct processing agent."""
    return route_input(state)


# =========================================================
# 2. PDF EXTRACTION CONDITIONAL EDGE
# =========================================================

def pdf_extraction_edge(
    state: FrontendWorkflowState,
) -> ProcessingRoute:
    """Route PDF extraction result to planning or error handling."""
    return route_after_pdf_extraction(state)


# =========================================================
# 3. VISION ANALYSIS CONDITIONAL EDGE
# =========================================================

def vision_analysis_edge(
    state: FrontendWorkflowState,
) -> ProcessingRoute:
    """Route vision analysis result to planning or error handling."""
    return route_after_vision(state)


# =========================================================
# 4. GENERIC AGENT ERROR EDGE
# =========================================================

def agent_error_edge(
    state: FrontendWorkflowState,
) -> GenericRoute:
    """Route generic agent execution results."""
    return route_after_node(state)


# =========================================================
# 5. BUILD VALIDATION CONDITIONAL EDGE
# =========================================================

def build_validation_edge(
    state: FrontendWorkflowState,
) -> BuildRoute:
    """Route successful builds to browser execution."""
    return route_after_build(state)


# =========================================================
# 6. BROWSER EXECUTION CONDITIONAL EDGE
# =========================================================

def browser_execution_edge(
    state: FrontendWorkflowState,
) -> BrowserRoute:
    """
    Route browser execution results.

    Browser success -> evaluation
    Browser failure -> refinement
    Retry exhausted -> workflow_error
    """
    return route_after_browser(state)


# =========================================================
# 7. EVALUATION CONDITIONAL EDGE
# =========================================================

def evaluation_edge(
    state: FrontendWorkflowState,
) -> EvaluationRoute:
    """Apply the evaluation quality gate."""
    return route_after_evaluation(state)


# =========================================================
# 8. REFINEMENT CONDITIONAL EDGE
# =========================================================

def refinement_edge(
    state: FrontendWorkflowState,
) -> RefinementRoute:
    """Route successful refinement back to build validation."""
    return route_after_refinement(state)


# =========================================================
# 9. PUBLIC EDGE REGISTRY
# =========================================================

EDGE_REGISTRY = {
    "input_router": input_router_edge,
    "pdf_extraction": pdf_extraction_edge,
    "vision_analysis": vision_analysis_edge,
    "agent_error": agent_error_edge,
    "build_validation": build_validation_edge,
    "browser_execution": browser_execution_edge,
    "evaluation": evaluation_edge,
    "refinement": refinement_edge,
}


# =========================================================
# 10. EDGE LOOKUP
# =========================================================

def get_edge(edge_name: str):
    """Retrieve an edge function by its registered name."""

    if edge_name not in EDGE_REGISTRY:
        raise ValueError(
            f"Unknown workflow edge: {edge_name}. "
            f"Available edges: {list(EDGE_REGISTRY.keys())}"
        )

    return EDGE_REGISTRY[edge_name]
