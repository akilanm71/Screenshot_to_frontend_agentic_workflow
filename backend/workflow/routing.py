
from typing import Any

from backend.workflow.state import FrontendWorkflowState


# =========================================================
# WORKFLOW CONFIGURATION
# =========================================================

DEFAULT_QUALITY_THRESHOLD = 85.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_REFINEMENT_ITERATIONS = 3


# =========================================================
# COMMON UTILITIES
# =========================================================

def _as_dict(value: Any) -> dict[str, Any]:
    """Safely normalize a result into a dictionary."""

    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def _has_errors(state: FrontendWorkflowState) -> bool:
    """Check whether the workflow has fatal errors."""

    return (
        bool(state.get("errors"))
        or state.get("status") == "failed"
    )


def _is_success(result: Any) -> bool:
    """Recognize common successful agent result formats."""

    data = _as_dict(result)

    if not data:
        return False

    status = str(
        data.get("status", "")
    ).lower().strip()

    if status in {
        "success",
        "completed",
        "passed",
        "ok",
    }:
        return True

    if status in {
        "failed",
        "error",
        "failure",
    }:
        return False

    if data.get("success") is True:
        return True

    if not status and data.get("success") is not False:
        return True

    return False


def _get_refinement_count(
    state: FrontendWorkflowState,
) -> int:
    """Safely retrieve refinement count."""

    try:
        return max(
            0,
            int(state.get("refinement_count", 0)),
        )
    except (TypeError, ValueError):
        return 0


def _get_max_refinement_iterations(
    state: FrontendWorkflowState,
) -> int:
    """Safely retrieve maximum refinement iterations."""

    try:
        return max(
            0,
            int(
                state.get(
                    "max_refinement_iterations",
                    DEFAULT_MAX_REFINEMENT_ITERATIONS,
                )
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_MAX_REFINEMENT_ITERATIONS


def _get_quality_score(
    state: FrontendWorkflowState,
) -> float | None:
    """Get evaluator score in either 0-1 or 0-100 format."""

    evaluation = _as_dict(
        state.get("evaluation_result")
    )

    raw_score = state.get("quality_score")

    if raw_score is None:
        raw_score = evaluation.get("quality_score")

    if raw_score is None:
        raw_score = evaluation.get("overall_score")

    if raw_score is None:
        raw_score = evaluation.get("score")

    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None

    if 0.0 <= score <= 1.0:
        score *= 100.0

    return score


# =========================================================
# 1. INPUT ROUTING
# =========================================================

def route_input(
    state: FrontendWorkflowState,
) -> str:
    """
    Image / screenshot -> vision_analysis
    PDF                -> pdf_extraction
    Unsupported       -> workflow_error
    """

    if _has_errors(state):
        return "workflow_error"

    classification = _as_dict(
        state.get("classification_result")
    )

    input_type = (
        state.get("input_type")
        or classification.get("input_format")
        or classification.get("input_type")
        or classification.get("file_type")
        or ""
    )

    normalized_type = str(
        input_type
    ).lower().strip()

    if normalized_type.startswith("."):
        normalized_type = normalized_type[1:]

    if normalized_type in {
        "image",
        "screenshot",
        "png",
        "jpg",
        "jpeg",
        "webp",
        "bmp",
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        return "vision_analysis"

    if normalized_type in {
        "pdf",
        "application/pdf",
    }:
        return "pdf_extraction"

    return "workflow_error"


# =========================================================
# 2. PDF ROUTING
# =========================================================

def route_after_pdf_extraction(
    state: FrontendWorkflowState,
) -> str:

    if _has_errors(state):
        return "workflow_error"

    extraction_result = state.get(
        "pdf_extraction_result"
    )

    if not _is_success(extraction_result):
        return "workflow_error"

    return "planning"


# =========================================================
# 3. VISION ROUTING
# =========================================================

def route_after_vision(
    state: FrontendWorkflowState,
) -> str:

    if _has_errors(state):
        return "workflow_error"

    vision_result = state.get(
        "vision_result"
    )

    if not _is_success(vision_result):
        return "workflow_error"

    return "planning"


# =========================================================
# 4. GENERIC NODE ROUTING
# =========================================================

def route_after_node(
    state: FrontendWorkflowState,
) -> str:
    """
    Generic route for nodes where failures are fatal.
    """

    if _has_errors(state):
        return "workflow_error"

    return "continue"


# =========================================================
# 5. BUILD VALIDATION ROUTING
# =========================================================

def route_after_build(
    state: FrontendWorkflowState,
) -> str:
    """
    Successful build -> browser_execution
    Failed build     -> workflow_error
    """

    if _has_errors(state):
        return "workflow_error"

    build_result = _as_dict(
        state.get("build_result")
    )

    if not build_result:
        return "workflow_error"

    if build_result.get("success") is True:
        return "browser_execution"

    return "workflow_error"


# =========================================================
# 6. BROWSER EXECUTION ROUTING
# =========================================================

def route_after_browser(
    state: FrontendWorkflowState,
) -> str:
    """
    Browser success -> evaluation

    Repairable browser failure -> refinement

    Missing diagnostics or exhausted refinement budget
    -> workflow_error.

    IMPORTANT:
    Browser failures must be preserved in browser_result.
    They must not be converted into fatal workflow errors
    before this router executes.
    """

    browser_result = _as_dict(
        state.get("browser_result")
    )

    # Missing browser diagnostics.
    if not browser_result:
        return "workflow_error"

    # Successful browser execution.
    if browser_result.get("success") is True:

        if _has_errors(state):
            return "workflow_error"

        return "evaluation"

    # Only explicitly reported failures are repair candidates.
    if browser_result.get("success") is not False:
        return "workflow_error"

    # Check refinement budget.
    refinement_count = _get_refinement_count(state)

    max_iterations = _get_max_refinement_iterations(
        state
    )

    if refinement_count >= max_iterations:
        return "workflow_error"

    # Browser failure should be repaired.
    return "refinement"


# =========================================================
# 7. EVALUATION ROUTING
# =========================================================

def route_after_evaluation(
    state: FrontendWorkflowState,
) -> str:
    """
    Quality gate.

    Score >= threshold -> finalize
    Score below threshold -> refinement
    Maximum iterations reached -> finalize, not approved
    Invalid evaluation -> workflow_error
    """

    if _has_errors(state):
        return "workflow_error"

    evaluation_result = _as_dict(
        state.get("evaluation_result")
    )

    if not evaluation_result:
        return "workflow_error"

    if not _is_success(evaluation_result):
        return "workflow_error"

    quality_score = _get_quality_score(state)

    if quality_score is None:
        return "workflow_error"

    try:
        quality_threshold = float(
            state.get(
                "quality_threshold",
                DEFAULT_QUALITY_THRESHOLD,
            )
        )
    except (TypeError, ValueError):
        quality_threshold = DEFAULT_QUALITY_THRESHOLD

    refinement_count = _get_refinement_count(state)

    max_iterations = _get_max_refinement_iterations(
        state
    )

    if quality_score >= quality_threshold:
        return "finalize"

    if refinement_count >= max_iterations:
        return "finalize"

    return "refinement"


# =========================================================
# 8. REFINEMENT ROUTING
# =========================================================

def route_after_refinement(
    state: FrontendWorkflowState,
) -> str:
    """
    Successful refinement -> build_validation

    Failed refinement -> workflow_error
    """

    if _has_errors(state):
        return "workflow_error"

    refinement_result = _as_dict(
        state.get("refinement_result")
    )

    if not _is_success(refinement_result):
        return "workflow_error"

    return "build_validation"


# =========================================================
# 9. WORKFLOW ERROR HANDLER
# =========================================================

def workflow_error_node(
    state: FrontendWorkflowState,
) -> dict[str, Any]:
    """
    Produce a structured workflow failure response.
    """

    errors = list(
        state.get("errors", [])
    )

    if not errors:
        errors.append(
            "Workflow stopped because a required result "
            "was missing, unsuccessful, or exceeded its "
            "retry/refinement limit."
        )

    error_details = list(
        state.get("error_details", [])
    )

    browser_result = _as_dict(
        state.get("browser_result")
    )

    return {
        "status": "failed",
        "current_agent": "workflow_error",
        "next_agent": "END",
        "errors": errors,
        "error_details": error_details,
        "final_output": {
            "workflow_id": state.get("workflow_id"),
            "project_name": state.get("project_name"),
            "status": "failed",
            "quality_score": state.get("quality_score"),
            "is_approved": False,
            "refinement_count": _get_refinement_count(
                state
            ),
            "retry_count": state.get("retry_count", 0),
            "browser_result": browser_result,
            "errors": errors,
            "error_details": error_details,
            "execution_metrics": state.get(
                "execution_metrics",
                {},
            ),
        },
    }
