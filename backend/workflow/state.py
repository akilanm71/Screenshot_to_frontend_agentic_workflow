
from typing import TypedDict, Any


class FrontendWorkflowState(TypedDict, total=False):

    # =====================================================
    # WORKFLOW INFORMATION
    # =====================================================

    workflow_id: str
    project_name: str
    status: str
    current_agent: str
    next_agent: str

    started_at: str
    completed_at: str

    # =====================================================
    # INPUT INFORMATION
    # =====================================================

    input_path: str
    input_type: str

    # =====================================================
    # AGENT OUTPUTS
    # =====================================================

    classification_result: dict[str, Any]
    vision_result: dict[str, Any]
    pdf_extraction_result: dict[str, Any]
    frontend_plan: dict[str, Any]

    # =====================================================
    # CODE GENERATION
    # =====================================================

    code_generation_result: dict[str, Any]
    generated_jsx: str
    generated_css: str
    build_result: dict[str, Any]

    # =====================================================
    # BROWSER EXECUTION
    # =====================================================

    browser_result: dict[str, Any]
    screenshot_path: str

    # =====================================================
    # EVALUATION
    # =====================================================

    evaluation_result: dict[str, Any]
    evaluation_history: list[dict[str, Any]]

    quality_score: float
    quality_threshold: float
    is_approved: bool

    # =====================================================
    # REFINEMENT AND RETRY MANAGEMENT
    # =====================================================

    refinement_result: dict[str, Any]
    refinement_count: int
    max_refinement_iterations: int

    retry_count: int
    max_retries: int
    retry_reason: str

    # =====================================================
    # ORCHESTRATION TRACKING
    # =====================================================

    completed_agents: list[str]

    # =====================================================
    # OBSERVABILITY AND ERROR HANDLING
    # =====================================================

    execution_metrics: dict[str, Any]
    errors: list[str]
    error_details: list[dict[str, Any]]

    repair_context: dict[str, Any]
    retry_reason: str  

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    final_output: dict[str, Any]
