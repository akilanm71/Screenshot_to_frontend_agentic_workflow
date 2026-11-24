
import inspect
import logging
import subprocess
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from backend.workflow.state import FrontendWorkflowState


logger = logging.getLogger(__name__)

DEFAULT_QUALITY_THRESHOLD = 85.0
DEFAULT_MAX_REFINEMENT_ITERATIONS = 3


# =========================================================
# COMMON UTILITIES
# =========================================================

def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_result(result: Any) -> Any:
    """Convert Pydantic objects and nested results into Python data."""

    if result is None:
        return {}

    if hasattr(result, "model_dump"):
        return normalize_result(result.model_dump())

    if hasattr(result, "dict") and callable(result.dict):
        return normalize_result(result.dict())

    if isinstance(result, dict):
        return {
            key: normalize_result(value)
            for key, value in result.items()
        }

    if isinstance(result, list):
        return [normalize_result(item) for item in result]

    if isinstance(result, tuple):
        return [normalize_result(item) for item in result]

    return result


def as_dict(value: Any) -> dict:
    value = normalize_result(value)
    return value if isinstance(value, dict) else {}


def get_source_data(state: FrontendWorkflowState) -> dict:
    return {
        "input_path": state.get("input_path", ""),
        "input_type": state.get("input_type", ""),
        "classification_result": state.get(
            "classification_result", {}
        ),
        "vision_result": state.get("vision_result", {}),
        "pdf_extraction_result": state.get(
            "pdf_extraction_result", {}
        ),
    }


def get_source_type(state: FrontendWorkflowState) -> str:
    """Return the source type expected by PlanningAgent."""

    input_path = str(
        state.get("input_path", "")
    ).lower()

    input_type = str(
        state.get("input_type", "")
    ).lower()

    classification = as_dict(
        state.get("classification_result", {})
    )

    classified_type = str(
        classification.get("file_type", "")
    ).lower()

    input_format = str(
        classification.get("input_format", "")
    ).lower()

    if (
        input_path.endswith(".pdf")
        or input_type in {"pdf", ".pdf"}
        or classified_type == "pdf"
        or input_format == ".pdf"
    ):
        return "pdf"

    if (
        input_path.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        )
        or input_type in {
            "image",
            "screenshot",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
        }
        or classified_type in {"image", "screenshot"}
    ):
        return "screenshot"

    raise ValueError(
        "Unable to determine source_type. "
        f"Input path: {input_path}, "
        f"input_type: {input_type}, "
        f"classified_type: {classified_type}"
    )


# =========================================================
# PROJECT DIRECTORY RESOLUTION
# =========================================================

def get_project_directory(state: FrontendWorkflowState) -> str:
    """
    Resolve the actual React application directory.

    Supported structures:

        project/
            package.json

    Or:

        project/
            frontend/
                package.json

    The returned path always points to the directory
    containing package.json.
    """

    refinement_result = as_dict(
        state.get("refinement_result", {})
    )

    code_result = as_dict(
        state.get("code_generation_result", {})
    )

    plan = as_dict(
        state.get("frontend_plan", {})
    )

    possible_paths = [
        # Canonical workflow path
        state.get("project_directory"),

        # Latest refinement output
        refinement_result.get("frontend_directory"),
        refinement_result.get("frontend_path"),
        refinement_result.get("project_dir"),
        refinement_result.get("project_directory"),
        refinement_result.get("output_directory"),
        refinement_result.get("project_path"),

        # Code generation output
        code_result.get("frontend_directory"),
        code_result.get("frontend_path"),
        code_result.get("project_dir"),
        code_result.get("project_directory"),
        code_result.get("output_directory"),
        code_result.get("project_path"),

        # Planning output
        plan.get("frontend_directory"),
        plan.get("frontend_path"),
        plan.get("project_dir"),
        plan.get("project_directory"),
        plan.get("output_directory"),
    ]

    checked_paths = set()

    for value in possible_paths:

        if not value:
            continue

        candidate = Path(str(value)).expanduser().resolve()

        if candidate in checked_paths:
            continue

        checked_paths.add(candidate)

        if not candidate.is_dir():
            continue

        # Case 1: Candidate is already the React app root.
        if (candidate / "package.json").is_file():
            logger.info(
                "Resolved React project directory: %s",
                candidate,
            )
            return str(candidate)

        # Case 2: React app is inside frontend/.
        frontend_directory = candidate / "frontend"

        if (
            frontend_directory.is_dir()
            and (frontend_directory / "package.json").is_file()
        ):
            frontend_directory = frontend_directory.resolve()

            logger.info(
                "Resolved nested React frontend directory: %s",
                frontend_directory,
            )

            return str(frontend_directory)

    raise ValueError(
        "React frontend directory not found. "
        "None of the project paths contains package.json. "
        "Check CodeGenerationAgent's returned project path."
    )


# =========================================================
# AGENT CONTEXT
# =========================================================

def build_agent_context(
    state: FrontendWorkflowState,
) -> dict[str, Any]:
    """Build common context passed to workflow agents."""

    source_data = get_source_data(state)

    frontend_plan = state.get("frontend_plan", {})

    code_result = as_dict(
        state.get("code_generation_result", {})
    )

    refinement_result = as_dict(
        state.get("refinement_result", {})
    )

    try:
        project_directory = get_project_directory(state)
    except ValueError:
        project_directory = ""

    # project_directory is the canonical React root.
    # Do not append another /frontend here.
    frontend_directory = project_directory

    context = {
        # Workflow
        "workflow_id": state.get("workflow_id", ""),
        "project_name": state.get("project_name", ""),
        "input_path": state.get("input_path", ""),
        "input_type": state.get("input_type", ""),

        # Source data
        "source_data": source_data,
        "classification_result": source_data[
            "classification_result"
        ],
        "vision_result": source_data["vision_result"],
        "pdf_extraction_result": source_data[
            "pdf_extraction_result"
        ],

        # File path aliases
        "file_path": state.get("input_path", ""),
        "image_path": state.get("input_path", ""),
        "pdf_path": state.get("input_path", ""),
        "source_path": state.get("input_path", ""),

        # Planning
        "frontend_plan": frontend_plan,
        "plan": frontend_plan,

        # Canonical generated project directory
        "project_directory": project_directory,
        "project_dir": project_directory,
        "project_path": project_directory,
        "frontend_directory": frontend_directory,

        # Generated source
        "generated_jsx": state.get("generated_jsx", ""),
        "generated_css": state.get("generated_css", ""),
        "code_generation_result": code_result,
        "refinement_result": refinement_result,

        # Build
        "build_result": state.get("build_result", {}),

        # Browser
        "browser_result": state.get("browser_result", {}),
        "screenshot_path": state.get("screenshot_path", ""),

        # Evaluation
        "evaluation_result": state.get(
            "evaluation_result", {}
        ),
        "evaluation_history": state.get(
            "evaluation_history", []
        ),
        "quality_score": state.get("quality_score", 0),
        "quality_threshold": state.get(
            "quality_threshold",
            DEFAULT_QUALITY_THRESHOLD,
        ),
        "is_approved": state.get("is_approved", False),

        # Refinement
      # Refinement
        "refinement_count": state.get("refinement_count", 0),
        "max_refinement_iterations": state.get(
    "max_refinement_iterations",
    DEFAULT_MAX_REFINEMENT_ITERATIONS,
),
"repair_context": state.get("repair_context", {}),
"retry_reason": state.get("retry_reason", ""),
    }

    context["source_type"] = get_source_type(state)

    return context


# =========================================================
# AGENT SIGNATURE ADAPTER
# =========================================================

def instantiate_agent(
    agent_class: type,
    context: dict[str, Any],
):
    """Instantiate an agent using supported constructor arguments."""

    signature = inspect.signature(agent_class.__init__)

    kwargs = {}
    missing = []

    for name, parameter in signature.parameters.items():

        if name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in context:
            kwargs[name] = context[name]

        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise RuntimeError(
            f"Cannot initialize {agent_class.__name__}. "
            f"Required constructor arguments not available: {missing}. "
            f"Constructor signature: {signature}"
        )

    return agent_class(**kwargs)


def invoke_agent_method(
    agent: Any,
    method_name: str,
    context: dict[str, Any],
    positional_fallback: Any = None,
):
    """Invoke an agent method using its declared parameters."""

    method = getattr(agent, method_name, None)

    if method is None or not callable(method):
        raise AttributeError(
            f"{agent.__class__.__name__} does not implement "
            f"callable method '{method_name}'."
        )

    signature = inspect.signature(method)

    kwargs = {}
    positional_only = []
    missing = []

    for name, parameter in signature.parameters.items():

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in context:

            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                positional_only.append(context[name])
            else:
                kwargs[name] = context[name]

        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:

        if (
            len(missing) == 1
            and positional_fallback is not None
        ):
            parameter_name = missing[0]
            parameter = signature.parameters[parameter_name]

            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                positional_only.append(positional_fallback)
            else:
                kwargs[parameter_name] = positional_fallback

        else:
            raise RuntimeError(
                f"Cannot invoke {agent.__class__.__name__}."
                f"{method_name}(). "
                f"Missing required arguments: {missing}. "
                f"Method signature: {signature}"
            )

    logger.info(
        "Invoking agent method | Agent: %s | Method: %s",
        agent.__class__.__name__,
        method_name,
    )

    return method(*positional_only, **kwargs)


# =========================================================
# EXECUTION METRICS
# =========================================================

def add_execution_metric(
    state: FrontendWorkflowState,
    agent_name: str,
    duration: float,
) -> dict:

    metrics = dict(
        state.get("execution_metrics", {})
    )

    agents = dict(
        metrics.get("agents", {})
    )

    previous = agents.get(agent_name, {})

    previous_duration = previous.get(
        "total_duration_seconds",
        previous.get("duration_seconds", 0),
    )

    execution_count = previous.get(
        "execution_count", 0
    ) + 1

    total_duration = previous_duration + duration

    agents[agent_name] = {
        "duration_seconds": round(duration, 4),
        "total_duration_seconds": round(total_duration, 4),
        "executed_at": current_timestamp(),
        "execution_count": execution_count,
    }

    metrics["agents"] = agents

    metrics["total_agent_duration_seconds"] = round(
        sum(
            item.get("total_duration_seconds", 0)
            for item in agents.values()
        ),
        4,
    )

    return metrics


# =========================================================
# ERROR HANDLING
# =========================================================

def build_error_update(
    state: FrontendWorkflowState,
    agent_name: str,
    exception: Exception,
    duration: float,
) -> dict:

    error_message = str(exception)

    errors = list(state.get("errors", []))
    errors.append(
        f"{agent_name}: {error_message}"
    )

    error_details = list(
        state.get("error_details", [])
    )

    error_details.append({
        "agent": agent_name,
        "error_type": type(exception).__name__,
        "error_message": error_message,
        "timestamp": current_timestamp(),
    })

    logger.exception(
        "Workflow node failed | Agent: %s",
        agent_name,
    )

    return {
        "status": "failed",
        "current_agent": agent_name,
        "errors": errors,
        "error_details": error_details,
        "execution_metrics": add_execution_metric(
            state,
            agent_name,
            duration,
        ),
    }


def execute_node(
    state: FrontendWorkflowState,
    agent_name: str,
    operation: Callable[[], dict],
) -> dict:
    """Common execution wrapper for workflow nodes."""

    started = time.perf_counter()

    try:
        result = normalize_result(operation())

        duration = time.perf_counter() - started

        if not isinstance(result, dict):
            result = {"result": result}

        if result.get("status") in ("failed", "error"):
            raise RuntimeError(
                result.get(
                    "error",
                    "Agent returned failed status.",
                )
            )

        completed_agents = list(
            state.get("completed_agents", [])
        )

        if agent_name not in completed_agents:
            completed_agents.append(agent_name)

        result["current_agent"] = agent_name
        result["completed_agents"] = completed_agents

        result["execution_metrics"] = add_execution_metric(
            state,
            agent_name,
            duration,
        )

        return result

    except Exception as exc:

        duration = time.perf_counter() - started

        return build_error_update(
            state,
            agent_name,
            exc,
            duration,
        )


# =========================================================
# 1. INPUT CLASSIFIER NODE
# =========================================================

def classify_input_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        from backend.agents.input_classifier import (
            InputClassifierAgent,
        )

        context = build_agent_context(state)

        agent = instantiate_agent(
            InputClassifierAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "classify",
            context,
            positional_fallback=state.get("input_path"),
        )

        classification = as_dict(result)

        input_type = (
            classification.get("input_format")
            or classification.get("file_type")
            or classification.get("source_type")
            or state.get("input_type", "")
        )

        if not input_type:
            raise ValueError(
                "Input classifier did not return a recognizable input type."
            )

        return {
            "classification_result": classification,
            "input_type": str(input_type).lower(),
            "status": "classification_completed",
        }

    return execute_node(
        state,
        "input_classifier",
        operation,
    )


# =========================================================
# 2. VISION ANALYSIS NODE
# =========================================================

def vision_analysis_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        from backend.agents.vision_agent import (
            VisionAnalysisAgent,
        )

        context = build_agent_context(state)

        agent = instantiate_agent(
            VisionAnalysisAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "analyze",
            context,
            positional_fallback=state.get("input_path"),
        )

        result = normalize_result(result)

        if not result:
            raise RuntimeError(
                "Vision Analysis Agent returned an empty result."
            )

        return {
            "vision_result": result,
            "status": "vision_analysis_completed",
        }

    return execute_node(
        state,
        "vision_analysis",
        operation,
    )


# =========================================================
# 3. PDF EXTRACTION NODE
# =========================================================

def pdf_extraction_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        from backend.agents.pdf_extraction_agent import (
            PDFExtractionAgent,
        )

        context = build_agent_context(state)

        agent = instantiate_agent(
            PDFExtractionAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "extract",
            context,
            positional_fallback=state.get("input_path"),
        )

        result = normalize_result(result)

        if not result:
            raise RuntimeError(
                "PDF Extraction Agent returned an empty result."
            )

        return {
            "pdf_extraction_result": result,
            "status": "pdf_extraction_completed",
        }

    return execute_node(
        state,
        "pdf_extraction",
        operation,
    )


# =========================================================
# 4. PLANNING NODE
# =========================================================

def planning_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        from backend.agents.planning_agent import PlanningAgent

        context = build_agent_context(state)

        source_type = get_source_type(state)
        context["source_type"] = source_type

        logger.info(
            "Planning source type | Workflow: %s | Source: %s",
            state.get("workflow_id"),
            source_type,
        )

        agent = instantiate_agent(
            PlanningAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "create_plan",
            context,
        )

        plan = as_dict(result)

        if not plan:
            raise RuntimeError(
                "Planning Agent returned an empty plan."
            )

        project_name = (
            plan.get("project_name")
            or state.get("project_name")
            or "generated-frontend"
        )

        return {
            "frontend_plan": plan,
            "project_name": project_name,
            "status": "planning_completed",
        }

    return execute_node(
        state,
        "planning_agent",
        operation,
    )


# =========================================================
# 5. CODE GENERATION NODE
# =========================================================

def code_generation_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        from backend.agents.code_agent import (
            CodeGenerationAgent,
        )

        context = build_agent_context(state)

        agent = instantiate_agent(
            CodeGenerationAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "generate",
            context,
            positional_fallback=state.get("frontend_plan", {}),
        )

        result = as_dict(result)

        generation_status = str(
            result.get("status", "")
        ).lower()

        generation_success = result.get("success")

        if (
            generation_status not in (
                "success",
                "completed",
                "passed",
            )
            and generation_success is not True
        ):
            raise RuntimeError(
                "Code Generation Agent did not report success. "
                f"Result: {result}"
            )

        # Resolve the project directory from the NEW
        # Code Generation result, not from an older state.
        path_state = {
            **state,
            "code_generation_result": result,
        }

        try:
            project_directory = get_project_directory(path_state)
        except ValueError:
            project_directory = ""

            logger.warning(
                "Code Generation completed but no React project "
                "directory could be resolved from its result."
            )

        return {
            "code_generation_result": result,
            "generated_jsx": result.get(
                "generated_jsx",
                state.get("generated_jsx", ""),
            ),
            "generated_css": result.get(
                "generated_css",
                state.get("generated_css", ""),
            ),
            "project_name": (
                result.get("project_name")
                or state.get("project_name", "")
            ),
            "project_directory": project_directory,
            "status": "code_generation_completed",
        }

    return execute_node(
        state,
        "code_generation",
        operation,
    )


# =========================================================
# 6. BUILD VALIDATION NODE
# =========================================================

def build_validation_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        # This returns the actual directory containing package.json.
        project_directory = get_project_directory(state)
        frontend_path = Path(project_directory)

        package_json = frontend_path / "package.json"

        if not package_json.is_file():
            raise FileNotFoundError(
                f"package.json not found: {package_json}"
            )

        logger.info(
            "Running frontend build | Directory: %s",
            frontend_path,
        )

        # Install dependencies only when node_modules is missing.
        if not (frontend_path / "node_modules").is_dir():

            logger.info(
                "node_modules missing. Installing dependencies."
            )

            install_result = subprocess.run(
                "npm.cmd install",
                cwd=str(frontend_path),
                capture_output=True,
                text=True,
                timeout=300,
                shell=True,
            )

            if install_result.returncode != 0:
                raise RuntimeError(
                    "npm install failed: "
                    + install_result.stderr[-3000:]
                )

        # Always build the exact current frontend directory.
        # Do not reuse a build result from another directory.
        build_result = subprocess.run(
            "npm.cmd run build",
            cwd=str(frontend_path),
            capture_output=True,
            text=True,
            timeout=300,
            shell=True,
        )

        result = {
            "success": build_result.returncode == 0,
            "return_code": build_result.returncode,
            "stdout": build_result.stdout[-5000:],
            "stderr": build_result.stderr[-5000:],
            "project_directory": str(frontend_path),
            "frontend_directory": str(frontend_path),
            "package_json": str(package_json),
            "executed_at": current_timestamp(),
        }

        if not result["success"]:
            raise RuntimeError(
                "Frontend production build failed: "
                + build_result.stderr[-3000:]
            )

        logger.info(
            "Frontend build successful | Directory: %s",
            frontend_path,
        )

        return {
            "build_result": result,
            "project_directory": str(frontend_path),
            "status": "build_completed",
        }

    return execute_node(
        state,
        "build_validation",
        operation,
    )


# =========================================================
# 7. BROWSER EXECUTION NODE
# =========================================================


# =========================================================
# 7. BROWSER EXECUTION NODE
# =========================================================


def browser_execution_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        import json

        from backend.agents.browser_agent import BrowserAgent

        # =====================================================
        # 1. RESOLVE PROJECT DIRECTORY
        # =====================================================

        project_directory = get_project_directory(state)
        frontend_path = Path(project_directory)

        package_json = frontend_path / "package.json"

        if not package_json.is_file():
            raise FileNotFoundError(
                f"Browser Agent cannot start. "
                f"package.json not found: {package_json}"
            )

        # =====================================================
        # 2. VALIDATE BUILD DIRECTORY
        # =====================================================

        build_result = as_dict(
            state.get("build_result", {})
        )

        built_directory = build_result.get(
            "project_directory"
        )

        if (
            built_directory
            and Path(built_directory).resolve()
            != frontend_path.resolve()
        ):
            raise RuntimeError(
                "Build and Browser directories do not match. "
                f"Build: {built_directory} | "
                f"Browser: {frontend_path}"
            )

        logger.info(
            "Starting Browser Agent | Project: %s",
            frontend_path,
        )

        # =====================================================
        # 3. PREPARE AGENT CONTEXT
        # =====================================================

        context = build_agent_context(state)

        context.update({
            "project_directory": str(frontend_path),
            "project_dir": str(frontend_path),
            "project_path": str(frontend_path),
            "frontend_directory": str(frontend_path),
        })

        # =====================================================
        # 4. INITIALIZE BROWSER AGENT
        # =====================================================

        agent = instantiate_agent(
            BrowserAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "run",
            context,
        )

        result = as_dict(result)

        browser_success = (
            result.get("success") is True
        )

        screenshot_path = (
            result.get("screenshot_path")
            or result.get("screenshot")
            or ""
        )

        # =====================================================
        # 5. HANDLE SUCCESS
        # =====================================================

        if browser_success:

            repair_context = {}
            retry_reason = ""

            logger.info(
                "Browser execution successful | Project: %s",
                frontend_path,
            )

        # =====================================================
        # 6. HANDLE BROWSER FAILURE
        # =====================================================

        else:

            runtime_error = as_dict(
                result.get("runtime_error", {})
            )

            error_message = (
                runtime_error.get("message")
                or result.get("error")
                or result.get("failure_reason")
                or result.get("message")
                or "Browser execution failed. "
                   "Inspect browser diagnostics."
            )

            failure_type = (
                result.get("failure_category")
                or runtime_error.get("category")
                or result.get("failure_type")
                or result.get("error_type")
                or "browser_runtime_failure"
            )

            # -------------------------------------------------
            # Load detailed repair context generated by
            # BrowserAgent, when the file is available.
            # -------------------------------------------------

            detailed_repair_context = {}

            repair_context_path = result.get(
                "repair_context_path"
            )

            if repair_context_path:

                repair_file = Path(
                    repair_context_path
                )

                if repair_file.is_file():

                    try:

                        with repair_file.open(
                            "r",
                            encoding="utf-8",
                        ) as file:

                            loaded_context = json.load(file)

                        if isinstance(
                            loaded_context,
                            dict,
                        ):
                            detailed_repair_context = (
                                loaded_context
                            )

                        logger.info(
                            "Loaded BrowserAgent repair context: %s",
                            repair_file,
                        )

                    except (
                        OSError,
                        json.JSONDecodeError,
                    ) as exc:

                        logger.warning(
                            "Could not load BrowserAgent "
                            "repair context: %s",
                            exc,
                        )

            # -------------------------------------------------
            # Extract detailed runtime diagnostics.
            # -------------------------------------------------

            detailed_runtime_error = as_dict(
                detailed_repair_context.get(
                    "runtime_error",
                    {},
                )
            )

            if not runtime_error:
                runtime_error = detailed_runtime_error

            missing_identifier = (
                runtime_error.get("missing_identifier")
                or detailed_repair_context.get(
                    "missing_identifier"
                )
                or result.get("missing_identifier")
                or ""
            )

            identifier_source_matches = (
                detailed_repair_context.get(
                    "identifier_source_matches"
                )
                or result.get(
                    "identifier_source_matches",
                    [],
                )
            )

            # -------------------------------------------------
            # Build complete repair context.
            # -------------------------------------------------

            repair_context = {
                "source": "browser_agent",

                "failure_type": failure_type,

                "failure_category": failure_type,

                "reason": error_message,

                "runtime_error": runtime_error,

                "missing_identifier": missing_identifier,

                "identifier_source_matches": (
                    identifier_source_matches
                ),

                "project_directory": str(
                    frontend_path
                ),

                "browser_result": result,

                "screenshot_path": str(
                    screenshot_path
                ),

                "repair_context_path": str(
                    repair_context_path or ""
                ),

                "refinement_iteration": (
                    state.get("refinement_count", 0) + 1
                ),
            }

            retry_reason = (
                f"Browser runtime failure: {error_message}"
            )

            logger.warning(
                "Browser execution failed. "
                "Preserving diagnostics for refinement | "
                "Project: %s | Failure: %s | "
                "Missing identifier: %s | Error: %s",
                frontend_path,
                failure_type,
                missing_identifier,
                error_message,
            )

        # =====================================================
        # 7. RETURN WORKFLOW STATE
        # =====================================================

        # IMPORTANT:
        # Do not return status="failed" here.
        # The workflow router must receive browser_result.success
        # and decide whether to proceed to evaluation or refinement.

        return {
            "browser_result": result,

            "repair_context": repair_context,

            "retry_reason": retry_reason,

            "screenshot_path": str(
                screenshot_path
            ),

            "project_directory": str(
                frontend_path
            ),

            "status": "browser_execution_completed",
        }

    return execute_node(
        state,
        "browser_agent",
        operation,
    )

# =========================================================
# 8. EVALUATOR NODE
# =========================================================

def evaluation_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        from backend.agents.evaluator_agent import EvaluatorAgent

        context = build_agent_context(state)

        agent = instantiate_agent(
            EvaluatorAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "evaluate",
            context,
        )

        result = as_dict(result)

        raw_score = None

        for key in (
            "quality_score",
            "overall_score",
            "score",
        ):
            if result.get(key) is not None:
                raw_score = result[key]
                break

        if raw_score is None:
            raise ValueError(
                "Evaluator did not return quality_score, "
                "overall_score, or score."
            )

        score = float(raw_score)

        if 0 <= score <= 1:
            score *= 100

        threshold = float(
            state.get(
                "quality_threshold",
                DEFAULT_QUALITY_THRESHOLD,
            )
        )

        evaluator_approval = result.get(
            "is_approved",
            result.get("passed", True),
        )

        approved = (
            score >= threshold
            and bool(evaluator_approval)
        )

        history = list(
            state.get("evaluation_history", [])
        )

        evaluation_record = {
            "iteration": len(history) + 1,
            "timestamp": current_timestamp(),
            "quality_score": round(score, 2),
            "quality_threshold": threshold,
            "is_approved": approved,
            "refinement_count": state.get(
                "refinement_count", 0
            ),
            "evaluation_result": result,
        }

        history.append(evaluation_record)

        logger.info(
            "Evaluation completed | Score: %.2f | "
            "Threshold: %.2f | Approved: %s",
            score,
            threshold,
            approved,
        )

        return {
            "evaluation_result": result,
            "evaluation_history": history,
            "quality_score": round(score, 2),
            "quality_threshold": threshold,
            "is_approved": approved,
            "status": "evaluation_completed",
        }

    return execute_node(
        state,
        "evaluator_agent",
        operation,
    )


# =========================================================
# 9. REFINEMENT NODE
# =========================================================


# =========================================================
# 9. REFINEMENT NODE
# =========================================================

def refinement_node(
    state: FrontendWorkflowState,
) -> dict:

    def operation():

        current_count = state.get("refinement_count", 0)

        max_iterations = state.get(
            "max_refinement_iterations",
            DEFAULT_MAX_REFINEMENT_ITERATIONS,
        )

        if current_count >= max_iterations:
            raise RuntimeError(
                "Maximum refinement iterations reached."
            )

        from backend.agents.refinement_agent import (
            RefinementAgent,
        )

        # Build the normal agent context.
        context = build_agent_context(state)

        # Explicitly pass browser failure information
        # to the RefinementAgent.
        context["repair_context"] = state.get(
            "repair_context", {}
        )

        context["browser_result"] = state.get(
            "browser_result", {}
        )

        context["retry_reason"] = state.get(
            "retry_reason", ""
        )

        logger.info(
            "Starting refinement | Iteration: %s/%s | "
            "Retry reason: %s",
            current_count + 1,
            max_iterations,
            context["retry_reason"],
        )

        # Instantiate the agent with the enriched context.
        agent = instantiate_agent(
            RefinementAgent,
            context,
        )

        result = invoke_agent_method(
            agent,
            "refine",
            context,
        )

        result = as_dict(result)

        if result.get("status") in ("failed", "error"):
            raise RuntimeError(
                f"Refinement Agent failed: {result}"
            )

        if result.get("success") is False:
            raise RuntimeError(
                f"Refinement Agent reported failure: {result}"
            )

        path_state = {
            **state,
            "refinement_result": result,
        }

        try:
            project_directory = get_project_directory(
                path_state
            )
        except ValueError:
            project_directory = state.get(
                "project_directory",
                "",
            )

        logger.info(
            "Refinement completed | Iteration: %s/%s | "
            "Project: %s",
            current_count + 1,
            max_iterations,
            project_directory,
        )

        return {
            "refinement_result": result,
            "project_directory": project_directory,
            "refinement_count": current_count + 1,
            "retry_count": state.get("retry_count", 0) + 1,
            "retry_reason": "",
            "repair_context": {},
            "status": "refinement_completed",
        }

    return execute_node(
        state,
        "refinement_agent",
        operation,
    )

# =========================================================
# 10. FINALIZE NODE
# =========================================================

def finalize_node(
    state: FrontendWorkflowState,
) -> dict:

    completed_at = current_timestamp()

    try:
        project_directory = get_project_directory(state)
    except ValueError:
        project_directory = state.get(
            "project_directory",
            "",
        )

    final_output = {
        "workflow_id": state.get("workflow_id"),
        "project_name": state.get("project_name"),
        "status": state.get("status"),
        "quality_score": state.get("quality_score", 0),
        "quality_threshold": state.get(
            "quality_threshold",
            DEFAULT_QUALITY_THRESHOLD,
        ),
        "is_approved": state.get("is_approved", False),
        "project_directory": project_directory,
        "screenshot_path": state.get("screenshot_path"),
        "evaluation_result": state.get(
            "evaluation_result", {}
        ),
        "evaluation_history": state.get(
            "evaluation_history", []
        ),
        "browser_result": state.get(
            "browser_result", {}
        ),
        "build_result": state.get(
            "build_result", {}
        ),
        "execution_metrics": state.get(
            "execution_metrics", {}
        ),
        "completed_agents": state.get(
            "completed_agents", []
        ),
        "refinement_count": state.get(
            "refinement_count", 0
        ),
        "errors": state.get("errors", []),
        "completed_at": completed_at,
    }

    return {
        "final_output": final_output,
        "completed_at": completed_at,
        "current_agent": "finalize",
        "status": (
            "completed"
            if state.get("is_approved", False)
            else "completed_with_warnings"
        ),
    }
