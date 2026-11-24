
import logging
import threading
import traceback
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from backend.api.dependencies import (
    UPLOAD_DIRECTORY,
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    validate_workflow_id,
)

from backend.workflow.workflow import (
    stream_workflow,
    get_workflow_state,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workflows",
    tags=["Workflows"],
)


# =========================================================
# WORKFLOW REGISTRY
# =========================================================

WORKFLOW_REGISTRY: dict[str, dict[str, Any]] = {}

WORKFLOW_LOCK = threading.RLock()


# =========================================================
# COMMON UTILITIES
# =========================================================

def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_state_values(snapshot: Any) -> dict[str, Any]:
    """
    Extract the state dictionary from a LangGraph snapshot.
    """

    if snapshot is None:
        return {}

    if isinstance(snapshot, dict):
        return snapshot

    values = getattr(snapshot, "values", None)

    if isinstance(values, dict):
        return values

    return {}


def serialize_workflow_status(
    workflow_id: str,
) -> dict[str, Any]:

    with WORKFLOW_LOCK:
        registry_item = WORKFLOW_REGISTRY.get(workflow_id)

        if registry_item is None:
            return {}

        result = dict(registry_item)

    # Retrieve the latest LangGraph checkpoint.
    try:
        snapshot = get_workflow_state(workflow_id)
        state = get_state_values(snapshot)

        if state:
            result.update({
                "status": state.get(
                    "status",
                    result.get("status", "running"),
                ),
                "current_agent": state.get(
                    "current_agent",
                    result.get("current_agent"),
                ),
                "completed_agents": state.get(
                    "completed_agents",
                    result.get("completed_agents", []),
                ),
                "errors": state.get(
                    "errors",
                    result.get("errors", []),
                ),
                "quality_score": state.get(
                    "quality_score",
                    result.get("quality_score"),
                ),
                "project_name": state.get(
                    "project_name",
                    result.get("project_name"),
                ),
                "final_output": state.get(
                    "final_output",
                    result.get("final_output"),
                ),
                "execution_metrics": state.get(
                    "execution_metrics",
                    result.get("execution_metrics", {}),
                ),
            })

    except Exception:
        logger.exception(
            "Unable to retrieve workflow checkpoint: %s",
            workflow_id,
        )

    return result


# =========================================================
# WORKFLOW BACKGROUND EXECUTION
# =========================================================

def run_workflow_background(
    initial_state: dict[str, Any],
) -> None:
    """
    Execute the existing LangGraph workflow and update
    the API registry as individual nodes execute.
    """

    workflow_id = initial_state["workflow_id"]

    logger.info(
        "Starting workflow execution: %s",
        workflow_id,
    )

    with WORKFLOW_LOCK:
        WORKFLOW_REGISTRY[workflow_id].update({
            "status": "running",
            "current_agent": "starting",
            "started_at": current_timestamp(),
        })

    try:

        # Stream real LangGraph node updates.
        for event in stream_workflow(initial_state):

            node_name = event.get("node", "unknown")
            node_output = event.get("output", {})

            if not isinstance(node_output, dict):
                node_output = {}

            current_agent = node_output.get(
                "current_agent",
                node_name,
            )

            completed_agents = node_output.get(
                "completed_agents",
                [],
            )

            with WORKFLOW_LOCK:

                registry_item = WORKFLOW_REGISTRY.get(
                    workflow_id
                )

                if registry_item is not None:

                    registry_item.update({
                        "status": "running",
                        "current_agent": current_agent,
                        "last_node": node_name,
                        "completed_agents": completed_agents,
                        "last_updated_at": current_timestamp(),
                    })

                    if "quality_score" in node_output:
                        registry_item["quality_score"] = (
                            node_output["quality_score"]
                        )

            logger.info(
                "Workflow %s | Node executed: %s",
                workflow_id,
                node_name,
            )

        # Read the final LangGraph state.
        snapshot = get_workflow_state(workflow_id)
        final_state = get_state_values(snapshot)

        final_status = final_state.get(
            "status",
            "completed",
        )

        if final_status not in {
            "completed",
            "completed_with_warnings",
            "failed",
        }:
            final_status = "completed"

        with WORKFLOW_LOCK:

            registry_item = WORKFLOW_REGISTRY.get(
                workflow_id
            )

            if registry_item is not None:

                registry_item.update({
                    "status": final_status,
                    "current_agent": final_state.get(
                        "current_agent",
                        "completed",
                    ),
                    "completed_agents": final_state.get(
                        "completed_agents",
                        registry_item.get(
                            "completed_agents",
                            [],
                        ),
                    ),
                    "project_name": final_state.get(
                        "project_name",
                    ),
                    "quality_score": final_state.get(
                        "quality_score",
                    ),
                    "errors": final_state.get(
                        "errors",
                        [],
                    ),
                    "final_output": final_state.get(
                        "final_output",
                    ),
                    "execution_metrics": final_state.get(
                        "execution_metrics",
                        {},
                    ),
                    "completed_at": current_timestamp(),
                })

        logger.info(
            "Workflow %s finished with status: %s",
            workflow_id,
            final_status,
        )

    except Exception as exc:

        logger.exception(
            "Workflow execution failed: %s",
            workflow_id,
        )

        error_message = str(exc)

        with WORKFLOW_LOCK:

            registry_item = WORKFLOW_REGISTRY.get(
                workflow_id
            )

            if registry_item is not None:

                registry_item.update({
                    "status": "failed",
                    "current_agent": "workflow_error",
                    "errors": [error_message],
                    "error_details": [{
                        "message": error_message,
                        "traceback": traceback.format_exc(),
                    }],
                    "completed_at": current_timestamp(),
                })


# =========================================================
# 1. SUBMIT WORKFLOW
# =========================================================

@router.post("")
async def create_workflow(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_query: str = Form(
        "Analyze the uploaded screenshot or PDF and generate a responsive React frontend using JSX and CSS."
    ),
    workflow_id: str = Form(default=""),
):

    # -----------------------------------------------------
    # Validate workflow ID
    # -----------------------------------------------------

    if not workflow_id:
        workflow_id = str(uuid.uuid4())

    try:
        workflow_id = validate_workflow_id(workflow_id)

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    with WORKFLOW_LOCK:
        if workflow_id in WORKFLOW_REGISTRY:
            raise HTTPException(
                status_code=409,
                detail="Workflow ID already exists.",
            )

    # -----------------------------------------------------
    # Validate uploaded file
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file extension: {extension}. "
                f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    # Read at most the configured limit plus one byte.
    content = await file.read(
        MAX_UPLOAD_SIZE_BYTES + 1
    )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the maximum allowed size.",
        )

    # -----------------------------------------------------
    # Save uploaded file
    # -----------------------------------------------------

    upload_directory = Path(UPLOAD_DIRECTORY)

    try:
        upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        input_path = (
            upload_directory
            / f"{workflow_id}{extension}"
        )

        input_path.write_bytes(content)

    except OSError as exc:

        logger.exception(
            "Unable to save uploaded file."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to save uploaded file.",
        ) from exc

    finally:
        await file.close()

    # -----------------------------------------------------
    # Determine input type
    # -----------------------------------------------------

    input_type = (
        "pdf"
        if extension == ".pdf"
        else "image"
    )

    # -----------------------------------------------------
    # Build initial LangGraph state
    # -----------------------------------------------------

    initial_state = {
        "workflow_id": workflow_id,
        "project_name": f"generated_project_{workflow_id[:8]}",
        "input_path": str(input_path.resolve()),
        "input_type": input_type,
        "user_query": user_query.strip(),

        "status": "queued",
        "current_agent": "queued",
        "next_agent": "",

        "started_at": current_timestamp(),
        "completed_at": "",

        "classification_result": {},
        "vision_result": {},
        "pdf_extraction_result": {},

        "frontend_plan": {},
        "plan": {},

        "code_generation_result": {},
        "generated_jsx": "",
        "generated_css": "",

        "build_result": {},
        "browser_result": {},
        "screenshot_path": "",

        "evaluation_result": {},
        "evaluation_history": [],

        "quality_score": 0.0,
        "quality_threshold": 85.0,
        "is_approved": False,

        "refinement_result": {},
        "refinement_count": 0,
        "max_refinement_iterations": 3,

        "retry_count": 0,
        "max_retries": 2,
        "retry_reason": "",

        "completed_agents": [],

        "execution_metrics": {},
        "errors": [],
        "error_details": [],

        "repair_context": {},
        "final_output": {},
    }

    # -----------------------------------------------------
    # Register workflow
    # -----------------------------------------------------

    with WORKFLOW_LOCK:

        WORKFLOW_REGISTRY[workflow_id] = {
            "workflow_id": workflow_id,
            "status": "queued",
            "current_agent": "queued",
            "completed_agents": [],
            "project_name": initial_state["project_name"],
            "input_filename": file.filename,
            "input_type": input_type,
            "created_at": current_timestamp(),
            "errors": [],
        }

    # -----------------------------------------------------
    # Schedule background execution
    # -----------------------------------------------------

    background_tasks.add_task(
        run_workflow_background,
        initial_state,
    )

    logger.info(
        "Workflow submitted: %s",
        workflow_id,
    )

    return {
        "success": True,
        "message": "Workflow submitted successfully.",
        "workflow_id": workflow_id,
        "status": "queued",
        "current_agent": "queued",
        "input_type": input_type,
        "input_filename": file.filename,
    }


# =========================================================
# 2. GET WORKFLOW STATUS
# =========================================================

@router.get("/{workflow_id}")
async def get_workflow_status(
    workflow_id: str,
):

    try:
        workflow_id = validate_workflow_id(workflow_id)

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    with WORKFLOW_LOCK:
        exists = workflow_id in WORKFLOW_REGISTRY

    if not exists:
        raise HTTPException(
            status_code=404,
            detail="Workflow not found.",
        )

    workflow = serialize_workflow_status(
        workflow_id
    )

    return {
        "success": True,
        "workflow_id": workflow_id,
        "status": workflow.get("status", "unknown"),
        "current_agent": workflow.get("current_agent"),
        "completed_agents": workflow.get(
            "completed_agents",
            [],
        ),
        "project_name": workflow.get("project_name"),
        "quality_score": workflow.get("quality_score"),
        "errors": workflow.get("errors", []),
        "execution_metrics": workflow.get(
            "execution_metrics",
            {},
        ),
        "created_at": workflow.get("created_at"),
        "started_at": workflow.get("started_at"),
        "completed_at": workflow.get("completed_at"),
    }


# =========================================================
# 3. GET WORKFLOW RESULT
# =========================================================

@router.get("/{workflow_id}/result")
async def get_workflow_result(
    workflow_id: str,
):

    try:
        workflow_id = validate_workflow_id(workflow_id)

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    with WORKFLOW_LOCK:
        exists = workflow_id in WORKFLOW_REGISTRY

    if not exists:
        raise HTTPException(
            status_code=404,
            detail="Workflow not found.",
        )

    workflow = serialize_workflow_status(
        workflow_id
    )

    status = workflow.get("status")

    if status in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail="Workflow is still running.",
        )

    try:
        snapshot = get_workflow_state(workflow_id)
        state = get_state_values(snapshot)

    except Exception as exc:

        logger.exception(
            "Unable to retrieve workflow result: %s",
            workflow_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve workflow result.",
        ) from exc

    final_output = state.get(
        "final_output",
        workflow.get("final_output", {}),
    )

    return {
        "success": status != "failed",
        "workflow_id": workflow_id,
        "status": status,
        "project_name": state.get(
            "project_name",
            workflow.get("project_name"),
        ),
        "final_output": final_output,
        "generated_jsx": state.get("generated_jsx", ""),
        "generated_css": state.get("generated_css", ""),
        "code_generation_result": state.get(
            "code_generation_result",
            {},
        ),
        "build_result": state.get("build_result", {}),
        "browser_result": state.get("browser_result", {}),
        "evaluation_result": state.get(
            "evaluation_result",
            {},
        ),
        "quality_score": state.get("quality_score"),
        "is_approved": state.get("is_approved", False),
        "errors": state.get("errors", []),
        "error_details": state.get("error_details", []),
        "execution_metrics": state.get(
            "execution_metrics",
            {},
        ),
    }
