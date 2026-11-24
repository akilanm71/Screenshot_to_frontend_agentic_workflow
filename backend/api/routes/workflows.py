
import inspect
import logging
import re
import uuid

from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder

from backend.api.dependencies import (
    get_upload_directory,
    get_workflow_runner,
    get_workflow_state_reader,
    validate_file_extension,
    validate_workflow_id,
    MAX_UPLOAD_SIZE_BYTES,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workflows",
    tags=["Workflows"],
)

WORKFLOW_REGISTRY: dict[str, dict[str, Any]] = {}


# =========================================================
# COMMON UTILITIES
# =========================================================

def normalize_project_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value)
    value = value.strip("-_").lower()
    return value[:80] or "generated-project"


async def read_state(state_reader, workflow_id: str):
    """Support both synchronous and asynchronous state readers."""

    result = state_reader(workflow_id)

    if inspect.isawaitable(result):
        result = await result

    return result


def snapshot_to_dict(snapshot: Any) -> dict:
    if snapshot is None:
        return {}

    if isinstance(snapshot, dict):
        return snapshot

    values = getattr(snapshot, "values", None)

    if isinstance(values, dict):
        return values

    return {}


def determine_workflow_status(result: Any) -> str:
    if not isinstance(result, dict):
        return "completed"

    result_status = str(result.get("status", "")).lower()

    if result_status == "failed":
        return "failed"

    if result_status in {
        "completed",
        "completed_with_warnings",
        "success",
    }:
        return result_status

    if result.get("errors") and not result.get("final_output"):
        return "failed"

    return "completed"


# =========================================================
# BACKGROUND WORKFLOW EXECUTION
# =========================================================

def execute_workflow(
    workflow_id: str,
    initial_state: dict,
):
    logger.info(
        "Workflow execution started | ID: %s",
        workflow_id,
    )

    WORKFLOW_REGISTRY[workflow_id]["status"] = "running"

    try:
        invoke_workflow = get_workflow_runner()

        config = {
            "configurable": {
                "thread_id": workflow_id,
            },
            "recursion_limit": 50,
        }

        result = invoke_workflow(
            initial_state=initial_state,
            config=config,
        )

        if inspect.isawaitable(result):
            raise TypeError(
                "Workflow runner returned a coroutine. "
                "The background executor expects a synchronous "
                "invoke_workflow function."
            )

        final_status = determine_workflow_status(result)

        WORKFLOW_REGISTRY[workflow_id].update({
            "status": final_status,
            "error": (
                "Workflow execution failed."
                if final_status == "failed"
                else None
            ),
        })

        logger.info(
            "Workflow execution finished | ID: %s | Status: %s",
            workflow_id,
            final_status,
        )

    except Exception as exc:
        logger.exception(
            "Workflow execution failed | ID: %s",
            workflow_id,
        )

        WORKFLOW_REGISTRY[workflow_id].update({
            "status": "failed",
            "error": str(exc),
        })


# =========================================================
# CREATE WORKFLOW
# =========================================================

@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_workflow(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_query: str = Form(default=""),
    workflow_id: str | None = Form(default=None),
    project_name: str | None = Form(default=None),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    original_filename = file.filename

    try:
        extension = validate_file_extension(original_filename).lower()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if extension not in {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    if workflow_id:
        workflow_id = validate_workflow_id(workflow_id)
    else:
        workflow_id = str(uuid.uuid4())

    if workflow_id in WORKFLOW_REGISTRY:
        raise HTTPException(
            status_code=409,
            detail="Workflow ID already exists.",
        )

    safe_filename = original_filename.replace("\\", "/")
    original_stem = Path(safe_filename).stem

    selected_project_name = normalize_project_name(
        project_name or original_stem
    )

    upload_directory = get_upload_directory()
    workflow_upload_directory = upload_directory / workflow_id

    workflow_upload_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = workflow_upload_directory / f"input{extension}"

    total_size = 0

    try:
        with input_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file exceeds the maximum size limit.",
                    )

                destination.write(chunk)

    except Exception:
        input_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()

    if total_size == 0:
        input_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    input_type = "pdf" if extension == ".pdf" else "image"

    initial_state = {
        "workflow_id": workflow_id,
        "project_name": selected_project_name,
        "input_path": str(input_path.resolve()),
        "input_type": input_type,
        "user_query": user_query.strip(),
        "status": "pending",
        "current_agent": None,
        "next_agent": None,
        "completed_agents": [],
        "errors": [],
        "error_details": [],
        "evaluation_history": [],
        "quality_score": 0.0,
        "quality_threshold": 85.0,
        "is_approved": False,
        "refinement_count": 0,
        "max_refinement_iterations": 3,
        "retry_count": 0,
        "max_retries": 2,
        "execution_metrics": {},
    }

    WORKFLOW_REGISTRY[workflow_id] = {
        "workflow_id": workflow_id,
        "project_name": selected_project_name,
        "input_path": str(input_path.resolve()),
        "source_type": input_type,
        "original_filename": original_filename,
        "user_query": user_query.strip(),
        "status": "pending",
        "error": None,
    }

    background_tasks.add_task(
        execute_workflow,
        workflow_id,
        initial_state,
    )

    logger.info(
        "Workflow accepted | ID: %s | Project: %s | Type: %s",
        workflow_id,
        selected_project_name,
        input_type,
    )

    return {
        "success": True,
        "message": "File uploaded successfully. Workflow accepted.",
        "workflow_id": workflow_id,
        "project_name": selected_project_name,
        "source_type": input_type,
        "status": "pending",
        "input_filename": original_filename,
        "file_size": total_size,
        "user_query": user_query.strip(),
    }


# =========================================================
# GET ALL WORKFLOWS
# =========================================================

@router.get("")
async def list_workflows():
    workflows = list(WORKFLOW_REGISTRY.values())

    return jsonable_encoder({
        "success": True,
        "total": len(workflows),
        "workflows": workflows,
    })


# =========================================================
# GET WORKFLOW STATUS
# =========================================================

@router.get("/{workflow_id}")
async def get_workflow_status(
    workflow_id: str,
    state_reader=Depends(get_workflow_state_reader),
):
    workflow_id = validate_workflow_id(workflow_id)

    registry_item = WORKFLOW_REGISTRY.get(workflow_id)

    if registry_item is None:
        raise HTTPException(
            status_code=404,
            detail="Workflow not found.",
        )

    response = {
        "success": True,
        "workflow_id": workflow_id,
        "project_name": registry_item["project_name"],
        "status": registry_item["status"],
        "source_type": registry_item["source_type"],
        "original_filename": registry_item["original_filename"],
        "error": registry_item["error"],
        "current_agent": None,
        "next_agent": None,
        "completed_agents": [],
        "quality_score": None,
        "refinement_count": None,
    }

    try:
        snapshot = await read_state(state_reader, workflow_id)
        values = snapshot_to_dict(snapshot)

        if values:
            response.update({
                "current_agent": values.get("current_agent"),
                "next_agent": values.get("next_agent"),
                "completed_agents": values.get("completed_agents", []),
                "quality_score": values.get("quality_score"),
                "refinement_count": values.get("refinement_count"),
            })

    except Exception:
        logger.debug(
            "Checkpoint not available yet | ID: %s",
            workflow_id,
            exc_info=True,
        )

    return jsonable_encoder(response)


# =========================================================
# GET WORKFLOW CHECKPOINT
# =========================================================

@router.get("/{workflow_id}/state")
async def get_workflow_checkpoint(
    workflow_id: str,
    state_reader=Depends(get_workflow_state_reader),
):
    workflow_id = validate_workflow_id(workflow_id)

    if workflow_id not in WORKFLOW_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail="Workflow not found.",
        )

    try:
        snapshot = await read_state(state_reader, workflow_id)
        values = snapshot_to_dict(snapshot)

        return jsonable_encoder({
            "success": True,
            "workflow_id": workflow_id,
            "message": (
                None if values else "No checkpoint state is available yet."
            ),
            "state": values,
            "next": getattr(snapshot, "next", None),
        })

    except Exception as exc:
        logger.exception(
            "Unable to retrieve workflow checkpoint | ID: %s",
            workflow_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve workflow checkpoint.",
        ) from exc


# =========================================================
# GET FINAL WORKFLOW RESULT
# =========================================================

@router.get("/{workflow_id}/result")
async def get_workflow_result(
    workflow_id: str,
    state_reader=Depends(get_workflow_state_reader),
):
    workflow_id = validate_workflow_id(workflow_id)

    registry_item = WORKFLOW_REGISTRY.get(workflow_id)

    if registry_item is None:
        raise HTTPException(
            status_code=404,
            detail="Workflow not found.",
        )

    if registry_item["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=(
                registry_item["error"]
                or "Workflow execution failed."
            ),
        )

    if registry_item["status"] not in {
        "completed",
        "completed_with_warnings",
    }:
        return {
            "success": False,
            "workflow_id": workflow_id,
            "status": registry_item["status"],
            "message": "Workflow has not completed yet.",
        }

    try:
        snapshot = await read_state(state_reader, workflow_id)
        values = snapshot_to_dict(snapshot)

        return jsonable_encoder({
            "success": True,
            "workflow_id": workflow_id,
            "project_name": registry_item["project_name"],
            "status": registry_item["status"],
            "quality_score": values.get("quality_score"),
            "evaluation_history": values.get(
                "evaluation_history",
                [],
            ),
            "final_output": values.get("final_output"),
        })

    except Exception as exc:
        logger.exception(
            "Unable to retrieve workflow result | ID: %s",
            workflow_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve workflow result.",
        ) from exc
