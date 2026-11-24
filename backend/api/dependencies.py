
import logging
import os

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from fastapi import HTTPException, Request, status

from backend.workflow.workflow import (
    invoke_workflow,
    get_workflow_state,
)


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATED_PROJECTS_DIR = (
    PROJECT_ROOT / "generated_projects"
)

UPLOAD_DIRECTORY = (
    PROJECT_ROOT / "uploads"
)


# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

try:
    MAX_UPLOAD_SIZE_MB = int(
        os.getenv("MAX_UPLOAD_SIZE_MB", "20")
    )
except ValueError:
    raise RuntimeError(
        "MAX_UPLOAD_SIZE_MB must be a valid integer."
    )

if MAX_UPLOAD_SIZE_MB <= 0:
    raise RuntimeError(
        "MAX_UPLOAD_SIZE_MB must be greater than zero."
    )

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)

ALLOWED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".pdf",
}


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

@lru_cache(maxsize=1)
def get_app_settings() -> dict:
    """
    Return cached application configuration.
    """

    return {
        "app_name": os.getenv(
            "APP_NAME",
            "Screenshot to Frontend Agentic AI",
        ),

        "environment": os.getenv(
            "APP_ENV",
            "development",
        ),

        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,

        "max_upload_size_bytes": MAX_UPLOAD_SIZE_BYTES,

        "allowed_extensions": sorted(
            ALLOWED_EXTENSIONS
        ),

        "upload_directory": str(
            UPLOAD_DIRECTORY
        ),

        "generated_projects_directory": str(
            GENERATED_PROJECTS_DIR
        ),
    }


# =========================================================
# WORKFLOW DEPENDENCIES
# =========================================================

def get_workflow_runner():
    """
    Return the existing compiled LangGraph workflow runner.

    The workflow is invoked by the background execution
    function in workflows.py.
    """

    if not callable(invoke_workflow):

        logger.error(
            "LangGraph workflow runner is not callable."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow service is unavailable.",
        )

    return invoke_workflow


def get_workflow_state_reader():
    """
    Return the LangGraph checkpoint state reader.
    """

    if not callable(get_workflow_state):

        logger.error(
            "Workflow state reader is not callable."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow state service is unavailable.",
        )

    return get_workflow_state


# =========================================================
# DIRECTORY DEPENDENCIES
# =========================================================

def get_upload_directory() -> Path:
    """
    Create and return the upload directory.
    """

    try:

        UPLOAD_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        return UPLOAD_DIRECTORY

    except OSError as exc:

        logger.exception(
            "Unable to initialize upload directory."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to access upload directory.",
        ) from exc


def get_generated_projects_directory() -> Path:
    """
    Create and return the generated projects directory.
    """

    try:

        GENERATED_PROJECTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        return GENERATED_PROJECTS_DIR

    except OSError as exc:

        logger.exception(
            "Unable to initialize generated projects directory."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to access generated projects directory.",
        ) from exc


# =========================================================
# WORKFLOW ID VALIDATION
# =========================================================

def validate_workflow_id(workflow_id: str) -> str:
    """
    Validate workflow ID.

    Supports UUIDs and safe alphanumeric identifiers
    containing hyphens and underscores.
    """

    if not isinstance(workflow_id, str):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow ID must be a string.",
        )

    workflow_id = workflow_id.strip()

    if not workflow_id or len(workflow_id) > 128:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow ID.",
        )

    allowed = all(
        character.isalnum() or character in "-_"
        for character in workflow_id
    )

    if not allowed:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow ID contains invalid characters.",
        )

    return workflow_id


# =========================================================
# FILE EXTENSION VALIDATION
# =========================================================

def validate_file_extension(filename: str) -> str:
    """
    Validate uploaded file extension.

    Supported:
        PNG
        JPG
        JPEG
        WEBP
        BMP
        PDF
    """

    if not filename or not filename.strip():

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )

    # Normalize Windows and Linux path separators.
    safe_filename = filename.replace("\\", "/")

    filename_only = Path(safe_filename).name

    extension = Path(
        filename_only
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported file type. "
                "Allowed: PNG, JPG, JPEG, WEBP, BMP and PDF."
            ),
        )

    return extension


# =========================================================
# UPLOAD SIZE VALIDATION
# =========================================================

def validate_upload_size(file_size: int) -> int:
    """
    Validate uploaded file size.

    Returns the validated file size.
    """

    if file_size <= 0:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if file_size > MAX_UPLOAD_SIZE_BYTES:

        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Maximum upload size is "
                f"{MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    return file_size


# =========================================================
# INPUT TYPE DETECTION
# =========================================================

def detect_input_type(filename: str) -> str:
    """
    Identify whether the uploaded file is an image or PDF.
    """

    extension = validate_file_extension(filename)

    if extension == ".pdf":
        return "pdf"

    return "image"


# =========================================================
# FASTAPI REQUEST DEPENDENCIES
# =========================================================

def get_request_id(request: Request) -> str:
    """
    Return the request tracking ID.
    """

    return getattr(
        request.state,
        "request_id",
        request.headers.get(
            "X-Request-ID",
            "unknown",
        ),
    )


def get_logger() -> logging.Logger:
    """
    Return the application logger.
    """

    return logger


# =========================================================
# STARTUP DIRECTORY INITIALIZATION
# =========================================================

def initialize_api_directories() -> None:
    """
    Initialize all required API directories during startup.
    """

    upload_directory = get_upload_directory()

    generated_projects_directory = (
        get_generated_projects_directory()
    )

    logger.info(
        (
            "API directories initialized | "
            "Uploads: %s | Projects: %s"
        ),
        upload_directory,
        generated_projects_directory,
    )


# =========================================================
# MODULE INITIALIZATION LOG
# =========================================================

logger.info(
    "API dependencies loaded | Project root: %s",
    PROJECT_ROOT,
)
