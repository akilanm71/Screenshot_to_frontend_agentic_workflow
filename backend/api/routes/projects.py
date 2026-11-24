
import logging
import re

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from backend.api.dependencies import (
    get_generated_projects_directory,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects",
    tags=["Projects"],
)


# =========================================================
# PROJECT UTILITIES
# =========================================================

def validate_project_id(project_id: str) -> str:
    """
    Validate project ID to prevent path traversal.
    """

    if (
        not project_id
        or len(project_id) > 100
        or not re.fullmatch(
            r"[a-zA-Z0-9_-]+",
            project_id,
        )
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID.",
        )

    return project_id


def resolve_project_directory(project_id: str) -> Path:
    """
    Resolve generated project directory safely.

    Supported structures:

        generated_projects/project-name/
        generated_projects/project-name/frontend/
    """

    project_id = validate_project_id(project_id)

    projects_root = (
        get_generated_projects_directory().resolve()
    )

    project_directory = (
        projects_root / project_id
    ).resolve()

    if not project_directory.is_relative_to(projects_root):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project directory.",
        )

    if not project_directory.is_dir():

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    # Structure 1: package.json directly in project folder.

    if (
        project_directory / "package.json"
    ).is_file():

        return project_directory

    # Structure 2: React frontend subdirectory.

    frontend_directory = (
        project_directory / "frontend"
    ).resolve()

    if not frontend_directory.is_relative_to(project_directory):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid frontend directory.",
        )

    if (
        frontend_directory / "package.json"
    ).is_file():

        return frontend_directory

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="React frontend project not found.",
    )


def get_project_metadata(
    project_directory: Path,
) -> dict:
    """
    Return common project metadata.
    """

    dist_directory = (
        project_directory / "dist"
    )

    source_directory = (
        project_directory / "src"
    )

    return {
        "frontend_directory": str(
            project_directory.resolve()
        ),

        "source_directory": str(
            source_directory.resolve()
        ),

        "build_directory": str(
            dist_directory.resolve()
        ),

        "package_json_exists": (
            project_directory / "package.json"
        ).is_file(),

        "source_directory_exists": (
            source_directory.is_dir()
        ),

        "build_available": (
            dist_directory / "index.html"
        ).is_file(),
    }


# =========================================================
# LIST PROJECTS
# =========================================================

@router.get(
    "",
    summary="List generated React projects",
)
async def list_projects():
    """
    Return all generated React projects.

    React dashboard can use this endpoint
    to display generated projects.
    """

    projects_root = (
        get_generated_projects_directory()
    )

    projects = []

    for directory in sorted(
        projects_root.iterdir(),
        key=lambda item: item.name.lower(),
    ):

        if not directory.is_dir():
            continue

        # Ignore directories without React package.json.

        if (
            directory / "package.json"
        ).is_file():

            frontend_directory = directory

        elif (
            directory / "frontend" / "package.json"
        ).is_file():

            frontend_directory = (
                directory / "frontend"
            )

        else:
            continue

        metadata = get_project_metadata(
            frontend_directory
        )

        projects.append({
            "project_id": directory.name,
            "project_name": directory.name,
            **metadata,
        })

    return {
        "success": True,
        "total": len(projects),
        "projects": projects,
    }


# =========================================================
# GET PROJECT DETAILS
# =========================================================

@router.get(
    "/{project_id}",
    summary="Get generated project details",
)
async def get_project(project_id: str):

    frontend_directory = (
        resolve_project_directory(project_id)
    )

    return {
        "success": True,
        "project_id": project_id,
        "project_name": project_id,
        **get_project_metadata(frontend_directory),
    }


# =========================================================
# LIST PRODUCTION BUILD FILES
# =========================================================

@router.get(
    "/{project_id}/files",
    summary="List production build files",
)
async def list_project_files(project_id: str):

    frontend_directory = (
        resolve_project_directory(project_id)
    )

    dist_directory = (
        frontend_directory / "dist"
    ).resolve()

    if not dist_directory.is_relative_to(
        frontend_directory.resolve()
    ):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid build directory.",
        )

    if not dist_directory.is_dir():

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Production build not found.",
        )

    files = []

    for file_path in sorted(
        dist_directory.rglob("*"),
        key=lambda item: str(item).lower(),
    ):

        if not file_path.is_file():
            continue

        # Do not expose files through symbolic links
        # that resolve outside the build directory.

        resolved_file = file_path.resolve()

        if not resolved_file.is_relative_to(dist_directory):
            continue

        files.append({
            "filename": file_path.name,

            "relative_path": str(
                file_path.relative_to(dist_directory)
            ).replace("\\", "/"),

            "size_bytes": file_path.stat().st_size,
        })

    return {
        "success": True,
        "project_id": project_id,
        "total_files": len(files),
        "files": files,
    }


# =========================================================
# PROJECT BUILD STATUS
# =========================================================

@router.get(
    "/{project_id}/build-status",
    summary="Get project production build status",
)
async def get_project_build_status(project_id: str):

    frontend_directory = (
        resolve_project_directory(project_id)
    )

    dist_directory = (
        frontend_directory / "dist"
    )

    index_file = (
        dist_directory / "index.html"
    )

    build_available = index_file.is_file()

    return {
        "success": True,
        "project_id": project_id,
        "build_available": build_available,

        "build_directory": str(
            dist_directory.resolve()
        ),

        "entry_file": (
            str(index_file.resolve())
            if build_available
            else None
        ),

        "message": (
            "Production build is available."
            if build_available
            else "Production build has not been generated."
        ),
    }
