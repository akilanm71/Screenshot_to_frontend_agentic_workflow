
import logging

from fastapi import APIRouter

router = APIRouter(tags=["Health"])

logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    return {
        "success": True,
        "status": "healthy",
        "service": "screenshot-to-frontend-api",
    }


@router.get("/ready")
async def readiness_check():
    return {
        "success": True,
        "status": "ready",
        "services": {
            "api": "available",
            "workflow": "configured",
        },
    }
