
import logging
import os
import time
import uuid

from contextlib import asynccontextmanager

from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.dependencies import initialize_api_directories


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

load_dotenv()

APP_NAME = os.getenv(
    "APP_NAME",
    "Screenshot to Frontend Agentic AI",
)

APP_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

APP_ENV = os.getenv("APP_ENV", "development")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:3000,"
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        DEFAULT_FRONTEND_ORIGINS,
    ).split(",")
    if origin.strip()
]


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(__name__)


# =========================================================
# APPLICATION LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("=" * 60)
    logger.info("Starting %s", APP_NAME)
    logger.info("Version: %s", APP_VERSION)
    logger.info("Environment: %s", APP_ENV)

    try:
        initialize_api_directories()
        logger.info("API directories initialized successfully")

    except Exception:
        logger.exception("API startup initialization failed")
        raise

    logger.info("Application startup completed")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down %s", APP_NAME)
    logger.info("Application shutdown completed")


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title=APP_NAME,
    description=(
        "Agentic AI platform that converts screenshots "
        "and PDF designs into functional React applications "
        "using LangGraph, LangChain and Ollama."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =========================================================
# CORS MIDDLEWARE - REACT FRONTEND
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Request-ID",
        "X-Process-Time",
    ],
)


# =========================================================
# REQUEST TRACKING MIDDLEWARE
# =========================================================

@app.middleware("http")
async def request_tracking_middleware(
    request: Request,
    call_next,
):

    request_id = (
    request.headers.get("X-Request-ID")
    or str(uuid.uuid4())
)

    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        process_time = time.perf_counter() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"

        logger.info(
            "Request completed | ID: %s | Method: %s | "
            "Path: %s | Status: %s | Duration: %.4fs",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            process_time,
        )

        return response

    except Exception:
        logger.exception(
            "Request failed | ID: %s | Path: %s",
            request_id,
            request.url.path,
        )
        raise


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/", tags=["Root"])
async def root():

    return {
        "success": True,
        "application": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "environment": APP_ENV,
        "documentation": "/docs",
        "health": "/health",
        "frontend_origins": FRONTEND_ORIGINS,
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health", tags=["Health"])
async def health_check():

    return {
        "success": True,
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION,
    }


# =========================================================
# READINESS CHECK
# =========================================================

@app.get("/ready", tags=["Health"])
async def readiness_check():

    return {
        "success": True,
        "status": "ready",
        "workflow_engine": "LangGraph",
    }


# =========================================================
# API INFORMATION
# =========================================================

@app.get(f"{API_PREFIX}/info", tags=["Information"])
async def api_information():

    return {
        "success": True,
        "application": APP_NAME,
        "version": APP_VERSION,
        "environment": APP_ENV,
        "api_prefix": API_PREFIX,
        "components": {
            "api_framework": "FastAPI",
            "workflow_engine": "LangGraph",
            "agent_framework": "LangChain",
            "llm_provider": "Ollama",
            "frontend": "React",
        },
        "endpoints": {
            "health": "/health",
            "information": f"{API_PREFIX}/info",
            "workflows": f"{API_PREFIX}/workflows",
            "projects": f"{API_PREFIX}/projects",
        },
    }


# =========================================================
# API ROUTER REGISTRATION
# =========================================================

def register_routers():

    from backend.api.routes import workflows
    from backend.api.routes import projects

    app.include_router(
        workflows.router,
        prefix=API_PREFIX,
    )

    app.include_router(
        projects.router,
        prefix=API_PREFIX,
    )

    logger.info("API routers registered successfully")


register_routers()


# =========================================================
# GLOBAL HTTP EXCEPTION HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    logger.exception(
        "Unhandled exception | Request ID: %s",
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "internal_server_error",
            "message": (
                "An unexpected error occurred "
                "while processing the request."
            ),
            "request_id": request_id,
        },
        headers={
            "X-Request-ID": request_id,
        },
    )
