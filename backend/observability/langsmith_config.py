import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def configure_langsmith():
    load_dotenv()

    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT")

    if not api_key or not project:
        raise EnvironmentError(
            "LANGSMITH_API_KEY or LANGSMITH_PROJECT missing in .env"
        )

    os.environ["LANGSMITH_TRACING"] = "true"

    logger.info("LangSmith tracing enabled: %s", project)

    return {
        "tracing_enabled": True,
        "project": project
    }