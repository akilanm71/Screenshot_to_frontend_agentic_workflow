
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:

    def __init__(self, output_dir: str = "generated_projects/observability"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.output_dir / "metrics.jsonl"

        self._lock = threading.Lock()

    @staticmethod
    def timestamp():
        return datetime.now(timezone.utc).isoformat()

    def record(self, event_type: str, data: Dict[str, Any]):
        event = {
            "event_type": event_type,
            "timestamp": self.timestamp(),
            **data
        }

        with self._lock:
            with open(self.metrics_file, "a", encoding="utf-8") as file:
                file.write(json.dumps(event, default=str) + "\n")

        logger.info("Recorded metric: %s", event_type)

    def record_llm_metrics(
        self,
        agent_name: str,
        model_name: str,
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None
    ):
        usage = usage or {}
        metadata = metadata or {}

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        eval_duration = metadata.get("eval_duration", 0)
        prompt_duration = metadata.get("prompt_eval_duration", 0)
        total_duration = metadata.get("total_duration", 0)
        load_duration = metadata.get("load_duration", 0)

        generation_seconds = eval_duration / 1_000_000_000

        tokens_per_second = (
            output_tokens / generation_seconds
            if generation_seconds > 0 else None
        )

        self.record("llm_call", {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get(
                "total_tokens",
                input_tokens + output_tokens
            ),
            "total_duration_ms": total_duration / 1_000_000,
            "load_duration_ms": load_duration / 1_000_000,
            "prompt_eval_duration_ms": prompt_duration / 1_000_000,
            "generation_duration_ms": eval_duration / 1_000_000,
            "tokens_per_second": tokens_per_second
        })

    def record_agent_metrics(
        self,
        agent_name: str,
        duration_ms: float,
        status: str,
        workflow_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        self.record("agent_execution", {
            "workflow_id": workflow_id,
            "agent_name": agent_name,
            "duration_ms": duration_ms,
            "status": status,
            "error": error
        })

    def record_workflow_metrics(
        self,
        workflow_id: str,
        duration_ms: float,
        status: str,
        total_agents: int = 0,
        refinement_count: int = 0
    ):
        self.record("workflow_execution", {
            "workflow_id": workflow_id,
            "duration_ms": duration_ms,
            "status": status,
            "total_agents": total_agents,
            "refinement_count": refinement_count
        })


metrics_collector = MetricsCollector()
