
from backend.observability.metrics_collector import metrics_collector

# 1. Simulate an LLM call
metrics_collector.record_llm_metrics(
    agent_name="planning_agent",
    model_name="qwen2.5vl:3b",
    usage={
        "input_tokens": 30,
        "output_tokens": 21,
        "total_tokens": 51
    },
    metadata={
        "total_duration": 8621725600,
        "load_duration": 8032394500,
        "prompt_eval_duration": 177015000,
        "eval_duration": 401240000
    },
    workflow_id="test_workflow_001"
)

# 2. Simulate agent execution
metrics_collector.record_agent_metrics(
    agent_name="planning_agent",
    duration_ms=4500,
    status="success",
    workflow_id="test_workflow_001"
)

# 3. Simulate workflow execution
metrics_collector.record_workflow_metrics(
    workflow_id="test_workflow_001",
    duration_ms=12000,
    status="success",
    total_agents=3,
    refinement_count=1
)

print("All metrics tests completed.")
