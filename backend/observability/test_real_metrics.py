
from backend.observability.llm_factory import (
    create_llm,
    invoke_llm
)

llm = create_llm(
    agent_name="metrics_test_agent"
)

response = invoke_llm(
    llm=llm,
    prompt="Explain what LangGraph is in two sentences.",
    agent_name="metrics_test_agent",
    workflow_id="workflow_metrics_001"
)

print("\nLLM RESPONSE:")
print(response.content)

print("\nTOKEN USAGE:")
print(response.usage_metadata)

print("\nMODEL METADATA:")
print(response.response_metadata)

print("\nMetrics recorded successfully.")
