from backend.observability.llm_factory import create_llm


llm = create_llm(
    agent_name="test_agent"
)

response = llm.invoke(
    "Explain what a React component is in one sentence.",
    config={
        "run_name": "test_agent_llm_call",
        "tags": [
            "test",
            "screenshot-to-frontend"
        ],
        "metadata": {
            "agent_name": "test_agent",
            "workflow_id": "test_workflow_001"
        }
    }
)

print("\nLLM RESPONSE:")
print(response.content)

print("\nTOKEN USAGE:")
print(response.usage_metadata)

print("\nMODEL METADATA:")
print(response.response_metadata)