
from pathlib import Path

from backend.workflow.workflow import (
    invoke_workflow,
    get_workflow_state,
)

# Actual input image path
INPUT_IMAGE = Path(
    r"D:\screenshot-to-frontend\generated_projects\sample-dashboard.png"
)

# Validate the input before starting LangGraph
if not INPUT_IMAGE.is_file():
    raise FileNotFoundError(
        f"Screenshot does not exist: {INPUT_IMAGE}"
    )

initial_state = {
    "workflow_id": "novadash-workflow-002",
    "project_name": "NovaDash Project Overview",
    "input_path": str(INPUT_IMAGE),
    "status": "started",
}

print("Input image:", INPUT_IMAGE)
print("Starting workflow...")

result = invoke_workflow(initial_state)

print("\n========== WORKFLOW RESULT ==========")
print(result)

print("\n========== CHECKPOINT STATE ==========")

checkpoint = get_workflow_state(
    "novadash-workflow-002"
)

print(checkpoint)
