import json
import logging
from pathlib import Path

from backend.agents.code_agent import CodeGenerationAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

PLAN_PATH = Path(
    "generated_projects/novadash-project-overview/frontend_plan.json"
)

OUTPUT_DIRECTORY = "generated_projects/code_generation_test"

WORKFLOW_ID = "novadash-codegen-isolated-001"


def main():
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(f"Plan not found: {PLAN_PATH}")

    with PLAN_PATH.open("r", encoding="utf-8") as file:
        frontend_plan = json.load(file)

    print("\n========== CODE GENERATION TEST ==========")
    print("Project:", frontend_plan.get("project_name"))
    print("Source type:", frontend_plan.get("source_type"))
    print("Output directory:", OUTPUT_DIRECTORY)
    print("Workflow ID:", WORKFLOW_ID)

    agent = CodeGenerationAgent(
        output_directory=OUTPUT_DIRECTORY,
        max_generation_attempts=2,
        max_llm_attempts=2,
        build_timeout=180,
    )

    result = agent.generate(
        frontend_plan=frontend_plan,
        workflow_id=WORKFLOW_ID,
    )

    print("\n========== GENERATION RESULT ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    status = result.get("status", "unknown")
    print("\nFinal status:", status)

    if status != "success":
        print("Generation did not complete successfully.")
        return

    print("\nCode generation completed successfully.")


if __name__ == "__main__":
    main()
