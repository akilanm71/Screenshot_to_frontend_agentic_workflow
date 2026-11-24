
import json
import logging
import os
import re

from pathlib import Path
from typing import Any, Optional

from ollama import Client
from pydantic import BaseModel, Field

from backend.config.settings import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================
# 1. PYDANTIC OUTPUT SCHEMAS
# ==================================================

class ComponentPlan(BaseModel):

    component_name: str
    component_type: str

    description: str

    content: Optional[str] = None

    layout_instructions: Optional[str] = None

    styling_instructions: Optional[str] = None

    props: dict[str, Any] = Field(
        default_factory=dict
    )

    source_reference: Optional[str] = None


class SectionPlan(BaseModel):

    section_name: str

    description: str

    layout_type: str

    components: list[ComponentPlan] = Field(
        default_factory=list
    )


class PagePlan(BaseModel):

    page_number: int

    page_title: str

    page_description: str

    layout_type: str

    sections: list[SectionPlan] = Field(
        default_factory=list
    )


class FrontendPlan(BaseModel):

    project_name: str

    source_type: str

    application_description: str

    frontend_framework: str = "React"

    styling_approach: str = "CSS"

    global_design_instructions: list[str] = Field(
        default_factory=list
    )

    reusable_components: list[str] = Field(
        default_factory=list
    )

    pages: list[PagePlan] = Field(
        default_factory=list
    )

    responsive_design_instructions: list[str] = Field(
        default_factory=list
    )

    implementation_notes: list[str] = Field(
        default_factory=list
    )


# ==================================================
# 2. PLANNING AGENT
# ==================================================

class PlanningAgent:

    def __init__(self):

        self.model = settings.llm_model

        self.client = Client(
            host=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434"
            )
        )

    # ------------------------------------------------
    # Validate input
    # ------------------------------------------------

    def _validate_input(
        self,
        source_type: str,
        source_data: dict
    ):

        supported_types = [
            "screenshot",
            "pdf"
        ]

        if source_type not in supported_types:

            raise ValueError(
                "source_type must be 'screenshot' or 'pdf'."
            )

        if not isinstance(source_data, dict):

            raise TypeError(
                "source_data must be a dictionary."
            )

        if not source_data:

            raise ValueError(
                "source_data cannot be empty."
            )

    # ------------------------------------------------
    # Build planning prompts
    # ------------------------------------------------

    def _build_prompt(
        self,
        source_type: str,
        source_data: dict
    ):

        system_prompt = """
You are an expert frontend architect and UI planning agent.

Your responsibility is to convert screenshot analysis or PDF
extraction data into a structured React frontend implementation plan.

You must return data matching the supplied Pydantic JSON schema.

GENERAL RULES:

1. Do not generate React, JavaScript, JSX or CSS source code.
2. Create a clear page, section and component hierarchy.
3. Identify reusable React components.
4. Preserve all important source content.
5. Never invent missing business information or values.
6. Include layout and styling instructions.
7. Include responsive design instructions.
8. Keep the plan practical for a React developer.
9. Use semantic component types and meaningful component names.
10. Put structured component data inside the props field.
11. Keep the original source order wherever possible.
12. Do not omit content merely to simplify the UI.

PDF-SPECIFIC RULES:

1. Organize content according to original PDF page numbers.
2. Preserve headings, paragraphs and bullet points.
3. Preserve every table row, column and cell value exactly.
4. Store complete table data inside component props.
5. Represent tables as structured rows and columns.
6. Do not convert tables into plain paragraphs.
7. Preserve image file paths exactly as provided.
8. Include image paths inside component props.
9. Identify the source PDF page using source_reference.
10. Use page dimensions when they are available.
11. Do not invent images, charts or diagrams that were not detected.

SCREENSHOT-SPECIFIC RULES:

1. Reconstruct the visible layout.
2. Use supplied visual positions and component descriptions.
3. Preserve component coordinates and dimensions.
4. Include background colors and text colors.
5. Include font sizes, font weights and border radius.
6. Include spacing and alignment instructions.
7. Preserve the visual hierarchy.
8. Do not invent visual properties that are not supplied.
9. Use source_reference to identify the original screenshot
   and its corresponding visual section.

COMPONENT RULES:

For each component, provide:

- component_name
- component_type
- description
- content, when applicable
- layout_instructions
- styling_instructions
- props
- source_reference

The props field should contain the actual data needed
by the React component.

For example, a table component should contain its complete
headers and rows in props, not just a description of the table.

Return only structured JSON matching the requested schema.
"""

        user_prompt = f"""
Create a complete frontend implementation plan.

SOURCE TYPE:
{source_type}

SOURCE DATA:
{json.dumps(source_data, ensure_ascii=False, indent=2)}

PROJECT REQUIREMENTS:

1. Create a meaningful project_name.
2. Set frontend_framework to React.
3. Set styling_approach to CSS.
4. Describe the application.
5. Identify global design instructions.
6. Identify reusable components.
7. Create pages in the original source order.
8. Organize each page into meaningful sections.
9. Create component plans for every important UI element.
10. Include actual component data in props.
11. Include responsive design instructions.
12. Include implementation notes.

For PDF input:

- Preserve the original page order.
- Use page_number from the extraction result.
- Preserve headings and paragraphs.
- Convert extracted tables into structured table components.
- Include complete table rows in component props.
- Include image paths when images are available.
- Use page-specific source references.

For screenshot input:

- Use the supplied visual analysis.
- Reconstruct the visible interface.
- Include component positioning and styling.
- Preserve the original visual hierarchy.

The generated plan must contain sufficient detail for a
separate React code-generation agent to implement the frontend
without guessing important content or layout details.
"""

        return system_prompt, user_prompt

    # ------------------------------------------------
    # Generate structured frontend plan
    # ------------------------------------------------

    def create_plan(
        self,
        source_type: str,
        source_data: dict,
        output_path: Optional[str] = None
    ) -> FrontendPlan:

        self._validate_input(
            source_type,
            source_data
        )

        system_prompt, user_prompt = self._build_prompt(
            source_type,
            source_data
        )

        logger.info("Planning Agent started")
        logger.info("Input type: %s", source_type)
        logger.info("Using model: %s", self.model)

        response = self.client.chat(

            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            format=FrontendPlan.model_json_schema(),

            options={
                "temperature": 0,
                "num_ctx": 8192
            }
        )

        # Extract Ollama response
        response_content = response.message.content

        if not response_content:
            raise ValueError(
                "Planning model returned an empty response."
            )

        # Validate JSON against Pydantic schema
        plan = FrontendPlan.model_validate_json(
            response_content
        )

        if not plan.pages:
            raise ValueError(
                "Planning Agent did not generate any pages."
            )

        # Save generated plan
        if output_path:

            output_file = Path(output_path)

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            output_file.write_text(
                plan.model_dump_json(indent=4),
                encoding="utf-8"
            )

            logger.info(
                "Frontend plan saved: %s",
                output_file
            )

        logger.info(
            "Planning completed successfully"
        )

        return plan


# ==================================================
# 3. LOCAL TEST EXECUTION
# ==================================================

if __name__ == "__main__":

    extraction_file = Path(
        "generated_projects/pdf_extraction/sample/"
        "extraction_result.json"
    )

    if not extraction_file.exists():

        raise FileNotFoundError(
            f"Extraction result not found: {extraction_file}"
        )

    # Load PDF extraction result
    with extraction_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        extraction_data = json.load(file)

    # Initialize Planning Agent
    agent = PlanningAgent()

    # Generate frontend plan
    result = agent.create_plan(

        source_type="pdf",

        source_data=extraction_data,

        output_path=(
            "generated_projects/planning/sample/"
            "frontend_plan.json"
        )
    )

    print("\n========== FRONTEND PLAN ==========")

    print(
        result.model_dump_json(indent=4)
    )

    print("\n========== SUMMARY ==========")

    print("Project:", result.project_name)
    print("Source:", result.source_type)
    print("Pages:", len(result.pages))

    print(
        "Total sections:",
        sum(
            len(page.sections)
            for page in result.pages
        )
    )

    print(
        "Total components:",
        sum(
            len(section.components)
            for page in result.pages
            for section in page.sections
        )
    )
