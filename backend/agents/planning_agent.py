
import json
import logging
import re

from pathlib import Path
from typing import Any, Optional
from uuid import uuid4
from time import perf_counter

from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.observability.llm_factory import (
    create_llm,
    invoke_llm
)
from backend.observability.metrics_collector import (
    metrics_collector
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# 1. PYDANTIC OUTPUT SCHEMAS
# =========================================================

class ComponentPlan(BaseModel):
    component_name: str
    component_type: str
    description: str

    content: Optional[str] = None
    layout_instructions: Optional[str] = None
    styling_instructions: Optional[str] = None

    props: dict[str, Any] = Field(default_factory=dict)

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


# =========================================================
# 2. PLANNING AGENT
# =========================================================

class PlanningAgent:

    def __init__(self):

        self.agent_name = "planning_agent"
        self.model = settings.llm_model

        self.max_generation_attempts = 2

        self.llm = create_llm(
            agent_name=self.agent_name,
            temperature=0,
            num_ctx=8192,
            num_predict=8192,
            model=self.model
        )

        logger.info(
            "Planning Agent initialized | Model: %s",
            self.model
        )

    # =====================================================
    # 3. INPUT VALIDATION
    # =====================================================

    def _validate_input(
        self,
        source_type: str,
        source_data: dict
    ):

        if source_type not in ("screenshot", "pdf"):
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

    # =====================================================
    # 4. SOURCE DATA SERIALIZATION
    # =====================================================

    @staticmethod
    def _serialize_source_data(
        source_data: dict
    ) -> str:

        return json.dumps(
            source_data,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    # =====================================================
    # 5. PLANNING PROMPTS
    # =====================================================

    def _build_prompt(
        self,
        source_type: str,
        source_data: dict,
        repair_feedback: Optional[list[str]] = None
    ):

        system_prompt = """
You are a senior frontend architect and structured UI planning agent.

Your task is to analyze screenshot or PDF extraction data and
produce a complete structured React frontend implementation plan.

Return valid JSON matching the supplied Pydantic schema.

========================================================
GENERAL RULES
========================================================

1. Return structured JSON only.
2. Do not generate JSX, JavaScript, HTML or CSS source code.
3. Preserve all important source content.
4. Never invent business information, names, values or percentages.
5. Preserve original source ordering.
6. Organize content into meaningful pages and sections.
7. Use semantic and reusable React component names.
8. Put actual display data inside component props.
9. Keep technical implementation instructions out of visible content.
10. Every component must have a meaningful description.
11. Include source references whenever available.
12. Do not combine unrelated UI elements into one large text component.

========================================================
PDF CONTENT STRUCTURING
========================================================

A PDF may contain headings, paragraphs, tables, lists, metrics,
project information, images and other structured content.

Separate these elements into appropriate components.

Never put an entire PDF page or document into one TextSection.

Never represent a detected table as a plain paragraph.

Never omit a section because the extraction format contains text.

Use the following conventions:

PAGE HEADER:
component_name: PageHeader
component_type: Header
props:
{
    "title": "Page title",
    "subtitle": "Optional subtitle"
}

PARAGRAPH:
component_name: TextSection
component_type: Text
props:
{
    "text": "Actual paragraph content"
}

METRICS:
component_name: MetricGrid
component_type: Metrics
props:
{
    "items": [
        {
            "label": "Total Views",
            "value": "24,780",
            "change": "+12.5%"
        }
    ]
}

TABLE:
component_name: TableComponent
component_type: Table
props:
{
    "table": {
        "headers": ["Project", "Status", "Progress"],
        "rows": [
            ["Website", "In progress", "92%"]
        ]
    }
}

ACTIVITY LIST:
component_name: ActivityList
component_type: List
props:
{
    "items": [
        "A new project was created."
    ]
}

PROJECT PROGRESS:
component_name: ProjectProgress
component_type: Progress
props:
{
    "items": [
        {
            "name": "Website",
            "status": "In progress",
            "progress": 92
        }
    ]
}

IMAGE:
component_name: ImageComponent
component_type: Image
props:
{
    "src": "original-image-path",
    "alt": "Image description"
}

========================================================
SCREENSHOT RULES
========================================================

For screenshot input:

1. Reconstruct the visible layout.
2. Preserve visual hierarchy.
3. Use supplied coordinates and dimensions.
4. Preserve detected colors, typography and spacing.
5. Include responsive design instructions.
6. Do not invent visual details that are not supplied.
7. Create separate components for distinct visible elements.
8. Preserve detected navigation, headers, cards, charts and lists.
9. Preserve the ordering of detected sections.
10. Use the supplied layout information.

========================================================
COMPONENT QUALITY
========================================================

Every component should include:

- component_name
- component_type
- description
- content when applicable
- layout_instructions
- styling_instructions
- props
- source_reference when available

Component props must contain sufficient actual data for
a separate React Code Generation Agent to implement the UI.

Do not provide only descriptions when actual source data exists.

========================================================
FINAL VALIDATION
========================================================

Before returning JSON:

1. Generate at least one page.
2. Every page must have at least one section.
3. Every section must have at least one component.
4. Preserve important source content.
5. Keep headings separate from paragraphs.
6. Keep tables structured.
7. Keep metrics structured.
8. Keep activities as individual list items.
9. Keep progress values structured.
10. Do not invent missing source values.
11. Ensure component props contain usable data.

Return only JSON matching the requested schema.
"""

        user_prompt = f"""
Create a complete structured frontend implementation plan.

SOURCE TYPE:
{source_type}

SOURCE DATA:
{self._serialize_source_data(source_data)}

PROJECT REQUIREMENTS:

- Use React.
- Use CSS.
- Create at least one page.
- Organize content into meaningful sections.
- Preserve original source information.
- Preserve original content ordering.
- Use separate components for distinct UI elements.
- Include actual display data inside component props.
- Preserve table rows and columns.
- Include responsive design instructions.
- Keep implementation notes separate from visible content.

The plan must be sufficiently detailed for a separate
React Code Generation Agent to implement the UI without
guessing important source content.

Return a non-empty pages array.
"""

        if repair_feedback:

            user_prompt += """

========================================================
PLAN QUALITY REPAIR REQUIRED
========================================================

The previous response had the following problems:

"""

            for issue in repair_feedback:
                user_prompt += f"- {issue}\n"

            user_prompt += """
Regenerate the complete plan.

The pages array must not be empty.
Every page must contain sections.
Every section must contain components.

Return valid JSON only.
"""

        return system_prompt, user_prompt

    # =====================================================
    # 6. JSON RESPONSE PARSING
    # =====================================================

    @staticmethod
    def _parse_response(
        response
    ) -> FrontendPlan:

        response_content = getattr(
            response,
            "content",
            None
        )

        if isinstance(response_content, list):
            response_content = "".join(
                str(item.get("text", ""))
                if isinstance(item, dict)
                else str(item)
                for item in response_content
            )

        if not isinstance(response_content, str):
            raise TypeError(
                "Planning model returned an unexpected response format."
            )

        response_content = response_content.strip()

        if not response_content:
            raise ValueError(
                "Planning model returned an empty response."
            )

        # Remove Markdown code fences.
        response_content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            response_content,
            flags=re.IGNORECASE
        ).strip()

        # Extract the outermost JSON object if additional text exists.
        if not response_content.startswith("{"):

            start = response_content.find("{")
            end = response_content.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise ValueError(
                    "No JSON object found in planning response."
                )

            response_content = response_content[start:end + 1]

        parsed_json = json.loads(response_content)

        return FrontendPlan.model_validate(
            parsed_json
        )

    # =====================================================
    # 7. PLAN QUALITY VALIDATION
    # =====================================================

    @staticmethod
    def _validate_plan_quality(
        plan: FrontendPlan,
        source_type: str
    ) -> list[str]:

        issues = []

        if not plan.pages:
            issues.append(
                "No pages were generated."
            )

        total_components = 0

        for page in plan.pages:

            if not page.sections:
                issues.append(
                    f"Page '{page.page_title}' has no sections."
                )

            for section in page.sections:

                if not section.components:
                    issues.append(
                        f"Section '{section.section_name}' "
                        "has no components."
                    )

                for component in section.components:

                    total_components += 1

                    component_name = (
                        component.component_name.lower()
                    )

                    props = component.props or {}

                    if component_name == "tablecomponent":

                        table = props.get("table", {})

                        if not isinstance(table, dict):
                            issues.append(
                                f"TableComponent in "
                                f"'{section.section_name}' "
                                "does not contain a structured table."
                            )

                        elif (
                            not table.get("headers")
                            or not table.get("rows")
                        ):
                            issues.append(
                                f"TableComponent in "
                                f"'{section.section_name}' "
                                "has missing headers or rows."
                            )

                    if component_name in (
                        "metricgrid",
                        "activitylist",
                        "projectprogress"
                    ):

                        items = props.get("items")

                        if (
                            not isinstance(items, list)
                            or not items
                        ):
                            issues.append(
                                f"{component.component_name} in "
                                f"'{section.section_name}' "
                                "has no structured items."
                            )

                    if component_name == "textsection":

                        text = props.get("text", "")

                        if isinstance(text, str):

                            line_count = len([
                                line for line in text.splitlines()
                                if line.strip()
                            ])

                            heading_count = len(
                                re.findall(
                                    r"(?im)^\s*(?:\d+[.)]\s+|#{1,6}\s+)",
                                    text
                                )
                            )

                            looks_like_whole_page = (
                                len(text) > 500
                                and line_count >= 10
                                and heading_count >= 2
                            )

                            if looks_like_whole_page:
                                issues.append(
                                    f"TextSection in "
                                    f"'{section.section_name}' "
                                    "appears to contain multiple sections."
                                )

        if total_components == 0:
            issues.append(
                "The plan contains no frontend components."
            )

        return list(dict.fromkeys(issues))

    # =====================================================
    # 8. SOURCE DATA FALLBACK HELPERS
    # =====================================================

    @staticmethod
    def _find_vision_data(
        source_data: dict
    ) -> dict:

        """
        Locate vision data inside common workflow result structures.
        """

        possible_keys = (
            "vision_result",
            "vision_analysis",
            "visual_analysis",
            "vision_data"
        )

        for key in possible_keys:

            value = source_data.get(key)

            if isinstance(value, dict) and value:
                return value

        return source_data

    @staticmethod
    def _extract_component_details(
        item: Any
    ) -> ComponentPlan:

        if isinstance(item, str):

            return ComponentPlan(
                component_name="ContentCard",
                component_type="Content",
                description=item,
                content=item,
                props={
                    "content": item
                }
            )

        if not isinstance(item, dict):

            return ComponentPlan(
                component_name="ContentCard",
                component_type="Content",
                description=str(item),
                content=str(item)
            )

        name = (
            item.get("component_name")
            or item.get("name")
            or item.get("label")
            or item.get("type")
            or "ContentCard"
        )

        component_type = (
            item.get("component_type")
            or item.get("type")
            or "Content"
        )

        description = (
            item.get("description")
            or item.get("label")
            or name
        )

        props = item.get("props", {})

        if not isinstance(props, dict):
            props = {}

        if not props:

            excluded = {
                "component_name",
                "component_type",
                "name",
                "type",
                "description"
            }

            props = {
                key: value
                for key, value in item.items()
                if key not in excluded
            }

        return ComponentPlan(
            component_name=str(name),
            component_type=str(component_type),
            description=str(description),
            content=item.get("content"),
            layout_instructions=item.get(
                "layout_instructions"
            ),
            styling_instructions=item.get(
                "styling_instructions"
            ),
            props=props,
            source_reference=item.get(
                "source_reference"
            )
        )

    # =====================================================
    # 9. FALLBACK PLAN GENERATION
    # =====================================================

    @classmethod
    def _build_fallback_plan(
        cls,
        source_type: str,
        source_data: dict
    ) -> FrontendPlan:

        logger.warning(
            "Building fallback frontend plan from source data."
        )

        vision = cls._find_vision_data(
            source_data
        )

        sections_data = vision.get(
            "sections",
            []
        )

        if not isinstance(sections_data, list):
            sections_data = []

        planned_sections = []

        # -------------------------------------------------
        # Build sections from detected vision components.
        # -------------------------------------------------

        for index, section in enumerate(sections_data):

            if not isinstance(section, dict):
                continue

            section_name = (
                section.get("section_name")
                or section.get("name")
                or section.get("title")
                or f"Section {index + 1}"
            )

            description = (
                section.get("description")
                or section_name
            )

            raw_components = (
                section.get("components", [])
            )

            if not isinstance(raw_components, list):
                raw_components = []

            planned_components = [
                cls._extract_component_details(item)
                for item in raw_components
            ]

            # Preserve a section even when the vision model
            # identified its name but did not identify components.
            if not planned_components:

                planned_components.append(
                    ComponentPlan(
                        component_name="SectionContent",
                        component_type="Content",
                        description=str(description),
                        content=str(description),
                        props={
                            "title": str(section_name),
                            "description": str(description)
                        }
                    )
                )

            planned_sections.append(
                SectionPlan(
                    section_name=str(section_name),
                    description=str(description),
                    layout_type=str(
                        section.get("layout_type")
                        or section.get("layout")
                        or "responsive"
                    ),
                    components=planned_components
                )
            )

        # -------------------------------------------------
        # Fallback for PDF extraction structures.
        # -------------------------------------------------

        if not planned_sections:

            pages_data = source_data.get(
                "pages",
                []
            )

            if isinstance(pages_data, list):

                for page_index, page in enumerate(pages_data):

                    if not isinstance(page, dict):
                        continue

                    page_text = (
                        page.get("text")
                        or page.get("content")
                        or ""
                    )

                    if not page_text:
                        continue

                    page_number = page.get(
                        "page_number",
                        page_index + 1
                    )

                    planned_sections.append(
                        SectionPlan(
                            section_name=f"Page {page_number} Content",
                            description="Extracted PDF page content.",
                            layout_type="document",
                            components=[
                                ComponentPlan(
                                    component_name="TextSection",
                                    component_type="Text",
                                    description="Extracted page text.",
                                    content=str(page_text),
                                    props={
                                        "text": str(page_text)
                                    }
                                )
                            ]
                        )
                    )

        # -------------------------------------------------
        # Generic source fallback.
        # -------------------------------------------------

        if not planned_sections:

            title = (
                vision.get("page_title")
                or vision.get("title")
                or "Generated Dashboard"
            )

            description = (
                vision.get("overall_layout")
                or "Content reconstructed from source data."
            )

            planned_sections = [
                SectionPlan(
                    section_name="Main Content",
                    description=str(description),
                    layout_type="responsive",
                    components=[
                        ComponentPlan(
                            component_name="ContentCard",
                            component_type="Content",
                            description="Source information",
                            props={
                                "title": str(title),
                                "content": json.dumps(
                                    source_data,
                                    ensure_ascii=False,
                                    default=str
                                )
                            }
                        )
                    ]
                )
            ]

        project_name = (
            vision.get("project_name")
            or vision.get("page_title")
            or vision.get("title")
            or "Generated Frontend Project"
        )

        page_title = (
            vision.get("page_title")
            or vision.get("title")
            or project_name
        )

        layout_type = (
            vision.get("overall_layout")
            or "responsive"
        )

        return FrontendPlan(
            project_name=str(project_name),
            source_type=source_type,
            application_description=(
                f"Frontend application reconstructed from "
                f"{source_type} source data."
            ),
            frontend_framework="React",
            styling_approach="CSS",
            global_design_instructions=[
                "Preserve the source visual hierarchy.",
                "Use responsive layouts.",
                "Use reusable React components.",
                "Preserve source content and component ordering."
            ],
            reusable_components=[
                "Header",
                "ContentCard",
                "SectionContent"
            ],
            pages=[
                PagePlan(
                    page_number=1,
                    page_title=str(page_title),
                    page_description=(
                        "Reconstructed from available source analysis."
                    ),
                    layout_type=str(layout_type),
                    sections=planned_sections
                )
            ],
            responsive_design_instructions=[
                "Use responsive desktop, tablet and mobile layouts.",
                "Avoid horizontal overflow.",
                "Adapt grids and navigation for smaller screens."
            ],
            implementation_notes=[
                "Fallback plan generated from structured source data.",
                "The LLM did not return a usable non-empty page structure.",
                "Review visual details that were unavailable in source analysis."
            ]
        )

    # =====================================================
    # 10. GENERATE PLAN
    # =====================================================

    def create_plan(
        self,
        source_type: str,
        source_data: dict,
        output_path: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> FrontendPlan:

        if workflow_id is None:
            workflow_id = f"planning-{uuid4().hex[:12]}"

        agent_start_time = perf_counter()
        status = "failed"

        try:

            self._validate_input(
                source_type,
                source_data
            )

            logger.info(
                "Planning Agent started | Type: %s | Workflow: %s",
                source_type,
                workflow_id
            )

            structured_llm = self.llm.bind(
                format=FrontendPlan.model_json_schema()
            )

            repair_feedback = None
            plan = None
            validation_passed = False

            # -------------------------------------------------
            # LLM GENERATION AND REPAIR LOOP
            # -------------------------------------------------

            for attempt in range(
                1,
                self.max_generation_attempts + 1
            ):

                logger.info(
                    "Planning generation attempt %s/%s",
                    attempt,
                    self.max_generation_attempts
                )

                system_prompt, user_prompt = self._build_prompt(
                    source_type=source_type,
                    source_data=source_data,
                    repair_feedback=repair_feedback
                )

                response = invoke_llm(
                    llm=structured_llm,
                    prompt=[
                        ("system", system_prompt),
                        ("human", user_prompt)
                    ],
                    agent_name=self.agent_name,
                    workflow_id=workflow_id
                )

                try:

                    plan = self._parse_response(
                        response
                    )

                    issues = self._validate_plan_quality(
                        plan,
                        source_type
                    )

                    if not plan.pages:

                        issues.insert(
                            0,
                            "No pages were generated. "
                            "Generate at least one complete page."
                        )

                    if not issues:

                        validation_passed = True

                        logger.info(
                            "Plan quality validation passed."
                        )

                        break

                    repair_feedback = issues

                    logger.warning(
                        "Plan quality validation found %s issue(s).",
                        len(issues)
                    )

                    for issue in issues:

                        logger.warning(
                            "PLAN QUALITY ISSUE: %s",
                            issue
                        )

                except Exception as parse_error:

                    plan = None

                    repair_feedback = [
                        "Return valid JSON matching the FrontendPlan schema.",
                        "The pages array must contain at least one page.",
                        "Every page must contain at least one section.",
                        "Every section must contain at least one component.",
                        f"Previous response error: {str(parse_error)}"
                    ]

                    logger.exception(
                        "Planning response parsing or validation failed."
                    )

            # -------------------------------------------------
            # FALLBACK WHEN LLM GENERATION FAILS
            # -------------------------------------------------

            if plan is None or not plan.pages:

                logger.warning(
                    "LLM did not generate valid pages. "
                    "Activating source-data fallback."
                )

                plan = self._build_fallback_plan(
                    source_type=source_type,
                    source_data=source_data
                )

                validation_passed = False

            if not plan.pages:

                raise ValueError(
                    "Planning Agent could not generate pages "
                    "from either LLM output or source data."
                )

            # -------------------------------------------------
            # FINAL STRUCTURAL VALIDATION
            # -------------------------------------------------

            final_issues = self._validate_plan_quality(
                plan,
                source_type
            )

            if final_issues:

                logger.warning(
                    "Final plan contains quality warnings: %s",
                    final_issues
                )

                if not validation_passed:

                    plan.implementation_notes.extend([
                        "Plan quality validation requires review.",
                        *final_issues
                    ])

            # -------------------------------------------------
            # SAVE PLAN
            # -------------------------------------------------

            if output_path:

                output_file = Path(
                    output_path
                )

                output_file.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                output_file.write_text(
                    plan.model_dump_json(
                        indent=4
                    ),
                    encoding="utf-8"
                )

                logger.info(
                    "Frontend plan saved: %s",
                    output_file
                )

            status = "success"

            logger.info(
                "Planning Agent completed successfully | "
                "Pages: %s | Quality passed: %s",
                len(plan.pages),
                validation_passed
            )

            return plan

        except Exception:

            logger.exception(
                "Planning Agent execution failed | Workflow: %s",
                workflow_id
            )

            raise

        finally:

            duration_ms = (
                perf_counter() - agent_start_time
            ) * 1000

            try:

                metrics_collector.record_agent_metrics(
                    agent_name=self.agent_name,
                    duration_ms=duration_ms,
                    status=status,
                    workflow_id=workflow_id
                )

            except Exception:

                logger.exception(
                    "Failed to record Planning Agent metrics."
                )

            logger.info(
                "Planning Agent execution time: %.2f ms",
                duration_ms
            )


# =========================================================
# 11. LOCAL TEST EXECUTION
# =========================================================

if __name__ == "__main__":

    extraction_file = Path(
        "generated_projects/pdf_extraction/sample/"
        "extraction_result.json"
    )

    if not extraction_file.exists():

        raise FileNotFoundError(
            f"Extraction result not found: {extraction_file}"
        )

    with extraction_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        extraction_data = json.load(file)

    agent = PlanningAgent()

    result = agent.create_plan(
        source_type="pdf",
        source_data=extraction_data,
        output_path=(
            "generated_projects/planning/sample/"
            "frontend_plan.json"
        ),
        workflow_id="novadash-planning-test-001"
    )

    print("\n========== FRONTEND PLAN ==========\n")

    print(
        result.model_dump_json(
            indent=4
        )
    )

    print("\n========== SUMMARY ==========\n")

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
