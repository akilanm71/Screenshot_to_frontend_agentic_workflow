
import json
import logging
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from backend.config.settings import settings
from backend.observability.llm_factory import create_llm, invoke_llm
from backend.observability.metrics_collector import metrics_collector


logger = logging.getLogger(__name__)


class CodeGenerationAgent:
    """Generate, normalize, validate, repair, and build React frontends."""

    AGENT_NAME = "code_generation_agent"

    IDENTITY_CSS = """
.generated-project-identity {
  min-height: 58px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 28px;
  color: #172033;
  background: #ffffff;
  border-bottom: 1px solid #e8ecf2;
  font-size: 15px;
  font-weight: 700;
}
.generated-project-identity-mark {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  border-radius: 10px;
  color: #ffffff;
  background: #635bff;
}
.generated-project-identity-name {
  overflow-wrap: anywhere;
}
"""

    def __init__(
        self,
        model: Optional[str] = None,
        output_directory: str = "generated_projects",
        max_generation_attempts: int = 3,
        max_llm_attempts: int = 2,
        build_timeout: int = 180,
    ):
        self.model = model or getattr(settings, "llm_model", None)
        self.output_directory = Path(output_directory)
        self.max_generation_attempts = max(1, max_generation_attempts)
        self.max_llm_attempts = max(1, max_llm_attempts)
        self.build_timeout = build_timeout

        self.llm = create_llm(
            agent_name=self.AGENT_NAME,
            temperature=0.2,
            num_ctx=8192,
            num_predict=4096,
            repeat_penalty=1.1,
            repeat_last_n=64,
            model=self.model,
        )

        logger.info("Code Generation Agent initialized")
        logger.info("Using model: %s", self.model)

    # ---------------------------------------------------------
    # PLAN NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def _to_dict(value: Any) -> dict:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "dict"):
            return value.dict()
        if hasattr(value, "__dict__"):
            return vars(value)

        raise TypeError(
            f"Unsupported frontend plan type: {type(value).__name__}"
        )

    @staticmethod
    def _deduplicate_items(items: list) -> list:
        result = []
        seen = set()

        for item in items:
            key = json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

            if key not in seen:
                seen.add(key)
                result.append(item)

        return result

    def _normalize_component(self, component: dict) -> dict:
        """
        Normalize component information from the frontend plan.

        Component type inference uses component_type, component_name,
        name, description, and section_name.

        This prevents components such as Search Input, Notification Icon,
        and Project Progress from becoming unsupported generic components.
        """

        result = dict(component)

        props = result.get("props")
        props = dict(props) if isinstance(props, dict) else {}

        transferable = (
            "items",
            "metrics",
            "activities",
            "projects",
            "headers",
            "rows",
            "labels",
            "values",
            "text",
            "title",
            "subtitle",
            "description",
            "content",
            "value",
            "name",
            "status",
            "change",
            "progress",
            "date",
            "message",
            "placeholder",
        )

        for key in transferable:
            if key not in props and key in result:
                props[key] = result[key]

        if props.get("content") is not None:
            props.setdefault("text", props["content"])

        searchable_parts = [
            result.get("component_type", ""),
            result.get("type", ""),
            result.get("component_name", ""),
            result.get("name", ""),
            result.get("description", ""),
            result.get("section_name", ""),
        ]

        searchable = " ".join(
            str(value)
            for value in searchable_parts
            if value is not None
        ).lower()

        compact = re.sub(r"[^a-z0-9]", "", searchable)

        # Order matters: more specific component types are checked first.
        if "piechart" in compact or "donutchart" in compact:
            component_type = "PieChart"

        elif "chart" in compact or "graph" in compact:
            component_type = "Chart"

        elif any(word in searchable for word in (
            "metric",
            "metrics",
            "statistic",
            "statistics",
            "kpi",
        )):
            component_type = "Metrics"

        elif "table" in searchable or "datagrid" in compact:
            component_type = "Table"

        elif "avatar" in searchable or "user profile" in searchable:
            component_type = "Avatar"

        elif any(word in searchable for word in (
            "notification",
            "icon",
        )):
            component_type = "Icon"

        elif any(word in searchable for word in (
            "search input",
            "search field",
            "input field",
            "text input",
        )):
            component_type = "Input"

        elif "input" in searchable:
            component_type = "Input"

        elif "button" in searchable:
            component_type = "Button"

        elif any(word in searchable for word in (
            "navigation",
            "nav menu",
            "navigation menu",
            "sidebar menu",
        )):
            component_type = "Menu"

        elif any(word in searchable for word in (
            "activity",
            "activities",
            "recent updates",
            "recent events",
            "list",
        )):
            component_type = "List"

        elif any(word in searchable for word in (
            "progress",
            "completion",
            "project status",
        )):
            component_type = "Progress"

        elif any(word in searchable for word in (
            "text",
            "paragraph",
            "description",
            "heading",
        )):
            component_type = "Text"

        else:
            component_type = (
                result.get("component_type")
                or result.get("type")
                or result.get("component_name")
                or result.get("name")
                or "Unknown"
            )

        result["component_type"] = str(component_type)

        # Normalize list-like component data.
        if not isinstance(props.get("items"), list):
            for alternative in (
                "metrics",
                "activities",
                "projects",
            ):
                if isinstance(props.get(alternative), list):
                    props["items"] = props[alternative]
                    break

        if isinstance(props.get("items"), list):
            props["items"] = self._deduplicate_items(
                props["items"]
            )
            result["items"] = props["items"]

        result["props"] = props

        return result

    def _convert_plan(self, frontend_plan: Any) -> dict:
        plan = self._to_dict(frontend_plan)

        if not isinstance(plan, dict):
            raise ValueError("Frontend plan must be a dictionary")

        if not isinstance(plan.get("pages"), list) or not plan["pages"]:
            raise ValueError(
                "Frontend plan must contain a non-empty pages list"
            )

        normalized = dict(plan)
        normalized.setdefault(
            "project_name",
            "Generated Frontend",
        )

        normalized_pages = []

        for page_index, page in enumerate(plan["pages"]):
            if not isinstance(page, dict):
                continue

            page_copy = dict(page)

            page_copy.setdefault(
                "page_number",
                page_index + 1,
            )

            page_copy.setdefault(
                "page_title",
                f"Page {page_index + 1}",
            )

            sections = page_copy.get("sections")
            sections = sections if isinstance(sections, list) else []

            normalized_sections = []

            for section_index, section in enumerate(sections):
                if not isinstance(section, dict):
                    continue

                section_copy = dict(section)

                section_copy.setdefault(
                    "section_name",
                    f"Section {section_index + 1}",
                )

                components = section_copy.get("components")
                components = (
                    components if isinstance(components, list) else []
                )

                normalized_components = []

                for component in components:
                    if not isinstance(component, dict):
                        continue

                    component_copy = dict(component)

                    # Make the section name available to normalization.
                    component_copy.setdefault(
                        "section_name",
                        section_copy["section_name"],
                    )

                    normalized_components.append(
                        self._normalize_component(component_copy)
                    )

                section_copy["components"] = normalized_components
                normalized_sections.append(section_copy)

            page_copy["sections"] = normalized_sections
            normalized_pages.append(page_copy)

        if not normalized_pages:
            raise ValueError("No valid pages found in frontend plan")

        normalized["pages"] = normalized_pages

        logger.info(
            "Normalized frontend plan: %s pages",
            len(normalized_pages),
        )

        return normalized

    @staticmethod
    def _serialize_plan(plan: dict) -> str:
        return json.dumps(
            plan,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    @staticmethod
    def _create_project_name(value: str) -> str:
        value = str(value or "generated-project").strip().lower()
        value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
        return value or "generated-project"

    # ---------------------------------------------------------
    # TEXT / CONTENT HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").lower()

        replacements = {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "\u00a0": " ",
            "&": " and ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r"[^a-z0-9$%+.]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_excluded_text(value: str) -> bool:
        normalized = CodeGenerationAgent._normalize_text(value)

        excluded = (
            "extraction test notes",
            "pdf extraction pipeline",
            "end of sample document",
            "sample pdf for testing",
            "test footer",
            "internal implementation notes",
            "developer instructions",
            "create a react frontend application",
            "generated frontend",
        )

        return any(item in normalized for item in excluded)

    def _extract_required_content(self, plan: dict) -> list[str]:
        """
        Extract visible text values for reporting and diagnostics.

        These values are not used as literal JSX-source requirements.
        BrowserAgent validates actual rendered DOM content.
        """

        required = []
        seen = set()

        def add(value):
            if value is None or isinstance(value, bool):
                return

            if not isinstance(value, (str, int, float)):
                return

            text = str(value).strip()
            normalized = self._normalize_text(text)

            if not normalized or self._is_excluded_text(text):
                return

            if normalized not in seen:
                seen.add(normalized)
                required.append(text)

        def visit_item(item):
            if isinstance(item, (str, int, float)):
                add(item)
                return

            if isinstance(item, list):
                for child in item:
                    visit_item(child)
                return

            if not isinstance(item, dict):
                return

            for key in (
                "label",
                "value",
                "change",
                "name",
                "status",
                "progress",
                "message",
                "text",
                "content",
                "headers",
                "rows",
                "items",
            ):
                if key in item:
                    visit_item(item[key])

        add(plan.get("project_name"))

        for page in plan.get("pages", []):
            add(page.get("page_title"))

            for section in page.get("sections", []):
                add(section.get("section_name"))

                for component in section.get("components", []):
                    props = component.get("props", {})

                    if component.get("content") is not None:
                        add(component["content"])

                    if isinstance(props, dict):
                        visit_item(props)

        return required

    # ---------------------------------------------------------
    # PROJECT IDENTITY
    # ---------------------------------------------------------

    def _ensure_project_identity(
        self,
        jsx: str,
        project_name: str,
    ) -> tuple[str, bool]:

        project_name = str(project_name or "").strip()

        if not project_name:
            return jsx, False

        if not jsx or not jsx.strip():
            raise ValueError("Cannot wrap empty JSX")

        if "__ProjectIdentityShell" in jsx:
            return jsx, True

        internal_component = "__GeneratedOriginalApp"
        matched = False

        named_patterns = (
            r"export\s+default\s+function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?=\()",
            r"export\s+default\s+class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b",
        )

        for pattern in named_patterns:
            match = re.search(pattern, jsx)

            if match:
                internal_component = match.group(1)

                jsx = re.sub(
                    pattern,
                    f"function {internal_component}",
                    jsx,
                    count=1,
                )

                matched = True
                break

        if not matched:
            expression_pattern = re.compile(
                r"export\s+default\s+([^;]+?)\s*;",
                re.DOTALL,
            )

            match = expression_pattern.search(jsx)

            if match:
                expression = match.group(1).strip()

                jsx = expression_pattern.sub(
                    f"const {internal_component} = {expression};",
                    jsx,
                    count=1,
                )

                matched = True

        if not matched:
            anonymous_pattern = re.compile(
                r"export\s+default\s+function\s*\("
            )

            if anonymous_pattern.search(jsx):
                jsx = anonymous_pattern.sub(
                    f"const {internal_component} = function (",
                    jsx,
                    count=1,
                )

                matched = True

        if not matched:
            raise ValueError(
                "Unable to locate a supported default React export"
            )

        name_json = json.dumps(
            project_name,
            ensure_ascii=False,
        )

        initial_json = json.dumps(
            project_name[0].upper(),
            ensure_ascii=False,
        )

        wrapper = f"""

// Automatically added project identity wrapper.
const __PROJECT_IDENTITY__ = {name_json};
const __PROJECT_INITIAL__ = {initial_json};

function __ProjectIdentityShell() {{
  return (
    <>
      <header className="generated-project-identity"
              aria-label="Project identity">
        <span className="generated-project-identity-mark">
          {{__PROJECT_INITIAL__}}
        </span>
        <span className="generated-project-identity-name">
          {{__PROJECT_IDENTITY__}}
        </span>
      </header>
      <{internal_component} />
    </>
  );
}}

export default __ProjectIdentityShell;
"""

        return jsx.rstrip() + "\n" + wrapper, True

    # ---------------------------------------------------------
    # SOURCE VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate_component_definitions(jsx: str) -> list[str]:
        used = set(
            re.findall(
                r"<([A-Z][A-Za-z0-9_]*)\b",
                jsx,
            )
        )

        defined = set(
            re.findall(
                r"(?:function|class)\s+([A-Z][A-Za-z0-9_]*)\b",
                jsx,
            )
        )

        defined.update(
            re.findall(
                r"(?:const|let|var)\s+([A-Z][A-Za-z0-9_]*)\s*=",
                jsx,
            )
        )

        imported = set()

        for match in re.findall(
            r"import\s+([\s\S]*?)\s+from\s+['\"][^'\"]+['\"]",
            jsx,
        ):
            imported.update(
                re.findall(
                    r"\b[A-Z][A-Za-z0-9_]*\b",
                    match,
                )
            )

        built_in = {
            "React",
            "Fragment",
            "Suspense",
            "StrictMode",
            "Profiler",
        }

        missing = sorted(
            used - defined - imported - built_in
        )

        return (
            ["Undefined React components: " + ", ".join(missing)]
            if missing else []
        )

    def _validate_component_data(self, plan: dict) -> list[str]:
        errors = []

        for page_index, page in enumerate(plan.get("pages", [])):
            for section_index, section in enumerate(
                page.get("sections", [])
            ):
                for component_index, component in enumerate(
                    section.get("components", [])
                ):
                    props = component.get("props", {})

                    if not isinstance(props, dict):
                        errors.append(
                            f"Page {page_index + 1}, section "
                            f"{section_index + 1}, component "
                            f"{component_index + 1}: props must be an object"
                        )
                        continue

                    component_type = str(
                        component.get("component_type", "")
                    ).lower()

                    if any(
                        word in component_type
                        for word in (
                            "metric",
                            "activity",
                            "list",
                            "progress",
                        )
                    ):
                        items = props.get("items", [])

                        if not isinstance(items, list):
                            errors.append(
                                f"Page {page_index + 1}, component "
                                f"{component_index + 1}: items must be a list"
                            )

        return errors

    def _validate_jsx(
        self,
        jsx: str,
        required_content: Optional[list[str]] = None,
        project_name: Optional[str] = None,
    ) -> list[str]:
        """
        Validate source structure only.

        Do not compare plan text against raw JSX source. React may render
        content from imported data, JSON constants, arrays, or expressions.
        BrowserAgent validates the actual rendered DOM.
        """

        errors = []

        if not jsx or not jsx.strip():
            return ["Generated JSX is empty"]

        if "export default" not in jsx:
            errors.append("React application has no default export")

        has_return = bool(
            re.search(
                r"\breturn\s*\(?\s*<",
                jsx,
                re.DOTALL,
            )
        )

        has_arrow_jsx = bool(
            re.search(
                r"=>\s*\(?\s*<(?:[A-Za-z]|>)",
                jsx,
                re.DOTALL,
            )
        )

        if not has_return and not has_arrow_jsx:
            errors.append(
                "React component does not appear to return JSX"
            )

        errors.extend(
            self._validate_component_definitions(jsx)
        )

        if project_name:
            if "__ProjectIdentityShell" not in jsx:
                errors.append(
                    "Project identity wrapper is missing"
                )

        return errors

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        content = getattr(response, "content", response)

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))

            return "\n".join(parts)

        return str(content or "")

    @staticmethod
    def _clean_code(content: str) -> str:
        content = str(content or "").strip()

        content = re.sub(
            r"^\s*```(?:jsx|javascript|js|css|html|json)?[ \t]*\r?\n?",
            "",
            content,
            flags=re.IGNORECASE,
        )

        content = re.sub(
            r"\r?\n?[ \t]*```\s*$",
            "",
            content,
        )

        return content.strip()

    def _invoke(self, prompt: str, workflow_id: str) -> str:
        last_error = None

        for attempt in range(1, self.max_llm_attempts + 1):
            try:
                response = invoke_llm(
                    self.llm,
                    prompt=[
                        (
                            "system",
                            "You are a senior frontend engineer. "
                            "Return complete valid source code. "
                            "Never return Markdown fences, explanations, "
                            "partial patches, or mixed JSX and CSS.",
                        ),
                        ("human", prompt),
                    ],
                    agent_name=self.AGENT_NAME,
                    workflow_id=workflow_id,
                )

                try:
                    metrics_collector.record_llm_metrics(
                        agent_name=self.AGENT_NAME,
                        model_name=str(self.model or "unknown"),
                        usage=getattr(
                            response,
                            "usage_metadata",
                            None,
                        ),
                        metadata=getattr(
                            response,
                            "response_metadata",
                            None,
                        ),
                        workflow_id=workflow_id,
                    )

                except Exception:
                    logger.debug(
                        "LLM metrics recording failed",
                        exc_info=True,
                    )

                result = self._clean_code(
                    self._extract_response_content(response)
                )

                if result:
                    return result

                raise ValueError(
                    "LLM returned an empty response"
                )

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "LLM attempt %s/%s failed: %s",
                    attempt,
                    self.max_llm_attempts,
                    exc,
                )

        raise RuntimeError(
            f"LLM invocation failed: {last_error}"
        )

    # ---------------------------------------------------------
    # JSX GENERATION
    # ---------------------------------------------------------

    def _build_jsx_prompt(
        self,
        plan: dict,
        repair_errors: Optional[list[str]] = None,
        previous_code: Optional[str] = None,
    ) -> str:

        repair = ""

        if repair_errors:
            repair = (
                "\n\nERRORS TO FIX:\n"
                + "\n".join(
                    f"- {error}"
                    for error in repair_errors
                )
            )

        previous = ""

        if previous_code:
            previous = (
                "\n\nPREVIOUS SOURCE TO REPAIR:\n"
                + previous_code[:18000]
            )

        return f"""
Generate a complete React application in one App.jsx file.

PROJECT:
{plan.get("project_name")}

NORMALIZED PLAN:
{self._serialize_plan(plan)}

DATA CONTRACT:
Every component has component_type and props.
props is always an object.
The normalized props contain migrated top-level values.
Use the plan data to render the actual visible content.

COMPONENT MAPPING:
- Text -> TextSection
- Metrics -> MetricGrid
- List -> ActivityList
- Progress -> ProjectProgress
- Table -> TableComponent
- Chart -> ChartComponent
- PieChart -> PieChartComponent
- Avatar -> AvatarComponent
- Icon -> IconComponent
- Input -> InputComponent
- Button -> ButtonComponent
- Menu -> NavigationMenu

REQUIREMENTS:
1. Return a complete React application.
2. Use React functional components.
3. Include exactly one default export.
4. Define every custom component used.
5. Never reference undefined components or variables.
6. Use safe defaults for optional properties.
7. Check arrays with Array.isArray before mapping.
8. Render every planned page using React state navigation.
9. Render every section on the selected page.
10. Render every component in each section.
11. Render every item in each component.
12. Preserve original visible labels, text, values and numbers.
13. Do not invent content.
14. Do not duplicate entries.
15. Do not render implementation notes or extraction instructions.
16. Do not render generic planner descriptions as page content.
17. Display the project name.
18. Use semantic HTML and accessible buttons.
19. Use useful CSS class names.
20. Make the layout responsive.
21. Do not use external image URLs.
22. Do not assume component_name is a React component.
23. Do not reference variables that have not been declared.
24. Use the normalized component_type values.

COMPONENT BEHAVIOR:
TextSection:
- Read props.text, props.content, or component.content.

MetricGrid:
- Read props.items.
- Render label, value and change.

ActivityList:
- Read props.items.
- Render each activity message.

ProjectProgress:
- Read props.items.
- Render name, status and progress.
- Clamp visual progress from 0 to 100.

TableComponent:
- Read props.headers and props.rows.
- Render a semantic table.

ChartComponent:
- Render chart data only when valid data is supplied.
- Do not invent chart values.

PAGE NAVIGATION:
- Use useState.
- Show one navigation button per planned page.
- Clicking a button changes the active page.
- Render only the selected page.

Before returning, check:
- All custom components are defined.
- All mapped values are arrays.
- Optional data has safe defaults.
- All planned pages can be selected.
- Project name is visible.
- Exactly one default export exists.
- Every referenced variable is declared.

Return only complete valid App.jsx source.
{repair}
{previous}
"""

    def _generate_jsx(
        self,
        plan: dict,
        workflow_id: str,
        repair_errors: Optional[list[str]] = None,
        previous_code: Optional[str] = None,
    ) -> str:

        return self._invoke(
            self._build_jsx_prompt(
                plan,
                repair_errors,
                previous_code,
            ),
            workflow_id,
        )

    # ---------------------------------------------------------
    # CSS
    # ---------------------------------------------------------

    def _build_css_prompt(self, plan: dict, jsx: str) -> str:
        return f"""
Generate a complete production-quality CSS stylesheet.

Project: {plan.get("project_name")}

React source:
{jsx[:16000]}

Requirements:
- Modern professional SaaS dashboard.
- Responsive desktop, tablet and mobile layouts.
- Consistent typography and spacing.
- Accessible contrast.
- Styled navigation and active page state.
- Styled metrics, activities, tables and progress bars.
- Include .generated-project-identity styles.
- Include styles for the JSX class names.
- Do not use Tailwind or external frameworks.
- Balance all CSS braces.
- Return CSS only without Markdown fences.
"""

    def _ensure_identity_css(self, css: str) -> str:
        css = str(css or "")

        if ".generated-project-identity" not in css:
            css = (
                css.rstrip()
                + "\n\n"
                + self.IDENTITY_CSS
            )

        return css

    def _generate_css(
        self,
        plan: dict,
        jsx: str,
        workflow_id: str,
    ) -> str:

        try:
            css = self._invoke(
                self._build_css_prompt(plan, jsx),
                workflow_id,
            )

            if len(css.strip()) > 100:
                return self._ensure_identity_css(css)

            logger.warning("LLM returned short CSS")

        except Exception:
            logger.exception("CSS generation failed")

        return self._fallback_css()

    # ---------------------------------------------------------
    # NPM / WINDOWS
    # ---------------------------------------------------------

    @staticmethod
    def _run_npm(
        frontend_directory: Path,
        arguments: list[str],
        timeout: int,
    ) -> dict:

        npm = (
            shutil.which("npm.cmd")
            or shutil.which("npm")
        )

        if not npm:
            return {
                "success": False,
                "error": "npm executable was not found in PATH",
            }

        try:
            if (
                platform.system() == "Windows"
                and npm.lower().endswith(".cmd")
            ):
                command = subprocess.list2cmdline(
                    [npm] + arguments
                )

                result = subprocess.run(
                    command,
                    cwd=str(frontend_directory),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=True,
                    encoding="utf-8",
                    errors="replace",
                )

            else:
                result = subprocess.run(
                    [npm] + arguments,
                    cwd=str(frontend_directory),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                    encoding="utf-8",
                    errors="replace",
                )

            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": result.stdout[-12000:],
                "stderr": result.stderr[-12000:],
                "error": (
                    ""
                    if result.returncode == 0
                    else (
                        result.stderr or result.stdout
                    )[-12000:]
                ),
            }

        except subprocess.TimeoutExpired as exc:
            return {
                "success": False,
                "error": f"npm command timed out: {exc}",
            }

        except Exception as exc:
            logger.exception("npm command failed")
            return {
                "success": False,
                "error": str(exc),
            }

    # ---------------------------------------------------------
    # CSS VALIDATION
    # ---------------------------------------------------------

    def _ensure_npm_dependencies(
        self,
        frontend_directory: Path,
    ) -> tuple[bool, str]:

        if (
            frontend_directory
            / "node_modules"
            / "postcss"
        ).exists():
            return True, ""

        result = self._run_npm(
            frontend_directory,
            ["install"],
            self.build_timeout,
        )

        if not result.get("success"):
            return (
                False,
                result.get("error", "npm install failed"),
            )

        if not (
            frontend_directory
            / "node_modules"
            / "postcss"
        ).exists():
            return (
                False,
                "PostCSS was not found after npm install",
            )

        return True, ""

    def _validate_css_syntax(
        self,
        css: str,
        frontend_directory: Path,
    ) -> dict:

        if not css.strip():
            return {
                "valid": False,
                "error": "Generated CSS is empty",
            }

        node = shutil.which("node")

        if not node:
            return {
                "valid": False,
                "error": "Node.js was not found",
            }

        dependencies_ok, error = self._ensure_npm_dependencies(
            frontend_directory
        )

        if not dependencies_ok:
            return {
                "valid": False,
                "error": error,
            }

        script = r"""
const postcss = require("postcss");
let css = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", chunk => css += chunk);
process.stdin.on("end", () => {
  try {
    postcss.parse(css);
    process.stdout.write(JSON.stringify({valid: true}));
  } catch (error) {
    process.stdout.write(JSON.stringify({
      valid: false,
      error: error.reason || error.message,
      line: error.line || null,
      column: error.column || null
    }));
    process.exitCode = 1;
  }
});
"""

        try:
            result = subprocess.run(
                [node, "-e", script],
                cwd=str(frontend_directory),
                input=css,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )

            output = (result.stdout or "").strip()

            if output:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    pass

            return {
                "valid": False,
                "error": (
                    result.stderr
                    or output
                    or "PostCSS returned no output"
                )[-5000:],
            }

        except Exception as exc:
            logger.exception("CSS validation failed")

            return {
                "valid": False,
                "error": str(exc),
            }

    def _repair_css_with_llm(
        self,
        css: str,
        validation_error: dict,
        workflow_id: str,
        attempt: int,
    ) -> str:

        prompt = f"""
Repair the complete CSS stylesheet.

VALIDATION ERROR:
{json.dumps(validation_error, indent=2, ensure_ascii=False)}

CURRENT CSS:
{css[:26000]}

Preserve existing selectors, visual design and responsive rules.
Fix syntax errors and unmatched braces.
Return the complete corrected CSS only. No Markdown fences.

Repair attempt: {attempt}
"""

        repaired = self._invoke(
            prompt,
            workflow_id,
        )

        if len(repaired.strip()) < 100:
            raise ValueError(
                "CSS repair returned incomplete source"
            )

        return self._ensure_identity_css(repaired)

    def _validate_and_repair_css(
        self,
        css: str,
        frontend_directory: Path,
        workflow_id: str,
        max_attempts: int = 3,
    ) -> tuple[str, bool, list[dict]]:

        history = []

        css = self._ensure_identity_css(
            self._clean_code(css)
        )

        for attempt in range(1, max_attempts + 1):
            validation = self._validate_css_syntax(
                css,
                frontend_directory,
            )

            history.append({
                "attempt": attempt,
                "validation": validation,
            })

            if validation.get("valid"):
                logger.info(
                    "CSS syntax validation passed"
                )
                return css, True, history

            logger.warning(
                "CSS validation failed: %s",
                validation,
            )

            if attempt == max_attempts:
                break

            try:
                css = self._repair_css_with_llm(
                    css,
                    validation,
                    workflow_id,
                    attempt,
                )

            except Exception as exc:
                history.append({
                    "attempt": attempt,
                    "repair_error": str(exc),
                })

                logger.exception(
                    "CSS repair failed"
                )
                break

        return css, False, history

    @staticmethod
    def _has_css_build_warnings(build_result: dict) -> bool:
        output = (
            str(build_result.get("stdout", ""))
            + "\n"
            + str(build_result.get("stderr", ""))
        ).lower()

        return any(
            phrase in output
            for phrase in (
                "css-syntax-error",
                "warnings when minifying css",
                'expected "}"',
                'unexpected "`"',
                'unbalanced "{"',
                "css parsing error",
            )
        )

    # ---------------------------------------------------------
    # FALLBACK JSX
    # ---------------------------------------------------------

    def _generate_fallback_jsx(self, plan: dict) -> str:
        """
        Generate a functional React UI from the normalized frontend plan.

        This fallback is used when LLM-generated JSX does not pass
        validation or the production build.
        """

        plan_json = json.dumps(
            plan,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        template = r'''
import React, { useState } from "react";

const frontendPlan = __FRONTEND_PLAN_JSON__;

function displayText(value) {
  if (value === null || value === undefined) return "";

  if (typeof value === "object") {
    return "";
  }

  return String(value);
}

function getItems(component) {
  const props = component?.props || {};

  if (Array.isArray(props.items)) return props.items;
  if (Array.isArray(component?.items)) return component.items;
  if (Array.isArray(props.metrics)) return props.metrics;
  if (Array.isArray(props.activities)) return props.activities;
  if (Array.isArray(props.projects)) return props.projects;

  return [];
}

function MetricGrid({ component }) {
  const items = getItems(component);

  return (
    <div className="metric-grid">
      {items.length ? (
        items.map((item, index) => {
          const data = item && typeof item === "object"
            ? item
            : { label: String(item), value: "—" };

          return (
            <article className="metric-card" key={index}>
              <div className="metric-label">
                {displayText(data.label || data.name || data.title)}
              </div>

              <div className="metric-value">
                {displayText(data.value ?? data.count ?? "—")}
              </div>

              {data.change !== undefined && (
                <div className="metric-change">
                  {displayText(data.change)}
                </div>
              )}
            </article>
          );
        })
      ) : (
        <div className="empty-component">
          No metric data available.
        </div>
      )}
    </div>
  );
}

function ActivityList({ component }) {
  const items = getItems(component);

  if (!items.length) {
    return (
      <div className="empty-component">
        No activity data available.
      </div>
    );
  }

  return (
    <div className="activity-list">
      {items.map((item, index) => {
        const message = item && typeof item === "object"
          ? item.message || item.text || item.title || item.name
          : item;

        return (
          <div className="activity-item" key={index}>
            <span className="activity-dot" />
            <span>{displayText(message)}</span>
          </div>
        );
      })}
    </div>
  );
}

function ProjectProgress({ component }) {
  const items = getItems(component);

  if (!items.length) {
    return (
      <div className="empty-component">
        No project progress data available.
      </div>
    );
  }

  return (
    <div className="project-list">
      {items.map((item, index) => {
        const data = item && typeof item === "object"
          ? item
          : { name: item };

        const raw = Number(
          data.progress ?? data.percentage ?? data.value ?? 0
        );

        const progress = Number.isFinite(raw)
          ? Math.max(0, Math.min(100, raw))
          : 0;

        return (
          <article className="project-item" key={index}>
            <div className="project-heading">
              <strong>
                {displayText(data.name || data.title || data.label)}
              </strong>

              <span>{displayText(data.status)}</span>
            </div>

            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: progress + "%" }}
              />
            </div>

            <div className="progress-value">
              {progress}%
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ChartComponent({ component, pie = false }) {
  const items = getItems(component);

  return (
    <div className="chart-placeholder">
      {pie ? (
        <div
          className="pie-chart-placeholder"
          aria-label="Chart placeholder"
        >
          <span>Chart data unavailable</span>
        </div>
      ) : (
        <div className="line-chart-placeholder">
          <div className="chart-grid-line" />
          <div className="chart-grid-line" />
          <div className="chart-grid-line" />

          <div className="chart-empty-message">
            {items.length
              ? "Chart data is available but requires chart rendering."
              : "Performance data unavailable."}
          </div>
        </div>
      )}
    </div>
  );
}

function TableComponent({ component }) {
  const props = component?.props || {};

  const headers = Array.isArray(props.headers)
    ? props.headers
    : [];

  const rows = Array.isArray(props.rows)
    ? props.rows
    : [];

  if (!headers.length && !rows.length) {
    return (
      <div className="empty-component">
        No table data available.
      </div>
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={index}>
                {displayText(header)}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {(Array.isArray(row) ? row : [row]).map(
                (cell, cellIndex) => (
                  <td key={cellIndex}>
                    {displayText(cell)}
                  </td>
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TextComponent({ component }) {
  const props = component?.props || {};

  const value =
    props.text ??
    (typeof props.content === "string" ? props.content : null) ??
    (typeof component?.content === "string"
      ? component.content
      : null) ??
    props.value ??
    "";

  const name = displayText(component?.component_name);

  return (
    <div className="text-section">
      {value ? (
        <p>{displayText(value)}</p>
      ) : (
        <p>{name}</p>
      )}
    </div>
  );
}

function SearchInput({ component }) {
  const props = component?.props || {};

  return (
    <div className="search-component">
      <input
        type="search"
        aria-label="Search dashboard"
        placeholder={
          displayText(props.placeholder) || "Search..."
        }
      />
    </div>
  );
}

function SearchButton({ component }) {
  const props = component?.props || {};

  const label = displayText(
    props.label || props.text || component?.component_name
  );

  return (
    <button
      className="search-button"
      type="button"
    >
      {label || "Search"}
    </button>
  );
}

function AvatarComponent({ component }) {
  const props = component?.props || {};

  const label = displayText(
    props.name ||
    props.label ||
    component?.component_name
  );

  const initial = (
    label.trim().charAt(0).toUpperCase() || "U"
  );

  return (
    <div
      className="avatar-component"
      title={label || "User profile"}
      aria-label={label || "User profile"}
    >
      {initial}
    </div>
  );
}

function NotificationIcon() {
  return (
    <button
      type="button"
      className="notification-button"
      aria-label="Notifications"
      title="Notifications"
    >
      <span aria-hidden="true">🔔</span>
      <span className="notification-dot" />
    </button>
  );
}

function NavigationMenu({ component }) {
  const description = String(
    component?.description || ""
  );

  const match = description.match(
    /(?:with|for)\s+(.+?)(?:\s+options)?(?:\.|$)/i
  );

  const labels = match
    ? match[1]
        .replace(/\s+and\s+/gi, ", ")
        .split(",")
        .map(value => value.trim())
        .filter(Boolean)
    : [];

  const menuItems = labels.length
    ? labels
    : ["Projects", "Analytics", "Users", "Settings"];

  return (
    <nav
      className="generated-menu"
      aria-label="Dashboard navigation"
    >
      {menuItems.map((label, index) => (
        <button
          type="button"
          key={index}
          className="generated-menu-item"
        >
          {label}
        </button>
      ))}
    </nav>
  );
}

function RenderComponent({ component }) {
  const safe = component || {};

  const type = String(
    safe.component_type || ""
  ).toLowerCase();

  return (
    <div className="generated-component">
      {type === "input" && (
        <SearchInput component={safe} />
      )}

      {type === "button" && (
        <SearchButton component={safe} />
      )}

      {type === "avatar" && (
        <AvatarComponent component={safe} />
      )}

      {type === "icon" && (
        <NotificationIcon />
      )}

      {type === "metrics" && (
        <MetricGrid component={safe} />
      )}

      {type === "chart" && (
        <ChartComponent component={safe} />
      )}

      {type === "piechart" && (
        <ChartComponent component={safe} pie />
      )}

      {type === "list" && (
        <ActivityList component={safe} />
      )}

      {type === "progress" && (
        <ProjectProgress component={safe} />
      )}

      {type === "table" && (
        <TableComponent component={safe} />
      )}

      {type === "menu" && (
        <NavigationMenu component={safe} />
      )}

      {type === "text" && (
        <TextComponent component={safe} />
      )}

      {![
        "input",
        "button",
        "avatar",
        "icon",
        "metrics",
        "chart",
        "piechart",
        "list",
        "progress",
        "table",
        "menu",
        "text"
      ].includes(type) && (
        <div className="empty-component">
          {displayText(safe.component_name)}
        </div>
      )}
    </div>
  );
}

function RenderPage({ page }) {
  const sections = Array.isArray(page?.sections)
    ? page.sections
    : [];

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>{displayText(page?.page_title)}</h1>
      </header>

      {sections.map((section, sectionIndex) => {
        const components = Array.isArray(section?.components)
          ? section.components
          : [];

        return (
          <section
            className="content-section"
            key={sectionIndex}
          >
            <h2>
              {displayText(section?.section_name)}
            </h2>

            {section?.description && (
              <p className="section-description">
                {displayText(section.description)}
              </p>
            )}

            <div className="section-components">
              {components.map(
                (component, componentIndex) => (
                  <article
                    className="component-card"
                    key={componentIndex}
                  >
                    <RenderComponent
                      component={component}
                    />
                  </article>
                )
              )}
            </div>
          </section>
        );
      })}
    </div>
  );
}

export default function App() {
  const pages = Array.isArray(frontendPlan.pages)
    ? frontendPlan.pages
    : [];

  const [activePage, setActivePage] = useState(0);

  const currentPage = pages[activePage] || null;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            {displayText(frontendPlan.project_name).charAt(0) || "A"}
          </div>

          <span>
            {displayText(frontendPlan.project_name)}
          </span>
        </div>

        <nav
          className="page-navigation"
          aria-label="Page navigation"
        >
          {pages.map((page, index) => (
            <button
              type="button"
              key={index}
              className={
                activePage === index
                  ? "nav-item active"
                  : "nav-item"
              }
              aria-current={
                activePage === index ? "page" : undefined
              }
              onClick={() => setActivePage(index)}
            >
              {displayText(
                page?.page_title || "Page " + (index + 1)
              )}
            </button>
          ))}
        </nav>
      </aside>

      <div className="main-panel">
        <div className="topbar">
          <span>Workspace</span>
          <span className="topbar-status">Dashboard</span>
        </div>

        {currentPage ? (
          <RenderPage page={currentPage} />
        ) : (
          <div className="empty-state">
            No pages were found.
          </div>
        )}
      </div>
    </main>
  );
}
'''

        return template.replace(
            "__FRONTEND_PLAN_JSON__",
            plan_json,
        )

    # ---------------------------------------------------------
    # FALLBACK CSS
    # ---------------------------------------------------------

    @staticmethod
    def _fallback_css() -> str:
        return r'''
* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #172033;
  background: #f5f7fb;
}

button { font: inherit; }

.app-shell {
  min-height: 100vh;
  display: flex;
}

.sidebar {
  width: 250px;
  flex-shrink: 0;
  background: #111827;
  color: #fff;
  padding: 28px 18px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 17px;
  font-weight: 700;
  line-height: 1.4;
  overflow-wrap: anywhere;
  margin-bottom: 42px;
}

.brand-mark {
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  border-radius: 11px;
  color: #fff;
  background: #635bff;
}

.page-navigation {
  display: grid;
  gap: 8px;
}

.nav-item {
  border: 0;
  border-radius: 9px;
  background: transparent;
  color: #cbd5e1;
  padding: 12px 13px;
  text-align: left;
  cursor: pointer;
}

.nav-item:hover,
.nav-item.active {
  background: #263247;
  color: #fff;
}

.main-panel {
  min-width: 0;
  flex: 1;
}

.topbar {
  height: 68px;
  padding: 0 36px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  border-bottom: 1px solid #e8ecf2;
  color: #64748b;
  font-size: 13px;
}

.topbar-status {
  color: #475569;
}

.page-content {
  max-width: 1440px;
  padding: 38px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 34px;
}

.page-header h1 {
  margin: 0 0 9px;
  color: #172033;
  font-size: clamp(25px, 3vw, 34px);
  letter-spacing: -1px;
}

.content-section {
  margin-bottom: 30px;
  padding: 25px;
  border: 1px solid #e8edf4;
  border-radius: 15px;
  background: #fff;
  box-shadow: 0 3px 12px rgba(25, 40, 70, 0.035);
}

.content-section > h2 {
  margin: 0 0 21px;
  font-size: 18px;
  color: #253047;
}

.text-section p {
  color: #596579;
  line-height: 1.8;
  margin: 0;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.metric-card {
  min-width: 0;
  padding: 20px;
  border: 1px solid #e9edf4;
  border-radius: 12px;
  background: #fff;
}

.metric-label {
  color: #748096;
  font-size: 13px;
  margin-bottom: 13px;
}

.metric-value {
  color: #172033;
  font-size: 27px;
  font-weight: 700;
  overflow-wrap: anywhere;
}

.metric-change {
  margin-top: 9px;
  color: #15966b;
  font-size: 12px;
  font-weight: 600;
}

.activity-list {
  display: grid;
}

.activity-item {
  display: flex;
  align-items: flex-start;
  gap: 13px;
  padding: 15px 0;
  border-bottom: 1px solid #eef1f5;
  color: #596579;
  line-height: 1.6;
}

.activity-item:last-child {
  border-bottom: 0;
}

.activity-dot {
  width: 9px;
  height: 9px;
  margin-top: 7px;
  flex-shrink: 0;
  border-radius: 50%;
  background: #635bff;
}

.project-list {
  display: grid;
  gap: 22px;
}

.project-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 15px;
  margin-bottom: 11px;
  color: #344054;
}

.project-heading span {
  color: #718096;
  font-size: 12px;
}

.progress-track {
  height: 8px;
  overflow: hidden;
  border-radius: 20px;
  background: #edf0f6;
}

.progress-fill {
  height: 100%;
  border-radius: inherit;
  background: #635bff;
}

.progress-value {
  margin-top: 7px;
  text-align: right;
  color: #748096;
  font-size: 12px;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}

th,
td {
  padding: 13px 15px;
  border-bottom: 1px solid #edf0f5;
}

th {
  color: #64748b;
  font-size: 12px;
  font-weight: 600;
  background: #f8fafc;
}

td {
  color: #475569;
  font-size: 13px;
}

.empty-state {
  padding: 60px 30px;
  text-align: center;
  color: #718096;
}

.generated-project-identity {
  min-height: 58px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 28px;
  color: #172033;
  background: #fff;
  border-bottom: 1px solid #e8ecf2;
  font-size: 15px;
  font-weight: 700;
}

.generated-project-identity-mark {
  width: 34px;
  height: 34px;
  display: grid;
  place-items: center;
  flex-shrink: 0;
  border-radius: 10px;
  color: #fff;
  background: #635bff;
}

.generated-project-identity-name {
  overflow-wrap: anywhere;
}

.search-component {
  display: flex;
  width: 100%;
  max-width: 460px;
  margin: 12px 0;
}

.search-component input {
  width: 100%;
  min-width: 0;
  padding: 12px 15px;
  border: 1px solid #dce2ec;
  border-radius: 9px;
  background: #fff;
  color: #172033;
  outline: none;
}

.search-component input:focus {
  border-color: #635bff;
  box-shadow: 0 0 0 3px rgba(99, 91, 255, .12);
}

.search-button,
.notification-button {
  border: 0;
  border-radius: 9px;
  padding: 10px 16px;
  background: #635bff;
  color: white;
  cursor: pointer;
}

.avatar-component {
  width: 44px;
  height: 44px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: #e8e7ff;
  color: #5146d8;
  font-weight: 700;
}

.notification-button {
  position: relative;
  background: #f0f1f7;
  color: #344054;
  font-size: 22px;
}

.notification-dot {
  position: absolute;
  top: 7px;
  right: 7px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #ef4444;
}

.section-components {
  display: grid;
  gap: 18px;
}

.component-card {
  min-width: 0;
}

.section-description {
  color: #718096;
  line-height: 1.6;
}

.empty-component {
  padding: 20px;
  color: #718096;
  background: #f8fafc;
  border: 1px dashed #dce2ec;
  border-radius: 10px;
}

.chart-placeholder {
  min-height: 190px;
  padding: 15px;
}

.line-chart-placeholder {
  position: relative;
  min-height: 170px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  border-left: 1px solid #dce2ec;
  border-bottom: 1px solid #dce2ec;
}

.chart-grid-line {
  width: 100%;
  border-top: 1px dashed #e5eaf1;
}

.chart-empty-message {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: #718096;
  font-size: 13px;
  text-align: center;
}

.pie-chart-placeholder {
  width: 150px;
  height: 150px;
  margin: 10px auto;
  display: grid;
  place-items: center;
  border: 20px solid #e9eaf5;
  border-radius: 50%;
  color: #718096;
  font-size: 11px;
  text-align: center;
}

.generated-menu {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.generated-menu-item {
  padding: 10px 15px;
  border: 1px solid #e5eaf1;
  border-radius: 8px;
  background: #fff;
  color: #475569;
  cursor: pointer;
}

@media (max-width: 1050px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .app-shell {
    display: block;
  }

  .sidebar {
    width: 100%;
    padding: 15px;
  }

  .brand {
    margin-bottom: 15px;
  }

  .page-navigation {
    display: flex;
    flex-wrap: wrap;
  }

  .nav-item {
    flex: 1 1 auto;
  }

  .topbar {
    padding: 0 18px;
  }

  .page-content {
    padding: 22px 15px;
  }

  .content-section {
    padding: 18px;
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }

  .project-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 5px;
  }

  .generated-project-identity {
    padding: 10px 16px;
  }
}
'''

    # ---------------------------------------------------------
    # PROJECT FILES
    # ---------------------------------------------------------

    def _create_project_files(
        self,
        project_directory: Path,
        jsx: str,
        css: str,
    ) -> dict:

        frontend = project_directory / "frontend"
        source = frontend / "src"
        source.mkdir(parents=True, exist_ok=True)

        files = {
            "App.jsx": source / "App.jsx",
            "App.css": source / "App.css",
            "main.jsx": source / "main.jsx",
            "index.html": frontend / "index.html",
            "package.json": frontend / "package.json",
            "vite.config.js": frontend / "vite.config.js",
        }

        files["App.jsx"].write_text(
            jsx,
            encoding="utf-8",
        )

        files["App.css"].write_text(
            self._ensure_identity_css(css),
            encoding="utf-8",
        )

        files["main.jsx"].write_text(
            """
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./App.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("React root element #root was not found");
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
""".strip(),
            encoding="utf-8",
        )

        files["index.html"].write_text(
            """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="theme-color" content="#f5f7fb" />
    <title>Generated Frontend</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
""".strip(),
            encoding="utf-8",
        )

        files["package.json"].write_text(
            json.dumps(
                {
                    "name": "generated-frontend",
                    "version": "1.0.0",
                    "private": True,
                    "type": "module",
                    "scripts": {
                        "dev": "vite",
                        "build": "vite build",
                        "preview": "vite preview",
                    },
                    "dependencies": {
                        "react": "^18.3.1",
                        "react-dom": "^18.3.1",
                    },
                    "devDependencies": {
                        "@vitejs/plugin-react": "^4.3.4",
                        "vite": "^5.4.11",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        files["vite.config.js"].write_text(
            """
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
});
""".strip(),
            encoding="utf-8",
        )

        return {
            key: str(path)
            for key, path in files.items()
        }

    # ---------------------------------------------------------
    # BUILD
    # ---------------------------------------------------------

    def _run_build(self, project_directory: Path) -> dict:
        frontend = project_directory / "frontend"

        install = self._run_npm(
            frontend,
            ["install"],
            self.build_timeout,
        )

        if not install.get("success"):
            return {
                "success": False,
                "stage": "npm_install",
                "error": install.get("error"),
                "stdout": install.get("stdout", ""),
                "stderr": install.get("stderr", ""),
            }

        build = self._run_npm(
            frontend,
            ["run", "build"],
            self.build_timeout,
        )

        return {
            "success": build.get("success", False),
            "stage": "build",
            "return_code": build.get("return_code"),
            "stdout": build.get("stdout", ""),
            "stderr": build.get("stderr", ""),
            "error": build.get("error", ""),
        }

    # ---------------------------------------------------------
    # REPORTS
    # ---------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, data: dict):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _save_validation_report(
        self,
        project_directory: Path,
        report: dict,
    ) -> str:

        path = (
            project_directory
            / "code_validation"
            / "validation_report.json"
        )

        self._write_json(path, report)

        return str(path)

    def _save_summary(
        self,
        project_directory: Path,
        summary: dict,
    ) -> str:

        path = (
            project_directory
            / "implementation_summary.json"
        )

        self._write_json(path, summary)

        return str(path)

    # ---------------------------------------------------------
    # GENERATION WORKFLOW
    # ---------------------------------------------------------

    def generate(
        self,
        frontend_plan: Any,
        workflow_id: Optional[str] = None,
    ) -> dict:

        started_at = time.time()
        workflow_id = workflow_id or "local_test"

        project_directory = None
        files = {}
        validation_errors = []
        css_repair_history = []
        attempt_history = []

        build_result = {
            "success": False,
            "error": "Build not executed",
        }

        generation_method = "llm"
        llm_generation_success = False
        status = "failed"
        plan = {}

        try:
            plan = self._convert_plan(frontend_plan)

            slug = self._create_project_name(
                plan.get("project_name")
            )

            project_directory = (
                self.output_directory / slug
            )

            project_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            required_content = self._extract_required_content(
                plan
            )

            plan_errors = self._validate_component_data(
                plan
            )

            if plan_errors:
                raise ValueError(
                    "Invalid normalized plan: "
                    + "; ".join(plan_errors)
                )

            logger.info(
                "Starting generation: %s",
                slug,
            )

            logger.info(
                "Extracted %s visible content values for browser validation",
                len(required_content),
            )

            previous_jsx = None
            repair_errors = []

            # -------------------------------------------------
            # LLM GENERATION AND REPAIR LOOP
            # -------------------------------------------------

            for attempt in range(
                1,
                self.max_generation_attempts + 1,
            ):

                record = {
                    "attempt": attempt,
                }

                logger.info(
                    "Generation / repair attempt %s/%s",
                    attempt,
                    self.max_generation_attempts,
                )

                try:
                    raw_jsx = self._generate_jsx(
                        plan,
                        workflow_id,
                        repair_errors,
                        previous_jsx,
                    )

                    jsx, identity_enforced = (
                        self._ensure_project_identity(
                            raw_jsx,
                            str(
                                plan.get("project_name", "")
                            ),
                        )
                    )

                    record["identity_enforced"] = (
                        identity_enforced
                    )

                    validation_errors = self._validate_jsx(
                        jsx,
                        required_content,
                        project_name=plan.get("project_name"),
                    )

                    record["source_validation_errors"] = (
                        validation_errors
                    )

                    if validation_errors:
                        record["status"] = (
                            "source_validation_failed"
                        )

                        attempt_history.append(record)

                        previous_jsx = raw_jsx
                        repair_errors = validation_errors

                        continue

                    css = self._generate_css(
                        plan,
                        jsx,
                        workflow_id,
                    )

                    files = self._create_project_files(
                        project_directory,
                        jsx,
                        css,
                    )

                    frontend = (
                        project_directory / "frontend"
                    )

                    css, css_valid, css_history = (
                        self._validate_and_repair_css(
                            css,
                            frontend,
                            workflow_id,
                            max_attempts=3,
                        )
                    )

                    record["css_validation"] = css_history

                    css_repair_history.extend(
                        css_history
                    )

                    if not css_valid:
                        record["status"] = (
                            "css_validation_failed"
                        )

                        attempt_history.append(record)

                        previous_jsx = raw_jsx

                        repair_errors = [
                            "CSS failed PostCSS validation.",
                            json.dumps(
                                css_history,
                                ensure_ascii=False,
                            ),
                            "Repair the complete React source if necessary.",
                        ]

                        continue

                    (
                        frontend / "src" / "App.css"
                    ).write_text(
                        css,
                        encoding="utf-8",
                    )

                    build_result = self._run_build(
                        project_directory
                    )

                    if self._has_css_build_warnings(
                        build_result
                    ):
                        build_result["success"] = False

                        build_result["error"] = (
                            "Vite reported CSS parsing/minification warnings"
                        )

                    record["build"] = build_result

                    if not build_result.get("success"):
                        record["status"] = "build_failed"

                        attempt_history.append(record)

                        previous_jsx = raw_jsx

                        repair_errors = [
                            "Vite production build failed.",
                            str(
                                build_result.get("error", "")
                            ),
                            "Repair the complete App.jsx source.",
                        ]

                        continue

                    record["status"] = "success"
                    record["llm_generation_success"] = True

                    attempt_history.append(record)

                    llm_generation_success = True
                    final_jsx = jsx
                    final_css = css
                    status = "success"

                    logger.info(
                        "LLM frontend passed source, CSS and Vite build checks"
                    )

                    break

                except Exception as exc:
                    logger.exception(
                        "Generation attempt %s failed",
                        attempt,
                    )

                    record["status"] = "exception"
                    record["error"] = str(exc)

                    attempt_history.append(record)

                    repair_errors = [
                        f"Generation exception: {exc}"
                    ]

            # -------------------------------------------------
            # PLAN-DRIVEN FALLBACK
            # -------------------------------------------------

            if status != "success":
                logger.warning(
                    "LLM generation did not pass. "
                    "Using plan-driven fallback."
                )

                generation_method = (
                    "plan_driven_fallback"
                )

                llm_generation_success = False

                final_jsx = self._generate_fallback_jsx(
                    plan
                )

                final_css = self._fallback_css()

                validation_errors = self._validate_jsx(
                    final_jsx,
                    required_content=None,
                    project_name=None,
                )

                if (
                    self._normalize_text(
                        plan.get("project_name")
                    )
                    not in self._normalize_text(final_jsx)
                ):
                    validation_errors.append(
                        "Fallback source does not contain the project name"
                    )

                files = self._create_project_files(
                    project_directory,
                    final_jsx,
                    final_css,
                )

                frontend = (
                    project_directory / "frontend"
                )

                final_css, css_valid, css_history = (
                    self._validate_and_repair_css(
                        final_css,
                        frontend,
                        workflow_id,
                        max_attempts=3,
                    )
                )

                css_repair_history.extend(
                    css_history
                )

                if css_valid:
                    (
                        frontend / "src" / "App.css"
                    ).write_text(
                        final_css,
                        encoding="utf-8",
                    )

                    build_result = self._run_build(
                        project_directory
                    )

                else:
                    validation_errors.append(
                        "Fallback CSS failed syntax validation"
                    )

                if self._has_css_build_warnings(
                    build_result
                ):
                    build_result["success"] = False

                    build_result["error"] = (
                        "Fallback build contains CSS parsing warnings"
                    )

                if (
                    not validation_errors
                    and css_valid
                    and build_result.get("success")
                ):
                    status = "success"
                else:
                    status = "failed"

            # -------------------------------------------------
            # VALIDATION REPORT
            # -------------------------------------------------

            fallback_used = (
                generation_method == "plan_driven_fallback"
            )

            validation_report = {
                "workflow_id": workflow_id,
                "project_name": plan.get("project_name"),
                "status": status,
                "generation_method": generation_method,
                "llm_generation_success": llm_generation_success,
                "fallback_used": fallback_used,
                "required_content_count": len(required_content),
                "required_content": required_content,
                "plan_validation_errors": plan_errors,
                "source_validation_errors": validation_errors,
                "browser_content_validation": (
                    "Pending BrowserAgent execution"
                ),
                "css_repair_history": css_repair_history,
                "build_result": build_result,
                "attempt_history": attempt_history,
            }

            validation_report_path = (
                self._save_validation_report(
                    project_directory,
                    validation_report,
                )
            )

            duration_ms = round(
                (time.time() - started_at) * 1000,
                2,
            )

            summary = {
                "workflow_id": workflow_id,
                "project_name": plan.get("project_name"),
                "status": status,
                "generation_method": generation_method,
                "llm_generation_success": llm_generation_success,
                "fallback_used": fallback_used,
                "model": self.model,
                "duration_ms": duration_ms,
                "project_directory": str(project_directory),
                "frontend_directory": str(
                    project_directory / "frontend"
                ),
                "files": files,
                "validation_report": validation_report_path,
                "build_result": build_result,
                "attempts": attempt_history,
                "css_repair_history": css_repair_history,
            }

            summary_path = self._save_summary(
                project_directory,
                summary,
            )

            try:
                metrics_collector.record_agent_metrics(
                    agent_name=self.AGENT_NAME,
                    duration_ms=duration_ms,
                    status=status,
                    workflow_id=workflow_id,
                    error=(
                        None
                        if status == "success"
                        else str(
                            validation_errors or build_result
                        )
                    ),
                )

            except Exception:
                logger.exception(
                    "Unable to record agent metrics"
                )

            return {
                "status": status,
                "success": status == "success",
                "workflow_id": workflow_id,
                "project_name": plan.get("project_name"),
                "generation_method": generation_method,
                "llm_generation_success": llm_generation_success,
                "fallback_used": fallback_used,
                "project_directory": str(project_directory),
                "frontend_directory": str(
                    project_directory / "frontend"
                ),
                "files": files,
                "validation_errors": validation_errors,
                "build_result": build_result,
                "validation_report_path": validation_report_path,
                "summary_path": summary_path,
                "css_repair_history": css_repair_history,
                "duration_ms": duration_ms,
            }

        except Exception as exc:
            logger.exception(
                "Code generation failed"
            )

            duration_ms = round(
                (time.time() - started_at) * 1000,
                2,
            )

            try:
                metrics_collector.record_agent_metrics(
                    agent_name=self.AGENT_NAME,
                    duration_ms=duration_ms,
                    status="failed",
                    workflow_id=workflow_id,
                    error=str(exc),
                )

            except Exception:
                logger.exception(
                    "Unable to record failure metrics"
                )

            return {
                "status": "failed",
                "success": False,
                "workflow_id": workflow_id,
                "project_directory": (
                    str(project_directory)
                    if project_directory
                    else None
                ),
                "files": files,
                "error": str(exc),
                "validation_errors": validation_errors,
                "build_result": build_result,
                "llm_generation_success": False,
                "fallback_used": False,
                "duration_ms": duration_ms,
            }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    plan_path = Path(
        "generated_projects/planning/sample/frontend_plan.json"
    )

    if not plan_path.exists():
        raise FileNotFoundError(
            f"Frontend plan not found: {plan_path}"
        )

    with plan_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        frontend_plan = json.load(file)

    agent = CodeGenerationAgent()

    result = agent.generate(
        frontend_plan=frontend_plan,
        workflow_id="novadash-code-generation",
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
