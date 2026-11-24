
import json
import logging
import os
import re
import shutil
import subprocess
import uuid

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

from backend.observability.llm_factory import create_llm, invoke_llm
from backend.observability.metrics_collector import metrics_collector


logger = logging.getLogger(__name__)


class RefinementAgent:
    """
    Diagnose browser/evaluation failures and safely repair generated
    React frontends.

    Supports:
    - Project-root and frontend-root path resolution.
    - Direct workflow repair_context integration.
    - Browser runtime and console diagnostics.
    - Missing-variable and data-flow repair.
    - Frontend plan discovery.
    - Required content validation.
    - JSX/CSS source validation.
    - Windows-compatible npm execution.
    - Production build validation.
    - Safe backup and rollback.
    - Rejected output preservation.
    - Observability and execution metrics.
    """

    def __init__(
        self,
        project_directory: str = "generated_projects/novadash-project-overview",
        model: Any = None,
        frontend_plan_path: Optional[str] = None,
        browser_output_directory: Optional[str] = None,
        max_retries: int = 1,
        max_content_retries: int = 2,
        strict_content_validation: bool = True,
        minimum_content_coverage: float = 90.0,
        build_timeout: int = 180,
        regenerate_css_only_when_needed: bool = True,
        repair_context: Optional[dict] = None,
        browser_result: Optional[dict] = None,
    ):

        self.agent_name = "refinement_agent"

        # -----------------------------------------------------
        # PROJECT AND FRONTEND DIRECTORY RESOLUTION
        # -----------------------------------------------------

        supplied_directory = Path(project_directory).resolve()

        def is_frontend_directory(path: Path) -> bool:
            return (
                (path / "package.json").is_file()
                or (path / "src" / "App.jsx").is_file()
            )

        # The caller may supply either:
        #   generated_projects/sample-dashboard
        # or:
        #   generated_projects/sample-dashboard/frontend

        if is_frontend_directory(supplied_directory):
            self.frontend_directory = supplied_directory
            self.project_directory = supplied_directory.parent

        elif is_frontend_directory(supplied_directory / "frontend"):
            self.project_directory = supplied_directory
            self.frontend_directory = supplied_directory / "frontend"

        else:
            self.project_directory = supplied_directory
            self.frontend_directory = supplied_directory / "frontend"

        self.frontend_plan_path = (
            Path(frontend_plan_path).resolve()
            if frontend_plan_path
            else None
        )

        self.browser_output_directory = (
            Path(browser_output_directory).resolve()
            if browser_output_directory
            else self.project_directory / "browser_output"
        )

        self.browser_results_path = (
            self.browser_output_directory / "browser_results.json"
        )

        self.runtime_repair_context_path = (
            self.browser_output_directory / "runtime_repair_context.json"
        )

        self.evaluation_report_path = (
            self.project_directory / "evaluation" / "evaluation_report.json"
        )

        self.jsx_path = self.frontend_directory / "src" / "App.jsx"
        self.css_path = self.frontend_directory / "src" / "App.css"
        self.package_json_path = self.frontend_directory / "package.json"

        self.refinement_directory = self.project_directory / "refinement"
        self.backup_directory = self.refinement_directory / "backups"
        self.rejected_directory = self.refinement_directory / "rejected_outputs"
        self.summary_path = self.refinement_directory / "refinement_summary.json"

        self.max_retries = max(0, int(max_retries))
        self.max_content_retries = max(0, int(max_content_retries))

        self.strict_content_validation = strict_content_validation

        self.minimum_content_coverage = max(
            0.0,
            min(100.0, float(minimum_content_coverage)),
        )

        self.build_timeout = max(30, int(build_timeout))

        self.regenerate_css_only_when_needed = (
            regenerate_css_only_when_needed
        )

        # Direct workflow inputs take priority over older JSON files.
        self.workflow_repair_context = (
            repair_context if isinstance(repair_context, dict) else {}
        )

        self.workflow_browser_result = (
            browser_result if isinstance(browser_result, dict) else {}
        )

        self.llm = model if model is not None else create_llm(
            agent_name=self.agent_name,
            temperature=0.1,
            num_ctx=16384,
            num_predict=8192,
        )

        self._last_summary = {}

        for directory in (
            self.refinement_directory,
            self.backup_directory,
            self.rejected_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Refinement Agent initialized")
        logger.info("Project directory: %s", self.project_directory)
        logger.info("Frontend directory: %s", self.frontend_directory)
        logger.info("App.jsx path: %s", self.jsx_path)
        logger.info("App.css path: %s", self.css_path)
        logger.info("Browser output directory: %s", self.browser_output_directory)

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    @staticmethod
    def _normalize_text(value: Any) -> str:
        value = str(value).lower()
        return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE)

    @staticmethod
    def _serialize_data(data: Any) -> str:
        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _read_json(path: Path) -> dict:

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object: {path}")

        return data

    @staticmethod
    def _read_text(path: Path) -> str:

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        return path.read_text(encoding="utf-8")

    @staticmethod
    def _extract_response_content(response: Any) -> str:

        if isinstance(response, str):
            return response.strip()

        content = getattr(response, "content", None)

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, str):
                    text_parts.append(item)

                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            if text_parts:
                return "\n".join(text_parts).strip()

        raise ValueError("LLM returned no text content.")

    @staticmethod
    def _remove_code_fences(code: str) -> str:

        if not isinstance(code, str):
            raise TypeError("Generated source must be a string.")

        code = code.replace("\ufeff", "")
        code = code.replace("\r\n", "\n").replace("\r", "\n")
        code = code.strip()

        if not code:
            return ""

        fence_pattern = re.compile(
            r"(?m)^[ \t]*```[ \t]*"
            r"[a-zA-Z0-9_+.-]*[ \t]*\n"
            r"(.*?)"
            r"\n[ \t]*```[ \t]*$",
            flags=re.DOTALL,
        )

        match = fence_pattern.search(code)

        if match:
            return match.group(1).strip()

        lines = code.splitlines()

        if lines and re.match(
            r"^[ \t]*```[a-zA-Z0-9_+.-]*[ \t]*$",
            lines[0],
        ):
            lines = lines[1:]

        if lines and re.match(
            r"^[ \t]*```[ \t]*$",
            lines[-1],
        ):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    @staticmethod
    def _as_dict(value: Any) -> dict:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _as_list(value: Any) -> list:

        if isinstance(value, list):
            return value

        if value is None:
            return []

        return [value]

    @staticmethod
    def _first_available(*values):

        for value in values:
            if value is not None and value != "":
                return value

        return None

    # =========================================================
    # PLAN DISCOVERY
    # =========================================================

    def _resolve_frontend_plan_path(self) -> Path:

        candidates = []

        if self.frontend_plan_path:
            candidates.append(self.frontend_plan_path)

        candidates.extend([
            self.project_directory / "frontend_plan.json",
            self.project_directory / "planning" / "frontend_plan.json",
            self.project_directory.parent / "planning" / "sample" / "frontend_plan.json",
            Path("generated_projects/planning/sample/frontend_plan.json").resolve(),
            Path("generated_projects/frontend_plan.json").resolve(),
        ])

        checked = set()

        for candidate in candidates:

            candidate = candidate.resolve()

            if candidate in checked:
                continue

            checked.add(candidate)

            if candidate.is_file():

                self.frontend_plan_path = candidate

                logger.info("Frontend plan found: %s", candidate)

                return candidate

        raise FileNotFoundError(
            "Could not locate frontend_plan.json. Checked: "
            + ", ".join(str(path) for path in checked)
        )

    # =========================================================
    # INPUT LOADING
    # =========================================================

    def _load_optional_json(self, path: Path) -> dict:

        if not path.exists():
            logger.warning("Optional input not found: %s", path)
            return {}

        try:
            return self._read_json(path)

        except Exception:
            logger.exception("Could not read optional JSON: %s", path)
            return {}

    def _load_inputs(self) -> dict:

        plan_path = self._resolve_frontend_plan_path()

        frontend_plan = self._read_json(plan_path)

        file_browser_results = self._load_optional_json(
            self.browser_results_path
        )

        file_repair_context = self._load_optional_json(
            self.runtime_repair_context_path
        )

        evaluation_report = self._load_optional_json(
            self.evaluation_report_path
        )

        browser_results = (
            self.workflow_browser_result
            if self.workflow_browser_result
            else file_browser_results
        )

        repair_context = (
            self.workflow_repair_context
            if self.workflow_repair_context
            else file_repair_context
        )

        # Some workflow implementations nest repair context inside
        # browser_result. Preserve it if a separate context was not supplied.
        if not repair_context and isinstance(browser_results, dict):

            nested_context = (
                browser_results.get("repair_context")
                or browser_results.get("runtime_repair_context")
            )

            if isinstance(nested_context, dict):
                repair_context = nested_context

        return {
            "frontend_plan": frontend_plan,
            "plan_path": str(plan_path),
            "browser_results": browser_results,
            "repair_context": repair_context,
            "evaluation_report": evaluation_report,
        }

    # =========================================================
    # CONTENT EXTRACTION
    # =========================================================

    def _extract_required_content(self, plan_data: Any) -> list[str]:

        required = []
        seen = set()

        ignored_keys = {
            "id",
            "component_id",
            "component_name",
            "component_type",
            "layout_type",
            "layout_instructions",
            "styling_instructions",
            "style",
            "styles",
            "design",
            "design_instructions",
            "global_design_instructions",
            "implementation_notes",
            "implementation",
            "source_reference",
            "source_file",
            "file_path",
            "file_name",
            "page_number",
            "metadata",
            "responsive_design_instructions",
            "instructions",
            "extraction_notes",
            "description",
        }

        def add_candidate(value: Any):

            if isinstance(value, bool) or value is None:
                return

            if isinstance(value, (int, float)):
                value = str(value)

            if not isinstance(value, str):
                return

            value = value.strip()

            if len(value) < 2:
                return

            normalized = self._normalize_text(value)

            if not normalized or normalized in seen:
                return

            seen.add(normalized)
            required.append(value)

        def traverse_props(data: Any):

            if isinstance(data, dict):

                for key, value in data.items():

                    if str(key).lower().strip() in ignored_keys:
                        continue

                    if isinstance(value, (dict, list)):
                        traverse_props(value)
                    else:
                        add_candidate(value)

            elif isinstance(data, list):

                for item in data:

                    if isinstance(item, (dict, list)):
                        traverse_props(item)
                    else:
                        add_candidate(item)

        def traverse_plan(data: Any):

            if not isinstance(data, dict):
                return

            add_candidate(data.get("project_name"))

            pages = data.get("pages", [])

            if not isinstance(pages, list):
                return

            for page in pages:

                if not isinstance(page, dict):
                    continue

                page_title = page.get("page_title")

                if (
                    isinstance(page_title, str)
                    and page_title.strip().lower()
                    not in {"page header", "page", "untitled"}
                ):
                    add_candidate(page_title)

                sections = page.get("sections", [])

                if not isinstance(sections, list):
                    continue

                for section in sections:

                    if not isinstance(section, dict):
                        continue

                    add_candidate(section.get("section_name"))

                    components = section.get("components", [])

                    if not isinstance(components, list):
                        continue

                    for component in components:

                        if not isinstance(component, dict):
                            continue

                        props = component.get("props", {})

                        if isinstance(props, (dict, list)):
                            traverse_props(props)

                        content = component.get("content")

                        if isinstance(content, str):
                            add_candidate(content)

        if (
            isinstance(plan_data, dict)
            and isinstance(plan_data.get("plan"), dict)
        ):
            traverse_plan(plan_data["plan"])
        else:
            traverse_plan(plan_data)

        logger.info(
            "Required visible content extracted: %s",
            len(required),
        )

        return required

    # =========================================================
    # BROWSER DIAGNOSTICS
    # =========================================================

    def _get_browser_diagnostics(
        self,
        browser_results: dict,
        repair_context: dict,
    ) -> dict:

        browser_results = self._as_dict(browser_results)
        repair_context = self._as_dict(repair_context)

        page = self._as_dict(
            browser_results.get("page_diagnostics")
        )

        repair_page = self._as_dict(
            repair_context.get("page_diagnostics")
        )

        semantic = self._as_dict(
            browser_results.get("semantic_validation")
        )

        if not semantic:
            semantic = self._as_dict(
                page.get("semantic_validation")
            )

        if not semantic:
            semantic = self._as_dict(
                repair_context.get("semantic_validation")
            )

        rendering = self._as_dict(
            browser_results.get("rendering_diagnostics")
        )

        network = self._as_dict(
            repair_context.get("network_diagnostics")
        )

        # Support direct and nested runtime error structures.
        runtime_error = self._as_dict(
            repair_context.get("runtime_error")
            or browser_results.get("runtime_error")
        )

        if not runtime_error:
            runtime_error = self._as_dict(
                rendering.get("runtime_error")
            )

        runtime_errors = self._first_available(
            browser_results.get("page_errors"),
            repair_context.get("page_errors"),
            rendering.get("page_errors"),
        )

        if not runtime_errors:
            runtime_errors = self._first_available(
                runtime_error.get("message"),
                runtime_error.get("error"),
                runtime_error.get("description"),
            )

        runtime_errors = self._as_list(runtime_errors)

        console_errors = self._first_available(
            browser_results.get("console_errors"),
            repair_context.get("console_errors"),
            rendering.get("console_errors"),
        )

        console_errors = self._as_list(console_errors)

        failed_requests = self._first_available(
            browser_results.get("failed_requests"),
            repair_context.get("failed_requests"),
            network.get("failed_requests"),
        )

        failed_requests = self._as_list(failed_requests)

        http_errors = self._first_available(
            browser_results.get("http_errors"),
            repair_context.get("http_errors"),
            network.get("http_errors"),
        )

        http_errors = self._as_list(http_errors)

        broken_images = self._first_available(
            browser_results.get("broken_images"),
            page.get("broken_images"),
            repair_page.get("broken_images"),
        )

        broken_images = self._as_list(broken_images)

        missing_content = semantic.get("missing_content", [])

        if not isinstance(missing_content, list):
            missing_content = []

        application_rendered = rendering.get(
            "application_rendered",
            page.get(
                "application_rendered",
                page.get("root_rendered", False),
            ),
        )

        vite_error_visible = page.get(
            "vite_error_visible",
            repair_page.get(
                "vite_error_visible",
                rendering.get("vite_error_visible", False),
            ),
        )

        semantic_status = semantic.get("status", "unknown")

        semantic_failed = (
            str(semantic_status).lower() == "failed"
            or bool(missing_content)
            or rendering.get("semantic_validation_failed", False)
        )

        runtime_failed = (
            rendering.get("runtime_failed", False)
            or not application_rendered
            or bool(runtime_errors)
            or bool(vite_error_visible)
        )

        workflow_error = self._first_available(
            repair_context.get("error"),
            repair_context.get("reason"),
            repair_context.get("failure_reason"),
            browser_results.get("error"),
            browser_results.get("failure"),
            runtime_error.get("message"),
            runtime_error.get("error"),
        ) or ""

        failure_category = self._first_available(
            repair_context.get("failure_category"),
            repair_context.get("failure_type"),
            browser_results.get("failure_category"),
            browser_results.get("failure_type"),
            runtime_error.get("category"),
        ) or "unknown"

        repair_instructions = self._first_available(
            repair_context.get("repair_instructions"),
            repair_context.get("suggested_fixes"),
            repair_context.get("instructions"),
        ) or []

        if isinstance(repair_instructions, str):
            repair_instructions = [repair_instructions]

        if not isinstance(repair_instructions, list):
            repair_instructions = []

        # Critical fields from the Browser Agent's repair context.
        missing_identifier = self._first_available(
            repair_context.get("missing_identifier"),
            runtime_error.get("missing_identifier"),
            browser_results.get("missing_identifier"),
        )

        source_matches = self._first_available(
            repair_context.get("source_matches"),
            runtime_error.get("source_matches"),
            [],
        )

        source_excerpts = self._first_available(
            repair_context.get("source_excerpts"),
            runtime_error.get("source_excerpts"),
            [],
        )

        if not isinstance(source_matches, list):
            source_matches = self._as_list(source_matches)

        if not isinstance(source_excerpts, list):
            source_excerpts = self._as_list(source_excerpts)

        return {
            "application_rendered": bool(application_rendered),
            "runtime_failed": bool(runtime_failed),
            "vite_error_visible": bool(vite_error_visible),
            "vite_error_text": page.get(
                "vite_error_text",
                repair_page.get("vite_error_text", ""),
            ),
            "semantic_failed": bool(semantic_failed),
            "semantic_status": semantic_status,
            "semantic_coverage": semantic.get("match_ratio"),
            "semantic_missing_content": missing_content,
            "runtime_errors": runtime_errors,
            "runtime_error": runtime_error,
            "missing_identifier": missing_identifier,
            "source_matches": source_matches,
            "source_excerpts": source_excerpts,
            "console_errors": console_errors,
            "failed_requests": failed_requests,
            "http_errors": http_errors,
            "broken_images": broken_images,
            "horizontal_overflow": bool(
                page.get(
                    "horizontal_overflow",
                    repair_page.get("horizontal_overflow", False),
                )
            ),
            "page_text_preview": page.get(
                "root_text_preview",
                repair_page.get("root_text_preview", ""),
            ),
            "dimensions": page.get(
                "dimensions",
                repair_page.get("dimensions", {}),
            ),
            "failure_category": failure_category,
            "repair_instructions": repair_instructions,
            "workflow_error": workflow_error,
            "workflow_retry_reason": repair_context.get(
                "retry_reason",
                "",
            ),
            "refinement_iteration": repair_context.get(
                "refinement_iteration",
                0,
            ),
            "browser_status": browser_results.get("status", "unknown"),
            "browser_success": browser_results.get("success"),
        }

    def _should_refine(
        self,
        diagnostics: dict,
        missing_content: list[str],
    ) -> tuple[bool, list[str]]:

        reasons = []

        if not diagnostics["application_rendered"]:
            reasons.append("React application is not rendering.")

        if diagnostics["runtime_failed"]:
            reasons.append("Runtime or React rendering failure detected.")

        if diagnostics["vite_error_visible"]:
            reasons.append("Vite error overlay detected.")

        if diagnostics["semantic_failed"]:
            reasons.append("Semantic content validation needs attention.")

        if missing_content:
            reasons.append(
                f"{len(missing_content)} required content items are missing."
            )

        if diagnostics["console_errors"]:
            reasons.append("Browser console errors were reported.")

        if diagnostics["broken_images"]:
            reasons.append("Broken images were detected.")

        if diagnostics["horizontal_overflow"]:
            reasons.append("Horizontal layout overflow was detected.")

        if diagnostics["http_errors"]:
            reasons.append("HTTP errors were reported.")

        if diagnostics["failed_requests"]:
            reasons.append("Failed network requests were reported.")

        if diagnostics["workflow_error"]:
            reasons.append(
                f"Workflow reported a browser failure: "
                f"{diagnostics['workflow_error']}"
            )

        if diagnostics["workflow_retry_reason"]:
            reasons.append(
                f"Workflow repair reason: "
                f"{diagnostics['workflow_retry_reason']}"
            )

        if diagnostics.get("missing_identifier"):
            reasons.append(
                "Browser Agent identified an unresolved identifier: "
                f"{diagnostics['missing_identifier']}."
            )

        return bool(reasons), reasons

    # =========================================================
    # CONTENT VALIDATION
    # =========================================================

    def _validate_required_content(
        self,
        jsx_code: str,
        required_content: list[str],
    ) -> dict:

        normalized_jsx = self._normalize_text(jsx_code)

        found = []
        missing = []

        for item in required_content:

            normalized_item = self._normalize_text(item)

            if not normalized_item:
                continue

            if normalized_item in normalized_jsx:
                found.append(item)
            else:
                missing.append(item)

        total = len(required_content)

        coverage = (
            len(found) / total * 100
            if total
            else 100.0
        )

        passed = coverage >= self.minimum_content_coverage

        return {
            "passed": passed,
            "validation_type": "jsx_source_presence",
            "note": (
                "This checks source presence only. Browser Agent must "
                "verify actual visible rendering after the build."
            ),
            "total_required_content": total,
            "found_count": len(found),
            "missing_count": len(missing),
            "coverage_percentage": round(coverage, 2),
            "minimum_coverage": self.minimum_content_coverage,
            "found_content": found,
            "missing_content": missing,
        }

    # =========================================================
    # PROMPTS
    # =========================================================

    def _build_jsx_prompt(
        self,
        frontend_plan: dict,
        browser_diagnostics: dict,
        evaluation_report: dict,
        current_jsx: str,
        missing_content: Optional[list[str]] = None,
        repair_context: Optional[dict] = None,
    ) -> str:

        missing_content = missing_content or []
        repair_context = repair_context or {}

        missing_identifier = (
            browser_diagnostics.get("missing_identifier")
            or "Not explicitly identified"
        )

        return f"""
You are a senior React engineer responsible for repairing an existing
production frontend.

Your task is to diagnose the actual browser failure and make the smallest
correct code change. You must fix the root cause, not merely suppress
the error.

FRONTEND PLAN:
{self._serialize_data(frontend_plan)}

BROWSER DIAGNOSTICS:
{self._serialize_data(browser_diagnostics)}

ORIGINAL BROWSER REPAIR CONTEXT:
{self._serialize_data(repair_context)}

EVALUATION REPORT:
{self._serialize_data(evaluation_report)}

CURRENT APP.JSX:
{current_jsx}

CONTENT THAT MUST BE VISIBLE:
{self._serialize_data(missing_content)}

CRITICAL IDENTIFIER REPAIR:

The Browser Agent identified this potentially undefined identifier:
{missing_identifier}

You MUST investigate this identifier and the original repair context.

If the identifier is "dashboard", for example, inspect every reference
to dashboard and determine where its data is intended to originate.

Do not assume that dashboard, normalizedPlan, or any other variable
exists merely because the JSX references it.

Use the following repair procedure:

1. Read the complete browser runtime error.
2. Inspect the missing identifier and its source matches.
3. Inspect the source excerpts supplied by the Browser Agent.
4. Locate every reference to the failing identifier in App.jsx.
5. Determine the intended data source from the frontend plan and
   existing application structure.
6. Check whether the value should be declared, imported, passed as a prop,
   returned from a hook, or derived from existing application data.
7. Implement a valid and consistent data flow.
8. Ensure every referenced variable, function, and component exists
   in the correct scope.
9. If the original component expects dashboard.header_sections,
   dashboard.main_content_sections, and dashboard.footer_sections,
   provide a valid dashboard data object with those properties only
   when the frontend plan and existing source support that structure.
10. Do not replace the missing object with arbitrary invented content.
11. Do not hide the component or remove the failing section.
12. Do not use optional chaining or empty fallback objects merely to
    conceal an undefined-variable error.
13. Do not introduce another undefined identifier while repairing it.
14. Preserve existing working state, event handlers, navigation,
    and component relationships.

MANDATORY APPLICATION REQUIREMENTS:

1. Preserve existing working functionality.
2. Preserve existing metric names and values.
3. Preserve existing activity items.
4. Preserve existing project names, statuses, and progress values.
5. Render every required section heading.
6. Render required content as visible React elements.
7. Never put required content only in comments.
8. Do not hide required content.
9. Do not invent values or data.
10. Fix the problems identified in diagnostics.
11. Preserve existing navigation and interactions.
12. Preserve the existing component structure where possible.
13. Use valid React JSX and functional components.
14. Preserve the CSS import: import "./App.css";
15. Return the COMPLETE App.jsx file.
16. Return source code only, without Markdown fences or explanations.

Do not remove existing content.
Do not rewrite the application unnecessarily.
Do not produce pseudocode.
Do not return partial components.
"""

    def _build_css_prompt(
        self,
        frontend_plan: dict,
        browser_diagnostics: dict,
        current_css: str,
        refined_jsx: str,
    ) -> str:

        return f"""
You are a senior frontend CSS engineer.

Repair only the identified styling and layout issues.

FRONTEND PLAN:
{self._serialize_data(frontend_plan)}

BROWSER DIAGNOSTICS:
{self._serialize_data(browser_diagnostics)}

REFINED APP.JSX:
{refined_jsx}

CURRENT APP.CSS:
{current_css}

REQUIREMENTS:

1. Preserve the existing visual identity.
2. Preserve all dashboard content.
3. Do not hide required elements.
4. Fix horizontal overflow if reported.
5. Fix actual layout problems described in diagnostics.
6. Keep the design responsive.
7. Avoid unnecessary global CSS changes.
8. Preserve existing colors, typography, and spacing where possible.
9. Use valid CSS.
10. Return the COMPLETE App.css file.
11. Return CSS source code only, without Markdown fences or explanations.
"""

    # =========================================================
    # LLM GENERATION
    # =========================================================

    def _generate_source(
        self,
        prompt: str,
        source_name: str,
        workflow_id: str,
    ) -> str:

        last_error = None

        for attempt in range(self.max_retries + 1):

            try:

                logger.info(
                    "Generating %s: attempt %s/%s",
                    source_name,
                    attempt + 1,
                    self.max_retries + 1,
                )

                response = invoke_llm(
                    llm=self.llm,
                    prompt=prompt,
                    agent_name=self.agent_name,
                    workflow_id=workflow_id,
                )

                generated = self._extract_response_content(response)
                generated = self._remove_code_fences(generated)

                if not generated:
                    raise ValueError(
                        f"LLM returned empty source for {source_name}."
                    )

                metadata = getattr(
                    response,
                    "response_metadata",
                    {},
                ) or {}

                if metadata.get("done_reason") == "length":
                    raise ValueError(
                        f"{source_name} generation was truncated."
                    )

                if re.search(r"(?m)^[ \t]*```", generated):
                    raise ValueError(
                        f"{source_name} still contains Markdown fences."
                    )

                logger.info(
                    "%s generated: %s characters",
                    source_name,
                    len(generated),
                )

                return generated

            except Exception as error:

                last_error = error

                logger.exception(
                    "Generation failed for %s on attempt %s.",
                    source_name,
                    attempt + 1,
                )

        raise RuntimeError(
            f"Could not generate {source_name}: {last_error}"
        )

    # =========================================================
    # SOURCE VALIDATION
    # =========================================================

    def _validate_generated_output(
        self,
        jsx_code: str,
        css_code: str,
    ) -> dict:

        errors = []
        warnings = []

        jsx_code = self._remove_code_fences(jsx_code)
        css_code = self._remove_code_fences(css_code)

        if not jsx_code.strip():
            errors.append("App.jsx is empty.")

        if not css_code.strip():
            errors.append("App.css is empty.")

        if jsx_code.strip() and not re.search(
            r"""import\s+['"].*\.css['"]""",
            jsx_code,
        ):
            jsx_code = 'import "./App.css";\n' + jsx_code
            warnings.append("CSS import was added automatically.")

        if css_code.count("{") != css_code.count("}"):
            errors.append("CSS braces are unbalanced.")

        if "export default" not in jsx_code:
            errors.append("Missing default export in App.jsx.")

        if not re.search(
            r"\bfunction\b|\bconst\b|\bclass\b",
            jsx_code,
        ):
            errors.append("No component declaration found.")

        if re.search(
            r"(?m)^[ \t]*```",
            jsx_code,
        ) or re.search(
            r"(?m)^[ \t]*```",
            css_code,
        ):
            errors.append("Generated source contains Markdown fences.")

        if "\x00" in jsx_code or "\x00" in css_code:
            errors.append("Generated source contains null characters.")

        if "<escape>" in jsx_code:
            errors.append("Unexpected escaped markup token found in JSX.")

        placeholder_patterns = (
            r"(?i)TODO_IMPLEMENT",
            r"(?i)YOUR_CODE_HERE",
            r"(?i)INSERT_CODE_HERE",
        )

        for pattern in placeholder_patterns:

            if re.search(pattern, jsx_code):
                errors.append(
                    f"Unresolved JSX placeholder detected: {pattern}"
                )

        # Basic guard for the previously observed undefined variable.
        # This is a heuristic check, not a complete JavaScript parser.
        for variable_name in ("normalizedPlan", "dashboard"):

            if re.search(rf"\b{variable_name}\b", jsx_code):

                declaration_patterns = (
                    rf"\b(?:const|let|var|function|class)\s+{variable_name}\b",
                    rf"\bimport\b[^;]*\b{variable_name}\b",
                    rf"\b{variable_name}\s*=",
                    rf"\b{variable_name}\s*:",
                )

                declared = any(
                    re.search(pattern, jsx_code)
                    for pattern in declaration_patterns
                )

                if not declared:
                    errors.append(
                        f"Potential undefined variable: {variable_name}. "
                        "It is referenced but no local declaration or "
                        "import was detected."
                    )

        return {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
            "jsx_code": jsx_code,
            "css_code": css_code,
        }

    # =========================================================
    # BACKUP AND FILE MANAGEMENT
    # =========================================================

    def _create_backup(self) -> str:

        backup_path = self.backup_directory / self._timestamp()
        backup_path.mkdir(parents=True, exist_ok=True)

        if self.jsx_path.exists():
            shutil.copy2(self.jsx_path, backup_path / "App.jsx")

        if self.css_path.exists():
            shutil.copy2(self.css_path, backup_path / "App.css")

        logger.info("Backup created: %s", backup_path)

        return str(backup_path)

    def _restore_backup(self, backup_path: str):

        backup_directory = Path(backup_path)

        backup_jsx = backup_directory / "App.jsx"
        backup_css = backup_directory / "App.css"

        if backup_jsx.exists():
            shutil.copy2(backup_jsx, self.jsx_path)

        if backup_css.exists():
            shutil.copy2(backup_css, self.css_path)

        logger.warning("Original frontend files restored from backup.")

    def _save_rejected_output(
        self,
        jsx_code: str,
        css_code: str,
        validation: dict,
    ) -> str:

        rejected_path = self.rejected_directory / self._timestamp()
        rejected_path.mkdir(parents=True, exist_ok=True)

        (rejected_path / "App.jsx").write_text(
            jsx_code,
            encoding="utf-8",
        )

        (rejected_path / "App.css").write_text(
            css_code,
            encoding="utf-8",
        )

        with (rejected_path / "validation.json").open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                validation,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        logger.warning("Rejected output saved: %s", rejected_path)

        return str(rejected_path)

    @staticmethod
    def _safe_write(path: Path, content: str):

        path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = path.with_suffix(path.suffix + ".tmp")

        temporary_path.write_text(
            content,
            encoding="utf-8",
        )

        temporary_path.replace(path)

    # =========================================================
    # NPM DISCOVERY
    # =========================================================

    def _find_npm_executable(self) -> Optional[str]:

        candidates = (
            ["npm.cmd", "npm.exe", "npm"]
            if os.name == "nt"
            else ["npm", "npm.cmd"]
        )

        for candidate in candidates:

            executable = shutil.which(candidate)

            if executable:
                logger.info("npm executable found: %s", executable)
                return executable

        if os.name == "nt":

            possible_paths = [
                Path(
                    os.environ.get(
                        "ProgramFiles",
                        "C:\\Program Files",
                    )
                ) / "nodejs" / "npm.cmd",

                Path(
                    os.environ.get(
                        "ProgramFiles(x86)",
                        "C:\\Program Files (x86)",
                    )
                ) / "nodejs" / "npm.cmd",

                Path(
                    os.environ.get("APPDATA", "")
                ) / "npm" / "npm.cmd",

                Path("D:/Program Files/npm.cmd"),
            ]

            for candidate in possible_paths:

                if candidate.is_file():
                    logger.info(
                        "npm found at fallback location: %s",
                        candidate,
                    )
                    return str(candidate)

        logger.warning("npm executable was not found in PATH.")

        return None

    # =========================================================
    # PRODUCTION BUILD
    # =========================================================

    def _run_production_build(self) -> dict:

        if not self.package_json_path.is_file():

            return {
                "status": "unavailable",
                "passed": False,
                "skipped": True,
                "error": "package.json not found.",
                "stdout": "",
                "stderr": "",
            }

        npm_executable = self._find_npm_executable()

        if not npm_executable:

            return {
                "status": "unavailable",
                "passed": False,
                "skipped": True,
                "error": "npm was not found in PATH.",
                "stdout": "",
                "stderr": "",
            }

        if os.name == "nt":
            command = f'"{npm_executable}" run build'
            use_shell = True
        else:
            command = [npm_executable, "run", "build"]
            use_shell = False

        logger.info("Running production build: %s", command)

        try:

            result = subprocess.run(
                command,
                cwd=str(self.frontend_directory),
                capture_output=True,
                text=True,
                timeout=self.build_timeout,
                shell=use_shell,
                encoding="utf-8",
                errors="replace",
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode == 0:

                logger.info("Production build passed.")

                return {
                    "status": "passed",
                    "passed": True,
                    "skipped": False,
                    "return_code": result.returncode,
                    "stdout": stdout[-5000:],
                    "stderr": stderr[-5000:],
                    "error": None,
                }

            logger.error(
                "Production build failed. Return code: %s",
                result.returncode,
            )

            return {
                "status": "failed",
                "passed": False,
                "skipped": False,
                "return_code": result.returncode,
                "stdout": stdout[-5000:],
                "stderr": stderr[-5000:],
                "error": "npm run build returned a non-zero exit code.",
            }

        except subprocess.TimeoutExpired as error:

            stdout = error.stdout or ""
            stderr = error.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")

            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            return {
                "status": "timed_out",
                "passed": False,
                "skipped": False,
                "error": (
                    f"Production build timed out after "
                    f"{self.build_timeout} seconds."
                ),
                "stdout": stdout[-3000:],
                "stderr": stderr[-3000:],
            }

        except FileNotFoundError as error:

            logger.exception("Could not launch npm build command.")

            return {
                "status": "unavailable",
                "passed": False,
                "skipped": True,
                "error": str(error),
                "stdout": "",
                "stderr": "",
            }

        except Exception as error:

            logger.exception("Production build execution failed.")

            return {
                "status": "error",
                "passed": False,
                "skipped": False,
                "error": str(error),
                "stdout": "",
                "stderr": "",
            }

    # =========================================================
    # SUMMARY
    # =========================================================

    def _save_summary(self, summary: dict):

        self.summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

        logger.info("Summary saved: %s", self.summary_path)

    def _save_failure_summary(
        self,
        workflow_id: str,
        error: Exception,
    ):

        summary = {
            "workflow_id": workflow_id,
            "agent_name": self.agent_name,
            "status": "failed",
            "timestamp": datetime.now().isoformat(),
            "project_directory": str(self.project_directory),
            "frontend_directory": str(self.frontend_directory),
            "error": str(error),
            "error_type": type(error).__name__,
            "changes_applied": False,
            "note": (
                "Review the error and rejected output, if available. "
                "Existing frontend files are preserved or restored "
                "when a backup was created."
            ),
        }

        if self._last_summary:
            summary.update(self._last_summary)
            summary["status"] = "failed"
            summary["error"] = str(error)
            summary["error_type"] = type(error).__name__

        try:
            self._save_summary(summary)

        except Exception:
            logger.exception("Could not save failure summary.")

    # =========================================================
    # CSS DECISION
    # =========================================================

    def _should_regenerate_css(self, diagnostics: dict) -> bool:

        if not self.regenerate_css_only_when_needed:
            return True

        if diagnostics.get("horizontal_overflow", False):
            return True

        failure_category = str(
            diagnostics.get("failure_category", "")
        ).strip().lower()

        repair_instructions = diagnostics.get(
            "repair_instructions",
            [],
        )

        if isinstance(repair_instructions, str):
            repair_instructions = [repair_instructions]

        repair_text = " ".join(
            str(item) for item in repair_instructions
        ).lower()

        combined = f"{failure_category} {repair_text}"

        css_specific_terms = (
            "css",
            "stylesheet",
            "horizontal overflow",
            "vertical overflow",
            "responsive layout",
            "responsive design",
            "broken layout",
            "layout overflow",
            "overlapping elements",
            "overlapping content",
            "incorrect spacing",
            "incorrect alignment",
            "styling issue",
            "style issue",
        )

        return any(term in combined for term in css_specific_terms)

    # =========================================================
    # MAIN REFINEMENT
    # =========================================================

    def _refine_impl(self, workflow_id: str) -> dict:

        logger.info("Starting refinement: %s", workflow_id)

        inputs = self._load_inputs()

        frontend_plan = inputs["frontend_plan"]
        browser_results = inputs["browser_results"]
        repair_context = inputs["repair_context"]
        evaluation_report = inputs["evaluation_report"]

        # -----------------------------------------------------
        # VERIFY FRONTEND FILES
        # -----------------------------------------------------

        if not self.jsx_path.is_file():
            raise FileNotFoundError(
                "Generated React source not found: "
                f"{self.jsx_path}. "
                "Check project_directory and frontend directory resolution."
            )

        if not self.css_path.is_file():
            raise FileNotFoundError(
                f"Generated CSS not found: {self.css_path}"
            )

        if not self.package_json_path.is_file():
            logger.warning(
                "package.json not found at expected frontend path: %s",
                self.package_json_path,
            )

        current_jsx = self._read_text(self.jsx_path)
        current_css = self._read_text(self.css_path)

        required_content = self._extract_required_content(frontend_plan)

        if not required_content:
            raise ValueError(
                "No visible content found in frontend_plan.json. "
                "Inspect the frontend plan schema."
            )

        browser_diagnostics = self._get_browser_diagnostics(
            browser_results,
            repair_context,
        )

        source_validation = self._validate_required_content(
            current_jsx,
            required_content,
        )

        missing_content = list(source_validation["missing_content"])

        browser_missing = browser_diagnostics.get(
            "semantic_missing_content",
            [],
        )

        existing_normalized = {
            self._normalize_text(existing)
            for existing in missing_content
        }

        for item in browser_missing:

            if isinstance(item, dict):
                value = item.get("value") or item.get("text")
            else:
                value = item

            if not isinstance(value, str):
                continue

            value = value.strip()

            if not value:
                continue

            normalized = self._normalize_text(value)

            if normalized not in existing_normalized:
                missing_content.append(value)
                existing_normalized.add(normalized)

        should_refine, reasons = self._should_refine(
            browser_diagnostics,
            missing_content,
        )

        logger.info("Required content count: %s", len(required_content))

        logger.info(
            "Current source coverage: %.2f%%",
            source_validation["coverage_percentage"],
        )

        logger.info(
            "Browser application rendered: %s",
            browser_diagnostics["application_rendered"],
        )

        logger.info(
            "Browser failure category: %s",
            browser_diagnostics["failure_category"],
        )

        logger.info(
            "Missing identifier: %s",
            browser_diagnostics.get("missing_identifier"),
        )

        # -----------------------------------------------------
        # NO-OP WHEN HEALTHY
        # -----------------------------------------------------

        if not should_refine:

            logger.info("No refinement required.")

            summary = {
                "workflow_id": workflow_id,
                "agent_name": self.agent_name,
                "status": "skipped",
                "reason": "No actionable browser or content issues detected.",
                "timestamp": datetime.now().isoformat(),
                "project_directory": str(self.project_directory),
                "frontend_directory": str(self.frontend_directory),
                "frontend_plan_path": inputs["plan_path"],
                "browser_diagnostics": browser_diagnostics,
                "content_validation": source_validation,
                "generated_files": {
                    "jsx": str(self.jsx_path),
                    "css": str(self.css_path),
                },
                "changes_applied": False,
                "next_step": "No refinement required.",
            }

            self._last_summary = summary
            self._save_summary(summary)

            return summary

        logger.warning("Refinement required: %s", reasons)

        # -----------------------------------------------------
        # GENERATE JSX
        # -----------------------------------------------------

        refined_jsx = current_jsx
        content_validation = None

        for attempt in range(self.max_content_retries + 1):

            prompt = self._build_jsx_prompt(
                frontend_plan=frontend_plan,
                browser_diagnostics=browser_diagnostics,
                evaluation_report=evaluation_report,
                current_jsx=refined_jsx,
                missing_content=missing_content,
                repair_context=repair_context,
            )

            refined_jsx = self._generate_source(
                prompt=prompt,
                source_name="App.jsx",
                workflow_id=workflow_id,
            )

            content_validation = self._validate_required_content(
                refined_jsx,
                required_content,
            )

            if content_validation["passed"]:
                break

            missing_content = content_validation["missing_content"]

            logger.warning(
                "JSX content validation failed on attempt %s. "
                "Missing %s items.",
                attempt + 1,
                len(missing_content),
            )

        if (
            self.strict_content_validation
            and not content_validation["passed"]
        ):

            rejected_path = self._save_rejected_output(
                refined_jsx,
                current_css,
                content_validation,
            )

            self._last_summary = {
                "rejected_output": rejected_path,
                "content_validation": content_validation,
                "refinement_reasons": reasons,
                "browser_diagnostics": browser_diagnostics,
                "missing_identifier": browser_diagnostics.get(
                    "missing_identifier"
                ),
            }

            raise ValueError(
                "Refinement rejected after content repair attempts. "
                f"Coverage: {content_validation['coverage_percentage']}%. "
                f"Rejected output: {rejected_path}"
            )

        # -----------------------------------------------------
        # CSS REGENERATION
        # -----------------------------------------------------

        regenerate_css = self._should_regenerate_css(
            browser_diagnostics
        )

        refined_css = current_css

        if regenerate_css:

            logger.info("CSS/layout issue detected. Regenerating CSS.")

            css_prompt = self._build_css_prompt(
                frontend_plan=frontend_plan,
                browser_diagnostics=browser_diagnostics,
                current_css=current_css,
                refined_jsx=refined_jsx,
            )

            refined_css = self._generate_source(
                prompt=css_prompt,
                source_name="App.css",
                workflow_id=workflow_id,
            )

        else:

            logger.info("Preserving existing App.css.")

        # -----------------------------------------------------
        # SOURCE VALIDATION
        # -----------------------------------------------------

        output_validation = self._validate_generated_output(
            refined_jsx,
            refined_css,
        )

        if not output_validation["passed"]:

            rejected_path = self._save_rejected_output(
                refined_jsx,
                refined_css,
                output_validation,
            )

            self._last_summary = {
                "rejected_output": rejected_path,
                "output_validation": output_validation,
                "refinement_reasons": reasons,
                "browser_diagnostics": browser_diagnostics,
                "missing_identifier": browser_diagnostics.get(
                    "missing_identifier"
                ),
            }

            raise ValueError(
                "Frontend source validation failed: "
                f"{output_validation['errors']}. "
                f"Rejected output: {rejected_path}"
            )

        refined_jsx = output_validation["jsx_code"]
        refined_css = output_validation["css_code"]

        # -----------------------------------------------------
        # BACKUP AND WRITE
        # -----------------------------------------------------

        backup_path = self._create_backup()

        try:

            self._safe_write(self.jsx_path, refined_jsx)
            self._safe_write(self.css_path, refined_css)

            logger.info("Updated App.jsx saved.")

            if regenerate_css:
                logger.info("Updated App.css saved.")
            else:
                logger.info("Existing App.css preserved.")

            # -------------------------------------------------
            # PRODUCTION BUILD
            # -------------------------------------------------

            build_result = self._run_production_build()

            if build_result["status"] in {
                "failed",
                "timed_out",
                "error",
            }:

                rejected_path = self._save_rejected_output(
                    refined_jsx,
                    refined_css,
                    {
                        "passed": False,
                        "reason": "Production build did not pass.",
                        "build_result": build_result,
                    },
                )

                raise RuntimeError(
                    "Refined frontend failed production build. "
                    "Original frontend files will be restored. "
                    f"Rejected output: {rejected_path}. "
                    f"Build details: {build_result.get('error')}"
                )

            if build_result["status"] == "unavailable":

                logger.warning(
                    "Production build could not be verified."
                )

        except Exception:

            self._restore_backup(backup_path)
            raise

        # -----------------------------------------------------
        # FINAL STATUS
        # -----------------------------------------------------

        if build_result["status"] == "passed":

            final_status = "success"

            next_step = (
                "Run Browser Agent again to verify actual browser rendering."
            )

        else:

            final_status = "build_unavailable"

            next_step = (
                "Source was preserved, but production compilation was "
                "not verified. Ensure npm is available, run npm run build, "
                "and execute Browser Agent."
            )

        summary = {
            "workflow_id": workflow_id,
            "agent_name": self.agent_name,
            "status": final_status,
            "timestamp": datetime.now().isoformat(),
            "project_directory": str(self.project_directory),
            "frontend_directory": str(self.frontend_directory),
            "frontend_plan_path": inputs["plan_path"],
            "browser_results_path": str(self.browser_results_path),
            "repair_context_path": str(
                self.runtime_repair_context_path
            ),
            "refinement_reasons": reasons,
            "browser_diagnostics": browser_diagnostics,
            "repair_context_received": bool(repair_context),
            "missing_identifier": browser_diagnostics.get(
                "missing_identifier"
            ),
            "generated_files": {
                "jsx": str(self.jsx_path),
                "css": str(self.css_path),
            },
            "backup_directory": backup_path,
            "css_regenerated": regenerate_css,
            "content_validation": content_validation,
            "output_validation": {
                "passed": output_validation["passed"],
                "errors": output_validation["errors"],
                "warnings": output_validation["warnings"],
            },
            "production_build": build_result,
            "changes_applied": True,
            "statistics": {
                "required_content_count": len(required_content),
                "found_content_count": content_validation["found_count"],
                "missing_content_count": content_validation["missing_count"],
                "content_coverage": content_validation["coverage_percentage"],
                "jsx_characters": len(refined_jsx),
                "css_characters": len(refined_css),
            },
            "next_step": next_step,
        }

        self._last_summary = summary
        self._save_summary(summary)

        logger.info(
            "Refinement completed. Status: %s",
            final_status,
        )

        return summary

    # =========================================================
    # PUBLIC ENTRY POINT
    # =========================================================

    def refine(
        self,
        workflow_id: Optional[str] = None,
    ) -> dict:

        workflow_id = workflow_id or (
            f"refinement-{uuid.uuid4().hex[:12]}"
        )

        start_time = perf_counter()

        status = "success"
        error_message = None

        try:

            result = self._refine_impl(workflow_id)

            status = result.get("status", "success")

            return result

        except Exception as error:

            status = "failed"
            error_message = str(error)

            logger.exception("Refinement Agent failed.")

            self._save_failure_summary(
                workflow_id=workflow_id,
                error=error,
            )

            raise

        finally:

            duration_ms = (
                perf_counter() - start_time
            ) * 1000

            try:

                metrics_collector.record_agent_metrics(
                    agent_name=self.agent_name,
                    duration_ms=duration_ms,
                    status=status,
                    workflow_id=workflow_id,
                    error=error_message,
                )

            except Exception:

                logger.exception(
                    "Failed to record agent metrics."
                )

            logger.info(
                "Refinement execution time: %.2f ms",
                duration_ms,
            )


# =============================================================
# STANDALONE EXECUTION
# =============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    agent = RefinementAgent(
        project_directory="generated_projects/novadash-project-overview",
        frontend_plan_path=(
            "generated_projects/planning/sample/frontend_plan.json"
        ),
        browser_output_directory=(
            "generated_projects/novadash-project-overview/browser_output"
        ),
        max_retries=1,
        max_content_retries=2,
        strict_content_validation=True,
        minimum_content_coverage=90.0,
        build_timeout=180,
        regenerate_css_only_when_needed=True,
    )

    result = agent.refine(
        workflow_id="novadash-refinement-test-008"
    )

    print("\nREFINEMENT RESULT\n")

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
            default=str,
        )
    )
