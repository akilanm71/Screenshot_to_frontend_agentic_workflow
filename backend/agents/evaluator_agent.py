
import json
import logging
import re
import base64
import mimetypes
import uuid

from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import fitz
from PIL import Image

from backend.config.settings import settings
from backend.observability.llm_factory import (
    create_llm,
    invoke_llm
)
from backend.observability.metrics_collector import metrics_collector


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# EVALUATOR AGENT
# ---------------------------------------------------------

class EvaluatorAgent:
    """
    Evaluates generated React frontend against original
    screenshots or PDF documents.

    Evaluation dimensions:
        1. Visual similarity
        2. Content completeness
        3. Table accuracy
        4. Layout accuracy
        5. Runtime quality

    Observability:
        - LangSmith LLM tracing
        - LLM token usage
        - LLM execution latency
        - Agent execution latency
        - Workflow ID tracking

    Outputs:
        - evaluation_report.json
        - evaluation_summary.txt
    """

    def __init__(
        self,
        project_directory: str = (
            "generated_projects/novadash-project-overview"
        ),
        source_path: Optional[str] = None,
        extraction_result_path: Optional[str] = (
            "generated_projects/pdf_extraction/sample/"
            "extraction_result.json"
        ),
        model: Optional[str] = None
    ):

        self.project_directory = Path(
            project_directory
        ).resolve()

        self.source_path = (
            Path(source_path).resolve()
            if source_path
            else None
        )

        self.extraction_result_path = (
            Path(extraction_result_path).resolve()
            if extraction_result_path
            else None
        )

        self.model = model or settings.llm_model

        self.agent_name = "evaluator_agent"

        # Centralized LangChain LLM initialization.
        self.llm = create_llm(
            agent_name=self.agent_name,
            temperature=0,
            num_ctx=8192,
            num_predict=1500,
            model=self.model
        )

        # Input files
        self.plan_path = (
            self.project_directory / "frontend_plan.json"
        )

        self.frontend_directory = (
            self.project_directory / "frontend"
        )

        self.app_jsx_path = (
            self.frontend_directory / "src" / "App.jsx"
        )

        self.browser_directory = (
            self.project_directory / "browser_output"
        )

        self.browser_screenshot_path = (
            self.browser_directory / "browser_screenshot.png"
        )

        self.browser_results_path = (
            self.browser_directory / "browser_results.json"
        )

        # Output files
        self.evaluation_directory = (
            self.project_directory / "evaluation"
        )

        self.source_pages_directory = (
            self.evaluation_directory / "source_pages"
        )

        self.report_path = (
            self.evaluation_directory / "evaluation_report.json"
        )

        self.summary_path = (
            self.evaluation_directory / "evaluation_summary.txt"
        )

        logger.info("Evaluator Agent initialized")
        logger.info("Project: %s", self.project_directory)
        logger.info("Model: %s", self.model)


    # -----------------------------------------------------
    # LOAD JSON
    # -----------------------------------------------------

    def _load_json(self, path: Path) -> dict[str, Any]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required JSON file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)


    # -----------------------------------------------------
    # RESOLVE SOURCE INPUT
    # -----------------------------------------------------

    def _resolve_source_path(
        self,
        extraction_data: Optional[dict[str, Any]]
    ) -> Path:
        """
        Resolves original screenshot/PDF path.

        Priority:
            1. Explicit source_path
            2. file_path from PDF extraction result
        """

        if self.source_path:
            return self.source_path

        if extraction_data:
            extracted_path = extraction_data.get("file_path")

            if extracted_path:
                return Path(extracted_path)

        raise ValueError(
            "Original source file was not provided. "
            "Pass source_path when creating EvaluatorAgent."
        )


    # -----------------------------------------------------
    # VALIDATE INPUT FILES
    # -----------------------------------------------------

    def _validate_inputs(self) -> dict[str, Any]:
        """
        Checks all required evaluation artifacts.
        """

        logger.info("Validating evaluator inputs")

        required_files = [
            self.plan_path,
            self.app_jsx_path,
            self.browser_screenshot_path,
            self.browser_results_path
        ]

        for path in required_files:

            if not path.exists():
                raise FileNotFoundError(
                    f"Required evaluation input missing: {path}"
                )

        plan_data = self._load_json(
            self.plan_path
        )

        browser_data = self._load_json(
            self.browser_results_path
        )

        extraction_data = None

        if (
            self.extraction_result_path
            and self.extraction_result_path.exists()
        ):
            extraction_data = self._load_json(
                self.extraction_result_path
            )

        source_path = self._resolve_source_path(
            extraction_data
        )

        if not source_path.exists():
            raise FileNotFoundError(
                f"Original source file not found: {source_path}"
            )

        logger.info("All required inputs validated")

        return {
            "plan": plan_data,
            "browser": browser_data,
            "extraction": extraction_data,
            "source_path": source_path
        }


    # -----------------------------------------------------
    # DETERMINE SOURCE TYPE
    # -----------------------------------------------------

    def _get_source_type(
        self,
        source_path: Path,
        plan_data: dict[str, Any]
    ) -> str:

        extension = source_path.suffix.lower()

        if extension == ".pdf":
            return "pdf"

        if extension in [".png", ".jpg", ".jpeg", ".webp"]:
            return "screenshot"

        source_type = plan_data.get(
            "source_type",
            "unknown"
        )

        if source_type not in ["pdf", "screenshot"]:
            raise ValueError(
                f"Unsupported source type: {source_type}"
            )

        return source_type


    # -----------------------------------------------------
    # PREPARE SOURCE IMAGES
    # -----------------------------------------------------

    def _prepare_source_images(
        self,
        source_path: Path,
        source_type: str
    ) -> list[Path]:
        """
        Converts source inputs into images.

        Screenshot:
            Uses the original image directly.

        PDF:
            Renders every PDF page independently.
        """

        self.source_pages_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        source_images = []

        if source_type == "screenshot":

            with Image.open(source_path) as image:
                image.verify()

            source_images.append(source_path)

            logger.info(
                "Original screenshot loaded: %s",
                source_path
            )

            return source_images

        logger.info("Rendering original PDF pages")

        pdf_document = fitz.open(source_path)

        try:

            for page_index in range(len(pdf_document)):

                page = pdf_document.load_page(
                    page_index
                )

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )

                image_path = (
                    self.source_pages_directory
                    / f"source_page_{page_index + 1:03d}.png"
                )

                pixmap.save(str(image_path))

                source_images.append(image_path)

                logger.info(
                    "Rendered PDF page %s/%s",
                    page_index + 1,
                    len(pdf_document)
                )

        finally:
            pdf_document.close()

        if not source_images:
            raise ValueError(
                "No pages could be rendered from the PDF."
            )

        return source_images


    # -----------------------------------------------------
    # TEXT NORMALIZATION
    # -----------------------------------------------------

    def _normalize_text(self, text: str) -> str:

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9]+",
            " ",
            text
        )

        return " ".join(text.split())


    # -----------------------------------------------------
    # EXTRACT EXPECTED CONTENT
    # -----------------------------------------------------

    def _extract_plan_content(
        self,
        plan_data: dict[str, Any]
    ) -> list[str]:
        """
        Extracts meaningful content strings from FrontendPlan.
        """

        expected = []

        def walk(value: Any):

            if isinstance(value, dict):

                for key, item in value.items():

                    if key in [
                        "content",
                        "page_title",
                        "section_name",
                        "description"
                    ] and isinstance(item, str):

                        expected.append(item)

                    walk(item)

            elif isinstance(value, list):

                for item in value:
                    walk(item)

        walk(plan_data)

        return expected


    # -----------------------------------------------------
    # EXTRACT PDF CONTENT
    # -----------------------------------------------------

    def _extract_pdf_content(
        self,
        extraction_data: Optional[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        """
        Extracts expected PDF text and table values.

        Returns:
            expected_content
            expected_table_values
        """

        expected_content = []
        expected_table_values = []

        if not extraction_data:
            return expected_content, expected_table_values

        pages = extraction_data.get(
            "pages",
            []
        )

        for page in pages:

            extracted_text = page.get(
                "extracted_text",
                ""
            )

            if extracted_text:

                for line in extracted_text.splitlines():

                    line = line.strip()

                    if line:
                        expected_content.append(line)

            tables = page.get(
                "tables",
                []
            )

            for table in tables:

                rows = table.get(
                    "rows",
                    []
                )

                for row in rows:

                    for cell in row:

                        if cell is not None:

                            cell_text = str(cell).strip()

                            if cell_text:
                                expected_table_values.append(
                                    cell_text
                                )

        return (
            expected_content,
            expected_table_values
        )


    # -----------------------------------------------------
    # LOAD GENERATED FRONTEND CODE
    # -----------------------------------------------------

    def _load_generated_code(self) -> str:

        if not self.app_jsx_path.exists():
            raise FileNotFoundError(
                f"Generated React code not found: "
                f"{self.app_jsx_path}"
            )

        return self.app_jsx_path.read_text(
            encoding="utf-8"
        )


    # -----------------------------------------------------
    # GET AVAILABLE BROWSER TEXT
    # -----------------------------------------------------

    def _get_browser_text(
        self,
        browser_data: dict[str, Any]
    ) -> str:
        """
        Reads text collected by the Browser Agent.
        """

        page_data = browser_data.get(
            "page",
            {}
        )

        return page_data.get(
            "body_text_preview",
            ""
        )


    # -----------------------------------------------------
    # CONTENT COMPLETENESS EVALUATION
    # -----------------------------------------------------

    def _evaluate_content(
        self,
        plan_data: dict[str, Any],
        extraction_data: Optional[dict[str, Any]],
        generated_code: str,
        browser_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Compares expected content against generated JSX
        and browser-rendered text.

        A match in JSX alone confirms content exists in
        source code, not necessarily that it rendered.
        """

        plan_content = self._extract_plan_content(
            plan_data
        )

        pdf_content, table_values = self._extract_pdf_content(
            extraction_data
        )

        expected_content = list(
            dict.fromkeys(
                plan_content + pdf_content
            )
        )

        browser_text = self._get_browser_text(
            browser_data
        )

        normalized_code = self._normalize_text(
            generated_code
        )

        normalized_browser = self._normalize_text(
            browser_text
        )

        matched_in_code = []
        matched_in_browser = []
        missing_content = []

        for item in expected_content:

            normalized_item = self._normalize_text(
                item
            )

            if len(normalized_item) < 3:
                continue

            code_match = (
                normalized_item in normalized_code
            )

            browser_match = (
                normalized_item in normalized_browser
            )

            if code_match:
                matched_in_code.append(item)

            if browser_match:
                matched_in_browser.append(item)

            if not code_match:
                missing_content.append(item)

        table_values = list(
            dict.fromkeys(table_values)
        )

        matched_table_values = []
        missing_table_values = []

        for value in table_values:

            normalized_value = self._normalize_text(
                value
            )

            if len(normalized_value) < 2:
                continue

            if (
                normalized_value in normalized_code
                or normalized_value in normalized_browser
            ):
                matched_table_values.append(value)

            else:
                missing_table_values.append(value)

        expected_count = len([
            item for item in expected_content
            if len(self._normalize_text(item)) >= 3
        ])

        matched_count = (
            expected_count - len(missing_content)
        )

        content_score = (
            round(
                (matched_count / expected_count) * 100,
                2
            )
            if expected_count
            else 100.0
        )

        expected_table_count = len([
            item for item in table_values
            if len(self._normalize_text(item)) >= 2
        ])

        matched_table_count = (
            expected_table_count - len(missing_table_values)
        )

        table_score = (
            round(
                (matched_table_count / expected_table_count) * 100,
                2
            )
            if expected_table_count
            else 100.0
        )

        return {
            "content_completeness_score": content_score,
            "table_accuracy_score": table_score,
            "expected_content_count": expected_count,
            "matched_content_count": matched_count,
            "matched_in_generated_code": matched_in_code,
            "matched_in_browser_text": matched_in_browser,
            "missing_content": missing_content,
            "expected_table_value_count": expected_table_count,
            "matched_table_value_count": matched_table_count,
            "matched_table_values": matched_table_values,
            "missing_table_values": missing_table_values,
            "browser_text_length": len(browser_text),
            "browser_text_note": (
                "Browser Agent currently stores only the first "
                "1000 characters of body text. Therefore, browser "
                "text matches are not exhaustive."
            )
        }


    # -----------------------------------------------------
    # BUILD VISUAL EVALUATION PROMPT
    # -----------------------------------------------------

    def _build_visual_prompt(
        self,
        source_type: str,
        page_number: int,
        total_pages: int,
        plan_data: dict[str, Any]
    ) -> str:

        return f"""
You are an expert UI visual quality evaluator.

You are given TWO images:

IMAGE 1:
Original source page or screenshot.

IMAGE 2:
Generated React application screenshot.

SOURCE TYPE:
{source_type}

SOURCE PAGE:
{page_number} of {total_pages}

PROJECT PLAN:
{json.dumps(plan_data, ensure_ascii=False)[:7000]}

Compare the images carefully.

Evaluate these dimensions:

1. Visual similarity:
   Overall visual resemblance between source and generated UI.

2. Layout accuracy:
   Position, arrangement, spacing and alignment of components.

3. Typography:
   Font hierarchy, font size, font weight and readability.

4. Color accuracy:
   Backgrounds, text colors, borders and visual contrast.

5. Component accuracy:
   Cards, sections, tables, navigation and other UI elements.

IMPORTANT:

- Do not judge only by content.
- Evaluate visual arrangement and styling.
- For PDF documents, consider the original page layout.
- The generated application may be a web representation
  rather than an exact pixel-for-pixel copy.
- Do not penalize responsive web layout unnecessarily.
- If the browser screenshot contains multiple pages,
  evaluate the relevant source content and its corresponding
  visible section.
- Do not invent visual defects.

Return ONLY valid JSON in this exact structure:

{{
    "visual_similarity": 0,
    "layout_accuracy": 0,
    "typography_accuracy": 0,
    "color_accuracy": 0,
    "component_accuracy": 0,
    "issues": [
        "Specific visual issue"
    ],
    "recommendations": [
        "Specific improvement"
    ]
}}

All scores must be between 0 and 100.
"""


    # -----------------------------------------------------
    # IMAGE ENCODING
    # -----------------------------------------------------

    def _image_to_data_url(
        self,
        image_path: Path
    ) -> str:
        """
        Converts image to base64 data URL for ChatOllama.
        """

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        mime_type, _ = mimetypes.guess_type(
            str(image_path)
        )

        if not mime_type:
            mime_type = "image/png"

        with image_path.open("rb") as image_file:

            encoded_image = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return (
            f"data:{mime_type};base64,{encoded_image}"
        )


    # -----------------------------------------------------
    # PARSE JSON FROM LLM
    # -----------------------------------------------------

    def _parse_llm_json(
        self,
        content: str
    ) -> dict[str, Any]:

        content = content.strip()

        try:
            return json.loads(content)

        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            content,
            flags=re.DOTALL
        )

        if not match:
            raise ValueError(
                "Vision model did not return valid JSON."
            )

        return json.loads(
            match.group(0)
        )


    # -----------------------------------------------------
    # NORMALIZE SCORE
    # -----------------------------------------------------

    def _safe_score(
        self,
        value: Any,
        default: float = 0
    ) -> float:

        try:
            score = float(value)

            return max(
                0.0,
                min(100.0, score)
            )

        except (TypeError, ValueError):
            return default


    # -----------------------------------------------------
    # VISUAL EVALUATION
    # -----------------------------------------------------

    def _evaluate_visual_page(
        self,
        source_image: Path,
        browser_screenshot: Path,
        page_number: int,
        total_pages: int,
        source_type: str,
        plan_data: dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Multimodal image evaluation using ChatOllama.

        Tracks individual LLM calls through LangSmith
        and the centralized metrics collector.
        """

        logger.info(
            "Evaluating visual similarity for page %s/%s",
            page_number,
            total_pages
        )

        prompt = self._build_visual_prompt(
            source_type=source_type,
            page_number=page_number,
            total_pages=total_pages,
            plan_data=plan_data
        )

        try:

            source_image_url = self._image_to_data_url(
                source_image
            )

            browser_image_url = self._image_to_data_url(
                browser_screenshot
            )

            human_content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": source_image_url
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": browser_image_url
                    }
                }
            ]

            response = invoke_llm(
                llm=self.llm,
                prompt=[
                    (
                        "system",
                        "You are a precise visual UI evaluator. "
                        "Compare the provided images and return "
                        "only the requested JSON."
                    ),
                    (
                        "human",
                        human_content
                    )
                ],
                agent_name=(
                    f"{self.agent_name}_visual_page_{page_number}"
                ),
                workflow_id=workflow_id
            )

            content = response.content or ""

            if isinstance(content, list):

                content = "\n".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                )

            result = self._parse_llm_json(
                content
            )

            logger.info(
                "Visual evaluation completed for page %s",
                page_number
            )

            return {
                "page_number": page_number,
                "status": "success",

                "visual_similarity": self._safe_score(
                    result.get("visual_similarity")
                ),

                "layout_accuracy": self._safe_score(
                    result.get("layout_accuracy")
                ),

                "typography_accuracy": self._safe_score(
                    result.get("typography_accuracy")
                ),

                "color_accuracy": self._safe_score(
                    result.get("color_accuracy")
                ),

                "component_accuracy": self._safe_score(
                    result.get("component_accuracy")
                ),

                "issues": result.get(
                    "issues",
                    []
                ),

                "recommendations": result.get(
                    "recommendations",
                    []
                )
            }

        except Exception as error:

            logger.exception(
                "Visual evaluation failed for page %s",
                page_number
            )

            return {
                "page_number": page_number,
                "status": "failed",
                "visual_similarity": None,
                "layout_accuracy": None,
                "typography_accuracy": None,
                "color_accuracy": None,
                "component_accuracy": None,
                "issues": [
                    f"Visual evaluation failed: {str(error)}"
                ],
                "recommendations": []
            }


    # -----------------------------------------------------
    # EVALUATE ALL VISUAL PAGES
    # -----------------------------------------------------

    def _evaluate_visual(
        self,
        source_images: list[Path],
        browser_screenshot: Path,
        source_type: str,
        plan_data: dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Evaluates every original source page against
        the generated browser screenshot.
        """

        page_results = []

        total_pages = len(source_images)

        for index, source_image in enumerate(
            source_images,
            start=1
        ):

            result = self._evaluate_visual_page(
                source_image=source_image,
                browser_screenshot=browser_screenshot,
                page_number=index,
                total_pages=total_pages,
                source_type=source_type,
                plan_data=plan_data,
                workflow_id=workflow_id
            )

            page_results.append(result)

        successful_results = [
            result
            for result in page_results
            if result["status"] == "success"
        ]

        if not successful_results:

            return {
                "status": "failed",
                "page_count": total_pages,
                "evaluated_page_count": 0,
                "visual_similarity_score": None,
                "layout_accuracy_score": None,
                "page_results": page_results
            }

        def average(key: str) -> float:

            values = [
                result[key]
                for result in successful_results
            ]

            return round(
                sum(values) / len(values),
                2
            )

        return {
            "status": (
                "success"
                if len(successful_results) == total_pages
                else "partial"
            ),
            "page_count": total_pages,
            "evaluated_page_count": len(successful_results),
            "visual_similarity_score": average(
                "visual_similarity"
            ),
            "layout_accuracy_score": average(
                "layout_accuracy"
            ),
            "typography_accuracy_score": average(
                "typography_accuracy"
            ),
            "color_accuracy_score": average(
                "color_accuracy"
            ),
            "component_accuracy_score": average(
                "component_accuracy"
            ),
            "page_results": page_results
        }


    # -----------------------------------------------------
    # RUNTIME EVALUATION
    # -----------------------------------------------------

    def _evaluate_runtime(
        self,
        browser_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Evaluates browser execution quality.
        """

        console_errors = browser_data.get(
            "console_errors",
            []
        )

        page_errors = browser_data.get(
            "page_errors",
            []
        )

        failed_requests = browser_data.get(
            "failed_requests",
            []
        )

        page_info = browser_data.get(
            "page",
            {}
        )

        http_status = page_info.get(
            "http_status"
        )

        total_errors = (
            len(console_errors)
            + len(page_errors)
            + len(failed_requests)
        )

        runtime_score = 100.0

        runtime_score -= len(console_errors) * 5
        runtime_score -= len(page_errors) * 15
        runtime_score -= len(failed_requests) * 5

        if http_status is not None and http_status >= 400:
            runtime_score -= 30

        runtime_score = max(
            0.0,
            runtime_score
        )

        return {
            "runtime_quality_score": round(
                runtime_score,
                2
            ),
            "http_status": http_status,
            "console_error_count": len(console_errors),
            "page_error_count": len(page_errors),
            "failed_request_count": len(failed_requests),
            "total_error_count": total_errors,
            "console_errors": console_errors,
            "page_errors": page_errors,
            "failed_requests": failed_requests
        }


    # -----------------------------------------------------
    # COLLECT ALL ISSUES
    # -----------------------------------------------------

    def _collect_issues(
        self,
        visual_results: dict[str, Any],
        content_results: dict[str, Any],
        runtime_results: dict[str, Any]
    ) -> list[dict[str, str]]:

        issues = []

        for page in visual_results.get(
            "page_results",
            []
        ):

            for issue in page.get(
                "issues",
                []
            ):

                issues.append({
                    "category": "visual",
                    "severity": "medium",
                    "description": str(issue)
                })

        for item in content_results.get(
            "missing_content",
            []
        ):

            issues.append({
                "category": "content",
                "severity": "high",
                "description": (
                    f"Expected content missing from generated "
                    f"React source: {item}"
                )
            })

        for item in content_results.get(
            "missing_table_values",
            []
        ):

            issues.append({
                "category": "table",
                "severity": "high",
                "description": (
                    f"Expected table value missing: {item}"
                )
            })

        for error in runtime_results.get(
            "page_errors",
            []
        ):

            issues.append({
                "category": "runtime",
                "severity": "high",
                "description": str(error)
            })

        for error in runtime_results.get(
            "console_errors",
            []
        ):

            issues.append({
                "category": "runtime",
                "severity": "medium",
                "description": str(error)
            })

        for request in runtime_results.get(
            "failed_requests",
            []
        ):

            issues.append({
                "category": "network",
                "severity": "medium",
                "description": str(request)
            })

        return issues


    # -----------------------------------------------------
    # COLLECT RECOMMENDATIONS
    # -----------------------------------------------------

    def _collect_recommendations(
        self,
        visual_results: dict[str, Any],
        content_results: dict[str, Any],
        runtime_results: dict[str, Any]
    ) -> list[str]:

        recommendations = []

        for page in visual_results.get(
            "page_results",
            []
        ):

            recommendations.extend(
                page.get(
                    "recommendations",
                    []
                )
            )

        if content_results.get("missing_content"):

            recommendations.append(
                "Restore all missing headings, paragraphs and "
                "document content in App.jsx."
            )

        if content_results.get("missing_table_values"):

            recommendations.append(
                "Correct the generated HTML tables to preserve "
                "all original table values and rows."
            )

        if runtime_results.get("page_error_count", 0):

            recommendations.append(
                "Fix JavaScript runtime errors before visual refinement."
            )

        if runtime_results.get("failed_request_count", 0):

            recommendations.append(
                "Investigate failed network requests and missing assets."
            )

        return list(
            dict.fromkeys(recommendations)
        )


    # -----------------------------------------------------
    # CALCULATE OVERALL SCORE
    # -----------------------------------------------------

    def _calculate_overall_score(
        self,
        visual_results: dict[str, Any],
        content_results: dict[str, Any],
        runtime_results: dict[str, Any]
    ) -> Optional[float]:
        """
        Weighted evaluation score.

        Weights:
            Visual similarity: 40%
            Content completeness: 30%
            Table accuracy: 15%
            Runtime quality: 5%
            Layout accuracy: 10%
        """

        visual = visual_results.get(
            "visual_similarity_score"
        )

        layout = visual_results.get(
            "layout_accuracy_score"
        )

        if visual is None or layout is None:
            return None

        content = content_results[
            "content_completeness_score"
        ]

        table = content_results[
            "table_accuracy_score"
        ]

        runtime = runtime_results[
            "runtime_quality_score"
        ]

        overall = (
            visual * 0.40
            + content * 0.30
            + table * 0.15
            + runtime * 0.05
            + layout * 0.10
        )

        return round(
            overall,
            2
        )


    # -----------------------------------------------------
    # SAVE EVALUATION REPORT
    # -----------------------------------------------------

    def _save_report(
        self,
        report: dict[str, Any]
    ) -> None:

        self.evaluation_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.report_path.write_text(
            json.dumps(
                report,
                indent=4,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        logger.info(
            "Evaluation report saved: %s",
            self.report_path
        )


    # -----------------------------------------------------
    # SAVE HUMAN-READABLE SUMMARY
    # -----------------------------------------------------

    def _save_summary(
        self,
        report: dict[str, Any]
    ) -> None:

        lines = [
            "FRONTEND EVALUATION SUMMARY",
            "=" * 50,
            "",
            f"Project: {report['project_name']}",
            f"Source type: {report['source_type']}",
            f"Overall score: {report['overall_score']}",
            "",
            "EVALUATION SCORES",
            "-" * 30,
            f"Visual similarity: "
            f"{report['visual_evaluation'].get('visual_similarity_score')}",
            f"Layout accuracy: "
            f"{report['visual_evaluation'].get('layout_accuracy_score')}",
            f"Content completeness: "
            f"{report['content_evaluation']['content_completeness_score']}",
            f"Table accuracy: "
            f"{report['content_evaluation']['table_accuracy_score']}",
            f"Runtime quality: "
            f"{report['runtime_evaluation']['runtime_quality_score']}",
            "",
            "ISSUES",
            "-" * 30
        ]

        for issue in report["issues"]:

            lines.append(
                f"[{issue['severity'].upper()}] "
                f"{issue['category']}: "
                f"{issue['description']}"
            )

        lines.extend([
            "",
            "RECOMMENDATIONS",
            "-" * 30
        ])

        for recommendation in report["recommendations"]:

            lines.append(
                f"- {recommendation}"
            )

        lines.extend([
            "",
            f"Refinement required: "
            f"{report['refinement_required']}"
        ])

        self.summary_path.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        logger.info(
            "Evaluation summary saved: %s",
            self.summary_path
        )


    # -----------------------------------------------------
    # INTERNAL EVALUATION WORKFLOW
    # -----------------------------------------------------

    def _evaluate_impl(
        self,
        workflow_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Main evaluation workflow.

        1. Validate inputs.
        2. Load original source.
        3. Render PDF pages if necessary.
        4. Compare source images with browser screenshot.
        5. Check content and table completeness.
        6. Check browser errors.
        7. Calculate scores.
        8. Save evaluation report.
        """

        # Step 1: Validate
        inputs = self._validate_inputs()

        plan_data = inputs["plan"]
        browser_data = inputs["browser"]
        extraction_data = inputs["extraction"]
        source_path = inputs["source_path"]

        source_type = self._get_source_type(
            source_path,
            plan_data
        )

        logger.info(
            "Source type detected: %s",
            source_type
        )

        # Step 2: Prepare original images
        source_images = self._prepare_source_images(
            source_path=source_path,
            source_type=source_type
        )

        logger.info(
            "Original source pages prepared: %s",
            len(source_images)
        )

        # Step 3: Visual evaluation
        visual_results = self._evaluate_visual(
            source_images=source_images,
            browser_screenshot=self.browser_screenshot_path,
            source_type=source_type,
            plan_data=plan_data,
            workflow_id=workflow_id
        )

        # Step 4: Load generated JSX
        generated_code = self._load_generated_code()

        # Step 5: Content evaluation
        content_results = self._evaluate_content(
            plan_data=plan_data,
            extraction_data=extraction_data,
            generated_code=generated_code,
            browser_data=browser_data
        )

        # Step 6: Runtime evaluation
        runtime_results = self._evaluate_runtime(
            browser_data
        )

        # Step 7: Consolidate issues
        issues = self._collect_issues(
            visual_results=visual_results,
            content_results=content_results,
            runtime_results=runtime_results
        )

        recommendations = self._collect_recommendations(
            visual_results=visual_results,
            content_results=content_results,
            runtime_results=runtime_results
        )

        # Step 8: Overall score
        overall_score = self._calculate_overall_score(
            visual_results=visual_results,
            content_results=content_results,
            runtime_results=runtime_results
        )

        refinement_required = (
            overall_score is None
            or overall_score < 90
            or len(issues) > 0
            or visual_results["status"] != "success"
        )

        report = {
            "project_name": plan_data.get(
                "project_name",
                self.project_directory.name
            ),
            "source_type": source_type,
            "evaluation_model": self.model,
            "overall_score": overall_score,
            "visual_evaluation": visual_results,
            "content_evaluation": content_results,
            "runtime_evaluation": runtime_results,
            "issues": issues,
            "recommendations": recommendations,
            "refinement_required": refinement_required,
            "evaluation_status": (
                "completed"
                if visual_results["status"] == "success"
                else "partial"
            )
        }

        # Step 9: Save reports
        self._save_report(report)

        self._save_summary(report)

        logger.info(
            "Evaluator Agent execution completed"
        )

        return report


    # -----------------------------------------------------
    # PUBLIC EVALUATION ENTRY POINT
    # -----------------------------------------------------

    def evaluate(
        self,
        workflow_id: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Public evaluation entry point.

        Tracks total agent execution time.
        """

        workflow_id = workflow_id or str(uuid.uuid4())

        start_time = perf_counter()
        status = "failed"

        logger.info(
            "Evaluator Agent execution started"
        )

        logger.info(
            "Workflow ID: %s",
            workflow_id
        )

        try:

            result = self._evaluate_impl(
                workflow_id=workflow_id
            )

            status = "success"

            logger.info(
                "Evaluator Agent completed successfully"
            )

            return result

        finally:

            duration_ms = (
                perf_counter() - start_time
            ) * 1000

            metrics_collector.record_agent_metrics(
                agent_name=self.agent_name,
                duration_ms=duration_ms,
                status=status,
                workflow_id=workflow_id
            )

            logger.info(
                "Evaluator Agent execution time: %.2f ms",
                duration_ms
            )


# ---------------------------------------------------------
# STANDALONE EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":

    logger.info("Starting Evaluator Agent")

    agent = EvaluatorAgent(
        project_directory=(
            "generated_projects/"
            "novadash-project-overview"
        ),
        extraction_result_path=(
            "generated_projects/"
            "pdf_extraction/"
            "sample/"
            "extraction_result.json"
        )
    )

    result = agent.evaluate(
        workflow_id="novadash-evaluator-test-001"
    )

    print("\n" + "=" * 60)
    print("EVALUATOR AGENT RESULT")
    print("=" * 60)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\nEvaluation report:")
    print(
        "generated_projects/novadash-project-overview/"
        "evaluation/evaluation_report.json"
    )
