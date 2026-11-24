
import json
import logging
import os
import re
import socket
import subprocess
import time

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
)


logger = logging.getLogger(__name__)


class BrowserAgent:
    """
    Browser validation agent for generated React/Vite applications.

    Responsibilities:
    - Start the generated frontend.
    - Open it using Playwright.
    - Detect JavaScript and browser errors.
    - Check whether React rendered content.
    - Perform optional semantic validation using frontend_plan.
    - Capture screenshots and diagnostic reports.

    This agent diagnoses problems. It does not modify generated source files.
    """

    def __init__(
        self,
        project_directory: Optional[str] = None,
        frontend_plan: Optional[dict] = None,
        plan: Optional[dict] = None,
        plan_path: Optional[str] = None,
        project_dir: Optional[str] = None,
        project_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5173,
        browser_name: str = "chromium",
        headless: bool = True,
        keep_server_running: bool = False,
        startup_timeout: int = 60,
        navigation_timeout: int = 30000,
        render_timeout: int = 15000,
        screenshot_timeout: int = 60000,
        viewport_width: int = 1440,
        viewport_height: int = 1000,
        semantic_validation: bool = True,
        minimum_content_match_ratio: float = 0.80,
        check_horizontal_overflow: bool = True,
        fail_on_console_errors: bool = True,
        reuse_existing_server: bool = False,
        **kwargs,
    ):
        selected_directory = (
            project_directory
            or project_dir
            or project_path
            or "generated_projects/sample-dashboard/frontend"
        )

        self.project_directory = Path(selected_directory).resolve()

        self.frontend_plan = (
            frontend_plan
            if isinstance(frontend_plan, dict)
            else plan if isinstance(plan, dict) else None
        )

        self.plan_path = (
            Path(plan_path).resolve()
            if plan_path
            else None
        )

        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"

        self.browser_name = browser_name
        self.headless = headless
        self.keep_server_running = keep_server_running

        self.startup_timeout = startup_timeout
        self.navigation_timeout = navigation_timeout
        self.render_timeout = render_timeout
        self.screenshot_timeout = screenshot_timeout

        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self.semantic_validation = semantic_validation
        self.minimum_content_match_ratio = minimum_content_match_ratio
        self.check_horizontal_overflow = check_horizontal_overflow
        self.fail_on_console_errors = fail_on_console_errors
        self.reuse_existing_server = reuse_existing_server

        self.source_directory = self.project_directory / "src"

        self.output_directory = (
            self.project_directory.parent / "browser_output"
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.screenshot_path = (
            self.output_directory / f"browser_{timestamp}.png"
        )

        self.report_path = (
            self.output_directory / f"browser_report_{timestamp}.json"
        )

        self.repair_context_path = (
            self.output_directory / f"repair_context_{timestamp}.json"
        )

        self.server_process = None
        self.owns_server = False

        self.page_errors = []
        self.console_errors = []
        self.failed_requests = []
        self.http_errors = []

        self.browser_report = {}

    # =========================================================
    # BASIC UTILITIES
    # =========================================================

    @staticmethod
    def _now():
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _write_json(path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                data,
                indent=4,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    def _is_port_available(self) -> bool:
        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as sock:
            sock.settimeout(1)

            return sock.connect_ex(
                (self.host, self.port)
            ) != 0

    # =========================================================
    # PROJECT VALIDATION
    # =========================================================

    def _validate_project(self) -> dict:

        required_files = [
            self.project_directory / "package.json",
            self.source_directory / "App.jsx",
            self.source_directory / "main.jsx",
        ]

        missing_files = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        package_json_path = (
            self.project_directory / "package.json"
        )

        package_data = {}

        if package_json_path.exists():
            try:
                package_data = json.loads(
                    package_json_path.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception as exc:
                missing_files.append(
                    f"Invalid package.json: {exc}"
                )

        scripts = package_data.get("scripts", {})

        if not scripts.get("dev"):
            missing_files.append(
                "package.json does not define scripts.dev"
            )

        node_modules = (
            self.project_directory / "node_modules"
        )

        if not node_modules.exists():
            missing_files.append(
                "node_modules directory is missing"
            )

        return {
            "valid": len(missing_files) == 0,
            "project_directory": str(
                self.project_directory
            ),
            "missing_files": missing_files,
            "package_name": package_data.get("name"),
            "dev_script": scripts.get("dev"),
        }

    # =========================================================
    # PLAN RESOLUTION
    # =========================================================

    def _load_plan(self) -> Optional[dict]:

        if isinstance(self.frontend_plan, dict):
            return self.frontend_plan

        if self.plan_path and self.plan_path.exists():
            try:
                loaded = json.loads(
                    self.plan_path.read_text(
                        encoding="utf-8"
                    )
                )

                if isinstance(loaded, dict):
                    return loaded

            except Exception:
                logger.exception(
                    "Could not load frontend plan"
                )

        # Intentionally do not load a default sample plan.
        # The workflow should supply frontend_plan.
        return None

    # =========================================================
    # PLAN CONTENT EXTRACTION
    # =========================================================

    def _extract_plan_text(self, plan: Optional[dict]) -> list:

        if not isinstance(plan, dict):
            return []

        ignored_keys = {
            "id",
            "key",
            "type",
            "component",
            "component_type",
            "class",
            "className",
            "style",
            "styles",
            "icon",
            "icon_name",
            "image",
            "image_url",
            "src",
            "href",
            "url",
            "metadata",
            "position",
            "layout",
            "width",
            "height",
            "margin",
            "padding",
            "gap",
        }

        values = []

        def walk(value, key=None):

            if key in ignored_keys:
                return

            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    walk(child_value, child_key)

            elif isinstance(value, list):
                for item in value:
                    walk(item)

            elif isinstance(value, str):
                cleaned = value.strip()

                if (
                    len(cleaned) >= 3
                    and not cleaned.startswith(("http://", "https://"))
                ):
                    values.append(cleaned)

        walk(plan)

        # Remove duplicates while preserving order.
        return list(dict.fromkeys(values))

    # =========================================================
    # SERVER MANAGEMENT
    # =========================================================

    def _start_server(self):

        if not self._is_port_available():

            if self.reuse_existing_server:
                logger.warning(
                    "Port %s is occupied. Reusing existing server "
                    "because reuse_existing_server=True.",
                    self.port,
                )
                return

            raise RuntimeError(
                f"Port {self.port} is already occupied. "
                "BrowserAgent will not silently test an unknown "
                "frontend. Stop the existing server or configure "
                "a different port."
            )

        if os.name == "nt":

            vite_cli = (
                self.project_directory
                / "node_modules"
                / "vite"
                / "bin"
                / "vite.js"
            )

            if not vite_cli.exists():
                raise FileNotFoundError(
                    f"Vite CLI not found: {vite_cli}"
                )

            command = [
                "node",
                str(vite_cli),
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--strictPort",
            ]

        else:

            command = [
                "npm",
                "run",
                "dev",
                "--",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--strictPort",
            ]

        logger.info(
            "Starting frontend: %s",
            self.project_directory,
        )

        self.server_process = subprocess.Popen(
            command,
            cwd=str(self.project_directory),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        self.owns_server = True

        start_time = time.time()

        while time.time() - start_time < self.startup_timeout:

            if self.server_process.poll() is not None:

                output = ""

                if self.server_process.stdout:
                    output = self.server_process.stdout.read()

                raise RuntimeError(
                    "Frontend server exited during startup.\n"
                    + output[-5000:]
                )

            try:
                import urllib.request

                with urllib.request.urlopen(
                    self.url,
                    timeout=2,
                ) as response:

                    if response.status == 200:
                        logger.info(
                            "Frontend server is ready: %s",
                            self.url,
                        )
                        return

            except Exception:
                time.sleep(1)

        raise TimeoutError(
            f"Frontend did not become ready within "
            f"{self.startup_timeout} seconds: {self.url}"
        )

    def _stop_server(self):

        if (
            self.server_process
            and self.owns_server
            and not self.keep_server_running
        ):

            if self.server_process.poll() is None:

                self.server_process.terminate()

                try:
                    self.server_process.wait(timeout=5)

                except subprocess.TimeoutExpired:
                    self.server_process.kill()

                    self.server_process.wait(timeout=5)

            logger.info("Frontend server stopped.")

    # =========================================================
    # BROWSER EVENT HANDLERS
    # =========================================================

    def _attach_browser_listeners(self, page):

        def on_page_error(error):

            self.page_errors.append({
                "message": str(error),
                "timestamp": self._now(),
            })

        def on_console(message):

            if message.type == "error":

                self.console_errors.append({
                    "message": message.text,
                    "timestamp": self._now(),
                })

        def on_request_failed(request):

            self.failed_requests.append({
                "url": request.url,
                "method": request.method,
                "failure": request.failure,
                "timestamp": self._now(),
            })

        def on_response(response):

            if response.status >= 400:

                self.http_errors.append({
                    "url": response.url,
                    "status": response.status,
                    "timestamp": self._now(),
                })

        page.on("pageerror", on_page_error)
        page.on("console", on_console)
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)

    # =========================================================
    # RUNTIME ERROR CLASSIFICATION
    # =========================================================

    def _classify_runtime_error(self) -> dict:

        messages = [
            item.get("message", "")
            for item in self.page_errors
        ]

        messages.extend(
            item.get("message", "")
            for item in self.console_errors
        )

        combined = "\n".join(messages)

        missing_identifier = None

        patterns = [
            r"([A-Za-z_$][\w$]*) is not defined",
            r"Can't find variable:\s*([A-Za-z_$][\w$]*)",
            r"ReferenceError:\s*([A-Za-z_$][\w$]*)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                combined,
                flags=re.IGNORECASE,
            )

            if match:
                missing_identifier = match.group(1)
                break

        if missing_identifier:

            return {
                "category": "undefined_variable",
                "message": combined,
                "missing_identifier": missing_identifier,
            }

        if "SyntaxError" in combined:

            return {
                "category": "syntax_error",
                "message": combined,
                "missing_identifier": None,
            }

        if "TypeError" in combined:

            return {
                "category": "type_error",
                "message": combined,
                "missing_identifier": None,
            }

        if combined:

            return {
                "category": "javascript_runtime_error",
                "message": combined,
                "missing_identifier": None,
            }

        return {
            "category": None,
            "message": None,
            "missing_identifier": None,
        }

    # =========================================================
    # SOURCE CODE DIAGNOSTICS
    # =========================================================

    def _search_source_identifier(
        self,
        identifier: Optional[str],
    ) -> list:

        if not identifier or not self.source_directory.exists():
            return []

        matches = []

        for file_path in self.source_directory.rglob("*"):

            if (
                not file_path.is_file()
                or file_path.suffix.lower() not in {
                    ".js",
                    ".jsx",
                    ".ts",
                    ".tsx",
                }
            ):
                continue

            try:
                lines = file_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()

            except Exception:
                continue

            for index, line in enumerate(lines):

                if re.search(
                    rf"\b{re.escape(identifier)}\b",
                    line,
                ):

                    start = max(0, index - 3)
                    end = min(len(lines), index + 4)

                    matches.append({
                        "file": str(file_path),
                        "line": index + 1,
                        "matching_line": line.strip(),
                        "source_excerpt": "\n".join(
                            f"{i + 1}: {lines[i]}"
                            for i in range(start, end)
                        ),
                    })

                    if len(matches) >= 30:
                        return matches

        return matches

    def _extract_stack_context(self) -> list:

        contexts = []

        stack_messages = [
            item.get("message", "")
            for item in self.page_errors
        ]

        for message in stack_messages:

            references = re.findall(
                r"App\.jsx:(\d+):(\d+)",
                message,
            )

            app_path = self.source_directory / "App.jsx"

            if not references or not app_path.exists():
                continue

            try:
                lines = app_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()

            except Exception:
                continue

            for line_number, column_number in references:

                line_index = int(line_number) - 1

                if not 0 <= line_index < len(lines):
                    continue

                start = max(0, line_index - 4)
                end = min(len(lines), line_index + 5)

                contexts.append({
                    "file": str(app_path),
                    "reported_line": int(line_number),
                    "reported_column": int(column_number),
                    "source_excerpt": "\n".join(
                        f"{i + 1}: {lines[i]}"
                        for i in range(start, end)
                    ),
                })

        return contexts

    # =========================================================
    # DOM VALIDATION
    # =========================================================

    def _inspect_dom(self, page) -> dict:

        return page.evaluate(
            """() => {
                const root = document.getElementById("root");

                if (!root) {
                    return {
                        root_exists: false,
                        root_rendered: false,
                        root_child_count: 0,
                        root_html_length: 0,
                        body_text: "",
                        body_height: document.body
                            ? document.body.scrollHeight : 0,
                        application_rendered: false,
                        image_count: 0,
                        broken_images: 0,
                        horizontal_overflow: false
                    };
                }

                const text = (root.innerText || "").trim();
                const rect = root.getBoundingClientRect();

                const images = Array.from(
                    root.querySelectorAll("img")
                );

                const brokenImages = images.filter(
                    img => !img.complete || img.naturalWidth === 0
                );

                return {
                    root_exists: true,
                    root_rendered: (
                        root.children.length > 0 &&
                        rect.width > 0 &&
                        rect.height > 0
                    ),
                    root_child_count: root.children.length,
                    root_html_length: root.innerHTML.length,
                    body_text: text.slice(0, 10000),
                    body_height: document.body.scrollHeight,
                    application_rendered: (
                        root.children.length > 0 &&
                        rect.height > 0
                    ),
                    image_count: images.length,
                    broken_images: brokenImages.length,
                    horizontal_overflow: (
                        document.documentElement.scrollWidth >
                        document.documentElement.clientWidth + 2
                    )
                };
            }"""
        )

    # =========================================================
    # SEMANTIC VALIDATION
    # =========================================================

    def _validate_semantic_content(
        self,
        dom_report: dict,
        plan: Optional[dict],
    ) -> dict:

        if not self.semantic_validation:

            return {
                "enabled": False,
                "checked": False,
                "expected_count": 0,
                "matched_count": 0,
                "missing_values": [],
                "match_ratio": None,
            }

        if not isinstance(plan, dict):

            return {
                "enabled": True,
                "checked": False,
                "reason": (
                    "No frontend_plan was supplied to BrowserAgent."
                ),
                "expected_count": 0,
                "matched_count": 0,
                "missing_values": [],
                "match_ratio": None,
            }

        expected_values = self._extract_plan_text(plan)

        rendered_text = (
            dom_report.get("body_text", "") or ""
        ).lower()

        matched = []
        missing = []

        for value in expected_values:

            normalized = value.lower().strip()

            if normalized and normalized in rendered_text:
                matched.append(value)
            else:
                missing.append(value)

        total = len(expected_values)

        ratio = (
            len(matched) / total
            if total
            else 1.0
        )

        return {
            "enabled": True,
            "checked": True,
            "expected_count": total,
            "matched_count": len(matched),
            "missing_values": missing,
            "matched_values": matched,
            "match_ratio": round(ratio, 4),
            "minimum_required_ratio": (
                self.minimum_content_match_ratio
            ),
            "passed": (
                ratio >= self.minimum_content_match_ratio
            ),
        }

    # =========================================================
    # REPAIR CONTEXT
    # =========================================================

    def _build_repair_context(
        self,
        runtime_error: dict,
        dom_report: dict,
        semantic_report: dict,
    ) -> dict:

        identifier = runtime_error.get(
            "missing_identifier"
        )

        return {
            "created_at": self._now(),
            "project_directory": str(
                self.project_directory
            ),
            "failure_category": runtime_error.get(
                "category"
            ),
            "runtime_error": runtime_error,
            "missing_identifier": identifier,
            "identifier_source_matches": (
                self._search_source_identifier(identifier)
            ),
            "stack_source_context": (
                self._extract_stack_context()
            ),
            "page_errors": self.page_errors,
            "console_errors": self.console_errors,
            "failed_requests": self.failed_requests,
            "http_errors": self.http_errors,
            "dom_report": dom_report,
            "semantic_validation": semantic_report,
            "screenshot_path": str(
                self.screenshot_path
            ),
            "recommended_action": (
                "Inspect the generated React source and fix the "
                "runtime error before repeating browser validation."
            ),
        }

    # =========================================================
    # MAIN EXECUTION
    # =========================================================

    def run(self) -> dict:

        started_at = self._now()
        start_clock = time.time()

        result = {
            "agent": "BrowserAgent",
            "status": "failed",
            "success": False,
            "started_at": started_at,
            "project_directory": str(
                self.project_directory
            ),
            "url": self.url,
            "screenshot_path": str(
                self.screenshot_path
            ),
            "report_path": str(
                self.report_path
            ),
            "repair_context_path": str(
                self.repair_context_path
            ),
            "failure_category": None,
            "error": None,
            "page_errors": [],
            "console_errors": [],
            "failed_requests": [],
            "http_errors": [],
        }

        try:

            # 1. Validate generated frontend.
            project_report = self._validate_project()

            result["project_validation"] = project_report

            if not project_report["valid"]:

                raise RuntimeError(
                    "Generated frontend project is incomplete: "
                    + "; ".join(
                        project_report["missing_files"]
                    )
                )

            # 2. Resolve the actual workflow plan.
            plan = self._load_plan()

            result["plan_available"] = (
                isinstance(plan, dict)
            )

            result["plan_source"] = (
                "frontend_plan"
                if isinstance(self.frontend_plan, dict)
                else "plan"
                if isinstance(self.frontend_plan, dict)
                else str(self.plan_path)
                if self.plan_path and self.plan_path.exists()
                else None
            )

            # 3. Start frontend.
            self._start_server()

            # 4. Launch browser.
            with sync_playwright() as playwright:

                browser_type = getattr(
                    playwright,
                    self.browser_name,
                )

                browser = browser_type.launch(
                    headless=self.headless
                )

                context = browser.new_context(
                    viewport={
                        "width": self.viewport_width,
                        "height": self.viewport_height,
                    },
                    ignore_https_errors=True,
                )

                page = context.new_page()

                page.set_default_timeout(
                    self.navigation_timeout
                )

                self._attach_browser_listeners(page)

                # 5. Open generated frontend.
                response = page.goto(
                    self.url,
                    wait_until="domcontentloaded",
                    timeout=self.navigation_timeout,
                )

                result["http_status"] = (
                    response.status if response else None
                )

                # Allow React time to mount.
                try:
                    page.wait_for_selector(
                        "#root",
                        state="attached",
                        timeout=self.render_timeout,
                    )

                    page.wait_for_timeout(2000)

                except PlaywrightTimeoutError:
                    logger.warning(
                        "React root did not appear before timeout."
                    )

                # 6. Inspect rendered DOM.
                dom_report = self._inspect_dom(page)

                result["dom_report"] = dom_report

                # 7. Semantic validation.
                semantic_report = (
                    self._validate_semantic_content(
                        dom_report,
                        plan,
                    )
                )

                result["semantic_validation"] = (
                    semantic_report
                )

                # 8. Screenshot.
                try:

                    page.screenshot(
                        path=str(self.screenshot_path),
                        full_page=True,
                        timeout=self.screenshot_timeout,
                    )

                except Exception as exc:

                    logger.exception(
                        "Screenshot capture failed."
                    )

                    result["screenshot_error"] = str(exc)

                # 9. Runtime error classification.
                runtime_error = (
                    self._classify_runtime_error()
                )

                result["runtime_error"] = runtime_error

                # 10. Browser diagnostics.
                result["page_errors"] = self.page_errors
                result["console_errors"] = self.console_errors
                result["failed_requests"] = self.failed_requests
                result["http_errors"] = self.http_errors

                # 11. Determine application state.
                has_runtime_error = bool(
                    self.page_errors
                    or (
                        self.fail_on_console_errors
                        and self.console_errors
                    )
                )

                application_rendered = bool(
                    dom_report.get("application_rendered")
                )

                if not application_rendered:

                    result["failure_category"] = (
                        runtime_error.get("category")
                        or "application_not_rendered"
                    )

                elif has_runtime_error:

                    result["failure_category"] = (
                        runtime_error.get("category")
                        or "javascript_runtime_error"
                    )

                elif (
                    semantic_report.get("checked")
                    and not semantic_report.get("passed", True)
                ):

                    result["failure_category"] = (
                        "semantic_validation_failed"
                    )

                elif (
                    self.check_horizontal_overflow
                    and dom_report.get("horizontal_overflow")
                ):

                    result["failure_category"] = (
                        "horizontal_overflow"
                    )

                elif (
                    self.check_horizontal_overflow
                    and dom_report.get("broken_images", 0) > 0
                ):

                    result["failure_category"] = (
                        "broken_images"
                    )

                else:

                    result["failure_category"] = None

                # 12. Build repair context when necessary.
                if result["failure_category"]:

                    repair_context = (
                        self._build_repair_context(
                            runtime_error,
                            dom_report,
                            semantic_report,
                        )
                    )

                    self._write_json(
                        self.repair_context_path,
                        repair_context,
                    )

                    result["status"] = (
                        "failed"
                        if (
                            not application_rendered
                            or has_runtime_error
                        )
                        else "completed_with_errors"
                    )

                    result["success"] = False

                else:

                    result["status"] = "completed"
                    result["success"] = True

                context.close()
                browser.close()

        except Exception as exc:

            logger.exception(
                "BrowserAgent execution failed."
            )

            result["status"] = "failed"
            result["success"] = False
            result["error"] = str(exc)

            if not result.get("failure_category"):
                result["failure_category"] = (
                    "browser_execution_error"
                )

            repair_context = {
                "created_at": self._now(),
                "project_directory": str(
                    self.project_directory
                ),
                "failure_category": (
                    result["failure_category"]
                ),
                "error": str(exc),
                "page_errors": self.page_errors,
                "console_errors": self.console_errors,
                "failed_requests": self.failed_requests,
                "http_errors": self.http_errors,
            }

            self._write_json(
                self.repair_context_path,
                repair_context,
            )

        finally:

            self._stop_server()

            result["completed_at"] = self._now()

            result["duration_seconds"] = round(
                time.time() - start_clock,
                3,
            )

            result["page_errors"] = self.page_errors
            result["console_errors"] = self.console_errors
            result["failed_requests"] = self.failed_requests
            result["http_errors"] = self.http_errors

            self._write_json(
                self.report_path,
                result,
            )

        self.browser_report = result

        return result


# =============================================================
# STANDALONE EXECUTION
# =============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    agent = BrowserAgent(
        project_directory=(
            "generated_projects/sample-dashboard/frontend"
        ),
        plan_path=None,
        headless=True,
        keep_server_running=False,
    )

    output = agent.run()

    print(
        json.dumps(
            output,
            indent=4,
            ensure_ascii=False,
            default=str,
        )
    )
