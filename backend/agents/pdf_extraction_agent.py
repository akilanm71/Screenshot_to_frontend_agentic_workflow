
import hashlib
import json

from pathlib import Path
from typing import Optional

import pymupdf
import pdfplumber

from pydantic import BaseModel, Field


# ==========================================
# 1. PYDANTIC OUTPUT SCHEMAS
# ==========================================

class ExtractedTable(BaseModel):
    table_number: int

    rows: list[list[Optional[str]]] = Field(
        default_factory=list
    )


class ExtractedImage(BaseModel):
    image_name: str
    image_path: str
    image_format: str
    width: int
    height: int


class PDFPageContent(BaseModel):
    page_number: int

    page_width: float
    page_height: float

    extracted_text: str

    extracted_text_length: int

    tables: list[ExtractedTable] = Field(
        default_factory=list
    )

    images: list[ExtractedImage] = Field(
        default_factory=list
    )

    requires_ocr: bool = False
    requires_vision: bool = False

    extraction_notes: list[str] = Field(
        default_factory=list
    )


class PDFExtractionResult(BaseModel):

    file_name: str
    file_path: str

    page_count: int

    total_text_characters: int

    total_tables: int
    total_images: int

    pdf_type: str

    pages: list[PDFPageContent] = Field(
        default_factory=list
    )

    output_directory: str


# ==========================================
# 2. PDF EXTRACTION AGENT
# ==========================================

class PDFExtractionAgent:

    def __init__(
        self,
        output_dir: str = "generated_projects/pdf_extraction",
        min_text_characters: int = 30
    ):

        self.output_dir = Path(output_dir)

        self.min_text_characters = min_text_characters

    # ======================================
    # Validate PDF
    # ======================================

    def _validate_pdf(self, pdf_path: str) -> Path:

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        if path.stat().st_size == 0:
            raise ValueError(
                "PDF file is empty."
            )

        return path

    # ======================================
    # Extract PDF text using PyMuPDF
    # ======================================

    def _extract_text(self, page) -> str:

        text = page.get_text("text")

        return text.strip()

    # ======================================
    # Extract tables using pdfplumber
    # ======================================

    def _extract_tables(self, pdf_page):

        extracted_tables = []

        tables = pdf_page.extract_tables() or []

        for index, table in enumerate(tables, start=1):

            cleaned_rows = []

            for row in table:

                cleaned_row = [
                    str(cell).strip() if cell is not None else None
                    for cell in row
                ]

                cleaned_rows.append(cleaned_row)

            extracted_tables.append(
                ExtractedTable(
                    table_number=index,
                    rows=cleaned_rows
                )
            )

        return extracted_tables

    # ======================================
    # Extract embedded images
    # ======================================

    def _extract_images(
        self,
        document,
        page,
        page_number: int,
        image_dir: Path,
        saved_images: dict
    ):

        extracted_images = []

        image_list = page.get_images(full=True)

        for image_index, image_info in enumerate(
            image_list,
            start=1
        ):

            xref = image_info[0]

            if xref in saved_images:

                extracted_images.append(
                    saved_images[xref]
                )

                continue

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]

            image_ext = image_data["ext"]

            image_width = image_data["width"]
            image_height = image_data["height"]

            image_hash = hashlib.sha256(
                image_bytes
            ).hexdigest()[:12]

            image_name = (
                f"page_{page_number}_"
                f"image_{image_index}_{image_hash}."
                f"{image_ext}"
            )

            image_path = image_dir / image_name

            image_path.write_bytes(image_bytes)

            extracted_image = ExtractedImage(
                image_name=image_name,
                image_path=str(image_path),
                image_format=image_ext,
                width=image_width,
                height=image_height
            )

            saved_images[xref] = extracted_image

            extracted_images.append(
                extracted_image
            )

        return extracted_images

    # ======================================
    # Main extraction method
    # ======================================

    def extract(self, pdf_path: str) -> PDFExtractionResult:

        path = self._validate_pdf(pdf_path)

        job_dir = (
            self.output_dir / path.stem
        )

        image_dir = job_dir / "images"

        image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        page_results = []

        total_text_characters = 0
        total_tables = 0
        total_images = 0

        saved_images = {}

        # Open PDF using PyMuPDF
        with pymupdf.open(str(path)) as document:

            if document.is_encrypted:
                raise ValueError(
                    "Encrypted PDFs are not supported. "
                    "Please provide an unlocked PDF."
                )

            page_count = len(document)

            if page_count == 0:
                raise ValueError(
                    "PDF contains no pages."
                )

            # Open same PDF using pdfplumber
            with pdfplumber.open(str(path)) as plumber_pdf:

                for page_index, page in enumerate(
                    document,
                    start=1
                ):

                    print(
                        f"Extracting page {page_index}/{page_count}"
                    )

                    # --------------------------
                    # Extract text
                    # --------------------------

                    text = self._extract_text(page)

                    text_length = len(text)

                    total_text_characters += text_length

                    # --------------------------
                    # Extract tables
                    # --------------------------

                    plumber_page = plumber_pdf.pages[
                        page_index - 1
                    ]

                    tables = self._extract_tables(
                        plumber_page
                    )

                    total_tables += len(tables)

                    # --------------------------
                    # Extract images
                    # --------------------------

                    images = self._extract_images(
                        document=document,
                        page=page,
                        page_number=page_index,
                        image_dir=image_dir,
                        saved_images=saved_images
                    )

                    total_images += len(images)

                    # --------------------------
                    # Page quality detection
                    # --------------------------

                    requires_ocr = (
                        text_length < self.min_text_characters
                    )

                    requires_vision = (
                        requires_ocr or len(images) > 0
                    )

                    notes = []

                    if requires_ocr:
                        notes.append(
                            "Low extracted text. OCR may be required."
                        )

                    if images:
                        notes.append(
                            "Embedded images found. "
                            "Visual analysis may be useful."
                        )

                    if tables:
                        notes.append(
                            "Tables detected on this page."
                        )

                    # --------------------------
                    # Create page result
                    # --------------------------

                    page_result = PDFPageContent(

                        page_number=page_index,

                        page_width=page.rect.width,
                        page_height=page.rect.height,

                        extracted_text=text,

                        extracted_text_length=text_length,

                        tables=tables,

                        images=images,

                        requires_ocr=requires_ocr,

                        requires_vision=requires_vision,

                        extraction_notes=notes
                    )

                    page_results.append(
                        page_result
                    )

        # ==================================
        # Classify PDF
        # ==================================

        pages_requiring_ocr = [
            page for page in page_results
            if page.requires_ocr
        ]

        if len(pages_requiring_ocr) == 0:

            pdf_type = "digital"

        elif len(pages_requiring_ocr) == len(page_results):

            pdf_type = "scanned"

        else:

            pdf_type = "mixed"

        # ==================================
        # Build final result
        # ==================================

        result = PDFExtractionResult(

            file_name=path.name,

            file_path=str(path.resolve()),

            page_count=page_count,

            total_text_characters=total_text_characters,

            total_tables=total_tables,

            total_images=total_images,

            pdf_type=pdf_type,

            pages=page_results,

            output_directory=str(job_dir.resolve())
        )

        # Save extraction result as JSON
        output_json = job_dir / "extraction_result.json"

        output_json.write_text(
            result.model_dump_json(indent=4),
            encoding="utf-8"
        )

        print(
            f"\nExtraction completed: {output_json}"
        )

        return result


# ==========================================
# 3. LOCAL TESTING
# ==========================================

if __name__ == "__main__":

    agent = PDFExtractionAgent()

    pdf_path = "uploads/sample.pdf"

    try:

        result = agent.extract(pdf_path)

        print("\nPDF EXTRACTION RESULT")

        print(
            result.model_dump_json(indent=4)
        )

        print("\nSUMMARY")

        print("File:", result.file_name)

        print("Pages:", result.page_count)

        print("PDF type:", result.pdf_type)

        print("Total text:", result.total_text_characters)

        print("Total tables:", result.total_tables)

        print("Total images:", result.total_images)

    except Exception as error:

        print(
            "PDF extraction failed:",
            error
        )
