
from pathlib import Path
from typing import Literal

import pymupdf  # PyMuPDF
from PIL import Image

from pydantic import BaseModel, Field


# -----------------------------------------
# 1. Classification output schema
# -----------------------------------------

class PageInfo(BaseModel):
    page_number: int
    extracted_text_length: int
    image_count: int
    requires_ocr: bool


class InputClassification(BaseModel):
    file_name: str
    file_type: Literal["image", "pdf"]
    input_format: str

    page_count: int = 1

    pdf_type: Literal[
        "digital",
        "scanned",
        "mixed",
        "not_applicable"
    ]

    processing_route: Literal[
        "screenshot_vision",
        "pdf_extraction",
        "pdf_extraction_and_vision"
    ]

    requires_ocr: bool = False

    page_details: list[PageInfo] = Field(
        default_factory=list
    )


# -----------------------------------------
# 2. Input Classification Agent
# -----------------------------------------

class InputClassifierAgent:

    SUPPORTED_IMAGE_FORMATS = {
        ".png", ".jpg", ".jpeg"
    }

    SUPPORTED_PDF_FORMATS = {
        ".pdf"
    }

    def __init__(
        self,
        min_text_characters: int = 30
    ):
        self.min_text_characters = min_text_characters

    # -------------------------------------
    # Main classification method
    # -------------------------------------

    def classify(self, file_path: str) -> InputClassification:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension in self.SUPPORTED_IMAGE_FORMATS:
            return self._classify_image(path)

        elif extension in self.SUPPORTED_PDF_FORMATS:
            return self._classify_pdf(path)

        else:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

    # -------------------------------------
    # Image classification
    # -------------------------------------

    def _classify_image(
        self,
        path: Path
    ) -> InputClassification:

        # Verify that the image is valid
        with Image.open(path) as image:
            image.verify()

        return InputClassification(
            file_name=path.name,
            file_type="image",
            input_format=path.suffix.lower(),
            page_count=1,
            pdf_type="not_applicable",
            processing_route="screenshot_vision",
            requires_ocr=False
        )

    # -------------------------------------
    # PDF classification
    # -------------------------------------

    def _classify_pdf(
        self,
        path: Path
    ) -> InputClassification:

        page_details = []

        try:
            with pymupdf.open(path) as document:

                if document.is_encrypted:
                    raise ValueError(
                        "Encrypted PDFs are not supported."
                    )

                page_count = len(document)

                if page_count == 0:
                    raise ValueError(
                        "PDF contains no pages."
                    )

                for index, page in enumerate(document):

                    # Extract text from current page
                    extracted_text = page.get_text("text")

                    text_length = len(
                        extracted_text.strip()
                    )

                    # Detect embedded images
                    images = page.get_images(
                        full=True
                    )

                    image_count = len(images)

                    # Determine whether OCR is needed
                    requires_ocr = (
                        text_length < self.min_text_characters
                    )

                    page_details.append(
                        PageInfo(
                            page_number=index + 1,
                            extracted_text_length=text_length,
                            image_count=image_count,
                            requires_ocr=requires_ocr
                        )
                    )

        except Exception as error:
            raise ValueError(
                f"Unable to process PDF: {error}"
            ) from error

        # ---------------------------------
        # Determine overall PDF type
        # ---------------------------------

        ocr_pages = [
            page for page in page_details
            if page.requires_ocr
        ]

        if len(ocr_pages) == 0:

            pdf_type = "digital"
            route = "pdf_extraction"
            requires_ocr = False

        elif len(ocr_pages) == len(page_details):

            pdf_type = "scanned"
            route = "pdf_extraction_and_vision"
            requires_ocr = True

        else:

            pdf_type = "mixed"
            route = "pdf_extraction_and_vision"
            requires_ocr = True

        return InputClassification(
            file_name=path.name,
            file_type="pdf",
            input_format=".pdf",
            page_count=len(page_details),
            pdf_type=pdf_type,
            processing_route=route,
            requires_ocr=requires_ocr,
            page_details=page_details
        )


# -----------------------------------------
# 3. Local testing
# -----------------------------------------

if __name__ == "__main__":

    agent = InputClassifierAgent()

    input_file = "uploads/sample.pdf"

    try:

        result = agent.classify(input_file)

        print("\nINPUT CLASSIFICATION")
        print(result.model_dump_json(indent=4))

    except Exception as error:

        print(f"Classification failed: {error}")
