from pathlib import Path
import fitz
import pytesseract
from PIL import Image
from langchain_core.documents import Document


import os
import shutil


def configure_tesseract():
    configured_path = os.getenv("TESSERACT_CMD")

    if configured_path:
        if not os.path.exists(configured_path):
            raise RuntimeError("TESSERACT_CMD points to a file that does not exist.")
        pytesseract.pytesseract.tesseract_cmd = configured_path
        return

    detected_path = shutil.which("tesseract")

    if detected_path:
        pytesseract.pytesseract.tesseract_cmd = detected_path
        return

    raise RuntimeError(
        "Tesseract OCR was not found. Install Tesseract and add it to PATH or set TESSERACT_CMD."
    )


def load_file(path: Path, document_hash: str) -> list[Document]:

    pdf = fitz.open(str(path))
    documents = []

    try:
        for page_number, page in enumerate(pdf):
            # Try normal text first

            text = page.get_text("text").strip()
            extraction_method = "text"

            # OCR this page if needed

            if len(text) < 30:
                matrix = fitz.Matrix(2.5, 2.5)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(image, lang="eng").strip()
                extraction_method = "ocr"

            # Skip empty pages

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "document_id": path.name,
                        "document_hash": document_hash,
                        "page": page_number,
                        "extraction_method": extraction_method,
                    },
                )
            )
    finally:
        pdf.close()

    return documents
