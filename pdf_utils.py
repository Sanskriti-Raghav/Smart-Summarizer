from typing import IO
from PyPDF2 import PdfReader


def extract_text_from_pdf(file_like: IO[bytes]) -> str:
    """
    Extracts text from a PDF using PyPDF2, skipping empty/unreadable pages.
    """
    reader = PdfReader(file_like)
    texts = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                texts.append(page_text)
        except Exception:
            # Skip unreadable page
            continue
    return "\n\n".join(texts).strip()
