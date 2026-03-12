import fitz  # PyMuPDF
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text as single string — kept for backward compatibility."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_pages_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text per page — preserves page numbers.
    Single PDF read — pages reused for both classification and chunking.
    Returns list of {page_number, text} dicts.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page_number": page_num,
                "text": text
            })
    doc.close()
    print(f"[extractor] Extracted {len(pages)} pages from {pdf_path}")
    return pages