"""
Refine 7 — PDF Validation
---------------------------
Validates uploaded PDF files before saving to disk or dispatching Celery task.
Checks:
1. File size — max 50MB
2. Magic bytes — must start with %PDF (real PDF, not renamed .txt or .exe)
3. Password protected — fitz raises exception on encrypted PDFs
4. Scanned only — PDF with pages but zero extractable text (image-only scan)

Single function: validate_pdf_bytes(contents, filename) → raises HTTPException if invalid.
Called in upload endpoint BEFORE writing to disk — bad files never touch storage.
"""

import fitz  # PyMuPDF
from fastapi import HTTPException

# Hard limits
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# PDF magic bytes — all real PDFs start with %PDF
PDF_MAGIC = b"%PDF"

# Minimum text chars across entire doc to be considered non-scanned
MIN_TEXT_CHARS = 50


def validate_pdf_bytes(contents: bytes, filename: str):
    """
    Validate PDF file contents before processing.
    Raises HTTPException with specific message if invalid.
    Returns silently if valid.

    Args:
        contents: raw file bytes
        filename: original filename (for error messages)
    """
    # 1. File size check
    size_mb = len(contents) / (1024 * 1024)
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is too large ({size_mb:.1f}MB). Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
        )

    # 2. Magic bytes — must start with %PDF
    if not contents.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' does not appear to be a valid PDF. Please upload a proper PDF file."
        )

    # 3. Open with PyMuPDF — catches corrupt files and password-protected PDFs
    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' could not be opened. It may be corrupt or in an unsupported format."
        )

    # 4. Password-protected check
    if doc.needs_pass:
        doc.close()
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is password-protected. Please upload an unlocked PDF."
        )

    # 5. Empty PDF — no pages at all
    if doc.page_count == 0:
        doc.close()
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' has no pages. Please upload a valid PDF."
        )

    # 6. Scanned-only PDF — has pages but zero extractable text
    total_text = ""
    for page in doc:
        total_text += page.get_text().strip()
        if len(total_text) >= MIN_TEXT_CHARS:
            break  # enough text found — stop early

    doc.close()

    if len(total_text) < MIN_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' appears to be a scanned image PDF with no extractable text. "
                   f"Please upload a text-based PDF or use OCR software to convert it first."
        )

    print(f"[pdf_validator] '{filename}' passed all checks ({size_mb:.1f}MB, {len(total_text)} chars extracted)")