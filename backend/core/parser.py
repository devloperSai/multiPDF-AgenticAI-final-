import os
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF using marker-pdf.
    Falls back to pymupdf if marker fails.
    """
    try:
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models

        models = load_all_models()
        full_text, _, _ = convert_single_pdf(pdf_path, models)
        return full_text

    except Exception as e:
        print(f"[marker-pdf] failed: {e}. Falling back to pymupdf.")
        return _fallback_extract(pdf_path)


def _fallback_extract(pdf_path: str) -> str:
    """Fallback extraction using pymupdf (fitz)."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise RuntimeError(f"Both marker-pdf and pymupdf failed: {e}")