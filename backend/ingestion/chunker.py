from typing import List, Dict


def chunk_text(text: str, pdf_id: str) -> List[Dict]:
    """Fallback chunker — no page number awareness."""
    return _fallback_chunk(text, pdf_id)


def chunk_pages(pages: List[Dict], pdf_id: str) -> List[Dict]:
    """
    Chunk text with page number awareness.
    Each chunk inherits the page number it started on.
    Single pass — no re-reading PDF.
    """
    chunks = []
    chunk_index = 0
    chunk_size = 1000
    overlap = 100

    for page in pages:
        page_number = page["page_number"]
        text = page["text"].strip()

        if not text:
            continue

        if len(text) <= chunk_size:
            chunks.append({
                "chunk_id": f"{pdf_id}_chunk_{chunk_index}",
                "pdf_id": pdf_id,
                "text": text,
                "chunk_index": chunk_index,
                "metadata": {
                    "category": "Text",
                    "page_number": page_number,
                }
            })
            chunk_index += 1
        else:
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text_slice = text[start:end].strip()
                if chunk_text_slice:
                    chunks.append({
                        "chunk_id": f"{pdf_id}_chunk_{chunk_index}",
                        "pdf_id": pdf_id,
                        "text": chunk_text_slice,
                        "chunk_index": chunk_index,
                        "metadata": {
                            "category": "Text",
                            "page_number": page_number,
                        }
                    })
                    chunk_index += 1
                start = end - overlap

    print(f"[chunker] {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def _fallback_chunk(text: str, pdf_id: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
    """Simple overlap-based fallback chunker."""
    chunks = []
    start = 0
    i = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_val = text[start:end].strip()

        if chunk_text_val:
            chunks.append({
                "chunk_id": f"{pdf_id}_chunk_{i}",
                "pdf_id": pdf_id,
                "text": chunk_text_val,
                "chunk_index": i,
                "metadata": {
                    "category": "Text",
                    "page_number": None,
                }
            })
        start = end - overlap
        i += 1

    return chunks