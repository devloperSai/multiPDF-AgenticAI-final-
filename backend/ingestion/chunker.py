"""
ingestion/chunker.py

Structure-aware chunker — splits by logical units instead of fixed character count.

Strategy (in order of priority):
  1. Detect numbered/named entity blocks (e.g. "1. Gateway of India", "## Section")
     and keep each block as one chunk — preserves address + description together
  2. Detect paragraph breaks (double newline) and group paragraphs into
     size-appropriate chunks
  3. Fall back to sentence-boundary splitting for dense text with no structure
  4. Hard cap at MAX_CHUNK_SIZE to prevent oversized chunks going to LLM

This means:
  - A cafe entry with name + description + address stays in ONE chunk
  - A research paper paragraph stays together
  - A legal clause stays together
  - Nothing gets split mid-sentence
"""

import re
from typing import List, Dict, Tuple

# ── Size limits ───────────────────────────────────────────────────────────────
MIN_CHUNK_SIZE  = 100    # ignore tiny fragments
MAX_CHUNK_SIZE  = 1500   # hard cap — no chunk sent to LLM exceeds this
TARGET_SIZE     = 800    # ideal chunk size — group paragraphs until this
OVERLAP_CHARS   = 150    # character overlap between consecutive chunks when splitting dense text


# ── Structural pattern detection ─────────────────────────────────────────────

# Numbered entity: "1. Name", "2. Name", "A.", "B." etc
_NUMBERED_ENTITY = re.compile(
    r'^(\d{1,2}\.|\w\.|#{1,3})\s+\S',
    re.MULTILINE
)

# Section headings: ALL CAPS line, or line ending with ":" alone
_SECTION_HEADING = re.compile(
    r'^([A-Z][A-Z\s&/\-]{4,}|.{3,80}:)\s*$',
    re.MULTILINE
)

# Structured field labels: "Address:", "Where:", "Cost for two:", etc
_FIELD_LABEL = re.compile(
    r'^(Address|Where|Location|Cost for two|Highlights|Must.haves?|Cuisine|Timings?|'
    r'Contact|Phone|Email|Website|URL|Price|Fee|Charges?|Tel|Fax)\s*:',
    re.IGNORECASE | re.MULTILINE
)


def _detect_structure(text: str) -> str:
    """
    Detect what kind of structure the text has.
    Returns: "numbered_entities" | "paragraphs" | "dense"
    """
    numbered_matches = len(_NUMBERED_ENTITY.findall(text))
    paragraph_breaks = text.count('\n\n')

    if numbered_matches >= 2:
        return "numbered_entities"
    if paragraph_breaks >= 2:
        return "paragraphs"
    return "dense"


def _split_by_numbered_entities(text: str) -> List[str]:
    """
    Split text at numbered/headed entity boundaries.
    Each entity (e.g. "1. Cafe Name\n...description...address...") becomes one block.
    Keeps structured data (name + info + address) together in one chunk.
    """
    # Find all entity start positions
    boundaries = [m.start() for m in _NUMBERED_ENTITY.finditer(text)]

    if not boundaries:
        return [text]

    blocks = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)

    # Prepend any text before the first numbered entity
    if boundaries[0] > 0:
        preamble = text[:boundaries[0]].strip()
        if len(preamble) >= MIN_CHUNK_SIZE:
            blocks.insert(0, preamble)

    return blocks


def _split_by_paragraphs(text: str) -> List[str]:
    """
    Group paragraphs into chunks of TARGET_SIZE.
    Paragraphs are separated by double newlines.
    Small paragraphs are merged until TARGET_SIZE is reached.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > MAX_CHUNK_SIZE and current:
            chunks.append('\n\n'.join(current))
            # Overlap: keep last paragraph in next chunk
            current = [current[-1]] if current else []
            current_len = len(current[0]) if current else 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def _split_dense_text(text: str) -> List[str]:
    """
    Split dense text (no structure) at sentence boundaries.
    Used as last resort for research paper paragraphs, legal text, etc.
    """
    # Split at sentence endings: . ! ? followed by space + capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)

        if current_len + s_len > MAX_CHUNK_SIZE and current:
            chunk_text = ' '.join(current)
            if len(chunk_text) >= MIN_CHUNK_SIZE:
                chunks.append(chunk_text)
            # Overlap: keep last 2 sentences
            overlap_sentences = current[-2:] if len(current) >= 2 else current[-1:]
            current = overlap_sentences
            current_len = sum(len(s) for s in current)

        current.append(sentence)
        current_len += s_len

    if current:
        chunk_text = ' '.join(current)
        if len(chunk_text) >= MIN_CHUNK_SIZE:
            chunks.append(chunk_text)

    return chunks if chunks else [text]


def _hard_cap_chunk(block: str) -> List[str]:
    """
    If a single block exceeds MAX_CHUNK_SIZE, split it further
    at the best available boundary within the cap.
    Preserves as much structure as possible.
    """
    if len(block) <= MAX_CHUNK_SIZE:
        return [block]

    # Try paragraph split first
    if '\n\n' in block:
        return _split_by_paragraphs(block)

    # Try sentence split
    return _split_dense_text(block)


def _chunk_page_text(text: str, page_number: int, pdf_id: str, start_index: int) -> List[Dict]:
    """
    Chunk a single page's text using structure-aware splitting.
    Returns list of chunk dicts.
    """
    text = text.strip()
    if not text or len(text) < MIN_CHUNK_SIZE:
        return []

    structure = _detect_structure(text)
    print(f"[chunker] Page {page_number}: structure={structure} | len={len(text)}")

    if structure == "numbered_entities":
        raw_blocks = _split_by_numbered_entities(text)
    elif structure == "paragraphs":
        raw_blocks = _split_by_paragraphs(text)
    else:
        raw_blocks = _split_dense_text(text)

    # Apply hard cap to any oversized blocks
    final_blocks = []
    for block in raw_blocks:
        if len(block) > MAX_CHUNK_SIZE:
            final_blocks.extend(_hard_cap_chunk(block))
        elif len(block) >= MIN_CHUNK_SIZE:
            final_blocks.append(block)

    # Build chunk dicts
    chunks = []
    for i, block in enumerate(final_blocks):
        chunks.append({
            "chunk_id":    f"{pdf_id}_chunk_{start_index + i}",
            "pdf_id":      pdf_id,
            "text":        block,
            "chunk_index": start_index + i,
            "metadata": {
                "category":    "Text",
                "page_number": page_number,
                "structure":   structure,
                "chunk_len":   len(block),
            }
        })

    return chunks


def chunk_pages(pages: List[Dict], pdf_id: str) -> List[Dict]:
    """
    Main entry point — chunk all pages of a PDF.
    Each chunk inherits the page number it started on.
    Uses structure-aware splitting per page.
    """
    all_chunks = []
    chunk_index = 0

    for page in pages:
        page_number = page["page_number"]
        text = page.get("text", "").strip()

        if not text:
            continue

        page_chunks = _chunk_page_text(text, page_number, pdf_id, chunk_index)
        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    print(f"[chunker] {len(all_chunks)} chunks from {len(pages)} pages (structure-aware)")
    return all_chunks


def chunk_text(text: str, pdf_id: str) -> List[Dict]:
    """
    Fallback entry point when no page info available.
    Wraps text as single page and delegates to chunk_pages.
    """
    fake_pages = [{"page_number": 1, "text": text}]
    return chunk_pages(fake_pages, pdf_id)


def _fallback_chunk(text: str, pdf_id: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Legacy fixed-size chunker kept for compatibility.
    Only used if structure-aware chunker is explicitly bypassed.
    """
    chunks = []
    start = 0
    i = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_val = text[start:end].strip()

        if chunk_text_val and len(chunk_text_val) >= MIN_CHUNK_SIZE:
            chunks.append({
                "chunk_id":    f"{pdf_id}_chunk_{i}",
                "pdf_id":      pdf_id,
                "text":        chunk_text_val,
                "chunk_index": i,
                "metadata": {
                    "category":    "Text",
                    "page_number": None,
                }
            })
        start = end - overlap
        i += 1

    return chunks