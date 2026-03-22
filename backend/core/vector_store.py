import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os

CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_store"
)

_client: chromadb.PersistentClient = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        print(f"[vector_store] ChromaDB initialized at: {CHROMA_DIR}")
    return _client


def _get_collection(session_id: str):
    client = _get_client()
    name = f"s-{session_id}"
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def _clean_excerpt(text: str, max_len: int = 300) -> str:
    if not text:
        return ""
    import re
    match = re.search(r'(?<=[.!?])\s+([A-Z])', text)
    if match and match.start() < len(text) // 2:
        cleaned = text[match.start():].strip()
    else:
        cleaned = text.strip()
    if len(cleaned) > max_len:
        truncated = cleaned[:max_len]
        last_space = truncated.rfind(" ")
        if last_space > max_len * 0.8:
            truncated = truncated[:last_space]
        cleaned = truncated + "…"
    return cleaned


def _reciprocal_rank_fusion(
    vector_chunks: List[Dict],
    bm25_chunks: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    Merge vector search and BM25 results using Reciprocal Rank Fusion.
    RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)
    Chunks in both lists get higher scores.
    k=60 is standard from the original RRF paper.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map:  Dict[str, Dict]  = {}

    for rank, chunk in enumerate(vector_chunks):
        key = chunk["text"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[key]  = chunk

    for rank, chunk in enumerate(bm25_chunks):
        key = chunk["text"][:100]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for key in sorted_keys:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        chunk["score"]     = chunk["rrf_score"]
        merged.append(chunk)

    return merged


def hybrid_search(
    session_id: str,
    query: str,
    query_embedding: List[float],
    top_k: int = 10,
    pdf_id: Optional[str] = None
) -> List[Dict]:
    """
    Hybrid search: vector + BM25 merged via Reciprocal Rank Fusion.
    Falls back gracefully if either method returns nothing.
    """
    from core.bm25_store import bm25_search

    fetch_k = top_k * 2

    if pdf_id:
        vector_chunks = query_chunks_by_pdf(
            session_id=session_id,
            query_embedding=query_embedding,
            pdf_id=pdf_id,
            top_k=fetch_k
        )
    else:
        vector_chunks = query_chunks(
            session_id=session_id,
            query_embedding=query_embedding,
            top_k=fetch_k
        )

    bm25_chunks = bm25_search(
        session_id=session_id,
        query=query,
        top_k=fetch_k,
        pdf_id=pdf_id
    )

    print(f"[hybrid] Vector: {len(vector_chunks)} | BM25: {len(bm25_chunks)} chunks")

    if not vector_chunks and not bm25_chunks:
        return []
    if not bm25_chunks:
        print("[hybrid] BM25 empty — using vector only")
        return vector_chunks[:top_k]
    if not vector_chunks:
        print("[hybrid] Vector empty — using BM25 only")
        return bm25_chunks[:top_k]

    merged = _reciprocal_rank_fusion(vector_chunks, bm25_chunks)
    print(f"[hybrid] After RRF: {len(merged)} unique chunks → top {top_k}")

    return merged[:top_k]


def store_chunks(
    session_id: str,
    chunks: List[Dict],
    embeddings: List[List[float]]
) -> int:
    if not chunks or not embeddings:
        raise ValueError("chunks and embeddings cannot be empty.")
    if len(chunks) != len(embeddings):
        raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

    collection = _get_collection(session_id)
    ids        = [c["chunk_id"] for c in chunks]
    documents  = [c["text"] for c in chunks]

    metadatas = []
    for c in chunks:
        chunk_meta = c.get("metadata", {}) or {}
        meta = {
            "pdf_id":      c["pdf_id"],
            "chunk_index": str(c["chunk_index"]),
            "page_number": str(chunk_meta.get("page_number", "")),
            "doc_type":    str(chunk_meta.get("doc_type", "general")),
        }
        metadatas.append(meta)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    # Invalidate BM25 cache — rebuilt lazily on next query
    from core.bm25_store import invalidate_bm25_index
    invalidate_bm25_index(session_id)

    return len(ids)


def query_chunks(
    session_id: str,
    query_embedding: List[float],
    top_k: int = 5
) -> List[Dict]:
    collection = _get_collection(session_id)
    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        distance   = results["distances"][0][i]
        similarity = round(1 - distance, 4)
        chunks.append({
            "text":     doc,
            "metadata": results["metadatas"][0][i],
            "score":    similarity
        })
    return chunks


def query_chunks_by_pdf(
    session_id: str,
    query_embedding: List[float],
    pdf_id: str,
    top_k: int = 5
) -> List[Dict]:
    collection = _get_collection(session_id)
    count = collection.count()
    if count == 0:
        return []

    n_results = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"pdf_id": pdf_id},
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        distance   = results["distances"][0][i]
        similarity = round(1 - distance, 4)
        chunks.append({
            "text":     doc,
            "metadata": results["metadatas"][0][i],
            "score":    similarity
        })
    return chunks


def get_session_pdf_ids(session_id: str) -> List[str]:
    collection = _get_collection(session_id)
    count = collection.count()
    if count == 0:
        return []

    results = collection.get(include=["metadatas"])
    pdf_ids = list(set(
        m["pdf_id"] for m in results["metadatas"] if m.get("pdf_id")
    ))
    print(f"[vector_store] Session {session_id} has {len(pdf_ids)} PDFs: {pdf_ids}")
    return pdf_ids


def multi_query_hybrid_search(
    session_id:      str,
    queries:         List[str],
    query_embeddings: List[List[float]],
    top_k:           int = 10,
) -> List[Dict]:
    """
    Multi-query retrieval — runs hybrid_search for each query variant,
    then merges all results via RRF for higher recall.

    WHY this works:
        Different query variants hit different chunks.
        "termination notice" finds clause-title chunks.
        "how many days to end contract" finds body-text chunks.
        RRF merges both — best chunks from all variants float to top.

    Args:
        session_id:       Session to search
        queries:          List of query strings [original, variant1, variant2...]
        query_embeddings: Corresponding embeddings — must match queries length
        top_k:            Final number of chunks to return after merge

    Returns:
        Deduplicated, RRF-merged chunks sorted by combined score.
    """
    if not queries or not query_embeddings:
        return []

    if len(queries) != len(query_embeddings):
        print(f"[vector_store] multi_query: query/embedding count mismatch — using first only")
        queries          = queries[:1]
        query_embeddings = query_embeddings[:1]

    # Single query — no need for multi-query overhead
    if len(queries) == 1:
        return hybrid_search(
            session_id=session_id,
            query=queries[0],
            query_embedding=query_embeddings[0],
            top_k=top_k
        )

    # Run hybrid search for each query variant
    all_result_lists = []
    for i, (query, embedding) in enumerate(zip(queries, query_embeddings)):
        results = hybrid_search(
            session_id=session_id,
            query=query,
            query_embedding=embedding,
            top_k=top_k
        )
        all_result_lists.append(results)
        print(f"[multi_query] Variant {i}: '{query[:40]}' → {len(results)} chunks")

    # Merge all result lists via RRF
    # Each result list is treated as a ranked list — position matters
    k   = 60  # standard RRF constant
    rrf_scores: Dict[str, float] = {}
    chunk_map:  Dict[str, Dict]  = {}

    for result_list in all_result_lists:
        for rank, chunk in enumerate(result_list):
            key = chunk["text"][:100]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk

    # Sort by combined RRF score
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    merged = []
    for key in sorted_keys[:top_k]:
        chunk = chunk_map[key].copy()
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        chunk["score"]     = chunk["rrf_score"]
        merged.append(chunk)

    print(f"[multi_query] Merged {sum(len(r) for r in all_result_lists)} results "
          f"from {len(queries)} variants → {len(merged)} unique chunks")
    return merged


def get_spread_chunks(
    session_id: str,
    pdf_id:     Optional[str] = None,
    max_chunks: int = 10
) -> List[Dict]:
    """
    Summary-specific retrieval — samples chunks evenly across all pages.

    WHY this exists:
        Standard hybrid search embeds the query ("summarize this pdf") and
        finds the most SIMILAR chunks. For a summary query, similarity is
        meaningless — every chunk is equally "relevant" to the word summarize.
        This causes random page selection, usually tail-end admin clauses.

    HOW this works:
        1. Fetch ALL chunks for the PDF from ChromaDB (no embedding needed)
        2. Group chunks by page_number
        3. Pick 1 representative chunk per page (longest = most content)
        4. Sample evenly across pages to get max_chunks total
        5. Sort final selection by page order so LLM reads document linearly

    Result: LLM gets chunks from page 1, 4, 7, 10, 13... of a 30-page doc
            instead of 5 random chunks from pages 19, 25, 29.

    Args:
        session_id: Session to retrieve from
        pdf_id:     Specific PDF to summarize. None = all PDFs in session
        max_chunks: Maximum chunks to return (default 10)

    Returns:
        List of chunk dicts sorted by page number, spread across document
    """
    collection = _get_collection(session_id)
    count      = collection.count()
    if count == 0:
        return []

    # Fetch ALL chunks — no embedding, no similarity
    where_filter = {"pdf_id": pdf_id} if pdf_id else None
    try:
        if where_filter:
            results = collection.get(
                where=where_filter,
                include=["documents", "metadatas"]
            )
        else:
            results = collection.get(include=["documents", "metadatas"])
    except Exception as e:
        print(f"[vector_store] get_spread_chunks failed: {e}")
        return []

    if not results["documents"]:
        return []

    # Build chunk list with page numbers
    all_chunks = []
    for i, doc in enumerate(results["documents"]):
        meta        = results["metadatas"][i] if results["metadatas"] else {}
        page_number = meta.get("page_number", "0")
        try:
            page_int = int(page_number)
        except (ValueError, TypeError):
            page_int = 0

        all_chunks.append({
            "text":     doc,
            "metadata": meta,
            "score":    0.01,   # neutral score — not similarity based
            "page_int": page_int
        })

    # Group by page — keep longest chunk per page (most content)
    pages: dict = {}
    for chunk in all_chunks:
        p = chunk["page_int"]
        if p not in pages or len(chunk["text"]) > len(pages[p]["text"]):
            pages[p] = chunk

    # Sort pages numerically
    sorted_pages = sorted(pages.keys())
    total_pages  = len(sorted_pages)

    print(f"[vector_store] get_spread_chunks: {len(all_chunks)} chunks across {total_pages} pages")

    if total_pages <= max_chunks:
        # Fewer pages than max_chunks — take one chunk per page
        selected_pages = sorted_pages
    else:
        # Sample evenly — pick max_chunks pages spread across document
        # Always include first and last page (intro + conclusion)
        step = (total_pages - 1) / (max_chunks - 1) if max_chunks > 1 else 1
        indices = set()
        indices.add(0)
        indices.add(total_pages - 1)
        for i in range(1, max_chunks - 1):
            indices.add(round(i * step))
        selected_pages = [sorted_pages[i] for i in sorted(indices)]

    # Build final chunk list in page order
    spread_chunks = [pages[p] for p in selected_pages]

    print(f"[vector_store] Spread selection: pages {[c['page_int'] for c in spread_chunks]}")
    return spread_chunks


def delete_pdf_chunks(session_id: str, pdf_id: str) -> int:
    collection    = _get_collection(session_id)
    results       = collection.get(where={"pdf_id": pdf_id})
    ids_to_delete = results["ids"]

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"[vector_store] Deleted {len(ids_to_delete)} chunks for pdf_id {pdf_id}")

    from core.bm25_store import invalidate_bm25_index
    invalidate_bm25_index(session_id)

    return len(ids_to_delete)


def delete_session_collection(session_id: str):
    client          = _get_client()
    collection_name = f"s-{session_id}"

    try:
        client.delete_collection(collection_name)
        print(f"[vector_store] Deleted collection '{collection_name}'")
    except Exception as e:
        print(f"[vector_store] Collection '{collection_name}' not found or already deleted: {e}")

    from core.bm25_store import invalidate_bm25_index
    invalidate_bm25_index(session_id)