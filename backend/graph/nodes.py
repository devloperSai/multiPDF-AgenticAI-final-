import os
from collections import defaultdict
from dotenv import load_dotenv
from core.embedder import embed_query
from graph.coref import resolve_coreferences
from core.vector_store import hybrid_search, get_session_pdf_ids, get_spread_chunks, multi_query_hybrid_search
from core.query_expander import expand_query
from core.hyde import generate_hyde_embedding
from core.semantic_cache import get_cached_answer, store_in_cache
from core.reranker import rerank_chunks
from core.citations_builder import build_citations
from graph.state import AgentState
from graph.router import classify_intent
from graph.model_router import get_llm_config, get_llm_config_from_chunks
from graph.fallback_llm import call_with_fallback, stream_with_fallback
from memory.context_builder import build_memory_context
from memory.session_store import get_session_doc_types, get_document_summaries
from models.database import SessionLocal

load_dotenv()

from config import cfg
MODEL                     = cfg.PRIMARY_LLM_MODEL
MAX_RETRIES               = cfg.MAX_RETRIES
HYDE_CONFIDENCE_THRESHOLD = cfg.HYDE_CONFIDENCE_THRESHOLD

# ── Enhancement #9: Answer Confidence Gating ─────────────────────────────────
# Source: "Add a confidence threshold — below it, tell user 'I found partial
#          information, here's what I know' instead of hallucinating a full answer."
#
# WHY retrieval_confidence (RRF score), not answer length:
#   The doc says "retrieval confidence is very low (0.3-0.4)" as the signal.
#   RRF scores in this system range 0.01-0.03 typically.
#   A score below 0.013 means even the best chunk barely ranked in both
#   vector + BM25 results — retrieval is genuinely uncertain.
#   Using answer length was wrong — a short answer can still be correct.
#   Using retrieval score is the honest signal of "did we find anything good?"
#
# NOTE: 0.013 is calibrated to RRF score range (0.01-0.03).
#   Equivalent to saying "top chunk ranked poorly in both vector and BM25".
CONFIDENCE_GATE_THRESHOLD = cfg.CONFIDENCE_GATE_THRESHOLD

# ── Enhancement #10: Chunk Deduplication ─────────────────────────────────────
# Source: "When multiple PDFs have overlapping content, same text gets retrieved
#          multiple times and sent to LLM. Add deduplication step after reranking."
#
# WHY Jaccard token similarity:
#   The doc didn't specify a method — Jaccard is fast (no model, no GPU),
#   works on token overlap, and is well-suited for detecting near-duplicate
#   text chunks which share most of the same words.
#   0.85 threshold = chunks must differ by at least 15% of their tokens to both be kept.
CHUNK_DEDUP_THRESHOLD = cfg.CHUNK_DEDUP_THRESHOLD

DOC_TYPE_DESCRIPTIONS = {
    "research":  "a research paper (focus on methodology, findings, and conclusions)",
    "legal":     "a legal document (focus on clauses, obligations, rights, and exact legal terms)",
    "financial": "a financial document (focus on exact figures, percentages, and financial metrics)",
    "general":   "a general document",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_comparison_system_suffix(doc_groups_with_types: dict) -> str:
    if not doc_groups_with_types:
        return ""
    lines = ["You are comparing multiple documents:"]
    for label, doc_type in doc_groups_with_types.items():
        description = DOC_TYPE_DESCRIPTIONS.get(doc_type, "a general document")
        lines.append(f"- {label} is {description}")
    lines.append("")
    lines.append("Apply the appropriate analytical lens for each document type when comparing.")
    lines.append("For legal documents: cite exact clauses. For research: cite findings. For financial: cite exact figures.")
    return "\n".join(lines)


def _build_prompt_config(
    question:      str,
    intent:        str,
    response_mode: str | None = None
) -> dict:
    """
    Build prompt style config.

    Priority order:
      1. response_mode — user's explicit UI selection (Enhancement #4)
         Overrides everything including intent-based styles.
         Exception: summary/comparison intents always keep their style
         because the retrieval strategy was already set for them.

      2. Intent-based style — summary and comparison have fixed styles
         because their retrieval was specifically tuned for those modes.

      3. Keyword detection — brief_signals, list_signals, detailed_signals
         Applied when no user mode is set and intent is factual.

      4. Balanced default — fallback for everything else.
    """
    # ── Priority 1: User explicit mode (Enhancement #4) ──────────────────────
    # Only applies to factual/out_of_scope intent.
    # summary and comparison keep their own style — user mode doesn't override
    # them because the retrieval pipeline was already configured for those.
    if response_mode and intent not in ("summary", "comparison"):
        if response_mode == "short":
            return {
                "style": "short",
                "instruction": "Answer in 1-2 sentences maximum. Be direct and precise. Cite the source using [Source N]."
            }
        elif response_mode == "verbatim":
            return {
                "style": "verbatim",
                "instruction": """Extract and return the EXACT relevant text from the document.
Do not paraphrase, summarize, or rephrase — quote directly.
If multiple passages are relevant, quote each one.
Cite every passage using [Source N] immediately after each quote."""
            }
        elif response_mode == "explanation":
            return {
                "style": "explanation",
                "instruction": """Give a comprehensive, well-structured explanation in this format:

Start with 1-2 sentences of overview.

Then use sections if needed — write a plain header line followed by bullet points:
South Goa:
- Site one
- Site two

North Goa:
- Site one
- Site two

Rules:
- Use ONLY "- " (dash space) for bullets. Never use *, +, or numbers.
- Each bullet on its own NEW LINE.
- Plain text section headers — no ## or ** or symbols.
- Do NOT use [Source N] inside bullets."""
            }
        elif response_mode == "bullets":
            return {
                "style": "bullets",
                "instruction": """Answer ONLY using this exact format — one item per line:
- Item one
- Item two
- Item three

Rules:
- Use ONLY "- " (dash space) to start each bullet. Never use *, +, or numbers.
- Each bullet on its own NEW LINE. Never put multiple items on the same line.
- If grouping by category, use a plain text header line (no symbols) then bullets below it.
- Keep each bullet concise — one fact per line.
- Do NOT use [Source N] inside bullets."""
            }
    # ─────────────────────────────────────────────────────────────────────────

    # ── Priority 2: Intent-based style ───────────────────────────────────────
    if intent == "summary":
        return {
            "style": "summary",
            "instruction": """Provide a comprehensive summary of the document content.
- Cover the main topic, key arguments, and conclusions
- Use clear sections with bullet points
- Cite sources inline using [Source N]"""
        }
    elif intent == "comparison":
        return {
            "style": "comparison",
            "instruction": """Compare and contrast the relevant concepts or documents.
- Use a structured format with clear sections
- Highlight similarities and differences explicitly
- Cite sources inline using [Source N]"""
        }

    # ── Priority 3: Keyword detection ────────────────────────────────────────
    q = question.lower()

    # Single-fact lookup signals — questions expecting one direct answer
    brief_signals    = ["brief", "short", "quick", "one line", "tldr", "in a sentence", "in short",
                        "what is the", "what was the", "how much", "how many", "when was",
                        "who is the", "who was the", "where is the", "what are the penalties",
                        "what is the amount", "what is the period", "what is the duration"]
    list_signals     = ["list", "what are", "enumerate", "give me all", "what were", "mention all"]
    # "what is" alone (without "the") stays in detailed — e.g. "what is arbitration" needs explanation
    detailed_signals = ["explain", "describe", "how does", "what is", "elaborate", "in detail", "tell me about"]
    address_signals  = ["address", "location", "where is", "where can i find", "located", "directions"]

    if any(s in q for s in address_signals):
        return {
            "style": "address",
            "instruction": """Extract and state the exact address or location as it appears in the document.
Look for fields labeled 'Address:', 'Where:', 'Location:' or similar.
State the address directly without any preamble. Cite the source using [Source N]."""
        }

    if any(s in q for s in brief_signals):
        return {"style": "concise", "instruction": "Give a brief, direct answer in 2-3 sentences maximum. Cite sources inline using [Source N]."}
    elif any(s in q for s in list_signals):
        return {"style": "list", "instruction": "Answer using a numbered or bulleted list. Be comprehensive. Cite sources inline using [Source N]."}
    elif any(s in q for s in detailed_signals):
        return {
            "style": "detailed",
            "instruction": """Give a comprehensive, well-structured answer:
- Start with a clear definition or overview
- Explain components or how it works
- Cover purpose and significance
- Use bullet points for multi-part information
- Cite every claim using [Source N] inline"""
        }

    # ── Priority 4: Balanced default ─────────────────────────────────────────
    return {"style": "balanced", "instruction": "Give a clear, complete answer. Use structure if the topic has multiple aspects. Cite sources inline using [Source N]."}


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Token-level Jaccard similarity between two strings.
    Returns fraction of shared tokens over union of all tokens.
    Fast — no model, no GPU.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _deduplicate_chunks(chunks: list) -> list:
    """
    Enhancement #10 — Remove near-duplicate chunks before prompt construction.

    Iterates chunks in ranked order (best first — already sorted by reranker).
    A chunk is dropped if it shares >= CHUNK_DEDUP_THRESHOLD (0.85) token overlap
    with any already-kept chunk.

    This handles the case where multiple PDFs contain the same boilerplate text,
    or the same passage is retrieved by both vector and BM25 paths.
    """
    if not chunks:
        return chunks

    kept = []
    for candidate in chunks:
        is_duplicate = False
        for kept_chunk in kept:
            sim = _jaccard_similarity(candidate["text"], kept_chunk["text"])
            if sim >= CHUNK_DEDUP_THRESHOLD:
                is_duplicate = True
                print(f"[dedup] Dropped chunk — Jaccard={sim:.2f} with existing chunk")
                break
        if not is_duplicate:
            kept.append(candidate)

    removed = len(chunks) - len(kept)
    if removed:
        print(f"[dedup] {len(chunks)} chunks → {len(kept)} kept ({removed} near-duplicates removed)")

    return kept


# ── Nodes ─────────────────────────────────────────────────────────────────────

def coref_node(state: AgentState) -> AgentState:
    """
    Enhancement #7 — Coreference Resolution

    Rewrites ambiguous questions before retrieval.
    Only fires when BOTH:
      a) question contains reference words (they, it, this, that...)
      b) session has >= 2 prior messages to resolve against

    Sets state["retrieval_question"] — used by retrieve_node for embedding.
    Original state["question"] preserved for display, DB, cache, prompt.

    Fallback: if rewrite fails → retrieval_question = original question.
    Pipeline never breaks regardless of LLM availability.
    """
    db = SessionLocal()
    try:
        retrieval_question = resolve_coreferences(
            question=state["question"],
            session_id=state["session_id"],
            db=db
        )
    finally:
        db.close()

    coref_rewritten = retrieval_question != state["question"]

    return {
        **state,
        "retrieval_question": retrieval_question,
        "coref_rewritten":    coref_rewritten,
    }


def _get_session_pdf_count(session_id: str) -> int:
    """Get number of PDFs in session — used for comparison intent guard."""
    try:
        pdf_ids = get_session_pdf_ids(session_id)
        return len(pdf_ids)
    except Exception:
        return 1  # safe default — assume 1 PDF


def route_node(state: AgentState) -> AgentState:
    pdf_count = _get_session_pdf_count(state["session_id"])
    intent    = classify_intent(state["question"], pdf_count=pdf_count)
    print(f"[router] Intent detected: {intent}")

    db = SessionLocal()
    try:
        doc_types = get_session_doc_types(db, state["session_id"])
    finally:
        db.close()

    llm_config = get_llm_config(doc_types)
    print(f"[router] Session doc types: {doc_types} | Fallback config: {llm_config['doc_type']}")

    question_embedding = embed_query(state["question"])

    # ── Cache bypass logic ────────────────────────────────────────────────────
    # Skip cache when:
    #   1. User explicitly selected a response mode (Short/Bullets/Explain/Verbatim)
    #      → they want freshly formatted answer, not a cached one from Auto mode
    #   2. Question starts with a format-modifying word
    #      → "explain what are..." should generate fresh explanation, not cached bullets
    FORMAT_BYPASS_WORDS = [
        "explain", "briefly", "summarize", "list", "describe",
        "give me a short", "give me a brief", "in short",
        "in detail", "elaborate", "break down", "outline"
    ]
    response_mode = state.get("response_mode")
    q_lower       = state["question"].lower().strip()
    bypass_cache  = (
        response_mode is not None
        or any(q_lower.startswith(w) for w in FORMAT_BYPASS_WORDS)
    )
    if bypass_cache:
        print(f"[cache] Bypass — mode={response_mode} | "
              f"format_word={any(q_lower.startswith(w) for w in FORMAT_BYPASS_WORDS)}")

    if intent != "out_of_scope" and not bypass_cache:
        cached = get_cached_answer(
            state["session_id"],
            state["question"],
            question_embedding,
            intent=intent
        )
        if cached:
            return {
                **state,
                "query_intent":         intent,
                "hyde_used":            False,
                "retrieval_confidence": 0.0,
                "doc_types":            doc_types,
                "llm_config":           llm_config,
                "question_embedding":   question_embedding,
                "answer":               cached["answer"],
                "citations":            cached["citations"],
                "cache_hit":            True,
                "cache_similarity":     cached["cache_similarity"],
                "is_sufficient":        True
            }

    return {
        **state,
        "query_intent":         intent,
        "hyde_used":            False,
        "retrieval_confidence": 0.0,
        "doc_types":            doc_types,
        "llm_config":           llm_config,
        "question_embedding":   question_embedding,
        "cache_hit":            False,
        "cache_similarity":     None
    }


def retrieve_node(state: AgentState) -> AgentState:
    """
    Retrieval strategy depends on intent:

    SUMMARY  → Spread retrieval (get_spread_chunks)
        Does NOT use embedding similarity — samples 1 chunk per page evenly
        across the entire document. Guarantees coverage of intro, body, conclusion.
        Embedding "summarize this pdf" finds random admin clauses — useless.

    COMPARISON → Per-PDF hybrid search
        Retrieves top_k chunks separately per PDF, groups by doc_label.
        Ensures both documents are represented equally in comparison.

    FACTUAL → Hybrid search (vector + BM25 + RRF)
        Standard retrieval — finds chunks most similar to the question.
        HyDE triggered if top RRF score below confidence threshold.
    """
    try:
        intent     = state.get("query_intent", "factual")
        session_id = state["session_id"]
        question   = state["question"]
        # Use rewritten question for retrieval if coref resolved references
        # Falls back to original question if coref_node didn't rewrite
        retrieval_question = state.get("retrieval_question") or question
        hyde_used  = False

        # ── Summary: spread retrieval across entire document ──────────────────
        if intent == "summary":
            pdf_ids = get_session_pdf_ids(session_id)
            if not pdf_ids:
                return {**state, "retrieved_chunks": [], "retrieval_confidence": 0.0,
                        "hyde_used": False, "error": "No documents in session"}

            if len(pdf_ids) == 1:
                # Single PDF — spread across all its pages
                chunks = get_spread_chunks(
                    session_id=session_id,
                    pdf_id=pdf_ids[0],
                    max_chunks=12
                )
                print(f"[retrieve] Summary spread: {len(chunks)} chunks from single PDF")
            else:
                # Multiple PDFs — spread per PDF, merge
                chunks_per_pdf = max(6, 12 // len(pdf_ids))
                all_chunks = []
                for i, pdf_id in enumerate(pdf_ids):
                    pdf_chunks = get_spread_chunks(
                        session_id=session_id,
                        pdf_id=pdf_id,
                        max_chunks=chunks_per_pdf
                    )
                    for chunk in pdf_chunks:
                        chunk["doc_label"] = f"Document {i+1} (PDF: {pdf_id[:8]}...)"
                    all_chunks.extend(pdf_chunks)
                    print(f"[retrieve] Summary spread PDF {pdf_id[:8]}...: {len(pdf_chunks)} chunks")
                chunks = all_chunks

            top_score = 0.01  # spread retrieval has no similarity score
            print(f"[retrieve] Summary spread complete: {len(chunks)} total chunks")

            return {
                **state,
                "retrieved_chunks":     chunks,
                "retrieval_confidence": top_score,
                "hyde_used":            False,
                "error":                None
            }

        # ── Comparison: per-PDF hybrid search ─────────────────────────────────
        top_k           = 10
        query_embedding = state.get("question_embedding") or embed_query(retrieval_question)

        if intent == "comparison":
            pdf_ids = get_session_pdf_ids(session_id)

            if len(pdf_ids) < 2:
                print(f"[retrieve] Comparison: only {len(pdf_ids)} PDF — falling back to standard")
                chunks = hybrid_search(
                    session_id=session_id,
                    query=question,
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            else:
                all_chunks = []
                per_pdf_k  = max(4, top_k // len(pdf_ids))
                for pdf_id in pdf_ids:
                    pdf_chunks = hybrid_search(
                        session_id=session_id,
                        query=retrieval_question,
                        query_embedding=query_embedding,
                        top_k=per_pdf_k,
                        pdf_id=pdf_id
                    )
                    for chunk in pdf_chunks:
                        chunk["doc_label"] = f"Document {pdf_ids.index(pdf_id) + 1} (PDF: {pdf_id[:8]}...)"
                    all_chunks.extend(pdf_chunks)
                    print(f"[retrieve] PDF {pdf_id[:8]}... → {len(pdf_chunks)} chunks")
                chunks = all_chunks

            top_score = max((c["score"] for c in chunks), default=0.0)
            print(f"[retrieve] Top score: {top_score:.4f} | Chunks: {len(chunks)} | Intent: {intent}")
            return {
                **state,
                "retrieved_chunks":     chunks,
                "retrieval_confidence": top_score,
                "hyde_used":            False,
                "error":                None
            }

        # ── Factual: query expansion + multi-query hybrid search ─────────────
        top_k = cfg.RETRIEVAL_TOP_K_FACTUAL

        # Step 1: Expand query into variants using T5 paraphrase model
        # "termination notice" → ["termination notice",
        #                          "how many days for contract termination",
        #                          "notice period to end the agreement"]
        # Falls back to [original] if model unavailable — zero degradation
        query_variants = expand_query(retrieval_question)

        if len(query_variants) > 1:
            # Step 2: Embed all variants
            query_embeddings_list = []
            for variant in query_variants:
                if variant == retrieval_question:
                    # Reuse already-computed embedding for original query
                    query_embeddings_list.append(query_embedding)
                else:
                    query_embeddings_list.append(embed_query(variant))

            # Step 3: Multi-query search — runs hybrid search per variant, merges via RRF
            chunks = multi_query_hybrid_search(
                session_id=session_id,
                queries=query_variants,
                query_embeddings=query_embeddings_list,
                top_k=top_k
            )
        else:
            # No expansion — single hybrid search as before
            chunks = hybrid_search(
                session_id=session_id,
                query=retrieval_question,
                query_embedding=query_embedding,
                top_k=top_k
            )

        top_score = max((c["score"] for c in chunks), default=0.0)
        print(f"[retrieve] Top score: {top_score:.4f} | Chunks: {len(chunks)} | "
              f"Intent: {intent} | Variants: {len(query_variants)}")

        # HyDE — triggered only for factual when confidence is still low
        # after query expansion (expansion failed OR document genuinely lacks info)
        if top_score < HYDE_CONFIDENCE_THRESHOLD:
            print(f"[retrieve] Low confidence ({top_score:.3f}) after expansion — triggering HyDE")
            hyde_embedding = generate_hyde_embedding(retrieval_question)
            chunks = hybrid_search(
                session_id=session_id,
                query=retrieval_question,
                query_embedding=hyde_embedding,
                top_k=top_k
            )
            hyde_used = True
            top_score = max((c["score"] for c in chunks), default=0.0)
            print(f"[retrieve] HyDE top score: {top_score:.4f}")

        return {
            **state,
            "retrieved_chunks":     chunks,
            "retrieval_confidence": top_score,
            "hyde_used":            hyde_used,
            "error":                None
        }

    except Exception as e:
        print(f"[retrieve] Error: {e}")
        return {**state, "retrieved_chunks": [], "error": str(e)}


def rerank_node(state: AgentState) -> AgentState:
    """
    Rerank chunks, apply threshold filter, then deduplicate.

    Pipeline inside this node:
      1. Cross-encoder reranking   — scores every chunk deeply
      2. Threshold filter (-8)     — drops garbage chunks (factual only)
      3. Chunk deduplication       — drops near-duplicates before prompt

    WHY threshold is skipped for summary/comparison:
      Summary queries like "summarize this pdf" are vague by design.
      Cross-encoder scores all chunks low (~-6 to -8) because no chunk
      specifically "answers" a summary request — that's expected.
      Applying the threshold would discard ALL chunks → empty answer.
      For summary/comparison: rank + dedup, but never threshold-filter.
    """
    chunks   = state["retrieved_chunks"]
    question = state["question"]
    intent   = state.get("query_intent", "factual")

    if not chunks:
        return {**state, "retrieved_chunks": []}

    # Step 1 + 2: rerank and threshold filter
    # Skip threshold for summary/comparison — vague queries score low by nature
    # top_k for summary = all chunks (spread retrieval already selected the right ones)
    # top_k for factual/comparison = 6 (standard)
    skip_threshold = intent in ("summary", "comparison")
    top_k_rerank   = len(chunks) if intent == "summary" else 6
    if skip_threshold:
        print(f"[reranker] Threshold skipped for intent={intent} | keeping all {top_k_rerank} chunks")
    reranked = rerank_chunks(question, chunks, top_k=top_k_rerank, skip_threshold=skip_threshold)

    # Minimum chunks guarantee — if threshold filtered everything out,
    # keep top 2 chunks regardless. This prevents "not available" responses
    # on vague questions where doc summary injection saves the answer.
    # Without at least some chunks, streaming path returns empty immediately.
    if not reranked and chunks:
        print(f"[reranker] All chunks filtered — keeping top 2 as fallback")
        reranked = sorted(chunks, key=lambda x: x.get("cross_score", x.get("score", 0)), reverse=True)[:2]

    # Step 3: Enhancement #10 — deduplicate
    deduped = _deduplicate_chunks(reranked)

    return {**state, "retrieved_chunks": deduped}


def generate_node(state: AgentState) -> AgentState:
    chunks     = state["retrieved_chunks"]
    intent     = state.get("query_intent", "factual")
    confidence = state.get("retrieval_confidence", 0.0)

    if not chunks:
        return {
            **state,
            "answer":        "This information is not available in the uploaded documents.",
            "citations":     [],
            "is_sufficient": True
        }

    # ── Enhancement #9: Answer Confidence Gating ──────────────────────────────
    # Source: "Add a confidence threshold — below it, tell user 'I found partial
    #          information, here's what I know' instead of hallucinating."
    #
    # If retrieval confidence is below threshold AND this is not a retry,
    # return a partial-information response instead of generating a full answer
    # from low-quality chunks. This prevents hallucination on weak retrieval.
    #
    # We still pass the chunks to build citations — so the user can see
    # what partial information WAS found.
    retry_count = state.get("retry_count", 0)
    if confidence < CONFIDENCE_GATE_THRESHOLD and retry_count == 0 and intent not in ("summary", "comparison"):
        print(f"[generate] Confidence gate triggered: {confidence:.4f} < {CONFIDENCE_GATE_THRESHOLD}")
        citations = build_citations(chunks)
        partial_answer = (
            "I found only partial information related to your question. "
            "Here is what I could find in the uploaded documents — "
            "it may not fully answer your question:\n\n"
            + "\n".join([f"- {c['text'][:200]}..." for c in chunks[:3]])
        )
        return {
            **state,
            "answer":        partial_answer,
            "citations":     citations,
            "is_sufficient": True   # don't retry — low retrieval score means doc genuinely lacks this info
        }
    # ──────────────────────────────────────────────────────────────────────────

    llm_config = get_llm_config_from_chunks(chunks)

    if llm_config.get("doc_type") == "general" and state.get("llm_config"):
        chunk_has_doc_type = any(c.get("metadata", {}).get("doc_type") for c in chunks)
        if not chunk_has_doc_type:
            llm_config = state["llm_config"]
            print(f"[generate] Falling back to session-level config: {llm_config.get('doc_type')}")

    db = SessionLocal()
    try:
        memory_context   = build_memory_context(db, state["session_id"])
        doc_summaries    = get_document_summaries(db, state["session_id"])
    finally:
        db.close()

    # ── Enhancement #15: Inject doc summaries for vague questions ─────────────
    # When user asks "what is this document about?" or "give me an overview",
    # we prepend stored summaries to the context so LLM has document-level
    # understanding without needing full retrieval.
    #
    # Always inject doc summaries — short (2-3 sentences) but gives LLM
    # document-level context for all question types. ~100 tokens cost.
    doc_summary_context = ""
    if doc_summaries:
        summary_lines = []
        for ds in doc_summaries:
            summary_lines.append(
                f"Document: {ds['filename']} ({ds['doc_type']})\n{ds['summary']}"
            )
        doc_summary_context = "\n\n".join(summary_lines)
        print(f"[generate] Injecting {len(doc_summaries)} doc summaries (low confidence retrieval)")

    doc_groups_with_types = {}

    if intent == "comparison" and any(c.get("doc_label") for c in chunks):
        doc_groups = defaultdict(list)
        for c in chunks:
            label = c.get("doc_label", "Document")
            doc_groups[label].append(c["text"])
            if label not in doc_groups_with_types:
                doc_groups_with_types[label] = c.get("metadata", {}).get("doc_type", "general")

        context_parts = []
        source_index  = 1
        for label, texts in doc_groups.items():
            context_parts.append(f"=== {label} ===")
            for text in texts:
                context_parts.append(f"[Source {source_index}]\n{text}")
                source_index += 1
        context_text = "\n\n".join(context_parts)
        print(f"[generate] Comparison doc types: {doc_groups_with_types}")
    else:
        context_text = "\n\n---\n\n".join(
            [f"[Source {i+1}]\n{c['text']}" for i, c in enumerate(chunks)]
        )

    prompt_config = _build_prompt_config(state["question"], intent, state.get("response_mode"))

    if intent == "comparison" and doc_groups_with_types:
        doc_type_suffix = _build_comparison_system_suffix(doc_groups_with_types)
    else:
        doc_type_suffix = llm_config.get("system_suffix", "")

    model       = llm_config.get("model", MODEL)
    temperature = llm_config.get("temperature", 0.2)
    max_tokens  = llm_config.get("max_tokens", 1024)
    doc_type    = llm_config.get("doc_type", "general")

    print(f"[generate] Style: {prompt_config['style']} | Doc type: {doc_type} | Temp: {temperature}")

    # ── Enhancement #4: inject user_instruction if set ──────────────────────
    user_instruction = state.get("user_instruction") or ""
    user_instruction_block = (
        f"\n\nUser preference: {user_instruction.strip()}"
        if user_instruction.strip() else ""
    )
    # ─────────────────────────────────────────────────────────────────────────

    system_prompt = f"""You are an expert document analyst and QA assistant.
Answer questions using ONLY the provided context from the uploaded documents.
Extract and state information exactly as it appears in the context — including addresses, names, numbers, and dates.
If the information is genuinely not present anywhere in the context, respond with exactly:
"This information is not available in the uploaded documents."
Never say "it can be inferred", "based on available information", or "it appears that".
Never speculate or derive answers. Never fabricate details.
Do NOT include [Source N] markers in your answer — citations are handled automatically.

{doc_type_suffix}{user_instruction_block}"""

    # Build final prompt — inject doc summaries if available
    summary_section = f"\n\nDocument summaries for reference:\n{doc_summary_context}" if doc_summary_context else ""

    user_prompt = f"""Conversation history:
{memory_context or "None"}{summary_section}

Context from documents:
{context_text}

Question: {state["question"]}

Answer style required:
{prompt_config['instruction']}

Answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    try:
        answer, provider = call_with_fallback(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not answer:
            raise ValueError("Empty response from LLM")

        if provider == "together":
            print(f"[generate] Used Together AI fallback")

        citations = build_citations(chunks)

        store_in_cache(
            state["session_id"],
            state["question"],
            state.get("question_embedding", []),
            answer,
            citations,
            intent=intent,
            response_mode=state.get("response_mode")
        )

        return {
            **state,
            "answer":        answer,
            "citations":     citations,
            "llm_config":    llm_config,
            "is_sufficient": True
        }

    except Exception as e:
        return {
            **state,
            "answer":        f"Generation failed: {e}",
            "citations":     [],
            "is_sufficient": False,
            "error":         str(e)
        }


def out_of_scope_node(state: AgentState) -> AgentState:
    print("[router] Out of scope — skipping retrieval and generation")
    return {
        **state,
        "answer":           "This information is not available in the uploaded documents.",
        "citations":        [],
        "is_sufficient":    True,
        "retrieved_chunks": [],
        "cache_hit":        False
    }


def sufficiency_check_node(state: AgentState) -> AgentState:
    answer      = state.get("answer", "")
    retry_count = state.get("retry_count", 0)

    insufficient_phrases = [
        "don't have enough information",
        "cannot find",
        "not in the context",
        "generation failed"
    ]

    is_insufficient = any(p in answer.lower() for p in insufficient_phrases)

    if is_insufficient and retry_count < MAX_RETRIES:
        return {**state, "is_sufficient": False, "retry_count": retry_count + 1}

    return {**state, "is_sufficient": True}