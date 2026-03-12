import os
from collections import defaultdict
from dotenv import load_dotenv
from core.embedder import embed_query
from core.vector_store import query_chunks, query_chunks_by_pdf, get_session_pdf_ids
from core.hyde import generate_hyde_embedding
from core.semantic_cache import get_cached_answer, store_in_cache
from core.reranker import rerank_chunks
from graph.state import AgentState
from graph.router import classify_intent
from graph.model_router import get_llm_config, get_llm_config_from_chunks
from graph.fallback_llm import call_with_fallback, stream_with_fallback
from memory.context_builder import build_memory_context
from memory.session_store import get_session_doc_types
from models.database import SessionLocal

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 2
HYDE_CONFIDENCE_THRESHOLD = 0.75

# Refine 3 — per-document type descriptions for comparison system prompt
DOC_TYPE_DESCRIPTIONS = {
    "research": "a research paper (focus on methodology, findings, and conclusions)",
    "legal": "a legal document (focus on clauses, obligations, rights, and exact legal terms)",
    "financial": "a financial document (focus on exact figures, percentages, and financial metrics)",
    "general": "a general document",
}


def _build_comparison_system_suffix(doc_groups_with_types: dict) -> str:
    """
    Refine 3 — Build dynamic system prompt suffix for comparison intent.
    doc_groups_with_types: {doc_label: doc_type}
    """
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


def _build_prompt_config(question: str, intent: str) -> dict:
    q = question.lower()

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

    brief_signals = ["brief", "short", "summarize", "quick", "one line", "tldr", "in a sentence", "in short"]
    list_signals = ["list", "what are", "enumerate", "give me all", "what were", "mention all"]
    detailed_signals = ["explain", "describe", "how does", "what is", "elaborate", "in detail", "tell me about"]

    if any(s in q for s in brief_signals):
        return {
            "style": "concise",
            "instruction": "Give a brief, direct answer in 2-3 sentences maximum. Cite sources inline using [Source N]."
        }
    elif any(s in q for s in list_signals):
        return {
            "style": "list",
            "instruction": "Answer using a numbered or bulleted list. Be comprehensive and cover all relevant points. Cite sources inline using [Source N]."
        }
    elif any(s in q for s in detailed_signals):
        return {
            "style": "detailed",
            "instruction": """Give a comprehensive, well-structured answer. Use this structure where applicable:
- Start with a clear definition or overview
- Explain the components or how it works
- Cover the purpose and significance
- Use bullet points or numbered lists for multi-part information
- Cite every claim using [Source N] inline"""
        }
    else:
        return {
            "style": "balanced",
            "instruction": "Give a clear, complete answer. Use structure if the topic has multiple aspects. Cite sources inline using [Source N]."
        }


def route_node(state: AgentState) -> AgentState:
    """
    Classify intent, fetch doc_types, embed question, check cache.
    llm_config set here is session-level fallback only.
    generate_node overrides with chunk-level config (Refine 2).
    """
    intent = classify_intent(state["question"])
    print(f"[router] Intent detected: {intent}")

    db = SessionLocal()
    try:
        doc_types = get_session_doc_types(db, state["session_id"])
    finally:
        db.close()

    llm_config = get_llm_config(doc_types)
    print(f"[router] Session doc types: {doc_types} | Fallback config: {llm_config['doc_type']}")

    question_embedding = embed_query(state["question"])

    if intent != "out_of_scope":
        cached = get_cached_answer(
            state["session_id"],
            state["question"],
            question_embedding
        )
        if cached:
            return {
                **state,
                "query_intent": intent,
                "hyde_used": False,
                "retrieval_confidence": 0.0,
                "doc_types": doc_types,
                "llm_config": llm_config,
                "question_embedding": question_embedding,
                "answer": cached["answer"],
                "citations": cached["citations"],
                "cache_hit": True,
                "cache_similarity": cached["cache_similarity"],
                "is_sufficient": True
            }

    return {
        **state,
        "query_intent": intent,
        "hyde_used": False,
        "retrieval_confidence": 0.0,
        "doc_types": doc_types,
        "llm_config": llm_config,
        "question_embedding": question_embedding,
        "cache_hit": False,
        "cache_similarity": None
    }


def retrieve_node(state: AgentState) -> AgentState:
    """
    Retrieve chunks with intent-aware strategy.
    Comparison: retrieves from each PDF separately and tags chunks.
    Other intents: standard retrieval across entire session.
    """
    try:
        intent = state.get("query_intent", "factual")
        top_k = 10 if intent in ("summary", "comparison") else 6
        query_embedding = state.get("question_embedding") or embed_query(state["question"])
        hyde_used = False

        if intent == "comparison":
            pdf_ids = get_session_pdf_ids(state["session_id"])

            if len(pdf_ids) < 2:
                print(f"[retrieve] Comparison requested but only {len(pdf_ids)} PDF(s) — falling back to standard retrieval")
                chunks = query_chunks(
                    session_id=state["session_id"],
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            else:
                all_chunks = []
                per_pdf_k = max(4, top_k // len(pdf_ids))

                for pdf_id in pdf_ids:
                    pdf_chunks = query_chunks_by_pdf(
                        session_id=state["session_id"],
                        query_embedding=query_embedding,
                        pdf_id=pdf_id,
                        top_k=per_pdf_k
                    )
                    for chunk in pdf_chunks:
                        chunk["doc_label"] = f"Document {pdf_ids.index(pdf_id) + 1} (PDF: {pdf_id[:8]}...)"
                    all_chunks.extend(pdf_chunks)
                    print(f"[retrieve] PDF {pdf_id[:8]}... → {len(pdf_chunks)} chunks")

                chunks = all_chunks

        else:
            chunks = query_chunks(
                session_id=state["session_id"],
                query_embedding=query_embedding,
                top_k=top_k
            )

        top_score = max((c["score"] for c in chunks), default=0.0)
        print(f"[retrieve] Top score: {top_score} | Chunks: {len(chunks)} | Intent: {intent}")

        if top_score < HYDE_CONFIDENCE_THRESHOLD and intent not in ("out_of_scope", "comparison"):
            print(f"[retrieve] Low confidence ({top_score:.3f}) — triggering HyDE")
            hyde_embedding = generate_hyde_embedding(state["question"])
            chunks = query_chunks(
                session_id=state["session_id"],
                query_embedding=hyde_embedding,
                top_k=top_k
            )
            hyde_used = True
            top_score = max((c["score"] for c in chunks), default=0.0)
            print(f"[retrieve] HyDE top score: {top_score}")

        return {
            **state,
            "retrieved_chunks": chunks,
            "retrieval_confidence": top_score,
            "hyde_used": hyde_used,
            "error": None
        }

    except Exception as e:
        return {**state, "retrieved_chunks": [], "error": str(e)}


def rerank_node(state: AgentState) -> AgentState:
    """
    Two-stage reranking:
    1. Filter chunks below 0.3 cosine similarity
    2. Re-score remaining chunks with cross-encoder
    """
    chunks = state["retrieved_chunks"]
    question = state["question"]

    filtered = [c for c in chunks if c.get("score", 0) > 0.3]

    if not filtered:
        return {**state, "retrieved_chunks": []}

    reranked = rerank_chunks(question, filtered, top_k=6)
    return {**state, "retrieved_chunks": reranked}


def generate_node(state: AgentState) -> AgentState:
    """
    Generate answer using doc-type-aware LLM configuration.
    Refine 2 — LLM config from retrieved chunk doc_types.
    Refine 3 — Comparison intent gets per-document system prompt.
    Refine 4 — Groq call wrapped with Together AI fallback.
    """
    chunks = state["retrieved_chunks"]
    intent = state.get("query_intent", "factual")

    if not chunks:
        return {
            **state,
            "answer": "This information is not available in the uploaded documents.",
            "citations": [],
            "is_sufficient": True
        }

    # Refine 2 — derive LLM config from actual retrieved chunks
    llm_config = get_llm_config_from_chunks(chunks)

    # Fallback to session-level config if chunks have no doc_type metadata
    if llm_config.get("doc_type") == "general" and state.get("llm_config"):
        chunk_has_doc_type = any(
            c.get("metadata", {}).get("doc_type")
            for c in chunks
        )
        if not chunk_has_doc_type:
            llm_config = state["llm_config"]
            print(f"[generate] Falling back to session-level config: {llm_config.get('doc_type')}")

    db = SessionLocal()
    try:
        memory_context = build_memory_context(db, state["session_id"])
    finally:
        db.close()

    # Build context + track doc_types per label for Refine 3
    doc_groups_with_types = {}

    if intent == "comparison" and any(c.get("doc_label") for c in chunks):
        doc_groups = defaultdict(list)
        for c in chunks:
            label = c.get("doc_label", "Document")
            doc_groups[label].append(c["text"])
            if label not in doc_groups_with_types:
                doc_groups_with_types[label] = c.get("metadata", {}).get("doc_type", "general")

        context_parts = []
        source_index = 1
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

    prompt_config = _build_prompt_config(state["question"], intent)

    # Refine 3 — comparison gets dynamic per-document suffix
    if intent == "comparison" and doc_groups_with_types:
        doc_type_suffix = _build_comparison_system_suffix(doc_groups_with_types)
        print(f"[generate] Comparison system suffix applied")
    else:
        doc_type_suffix = llm_config.get("system_suffix", "")

    model = llm_config.get("model", MODEL)
    temperature = llm_config.get("temperature", 0.2)
    max_tokens = llm_config.get("max_tokens", 1024)
    doc_type = llm_config.get("doc_type", "general")

    print(f"[generate] Style: {prompt_config['style']} | Doc type: {doc_type} | Temp: {temperature}")

    system_prompt = f"""You are an expert document analyst and QA assistant.
Answer questions using ONLY the provided context from the uploaded documents.
If the answer is not in the context, respond with exactly one sentence: "This information is not available in the uploaded documents."
Do not elaborate, explain, or pad the response when the answer is not found.
Never fabricate or assume information not explicitly present in the context.
Always cite sources using [Source N] notation inline for every claim.

{doc_type_suffix}"""

    user_prompt = f"""Conversation history:
{memory_context or "None"}

Context from documents:
{context_text}

Question: {state["question"]}

Answer style required:
{prompt_config['instruction']}

Answer:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # Refine 4 — Groq with Together AI fallback
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

        citations = [
            {
                "source_index": i + 1,
                "pdf_id": c["metadata"].get("pdf_id"),
                "chunk_index": c["metadata"].get("chunk_index"),
                "page_number": c["metadata"].get("page_number", None),
                "score": round(c["score"], 4),
                "excerpt": c["text"][:200]
            }
            for i, c in enumerate(chunks)
        ]

        store_in_cache(
            state["session_id"],
            state["question"],
            state.get("question_embedding", []),
            answer,
            citations
        )

        return {
            **state,
            "answer": answer,
            "citations": citations,
            "llm_config": llm_config,
            "is_sufficient": True
        }

    except Exception as e:
        return {
            **state,
            "answer": f"Generation failed: {e}",
            "citations": [],
            "is_sufficient": False,
            "error": str(e)
        }


def out_of_scope_node(state: AgentState) -> AgentState:
    """Fast exit for out-of-scope questions."""
    print("[router] Out of scope — skipping retrieval and generation")
    return {
        **state,
        "answer": "This information is not available in the uploaded documents.",
        "citations": [],
        "is_sufficient": True,
        "retrieved_chunks": [],
        "cache_hit": False
    }


def sufficiency_check_node(state: AgentState) -> AgentState:
    """Check if answer is adequate or needs retry."""
    answer = state.get("answer", "")
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