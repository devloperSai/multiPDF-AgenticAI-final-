import threading
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from pydantic import BaseModel
from models.database import get_db, SessionLocal
from memory.session_store import get_session, update_session_title
from memory.chat_history import save_message
from graph.graph import qa_graph
from fastapi.responses import StreamingResponse
from api.input_validator import validate_question  # Refine 6
import json

router = APIRouter(prefix="/qa", tags=["qa"])


class AskRequest(BaseModel):
    session_id: str
    question: str


def _run_ragas_background(session_id: str, question: str, answer: str, context_texts: list):
    """Run RAGAS in background thread — does not block user response."""
    try:
        from pipeline.evaluator import evaluate_response
        from models.schema import Message

        scores = evaluate_response(question, answer, context_texts)
        valid = [v for v in scores.values() if v is not None]
        avg = round(sum(valid) / len(valid), 4) if valid else None

        db = SessionLocal()
        try:
            msg = (
                db.query(Message)
                .filter(
                    Message.session_id == session_id,
                    Message.role == "assistant"
                )
                .order_by(Message.created_at.desc())
                .first()
            )
            if msg and avg is not None:
                msg.ragas_score = avg
                db.commit()
                print(f"[ragas] Score {avg} saved to message {msg.id}")
        finally:
            db.close()

    except Exception as e:
        print(f"[ragas] Background thread failed: {e}")


def _check_documents_ready(db: DBSession, session_id: str) -> bool:
    """
    Refine 1 — Check if session has at least one fully processed document.
    Prevents questions being asked before any PDF is ready.
    """
    from models.schema import Document
    doc = db.query(Document).filter(
        Document.session_id == session_id,
        Document.status.in_(["completed", "complete"])
    ).first()
    return doc is not None


@router.post("/ask")
def ask(request: AskRequest, db: DBSession = Depends(get_db)):
    # Refine 6 — validate question before anything else
    validate_question(request.question)

    session = get_session(db, request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not _check_documents_ready(db, request.session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents ready. Please wait for PDF processing to complete before asking questions."
        )

    save_message(db, request.session_id, "user", request.question)
    update_session_title(db, request.session_id, request.question)

    initial_state = {
        "question": request.question,
        "session_id": request.session_id,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "ragas_score": None,
        "retry_count": 0,
        "is_sufficient": False,
        "error": None,
        "query_intent": None,
        "hyde_used": False,
        "retrieval_confidence": 0.0,
        "doc_types": [],
        "llm_config": None,
        "cache_hit": False,
        "cache_similarity": None,
        "question_embedding": None
    }

    try:
        final_state = qa_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    answer = final_state.get("answer", "No answer generated.")
    citations = final_state.get("citations", [])
    retry_count = final_state.get("retry_count", 0)
    query_intent = final_state.get("query_intent")
    hyde_used = final_state.get("hyde_used", False)
    retrieval_confidence = final_state.get("retrieval_confidence", 0.0)

    save_message(
        db,
        request.session_id,
        "assistant",
        answer,
        citations=citations,
        ragas_score=None
    )

    context_texts = [c["excerpt"] for c in citations if c.get("excerpt")]
    if (
        context_texts
        and answer
        and "failed" not in answer.lower()
        and query_intent != "out_of_scope"
        and not final_state.get("cache_hit")
    ):
        thread = threading.Thread(
            target=_run_ragas_background,
            args=(request.session_id, request.question, answer, context_texts),
            daemon=True
        )
        thread.start()
        print("[ragas] Background evaluation started")

    return {
        "session_id": request.session_id,
        "question": request.question,
        "answer": answer,
        "citations": citations,
        "retry_count": retry_count,
        "query_intent": query_intent,
        "hyde_used": hyde_used,
        "retrieval_confidence": retrieval_confidence,
        "doc_type": final_state.get("llm_config", {}).get("doc_type"),
        "doc_types": final_state.get("doc_types", []),
        "cache_hit": final_state.get("cache_hit", False),
        "cache_similarity": final_state.get("cache_similarity"),
        "ragas_status": "evaluating in background" if query_intent != "out_of_scope" and not final_state.get("cache_hit") else "skipped"
    }


@router.post("/ask/stream")
async def ask_stream(request: AskRequest, db: DBSession = Depends(get_db)):
    """
    Streaming version of /ask.
    Returns Server-Sent Events (SSE) — tokens appear word by word.
    """
    # Refine 6 — validate question before anything else
    validate_question(request.question)

    session = get_session(db, request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not _check_documents_ready(db, request.session_id):
        raise HTTPException(
            status_code=400,
            detail="No documents ready. Please wait for PDF processing to complete before asking questions."
        )

    save_message(db, request.session_id, "user", request.question)
    update_session_title(db, request.session_id, request.question)

    initial_state = {
        "question": request.question,
        "session_id": request.session_id,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "ragas_score": None,
        "retry_count": 0,
        "is_sufficient": False,
        "error": None,
        "query_intent": None,
        "hyde_used": False,
        "retrieval_confidence": 0.0,
        "doc_types": [],
        "llm_config": None,
        "cache_hit": False,
        "cache_similarity": None,
        "question_embedding": None
    }

    async def event_generator():
        from graph.nodes import route_node, retrieve_node, rerank_node
        from graph.nodes import _build_prompt_config, _build_comparison_system_suffix
        from graph.model_router import get_llm_config_from_chunks
        from graph.fallback_llm import stream_with_fallback
        from core.semantic_cache import store_in_cache
        from memory.context_builder import build_memory_context
        from models.database import SessionLocal as SL
        from collections import defaultdict
        import os

        state = initial_state.copy()
        state = route_node(state)

        if state.get("cache_hit"):
            answer = state["answer"]
            citations = state["citations"]
            for word in answer.split(" "):
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'cache_hit': True, 'query_intent': state.get('query_intent')})}\n\n"
            return

        if state.get("query_intent") == "out_of_scope":
            msg = "This information is not available in the uploaded documents."
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'citations': [], 'cache_hit': False, 'query_intent': 'out_of_scope'})}\n\n"
            return

        state = retrieve_node(state)
        state = rerank_node(state)
        chunks = state["retrieved_chunks"]

        if not chunks:
            msg = "This information is not available in the uploaded documents."
            yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'citations': [], 'cache_hit': False, 'query_intent': state.get('query_intent')})}\n\n"
            return

        db2 = SL()
        try:
            memory_context = build_memory_context(db2, request.session_id)
        finally:
            db2.close()

        intent = state.get("query_intent", "factual")

        # Refine 2 — derive LLM config from retrieved chunk doc_types
        llm_config = get_llm_config_from_chunks(chunks)
        if llm_config.get("doc_type") == "general" and state.get("llm_config"):
            chunk_has_doc_type = any(c.get("metadata", {}).get("doc_type") for c in chunks)
            if not chunk_has_doc_type:
                llm_config = state["llm_config"]
                print(f"[stream] Falling back to session-level config: {llm_config.get('doc_type')}")

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
            print(f"[stream] Comparison doc types: {doc_groups_with_types}")
        else:
            context_text = "\n\n---\n\n".join(
                [f"[Source {i+1}]\n{c['text']}" for i, c in enumerate(chunks)]
            )

        prompt_config = _build_prompt_config(state["question"], intent)

        # Refine 3 — comparison gets dynamic per-document suffix
        if intent == "comparison" and doc_groups_with_types:
            doc_type_suffix = _build_comparison_system_suffix(doc_groups_with_types)
            print(f"[stream] Comparison system suffix applied")
        else:
            doc_type_suffix = llm_config.get("system_suffix", "")

        model = llm_config.get("model", "llama-3.3-70b-versatile")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 1024)

        print(f"[stream] Doc type: {llm_config.get('doc_type')} | Temp: {temperature} | Intent: {intent}")

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

        full_answer = ""
        provider_used = "groq"

        try:
            # Refine 4 — stream with Together AI fallback
            for delta, provider in stream_with_fallback(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                full_answer += delta
                provider_used = provider
                yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        if provider_used == "together":
            print(f"[stream] Used Together AI fallback")

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

        save_message(
            db,
            request.session_id,
            "assistant",
            full_answer,
            citations=citations,
            ragas_score=None
        )

        store_in_cache(
            request.session_id,
            request.question,
            state.get("question_embedding", []),
            full_answer,
            citations
        )

        context_texts = [c["excerpt"] for c in citations if c.get("excerpt")]
        if context_texts and full_answer and intent != "out_of_scope":
            thread = threading.Thread(
                target=_run_ragas_background,
                args=(request.session_id, request.question, full_answer, context_texts),
                daemon=True
            )
            thread.start()

        yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'cache_hit': False, 'query_intent': intent, 'hyde_used': state.get('hyde_used', False), 'retrieval_confidence': state.get('retrieval_confidence', 0.0), 'doc_type': llm_config.get('doc_type'), 'provider': provider_used})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )