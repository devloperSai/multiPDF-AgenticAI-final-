from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    question:   str
    session_id: str

    # Retrieval
    retrieved_chunks:     List[dict]
    retrieval_confidence: float
    hyde_used:            bool
    question_embedding:   Optional[list]

    # Generation
    answer:      str
    citations:   List[dict]
    ragas_score: Optional[float]

    # Flow control
    retry_count:  int
    is_sufficient: bool
    error:         Optional[str]

    # Intent — WHAT to retrieve and how much
    # factual | summary | comparison | out_of_scope
    query_intent: Optional[str]

    # ── Enhancement #4: Response mode — HOW to format the answer ─────────────
    # Set by user via UI mode selector (short/verbatim/explanation/bullets).
    # None = not set by user → keyword detection in _build_prompt_config applies.
    #
    # Priority in _build_prompt_config:
    #   1. response_mode (user explicit choice)  ← highest priority
    #   2. intent (summary/comparison force their own style)
    #   3. keyword detection (brief_signals, list_signals etc.)
    #   4. balanced default
    #
    # user_instruction = free text from the persistent instruction box.
    # Injected into system prompt as additional context for all answers.
    response_mode:    Optional[str]   # "short"|"verbatim"|"explanation"|"bullets"|None
    user_instruction: Optional[str]   # free text persistent instruction, or None
    # ─────────────────────────────────────────────────────────────────────────

    # Doc type routing
    doc_types:  List[str]
    llm_config: Optional[dict]

    # ── Enhancement #7: Coreference resolution ───────────────────────────────
    # retrieval_question = rewritten version of question for embedding + BM25.
    # Set by coref_node if pronouns detected, otherwise = original question.
    # Original question always preserved — used for display, DB, cache, prompt.
    # retrieval_question used ONLY inside retrieve_node for vector + BM25 search.
    retrieval_question: Optional[str]
    coref_rewritten:    bool            # True if question was actually rewritten
    # ─────────────────────────────────────────────────────────────────────────

    # Enhancement #14 — Query Expansion
    expanded_queries: Optional[List[str]]   # [original, variant1, variant2...]

    # Cache
    cache_hit:        bool
    cache_similarity: Optional[float]