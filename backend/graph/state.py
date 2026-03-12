from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    question: str
    session_id: str
    retrieved_chunks: List[dict]
    answer: str
    citations: List[dict]
    ragas_score: Optional[float]
    retry_count: int
    is_sufficient: bool
    error: Optional[str]
    # Phase 7
    query_intent: Optional[str]
    hyde_used: bool
    retrieval_confidence: float
    # Phase 8
    doc_types: List[str]
    llm_config: Optional[dict]
    # Phase 9
    cache_hit: bool
    cache_similarity: Optional[float]
    question_embedding: Optional[list]