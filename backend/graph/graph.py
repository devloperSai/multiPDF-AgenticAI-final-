from langgraph.graph import StateGraph, END
from graph.state import AgentState
from graph.nodes import (
    route_node,
    retrieve_node,
    rerank_node,
    generate_node,
    out_of_scope_node,
    sufficiency_check_node
)
from graph.agent_node import agentic_generate_node


def _route_after_routing(state: AgentState) -> str:
    if state.get("cache_hit"):
        return "cache_hit"
    if state.get("query_intent") == "out_of_scope":
        return "out_of_scope"
    return "retrieve"


def _route_after_rerank(state: AgentState) -> str:
    """Route to agent for complex queries, standard generate for simple ones."""
    intent = state.get("query_intent", "factual")
    if intent in ("comparison", "summary"):
        return "agent_generate"
    return "generate"


def _should_retry(state: AgentState) -> str:
    if not state.get("is_sufficient") and state.get("retry_count", 0) < 2:
        return "retrieve"
    return END


def _cache_hit_node(state: AgentState) -> AgentState:
    print(f"[cache] Returning cached answer (similarity={state.get('cache_similarity')})")
    return state


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route", route_node)
    graph.add_node("cache_hit", _cache_hit_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("agent_generate", agentic_generate_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("sufficiency_check", sufficiency_check_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        _route_after_routing,
        {
            "cache_hit": "cache_hit",
            "out_of_scope": "out_of_scope",
            "retrieve": "retrieve"
        }
    )

    graph.add_edge("cache_hit", END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("retrieve", "rerank")

    graph.add_conditional_edges(
        "rerank",
        _route_after_rerank,
        {
            "agent_generate": "agent_generate",
            "generate": "generate"
        }
    )

    graph.add_edge("agent_generate", "sufficiency_check")
    graph.add_edge("generate", "sufficiency_check")

    graph.add_conditional_edges(
        "sufficiency_check",
        _should_retry,
        {
            "retrieve": "retrieve",
            END: END
        }
    )

    return graph.compile()


qa_graph = build_graph()