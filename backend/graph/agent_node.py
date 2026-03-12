import os
import re
from groq import Groq
from dotenv import load_dotenv
from core.tools import TOOLS, TOOL_DESCRIPTIONS
from graph.state import AgentState
from graph.model_router import get_llm_config
from memory.context_builder import build_memory_context
from models.database import SessionLocal

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MAX_TOOL_CALLS = 5  # prevent infinite loops


def _parse_tool_call(line: str):
    """
    Parse TOOL_CALL: tool_name(arg1, arg2) format.
    Returns (tool_name, args_list) or None.
    """
    match = re.match(r'TOOL_CALL:\s*(\w+)\((.*)\)', line.strip())
    if not match:
        return None

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    # Parse arguments
    args = []
    if args_str:
        for arg in args_str.split(","):
            arg = arg.strip().strip('"').strip("'")
            # Try int, then float, then string
            try:
                args.append(int(arg))
            except ValueError:
                try:
                    args.append(float(arg))
                except ValueError:
                    args.append(arg)

    return tool_name, args


def _execute_tool(tool_name: str, args: list, session_id: str) -> str:
    """Execute a tool and return formatted result string."""
    if tool_name not in TOOLS:
        return f"[Tool Error] Unknown tool: {tool_name}"

    tool_fn = TOOLS[tool_name]

    try:
        # Always inject session_id as first arg for document tools
        if tool_name in ("search_document", "get_page", "summarize_document"):
            result = tool_fn(session_id, *args)
        else:
            result = tool_fn(*args)

        # Format result as readable string
        if isinstance(result, list):
            parts = []
            for item in result:
                if "error" in item:
                    parts.append(f"Error: {item['error']}")
                elif "text" in item:
                    page = item.get("page_number", "?")
                    parts.append(f"[Page {page}] {item['text'][:300]}")
            return "\n\n".join(parts) if parts else "No results found."
        elif isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            elif "result" in result:
                return f"Result: {result['result']}"
            return str(result)
        return str(result)

    except Exception as e:
        return f"[Tool Error] {tool_name} failed: {e}"


def agentic_generate_node(state: AgentState) -> AgentState:
    """
    Tool-use agent node.
    LLM decides which tools to call, executes them, then generates final answer.
    Replaces generate_node for complex queries.
    """
    chunks = state["retrieved_chunks"]
    intent = state.get("query_intent", "factual")
    llm_config = state.get("llm_config", {})
    session_id = state["session_id"]

    # Build initial context from retrieved chunks
    if chunks:
        initial_context = "\n\n---\n\n".join(
            [f"[Source {i+1} | Page {c['metadata'].get('page_number', '?')}]\n{c['text']}"
             for i, c in enumerate(chunks)]
        )
    else:
        initial_context = "No initial context retrieved."

    db = SessionLocal()
    try:
        memory_context = build_memory_context(db, session_id)
    finally:
        db.close()

    model = llm_config.get("model", "llama-3.3-70b-versatile")
    temperature = llm_config.get("temperature", 0.2)
    max_tokens = llm_config.get("max_tokens", 1500)
    doc_type_suffix = llm_config.get("system_suffix", "")

    system_prompt = f"""You are an expert document analyst with access to tools for searching and analyzing documents.

{TOOL_DESCRIPTIONS}

Rules:
- Use tools when the initial context is insufficient or you need specific information
- Always cite sources using [Source N] or [Page N] notation
- After using tools, synthesize all gathered information into a comprehensive answer
- If information is not available in the documents, say so clearly
- Never fabricate information

{doc_type_suffix}"""

    # Start conversation with initial context
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""Conversation history:
{memory_context or "None"}

Initial context from documents:
{initial_context}

Question: {state["question"]}

Think step by step. Use tools if you need more specific information. Then provide your final answer."""}
    ]

    tool_call_count = 0
    full_answer = ""
    tool_results_log = []

    # Agentic loop
    while tool_call_count <= MAX_TOOL_CALLS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response_text = response.choices[0].message.content.strip()

        except Exception as e:
            return {
                **state,
                "answer": f"Agent failed: {e}",
                "citations": [],
                "is_sufficient": False,
                "error": str(e)
            }

        # Check for tool calls in response
        lines = response_text.split("\n")
        tool_called = False

        for line in lines:
            if line.strip().startswith("TOOL_CALL:"):
                parsed = _parse_tool_call(line)
                if parsed:
                    tool_name, args = parsed
                    print(f"[agent] Tool call: {tool_name}({args})")

                    tool_result = _execute_tool(tool_name, args, session_id)
                    tool_results_log.append({
                        "tool": tool_name,
                        "args": args,
                        "result_preview": tool_result[:100]
                    })

                    # Add tool exchange to conversation
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": f"Tool result for {tool_name}:\n{tool_result}\n\nContinue with your analysis."
                    })

                    tool_call_count += 1
                    tool_called = True
                    break

        if not tool_called:
            # No tool call — this is the final answer
            full_answer = response_text
            print(f"[agent] Final answer after {tool_call_count} tool calls")
            break

        if tool_call_count >= MAX_TOOL_CALLS:
            # Force final answer
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "You have reached the maximum tool calls. Please provide your final answer now based on all gathered information."
            })
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            full_answer = response.choices[0].message.content.strip()
            break

    # Build citations from initial chunks
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

    print(f"[agent] Tools used: {[t['tool'] for t in tool_results_log]}")

    return {
        **state,
        "answer": full_answer,
        "citations": citations,
        "is_sufficient": True,
        "error": None
    }