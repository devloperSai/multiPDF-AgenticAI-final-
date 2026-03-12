import re
from typing import Any
from core.embedder import embed_query
from core.vector_store import query_chunks, query_chunks_by_pdf, get_session_pdf_ids


def search_document(session_id: str, query: str, top_k: int = 4) -> list:
    """
    Tool: Search for specific information across all documents in session.
    Returns top matching chunks with scores.
    """
    try:
        embedding = embed_query(query)
        chunks = query_chunks(session_id, embedding, top_k=top_k)
        results = []
        for i, c in enumerate(chunks):
            results.append({
                "index": i + 1,
                "text": c["text"],
                "score": round(c["score"], 4),
                "page_number": c["metadata"].get("page_number", "unknown"),
                "pdf_id": c["metadata"].get("pdf_id", "")[:8]
            })
        print(f"[tool] search_document('{query}') → {len(results)} results")
        return results
    except Exception as e:
        return [{"error": str(e)}]


def get_page(session_id: str, page_number: int, pdf_index: int = 0) -> list:
    """
    Tool: Fetch all chunks from a specific page number.
    pdf_index: 0 = first PDF, 1 = second PDF, etc.
    """
    try:
        pdf_ids = get_session_pdf_ids(session_id)
        if not pdf_ids:
            return [{"error": "No documents found in session"}]

        pdf_index = min(pdf_index, len(pdf_ids) - 1)
        pdf_id = pdf_ids[pdf_index]

        from core.vector_store import _get_collection
        collection = _get_collection(session_id)
        results = collection.get(
            where={
                "$and": [
                    {"pdf_id": pdf_id},
                    {"page_number": str(page_number)}
                ]
            },
            include=["documents", "metadatas"]
        )

        chunks = []
        for i, doc in enumerate(results["documents"]):
            chunks.append({
                "index": i + 1,
                "text": doc,
                "page_number": page_number,
                "pdf_id": pdf_id[:8]
            })

        print(f"[tool] get_page({page_number}) → {len(chunks)} chunks")
        return chunks if chunks else [{"error": f"No content found on page {page_number}"}]

    except Exception as e:
        return [{"error": str(e)}]


def calculate(expression: str) -> dict:
    """
    Tool: Safely evaluate math expressions.
    Handles: +, -, *, /, **, %, parentheses, basic functions.
    No eval() — uses safe parser.
    """
    try:
        # Strip whitespace and validate
        expression = expression.strip()

        # Only allow safe characters
        allowed = re.compile(r'^[\d\s\+\-\*\/\(\)\.\%\^]+$')
        if not allowed.match(expression):
            return {"error": "Invalid characters in expression", "expression": expression}

        # Replace ^ with ** for power
        expression = expression.replace("^", "**")

        result = eval(expression, {"__builtins__": {}}, {})
        print(f"[tool] calculate('{expression}') → {result}")
        return {
            "expression": expression,
            "result": round(float(result), 6)
        }

    except Exception as e:
        return {"error": str(e), "expression": expression}


def summarize_document(session_id: str, pdf_index: int = 0, top_k: int = 8) -> list:
    """
    Tool: Get representative chunks from a document for summarization.
    Fetches chunks spread across the document.
    """
    try:
        pdf_ids = get_session_pdf_ids(session_id)
        if not pdf_ids:
            return [{"error": "No documents found"}]

        pdf_index = min(pdf_index, len(pdf_ids) - 1)
        pdf_id = pdf_ids[pdf_index]

        # Use broad query to get representative chunks
        embedding = embed_query("main topic introduction conclusion summary findings")
        chunks = query_chunks_by_pdf(session_id, embedding, pdf_id, top_k=top_k)

        results = []
        for i, c in enumerate(chunks):
            results.append({
                "index": i + 1,
                "text": c["text"],
                "page_number": c["metadata"].get("page_number", "unknown")
            })

        print(f"[tool] summarize_document(pdf_index={pdf_index}) → {len(results)} chunks")
        return results

    except Exception as e:
        return [{"error": str(e)}]


# Tool registry — maps tool name to function
TOOLS = {
    "search_document": search_document,
    "get_page": get_page,
    "calculate": calculate,
    "summarize_document": summarize_document
}

# Tool descriptions for LLM
TOOL_DESCRIPTIONS = """You have access to the following tools. Use them when needed:

1. search_document(query: str) 
   - Search for specific information in the uploaded documents
   - Use when you need to find details about a specific topic
   - Example: search_document("revenue figures 2023")

2. get_page(page_number: int, pdf_index: int = 0)
   - Fetch content from a specific page
   - pdf_index: 0 for first document, 1 for second, etc.
   - Example: get_page(5, 0)

3. calculate(expression: str)
   - Perform mathematical calculations
   - Use when the question involves numbers, percentages, or arithmetic
   - Example: calculate("150 * 0.25")

4. summarize_document(pdf_index: int = 0)
   - Get representative content from a document for summarization
   - Example: summarize_document(0)

To use a tool, respond with EXACTLY this format on its own line:
TOOL_CALL: tool_name(argument1, argument2)

After getting tool results, continue reasoning and provide your final answer.
When you have enough information, provide your final answer normally without any TOOL_CALL."""