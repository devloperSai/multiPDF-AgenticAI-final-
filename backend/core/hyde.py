"""
HyDE — Hypothetical Document Embedding
---------------------------------------
Instead of embedding the question directly:
  1. Ask LLM to generate a hypothetical passage that would answer it
  2. Embed that hypothetical passage
  3. Use that embedding for retrieval

Why this works:
  A hypothetical answer is semantically closer to actual document chunks
  than the question itself. "What is the termination notice period?" embeds
  close to other questions. A passage like "The termination notice period is
  30 days as per clause 6.2..." embeds close to actual contract text.

Only triggered when retrieval confidence is low (RRF score < 0.02).

Fallback chain (no crash on any failure):
  call_with_fallback (Groq → Together → OpenAI → Anthropic)
      → if all providers fail → embed original question directly
  The system always returns a valid embedding — never raises.
"""

from core.embedder import embed_query


def generate_hyde_embedding(question: str) -> list:
    """
    Generate a HyDE embedding for the question.

    Uses call_with_fallback — same provider chain as answer generation.
    Previously used a hardcoded Groq client which bypassed the fallback
    system entirely. If Groq was down, HyDE failed silently with no retry.

    Args:
        question: The user's question to generate a hypothetical answer for.

    Returns:
        Embedding vector — either of the hypothetical answer (success)
        or of the original question (all providers failed).
    """
    try:
        from graph.fallback_llm import call_with_fallback

        hypothetical_answer, provider = call_with_fallback(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document passage generator. "
                        "Generate a realistic, factual passage that directly answers the question below. "
                        "Write it as if extracted from an actual document — formal, precise, no preamble. "
                        "Do not say 'hypothetically' or 'this passage'. Just write the passage."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nPassage:"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=cfg.HYDE_TEMPERATURE,
            max_tokens=cfg.HYDE_MAX_TOKENS,
        )

        hypothetical_answer = hypothetical_answer.strip()
        print(f"[hyde] Generated via {provider}: {hypothetical_answer[:100]}...")

        return embed_query(hypothetical_answer)

    except Exception as e:
        # All providers failed — fall back to embedding the question directly
        # This is safe — retrieval continues with original question embedding
        print(f"[hyde] All providers failed: {e} — falling back to question embedding")
        return embed_query(question)