"""
Refine 4 — Together AI Fallback LLM
------------------------------------
Called ONLY when Groq API fails (rate limit, timeout, server error).
Uses same Llama 3.3 70B model for consistent output quality.
Never called on successful Groq responses — zero overhead on happy path.

Usage:
    from graph.fallback_llm import call_with_fallback

    answer = call_with_fallback(
        messages=[...],
        temperature=0.2,
        max_tokens=1024,
        stream=False
    )
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Together AI uses OpenAI-compatible API
# No new SDK needed — works with openai package already installed
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def _call_groq(client, messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """Standard Groq call — returns answer string."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def _call_together(messages: list, temperature: float, max_tokens: int) -> str:
    """
    Together AI fallback call.
    Uses OpenAI-compatible client pointed at Together's endpoint.
    Same interface as Groq — drop-in replacement.
    """
    try:
        from openai import OpenAI

        together_client = OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url=TOGETHER_BASE_URL
        )

        response = together_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise RuntimeError(f"Together AI fallback also failed: {e}")


def call_with_fallback(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    """
    Try Groq first. If it fails, fall back to Together AI.

    Returns:
        (answer: str, provider: str) — provider is "groq" or "together"

    Raises:
        RuntimeError if both providers fail.
    """
    from groq import Groq

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        answer = _call_groq(groq_client, messages, model, temperature, max_tokens)
        return answer, "groq"

    except Exception as groq_error:
        print(f"[fallback_llm] Groq failed: {groq_error} — trying Together AI")

        try:
            answer = _call_together(messages, temperature, max_tokens)
            print(f"[fallback_llm] Together AI succeeded")
            return answer, "together"

        except Exception as together_error:
            raise RuntimeError(
                f"Both providers failed.\n"
                f"Groq: {groq_error}\n"
                f"Together: {together_error}"
            )


def stream_with_fallback(
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
):
    """
    Streaming version — try Groq stream first.
    If Groq fails before stream starts, fall back to Together AI stream.
    If Groq fails mid-stream, that partial output is already sent — 
    we do NOT retry mid-stream as tokens already reached the client.

    Yields: (delta: str, provider: str)
    """
    from groq import Groq

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Try Groq streaming
    try:
        stream = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        # Yield first chunk to confirm stream started
        first_chunk = True
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_chunk:
                    print(f"[fallback_llm] Groq stream started")
                    first_chunk = False
                yield delta, "groq"
        return

    except Exception as groq_error:
        print(f"[fallback_llm] Groq stream failed before start: {groq_error} — trying Together AI stream")

    # Groq failed before yielding anything — try Together AI stream
    try:
        from openai import OpenAI

        together_client = OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url=TOGETHER_BASE_URL
        )

        stream = together_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        print(f"[fallback_llm] Together AI stream started")
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta, "together"

    except Exception as together_error:
        raise RuntimeError(
            f"Both stream providers failed.\n"
            f"Groq: {groq_error}\n"
            f"Together: {together_error}"
        )