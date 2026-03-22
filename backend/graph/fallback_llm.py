"""
Multi-Provider LLM Dispatcher
-------------------------------
Production provider chain — tries each provider in order.
Falls back on rate limit, quota exhaustion, or any failure.

Provider chain (tried in order):
  1. Groq          — free, fast, primary
  2. Together AI   — free tier, first fallback
  3. OpenAI        — paid, optional (only if OPENAI_API_KEY in .env)
  4. Anthropic     — paid, optional (only if ANTHROPIC_API_KEY in .env)

Key design decisions:
  - Groq stays primary — preserves free quota for as long as possible
  - Paid providers (OpenAI, Anthropic) only join chain if API key exists in .env
  - No key = not in chain = never called = $0 cost
  - Adding a new provider later = one line in .env, zero code changes
  - Rate limit / quota exhaustion is detected and triggers next provider
  - Both streaming and non-streaming paths supported

Fallback note for hyde.py and context_builder.py:
  These also call call_with_fallback — they benefit from this chain too.
  If Groq is exhausted, HyDE and memory summarization fall to Together AI
  automatically. No separate handling needed.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Provider configs ──────────────────────────────────────────────────────────
# Model used per provider
PROVIDER_MODELS = {
    "groq":      "llama-3.3-70b-versatile",
    "together":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "cerebras":  "llama3.1-8b",        # Cerebras free tier — confirmed available
    "gemini":    "gemini-2.0-flash",   # free: 1M tokens/day
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


TOGETHER_BASE_URL = "https://api.together.xyz/v1"

# Errors that mean "quota/rate limit" vs "hard failure"
# On quota errors → try next provider
# On hard errors (auth, bad request) → try next provider too (might work elsewhere)
RATE_LIMIT_KEYWORDS = [
    "rate limit", "rate_limit", "quota", "exhausted",
    "too many requests", "429", "limit exceeded"
]


def _is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a quota/rate limit — vs a hard failure."""
    msg = str(error).lower()
    return any(kw in msg for kw in RATE_LIMIT_KEYWORDS)


def _build_provider_chain() -> list:
    """
    Build provider chain from available API keys.
    Chain order: Groq → Together → Gemini → OpenAI → Anthropic

    Groq:    always included (free, primary)
    Together: always included (paid trial)
    Gemini:  included if GEMINI_API_KEY in .env (free: 1M tokens/day)
    OpenAI:  included if OPENAI_API_KEY in .env (paid)
    Anthropic: included if ANTHROPIC_API_KEY in .env (paid)
    """
    chain = ["groq", "together"]

    if os.getenv("CEREBRAS_API_KEY"):
        chain.append("cerebras")
        print(f"[fallback_llm] Cerebras added to provider chain (free tier)")

    if os.getenv("GEMINI_API_KEY"):
        chain.append("gemini")
        print(f"[fallback_llm] Gemini added to provider chain (free tier)")

    if os.getenv("OPENAI_API_KEY"):
        chain.append("openai")
        print(f"[fallback_llm] OpenAI added to provider chain")

    if os.getenv("ANTHROPIC_API_KEY"):
        chain.append("anthropic")
        print(f"[fallback_llm] Anthropic added to provider chain")

    print(f"[fallback_llm] Provider chain: {' → '.join(chain)}")
    return chain


# Build chain once at module load — doesn't change at runtime
_PROVIDER_CHAIN = _build_provider_chain()


# ── Individual provider callers ───────────────────────────────────────────────

def _call_groq(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def _call_together(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url=TOGETHER_BASE_URL
    )
    response = client.chat.completions.create(
        model=PROVIDER_MODELS["together"],   # Together uses its own model name
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def _call_openai(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=PROVIDER_MODELS["openai"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Anthropic uses separate system param — extract from messages
    system_content = ""
    user_messages  = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append(msg)

    response = client.messages.create(
        model=PROVIDER_MODELS["anthropic"],
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_content,
        messages=user_messages
    )
    return response.content[0].text.strip()


def _call_cerebras(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """
    Cerebras via OpenAI-compatible API.
    Free tier — same Llama 3.3 70B as Groq, extremely fast inference.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url=CEREBRAS_BASE_URL
    )
    response = client.chat.completions.create(
        model=PROVIDER_MODELS["cerebras"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def _call_gemini(messages: list, model: str, temperature: float, max_tokens: int) -> str:
    """
    Gemini via native google-generativeai library.
    Free tier: 1M tokens/day, 1500 requests/day.
    Converts OpenAI message format to Gemini format.
    """
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel(
        model_name=PROVIDER_MODELS["gemini"],
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )

    # Convert OpenAI messages to Gemini format
    # System message → prepend to first user message
    system_content = ""
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            content = f"{system_content}\n\n{msg['content']}" if system_content and not gemini_messages else msg["content"]
            gemini_messages.append({"role": "user", "parts": [content]})
            system_content = ""  # only prepend once
        elif msg["role"] == "assistant":
            gemini_messages.append({"role": "model", "parts": [msg["content"]]})

    if not gemini_messages:
        gemini_messages = [{"role": "user", "parts": [system_content]}]

    response = gemini_model.generate_content(gemini_messages)
    return response.text.strip()


def _call_provider(
    provider:    str,
    messages:    list,
    model:       str,
    temperature: float,
    max_tokens:  int
) -> str:
    """Dispatch to the correct provider caller."""
    if provider == "groq":
        return _call_groq(messages, model, temperature, max_tokens)
    elif provider == "together":
        return _call_together(messages, model, temperature, max_tokens)
    elif provider == "cerebras":
        return _call_cerebras(messages, model, temperature, max_tokens)
    elif provider == "gemini":
        return _call_gemini(messages, model, temperature, max_tokens)
    elif provider == "openai":
        return _call_openai(messages, model, temperature, max_tokens)
    elif provider == "anthropic":
        return _call_anthropic(messages, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Streaming callers ─────────────────────────────────────────────────────────

def _stream_groq(messages, model, temperature, max_tokens):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    stream = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_together(messages, model, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url=TOGETHER_BASE_URL)
    stream = client.chat.completions.create(
        model=PROVIDER_MODELS["together"], messages=messages,
        temperature=temperature, max_tokens=max_tokens, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_cerebras(messages, model, temperature, max_tokens):
    """Cerebras streaming via OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("CEREBRAS_API_KEY"),
        base_url=CEREBRAS_BASE_URL
    )
    stream = client.chat.completions.create(
        model=PROVIDER_MODELS["cerebras"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_gemini(messages, model, temperature, max_tokens):
    """Gemini streaming via native google-generativeai library."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel(
        model_name=PROVIDER_MODELS["gemini"],
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    )

    # Convert OpenAI messages to Gemini format
    system_content = ""
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            content = f"{system_content}\n\n{msg['content']}" if system_content and not gemini_messages else msg["content"]
            gemini_messages.append({"role": "user", "parts": [content]})
            system_content = ""
        elif msg["role"] == "assistant":
            gemini_messages.append({"role": "model", "parts": [msg["content"]]})

    if not gemini_messages:
        gemini_messages = [{"role": "user", "parts": [system_content]}]

    for chunk in gemini_model.generate_content(gemini_messages, stream=True):
        if chunk.text:
            yield chunk.text


def _stream_openai(messages, model, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    stream = client.chat.completions.create(
        model=PROVIDER_MODELS["openai"], messages=messages,
        temperature=temperature, max_tokens=max_tokens, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_anthropic(messages, model, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_content = ""
    user_messages  = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append(msg)

    with client.messages.stream(
        model=PROVIDER_MODELS["anthropic"],
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_content,
        messages=user_messages
    ) as stream:
        for text in stream.text_stream:
            yield text


def _stream_provider(provider, messages, model, temperature, max_tokens):
    """Dispatch to the correct streaming provider."""
    if provider == "groq":
        yield from _stream_groq(messages, model, temperature, max_tokens)
    elif provider == "together":
        yield from _stream_together(messages, model, temperature, max_tokens)
    elif provider == "cerebras":
        yield from _stream_cerebras(messages, model, temperature, max_tokens)
    elif provider == "gemini":
        yield from _stream_gemini(messages, model, temperature, max_tokens)
    elif provider == "openai":
        yield from _stream_openai(messages, model, temperature, max_tokens)
    elif provider == "anthropic":
        yield from _stream_anthropic(messages, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Public interface ──────────────────────────────────────────────────────────

def call_with_fallback(
    messages:    list,
    model:       str,
    temperature: float,
    max_tokens:  int,
) -> tuple[str, str]:
    """
    Try each provider in chain. Return (answer, provider_name).

    Provider chain: Groq → Together → OpenAI* → Anthropic*
    (* only if API key exists in .env)

    On rate limit or any error → log and try next provider.
    If all fail → raise RuntimeError with full error log.
    """
    errors = []

    for provider in _PROVIDER_CHAIN:
        try:
            answer = _call_provider(provider, messages, model, temperature, max_tokens)
            if provider != "groq":
                print(f"[fallback_llm] Used {provider} (Groq unavailable)")
            return answer, provider

        except Exception as e:
            reason = "rate limit" if _is_rate_limit_error(e) else "error"
            print(f"[fallback_llm] {provider} {reason}: {e} — trying next provider")
            errors.append(f"{provider}: {e}")
            continue

    raise RuntimeError(
        f"All providers failed.\n" + "\n".join(errors)
    )


def stream_with_fallback(
    messages:    list,
    model:       str,
    temperature: float,
    max_tokens:  int,
):
    """
    Streaming version — try each provider in chain.
    Yields (delta, provider) tuples.

    If a provider fails BEFORE yielding anything → try next.
    If a provider fails MID-STREAM → partial output already sent to client,
    cannot retry. Log the error and stop.
    """
    errors = []

    for provider in _PROVIDER_CHAIN:
        try:
            first_chunk = True
            for delta in _stream_provider(provider, messages, model, temperature, max_tokens):
                if first_chunk:
                    if provider != "groq":
                        print(f"[fallback_llm] Streaming via {provider} (Groq unavailable)")
                    first_chunk = False
                yield delta, provider
            return  # stream completed successfully

        except Exception as e:
            reason = "rate limit" if _is_rate_limit_error(e) else "error"
            print(f"[fallback_llm] {provider} stream {reason}: {e} — trying next")
            errors.append(f"{provider}: {e}")
            continue

    raise RuntimeError(
        f"All providers failed for streaming.\n" + "\n".join(errors)
    )