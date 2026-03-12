import os
from groq import Groq
from dotenv import load_dotenv
from core.embedder import embed_query

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_hyde_embedding(question: str) -> list:
    """
    HyDE: Hypothetical Document Embedding.
    
    Instead of embedding the question directly, we:
    1. Ask the LLM to generate a hypothetical answer
    2. Embed that hypothetical answer
    3. Use that embedding for retrieval
    
    Why this works: A hypothetical answer is semantically closer
    to actual document chunks than the question itself.
    Only triggered when retrieval confidence is low (< 0.75).
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Generate a detailed hypothetical passage that would answer the following question. Write it as if it were extracted from a document. Do not say 'hypothetically' — just write the passage directly."
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nHypothetical passage:"
                }
            ],
            temperature=0.5,
            max_tokens=300
        )

        hypothetical_answer = response.choices[0].message.content.strip()
        print(f"[hyde] Generated hypothetical answer: {hypothetical_answer[:100]}...")

        # Embed the hypothetical answer instead of the question
        hyde_embedding = embed_query(hypothetical_answer)
        return hyde_embedding

    except Exception as e:
        print(f"[hyde] Failed, falling back to question embedding: {e}")
        return embed_query(question)