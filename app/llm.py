"""
LLM generation layer. Supports Groq (default, matches the candidate's prior
stack) and Anthropic as a drop-in alternative, selected via LLM_PROVIDER.
"""
from typing import List, Dict
from app.config import settings

SYSTEM_PROMPT = """
You are ByteVox's AI Document Assistant.

Your job is to answer questions ONLY using the provided document context.

Rules:
1. Use only the retrieved context.
2. Never use outside knowledge.
3. If the answer is not present in the context, respond exactly:
   "I don't have enough information in the indexed documents."
4. Be concise and accurate.
5. Prefer bullet points when listing information.
6. Mention the source filename(s) naturally in your answer.
7. If multiple sources contain relevant information, combine them into one answer.
8. Do not fabricate facts or make assumptions.
"""


def _build_user_prompt(question: str, chunks: List[Dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def generate_answer(question: str, chunks: List[Dict]) -> str:
    prompt = _build_user_prompt(question, chunks)

    if settings.llm_provider == "groq":
        return _generate_groq(prompt)
    elif settings.llm_provider == "anthropic":
        return _generate_anthropic(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def _generate_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content


def _generate_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")
