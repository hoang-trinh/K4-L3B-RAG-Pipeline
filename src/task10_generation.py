"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env (Groq, OpenAI, Gemini, Anthropic).
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

import os
from dotenv import load_dotenv

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (giảm lost-in-the-middle)."""
    if len(chunks) <= 2:
        return [dict(chunk) for chunk in chunks]
    front = chunks[::2]
    back = chunks[1::2]
    reordered = front + back[::-1]
    return [dict(chunk) for chunk in reordered]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "")
        source = metadata.get("source", "")
        content = chunk.get("content", "")
        parts.append(
            f"[Document {index} | Title: {title} | Source: {source}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi Groq, OpenAI, Gemini hoặc Anthropic theo cấu hình LLM_PROVIDER."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER).lower()
    model = os.getenv("LLM_MODEL", LLM_MODEL)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Please set GROQ_API_KEY in your .env file.")
        model_name = model or "openai/gpt-oss-120b"
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content or ""

    elif provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env.")
        client = OpenAI(api_key=api_key)
        model_name = model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    elif provider == "gemini":
        from google import genai
        from google.genai import types
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing in .env.")
        client = genai.Client(api_key=api_key)
        model_name = model or "gemini-2.0-flash"
        response = client.models.generate_content(
            model=model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
        return response.text or ""

    elif provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is missing in .env.")
        client = anthropic.Anthropic(api_key=api_key)
        model_name = model or "claude-3-5-haiku-20241022"
        response = client.messages.create(
            model=model_name,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=TEMPERATURE,
            max_tokens=2048,
        )
        return response.content[0].text

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Supported providers: groq, openai, gemini, anthropic."
        )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult gồm answer, sources và retrieval_source."""
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
    except Exception as exc:
        retrieval_source = chunks[0].get("retrieval_method", "none") if chunks else "none"
        if retrieval_source not in {"hybrid", "pageindex", "none"}:
            retrieval_source = "hybrid"
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": chunks,
            "retrieval_source": retrieval_source,
        }

    retrieval_source = chunks[0].get("retrieval_method", "hybrid")
    if retrieval_source not in {"hybrid", "pageindex", "none"}:
        retrieval_source = "hybrid"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sample_query = "Quảng cáo Shopee là gì?"
    print(f"--- Đặt câu hỏi thử nghiệm: '{sample_query}' ---")
    result = generate_with_citation(sample_query)
    print(f"\n[Câu trả lời từ Groq ({LLM_MODEL or 'openai/gpt-oss-120b'})]:\n{result['answer']}")
    print(f"\n[Số tài liệu trích dẫn]: {len(result['sources'])} (Phương thức: {result['retrieval_source']})")
