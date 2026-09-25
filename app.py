import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Đảm bảo cả project root và src/ đều nằm trong sys.path để tương thích mọi linter và runner
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
for path_str in (str(ROOT_DIR), str(SRC_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from task10_generation import generate_with_citation
except ImportError:
    from src.task10_generation import generate_with_citation

load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot (Groq API)",
    page_icon="🤖",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

llm_provider = os.getenv("LLM_PROVIDER", "groq")
llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
emb_provider = os.getenv("EMBEDDING_PROVIDER", "gemini")

with st.sidebar:
    st.title("⚙️ Cấu hình RAG")
    st.info(f"**LLM Provider:** `{llm_provider.upper()}`\n\n**LLM Model:** `{llm_model}`\n\n**Embedding:** `{emb_provider}`")
    top_k = st.slider("Số lượng chunks tham khảo (top_k)", 3, 10, 5)
    
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

st.title("💬 RAG Chatbot with Citation")
st.caption(f"Hệ thống hỏi đáp tài liệu sử dụng **Groq API Cloud ({llm_model})** siêu tốc.")

def render_sources(sources: list[dict], retrieval_source: str = ""):
    if not sources:
        return
    with st.expander(f"📚 Xem {len(sources)} nguồn tài liệu tham khảo (Phương pháp: {retrieval_source or 'N/A'})"):
        for idx, chunk in enumerate(sources, 1):
            meta = chunk.get("metadata", {})
            title = meta.get("title", f"Tài liệu {idx}")
            source = meta.get("source", "N/A")
            url = meta.get("url")
            score = chunk.get("score", 0.0)
            method = chunk.get("retrieval_method", "N/A")

            st.markdown(f"**[{idx}] {title}**")
            st.caption(f"📁 Tệp: `{source}` | 🔍 Phương thức: `{method}` | ⭐ Điểm: `{score:.4f}`")
            if url:
                st.markdown(f"🔗 [Liên kết nguồn]({url})")
            content_preview = chunk.get("content", "").strip()
            if len(content_preview) > 350:
                content_preview = content_preview[:350] + "..."
            st.text(content_preview)
            st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"], message.get("retrieval_source", ""))

query = st.chat_input("Nhập câu hỏi của bạn về tài liệu...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm thông tin và tạo câu trả lời với Groq..."):
            result = generate_with_citation(query, top_k=top_k)
            answer = result.get("answer", "Không nhận được câu trả lời.")
            sources = result.get("sources", [])
            retrieval_source = result.get("retrieval_source", "none")

            st.markdown(answer)
            render_sources(sources, retrieval_source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
