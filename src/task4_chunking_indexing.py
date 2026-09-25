"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_embedding_model = None


def get_embedding_model():
    """Lazy load sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Dispatch theo EMBEDDING_PROVIDER trong .env."""
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()

    if provider == "groq":
        raise ValueError(
            "Groq API hiện chưa cung cấp endpoint embeddings. "
            "Để chạy cloud không tốn phí, hãy đặt EMBEDDING_PROVIDER=gemini (dùng text-embedding-004 miễn phí) "
            "hoặc EMBEDDING_PROVIDER=openai hoặc EMBEDDING_PROVIDER=sentence_transformers (chạy local)."
        )

    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment or .env file.")
        client = OpenAI(api_key=api_key)
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        if "bge" in model.lower() or not model:
            model = "text-embedding-3-small"
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    elif provider == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
        client = genai.Client(api_key=api_key)
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
        if "bge" in model.lower() or not model:
            model = "text-embedding-004"

        try:
            res = client.models.embed_content(model=model, contents=texts)
            return [e.values for e in res.embeddings]
        except Exception:
            embeddings = []
            for text in texts:
                res = client.models.embed_content(model=model, contents=text)
                embeddings.append(res.embeddings[0].values)
            return embeddings
    else:
        # Default: sentence_transformers
        model = get_embedding_model()
        vectors = model.encode(texts, show_progress_bar=False)
        return vectors.tolist()


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    class CustomEmbeddingFunction(chromadb.EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input: list[str]) -> list[list[float]]:
            return embed_texts(input)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=CustomEmbeddingFunction(),
    )


def preprocess_text(content: str, doc_type: str = "legal") -> str:
    """Xử lý và làm sạch dữ liệu văn bản trước khi chunking:
    1. Loại bỏ các ký tự ngắt trang (form-feed \x0c), zero-width space, null bytes.
    2. Loại bỏ rác crawl (thẻ ảnh Markdown rỗng, menu điều hướng lặp).
    3. Chuẩn hóa các gạch đầu dòng (bullet points) và khoảng cách dòng thừa.
    """
    if not content:
        return ""

    import re

    # 1. Loại bỏ ký tự đặc biệt / form-feed / zero-width space
    text = content.replace("\x0c", "\n").replace("\ufeff", "").replace("\u200b", "").replace("\r\n", "\n")

    # 2. Loại bỏ thẻ ảnh markdown: ![](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # 3. Chuẩn hóa các biểu tượng gạch đầu dòng
    text = re.sub(r"^[ \t]*[●■◆▪★]\s*", "- ", text, flags=re.MULTILINE)

    # 4. Loại bỏ các dòng điều hướng rác từ crawler đối với news
    if doc_type == "news":
        clean_lines = []
        for line in text.splitlines():
            s = line.strip()
            # Bỏ qua các menu điều hướng không có giá trị nội dung
            if s in ("Trang Chủ", "Loại", "Tìm Hiểu", "Tin Mới", "Tạo Chiến dịch", "Sự kiện", "Khoá học", "Mới"):
                continue
            clean_lines.append(line)
        text = "\n".join(clean_lines)

    # 5. Chuẩn hóa khoảng trắng và dòng trống liên tiếp (tối đa 2 dòng ngắt đoạn)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_documents() -> list[dict]:
    """Đọc Markdown, tiền xử lý làm sạch văn bản và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        raw_content = path.read_text(encoding="utf-8")
        
        # Tiền xử lý làm sạch nội dung trước khi chunking
        content = preprocess_text(raw_content, doc_type=doc_type)
        if not content:
            continue

        title = path.stem
        url = None

        # Trích xuất title và source url nếu có trong Header
        for line in raw_content.splitlines()[:10]:
            trimmed = line.strip()
            if trimmed.startswith("# ") and title == path.stem:
                extracted_title = trimmed[2:].strip()
                if extracted_title:
                    title = extracted_title
            elif trimmed.startswith("**Source:**"):
                url_val = trimmed.split("**Source:**", 1)[1].strip()
                if url_val:
                    url = url_val

        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        splits = splitter.split_text(document["content"])
        for index, text in enumerate(splits):
            chunk_metadata = {**document["metadata"], "chunk_index": index}
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": chunk_metadata,
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    collection = get_collection()
    batch_size = 200
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        metadatas = []
        for chunk in batch:
            m = dict(chunk["metadata"])
            if m.get("url") is None:
                m["url"] = ""
            metadatas.append(m)

        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=metadatas,
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print("Embedding completed")
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks to ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    run_pipeline()
