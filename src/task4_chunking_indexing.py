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
        embeddings = []
        for text in texts:
            res = client.models.embed_content(model=model, contents=text)
            embeddings.append(res.embedding.values)
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


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8")
        title = path.stem
        url = None

        # Trích xuất title và source url nếu có trong Header
        for line in content.splitlines()[:10]:
            trimmed = line.strip()
            if trimmed.startswith("# ") and not title:
                title = trimmed[2:].strip()
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
