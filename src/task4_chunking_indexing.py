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
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()
    if provider == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = ("st", SentenceTransformer(EMBEDDING_MODEL))
    elif provider == "openai":
        from openai import OpenAI
        _EMBEDDER = ("openai", OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    elif provider == "gemini":
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        _EMBEDDER = ("gemini", genai)
    else:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = ("st", SentenceTransformer(EMBEDDING_MODEL))
    return _EMBEDDER


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo vector embedding cho danh sách văn bản theo provider trong .env."""
    if not texts:
        return []
    provider_type, client = _get_embedder()
    if provider_type == "st":
        embeddings = client.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    elif provider_type == "openai":
        model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        response = client.embeddings.create(input=texts, model=model_name)
        return [item.embedding for item in response.data]
    elif provider_type == "gemini":
        model_name = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        results = []
        for text in texts:
            res = client.embed_content(model=model_name, content=text)
            results.append(res["embedding"])
        return results
    raise ValueError(f"Unsupported provider: {provider_type}")


def get_collection():
    """Mở hoặc tạo Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown trong data/standardized/ và trả về danh sách Document."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        title = path.stem
        url = None

        # Trích xuất title và url từ header nếu có
        for line in content.splitlines()[:6]:
            if line.startswith("# ") and title == path.stem:
                title = line[2:].strip()
            elif line.startswith("**Source:**"):
                raw_url = line.replace("**Source:**", "").strip()
                if raw_url:
                    url = raw_url

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
    """Chia Document thành chunks có id và chunk_index theo contract."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        splits = [s.strip() for s in splitter.split_text(document["content"]) if s.strip()]
        for index, text in enumerate(splits):
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": text,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": index,
                },
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk theo batch."""
    if not chunks:
        return []
    batch_size = 32
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["content"] for c in batch]
        vectors = embed_texts(texts)
        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
        )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")
    
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    
    print("Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)
    
    print("Indexing to ChromaDB...")
    index_to_vectorstore(embedded_chunks)
    print(f"[OK] Indexed {len(embedded_chunks)} chunks into ChromaDB ({COLLECTION_NAME})")


if __name__ == "__main__":
    run_pipeline()
