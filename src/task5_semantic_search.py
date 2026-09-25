"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

try:
    from .task4_chunking_indexing import embed_texts, get_collection
except ImportError:
    from task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    query_vector = embed_texts([query])[0]
    collection = get_collection()
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    if response and response.get("ids") and response["ids"] and response["ids"][0]:
        for item_id, content, raw_meta, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            meta = dict(raw_meta) if raw_meta else {}
            if not meta.get("url"):
                meta["url"] = None
            if "chunk_index" in meta and not isinstance(meta["chunk_index"], int):
                meta["chunk_index"] = int(meta["chunk_index"])

            results.append({
                "id": item_id,
                "content": content,
                "score": max(0.0, 1.0 - float(distance)),
                "metadata": meta,
                "retrieval_method": "dense",
            })

    # Sort descending by score, deduplicate by ID, cap at top_k
    seen_ids = set()
    unique_results = []
    for item in sorted(results, key=lambda x: x["score"], reverse=True):
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_results.append(item)
        if len(unique_results) >= top_k:
            break

    return unique_results


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    query = "chinh sach van chuyen"
    results = semantic_search(query, top_k=3)
    print(f"Found {len(results)} results for query: '{query}'")
    for r in results:
        print(f"- ID: {r['id']} | Score: {r['score']:.4f} | Source: {r['metadata'].get('source')}")


