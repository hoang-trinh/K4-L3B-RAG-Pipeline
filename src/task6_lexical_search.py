"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


import math
import numpy as np
from rank_bm25 import BM25Okapi
from .task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []


class PatchedBM25Okapi(BM25Okapi):
    """BM25Okapi với Lucene-style non-negative IDF để tránh idf <= 0 trên corpus nhỏ."""
    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))


def get_corpus() -> list[dict]:
    """Lấy hoặc load corpus chunks."""
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [item["content"].lower().split() for item in corpus]
    return PatchedBM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    import src.task6_lexical_search as lexical_mod
    corpus = lexical_mod.CORPUS if lexical_mod.CORPUS else get_corpus()
    if not corpus:
        return []

    tokens = query.lower().split()
    if not tokens:
        return []

    bm25 = build_bm25_index(corpus)
    scores = bm25.get_scores(tokens)
    indices = np.argsort(scores)[::-1]

    results = []
    seen_ids = set()
    for index in indices:
        if scores[index] <= 0:
            continue
        item = corpus[index]
        if item["id"] in seen_ids:
            continue
        seen_ids.add(item["id"])

        meta = dict(item["metadata"])
        if not meta.get("url"):
            meta["url"] = None

        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": meta,
            "retrieval_method": "bm25",
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
