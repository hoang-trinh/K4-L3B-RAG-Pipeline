"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_FILE = Path(__file__).parent.parent / "data" / "pageindex_cache.json"
TEMP_PDF_DIR = Path(__file__).parent.parent / "data" / ".temp_pdf"

logger = logging.getLogger(__name__)


def load_cache() -> dict:
    """Đọc cache mapping tài liệu -> document ID từ file JSON."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Không thể đọc cache file %s: %s", CACHE_FILE, e)
    return {}


def save_cache(cache_data: dict) -> None:
    """Lưu cache mapping vào file JSON."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_markdown_to_pdf(content: str, output_path: Path) -> Path:
    """Chuyển đổi nội dung Markdown sang file PDF bằng fpdf2 để upload lên PageIndex."""
    from fpdf import FPDF

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.add_page()

    # Làm sạch ký tự đặc biệt / BOM trước khi render PDF
    clean_content = content.replace("\ufeff", "").replace("\u200b", "").replace("\r\n", "\n")

    font_path = "C:/Windows/Fonts/arial.ttf"
    has_unicode_font = False
    if os.path.exists(font_path):
        try:
            pdf.add_font("ArialUnicode", "", font_path)
            pdf.set_font("ArialUnicode", size=10)
            has_unicode_font = True
        except Exception:
            pdf.set_font("Helvetica", size=10)
    else:
        pdf.set_font("Helvetica", size=10)

    for line in clean_content.splitlines():
        line_clean = line.strip()
        if not line_clean:
            pdf.ln(4)
            continue
        try:
            if not has_unicode_font:
                line_clean = line_clean.encode("ascii", "replace").decode("ascii")
            pdf.multi_cell(w=pdf.epw, h=5, text=line_clean)
        except Exception:
            safe_text = line_clean.encode("ascii", "replace").decode("ascii")
            try:
                pdf.multi_cell(w=pdf.epw, h=5, text=safe_text)
            except Exception:
                pass

    pdf.output(str(output_path))
    return output_path


def extract_metadata(md_path: Path, raw_content: str) -> dict:
    """Trích xuất metadata (title, url, doc_type, source) từ Markdown file."""
    doc_type = "legal" if "legal" in md_path.parts else "news"
    title = md_path.stem
    url = None

    for line in raw_content.splitlines()[:10]:
        trimmed = line.strip()
        if trimmed.startswith("# ") and title == md_path.stem:
            extracted_title = trimmed[2:].strip()
            if extracted_title:
                title = extracted_title
        elif trimmed.startswith("**Source:**"):
            url_val = trimmed.split("**Source:**", 1)[1].strip()
            if url_val:
                url = url_val

    return {
        "source": md_path.name,
        "title": title,
        "doc_type": doc_type,
        "url": url,
    }


def upload_documents() -> None:
    """Upload tài liệu lên PageIndex và lưu document IDs để tái sử dụng."""
    if not PAGEINDEX_API_KEY:
        print("[PageIndex] PAGEINDEX_API_KEY chưa được thiết lập trong .env. Bỏ qua upload.")
        return

    try:
        from pageindex import PageIndexClient
    except ImportError:
        logger.error("Chưa cài đặt package pageindex. Hãy chạy `pip install pageindex`.")
        return

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    cache = load_cache()
    docs_cache = cache.setdefault("documents", {})

    uploaded_count = 0
    for md_path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_path.name.startswith("."):
            continue

        rel_key = md_path.relative_to(STANDARDIZED_DIR).as_posix()
        if rel_key in docs_cache and docs_cache[rel_key].get("doc_id"):
            continue

        raw_content = md_path.read_text(encoding="utf-8")
        meta = extract_metadata(md_path, raw_content)

        # Ưu tiên sử dụng file PDF gốc trong landing/legal nếu có
        candidate_pdf = LANDING_LEGAL_DIR / f"{md_path.stem}.pdf"
        if candidate_pdf.exists():
            pdf_path = candidate_pdf
        else:
            temp_pdf = TEMP_PDF_DIR / f"{md_path.stem}.pdf"
            pdf_path = convert_markdown_to_pdf(raw_content, temp_pdf)

        try:
            print(f"[PageIndex] Uploading {md_path.name}...")
            response = client.submit_document(file_path=str(pdf_path))
            doc_id = response.get("doc_id")
            if doc_id:
                docs_cache[rel_key] = {
                    "doc_id": doc_id,
                    "source": meta["source"],
                    "title": meta["title"],
                    "doc_type": meta["doc_type"],
                    "url": meta["url"],
                }
                save_cache(cache)
                uploaded_count += 1
                print(f"[PageIndex] Đã upload thành công {md_path.name} -> doc_id: {doc_id}")
        except Exception as e:
            logger.warning("[PageIndex] Lỗi khi upload %s: %s", md_path.name, e)

    print(f"[PageIndex] Hoàn tất upload. Đã upload mới {uploaded_count} tài liệu.")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Truy vấn PageIndex và trả về danh sách SearchResult theo chuẩn."""
    if not PAGEINDEX_API_KEY:
        return []

    cache = load_cache()
    docs_cache = cache.get("documents", {})
    if not docs_cache:
        return []

    try:
        from pageindex import PageIndexClient
    except ImportError:
        logger.error("Package pageindex chưa được cài đặt.")
        return []

    try:
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        all_results: list[dict] = []

        for rel_key, doc_info in docs_cache.items():
            doc_id = doc_info.get("doc_id")
            if not doc_id:
                continue

            try:
                # Gửi truy vấn retrieval
                resp = client.submit_query(doc_id=doc_id, query=query)
                retrieval_id = resp.get("retrieval_id")
                if not retrieval_id:
                    continue

                # Poll kết quả retrieval với timeout giới hạn (tối đa 5s cho mỗi document)
                start_time = time.time()
                retrieval_res = None
                while time.time() - start_time < 5.0:
                    retrieval_res = client.get_retrieval(retrieval_id)
                    status = retrieval_res.get("status", "").lower()
                    if status in ("completed", "ready", "done", "success"):
                        break
                    elif status in ("failed", "error"):
                        retrieval_res = None
                        break
                    time.sleep(0.5)

                if not retrieval_res:
                    continue

                # Trích xuất retrieved nodes từ response PageIndex
                nodes = (
                    retrieval_res.get("retrieved_nodes")
                    or retrieval_res.get("retrieval_nodes")
                    or retrieval_res.get("nodes")
                    or retrieval_res.get("results")
                    or []
                )

                for idx, node in enumerate(nodes):
                    content = ""
                    if isinstance(node, str):
                        content = node.strip()
                    elif isinstance(node, dict):
                        content = (
                            node.get("text")
                            or node.get("content")
                            or node.get("summary")
                            or ""
                        ).strip()

                    if not content:
                        continue

                    raw_score = node.get("score") if isinstance(node, dict) else None
                    if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                        score = float(raw_score)
                    else:
                        # Gán score giảm dần theo thứ tự nếu API không trả numeric score
                        score = max(0.1, 1.0 - (len(all_results) * 0.05))

                    chunk_idx = idx
                    if isinstance(node, dict) and "page_index" in node:
                        try:
                            chunk_idx = max(0, int(node["page_index"]))
                        except (ValueError, TypeError):
                            chunk_idx = idx

                    node_id = (
                        node.get("node_id", f"node-{idx}")
                        if isinstance(node, dict)
                        else f"node-{idx}"
                    )
                    item_id = f"pageindex-{doc_id}-{node_id}"

                    all_results.append({
                        "id": item_id,
                        "content": content,
                        "score": score,
                        "metadata": {
                            "source": doc_info.get("source", "unknown.md"),
                            "title": doc_info.get("title", "Tài liệu"),
                            "doc_type": doc_info.get("doc_type", "legal"),
                            "url": doc_info.get("url"),
                            "chunk_index": chunk_idx,
                        },
                        "retrieval_method": "pageindex",
                    })
            except Exception as e:
                logger.warning("[PageIndex] Lỗi khi query doc %s: %s", doc_id, e)
                continue

        # Sắp xếp giảm dần theo score và lọc trùng lặp ID
        seen_ids = set()
        unique_results = []
        for item in sorted(all_results, key=lambda x: x["score"], reverse=True):
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                unique_results.append(item)
            if len(unique_results) >= top_k:
                break

        return unique_results

    except Exception as e:
        logger.error("[PageIndex] Lỗi trong quá trình tìm kiếm pageindex: %s", e)
        return []


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    upload_documents()
    test_query = "quy định hoàn tiền"
    res = pageindex_search(test_query, top_k=3)
    print(f"PageIndex Search kết quả cho '{test_query}': {len(res)} items")
    for r in res:
        print(f"- ID: {r['id']} | Score: {r['score']:.4f} | Source: {r['metadata']['source']}")
