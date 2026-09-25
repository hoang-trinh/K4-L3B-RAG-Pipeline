"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path
import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

SOURCES: dict[str, str] = {
    # Thêm URL dự phòng nếu cần tải tự động
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Kiểm tra tài liệu đã có hoặc tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    existing_files = [
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".doc"] and not f.name.startswith(".")
    ]

    if len(existing_files) >= 3:
        print(f"[OK] Da co {len(existing_files)} tai lieu hop le trong {DATA_DIR}:")
        for f in existing_files:
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        return

    for filename, url in SOURCES.items():
        dest = DATA_DIR / filename
        if not dest.exists():
            print(f"Downloading {filename} from {url}...")
            response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            dest.write_bytes(response.content)
            print(f"Saved: {filename}")


if __name__ == "__main__":
    setup_directory()
    download_documents()

