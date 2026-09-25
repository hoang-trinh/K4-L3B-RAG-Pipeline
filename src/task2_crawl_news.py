"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://ads.shopee.vn/news/557",
    "https://ads.shopee.vn/news/1447",
    "https://ads.shopee.vn/news/1293",
    "https://ads.shopee.vn/news/1225",
    "https://ads.shopee.vn/news/713",
    "https://ads.shopee.vn/news/353"
]


async def crawl_article(url: str) -> dict:
    from datetime import datetime
    import html
    import httpx
    from bs4 import BeautifulSoup

    # Try Crawl4AI first
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success and result.markdown and len(result.markdown.strip()) >= 200:
                title = result.metadata.get("title", "")
                if " | " in title:
                    title = title.split(" | ")[0].strip()
                if not title:
                    title = "Shopee Ads Article"
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown.strip(),
                }
    except Exception as e:
        print(f"Crawl4AI notice for {url}: {e}, falling back to direct extraction...")

    # Fallback: direct HTTP request + BeautifulSoup & JSON-LD / HTML parsing
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        if " | " in title:
            title = title.split(" | ")[0].strip()
        if not title:
            title = "Shopee Ads Article"

        # Extract content: check JSON-LD Schema first (Shopee Ads stores clean articleBody)
        content_markdown = ""
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                ld = json.loads(script.string)
                if isinstance(ld, dict) and "articleBody" in ld:
                    body_data = ld["articleBody"]
                    if isinstance(body_data, str):
                        try:
                            inner = json.loads(body_data)
                            if isinstance(inner, dict):
                                parts = [str(v) for v in inner.values() if v]
                                content_markdown = "\n\n".join(parts)
                            else:
                                content_markdown = body_data
                        except Exception:
                            content_markdown = body_data
                    if content_markdown and len(content_markdown.strip()) >= 200:
                        break
            except Exception:
                continue

        # If JSON-LD not available or short, extract text from HTML elements
        if not content_markdown or len(content_markdown.strip()) < 200:
            main_el = soup.find("div", id="main-app") or soup.find("article") or soup.find("main") or soup.body
            if main_el:
                for unwanted in main_el(["script", "style", "nav", "header", "footer"]):
                    unwanted.decompose()
                content_markdown = main_el.get_text(separator="\n\n", strip=True)

        content_markdown = html.unescape(content_markdown.strip())
        if len(content_markdown) < 200:
            raise ValueError(f"Content extracted from {url} is too short ({len(content_markdown)} chars)")

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content_markdown,
        }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
