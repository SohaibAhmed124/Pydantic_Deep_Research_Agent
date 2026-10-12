import asyncio
import re
import time
from typing import Any, Dict, List, Optional
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from ddgs.ddgs import DDGS

# Reputable domains for ranking relevance
PRIMARY_FINANCIAL_DOMAINS = [
    "sec.gov",
    "investor.",
    "ir.",
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "seekingalpha.com",
]


def score_source_relevance(url: str, is_financial: bool = False) -> int:
    """Score the credibility/priority of a source URL."""
    score = 0
    url_lower = url.lower()
    if is_financial:
        for domain in PRIMARY_FINANCIAL_DOMAINS:
            if domain in url_lower:
                score += 10
                break
    if any(junk in url_lower for junk in ["pinterest", "quora", "reddit", "youtube"]):
        score -= 5
    return score


def _sync_ddg_text(query: str, max_results: int) -> List[Dict[str, str]]:
    """Synchronous DuckDuckGo search execution."""
    client = DDGS()
    raw = client.text(query, max_results=max_results)
    results = []
    for item in raw:
        results.append({
            "title": str(item.get("title", "")).strip(),
            "href": str(item.get("href", "")).strip(),
            "body": str(item.get("body", "")).strip(),
        })
    return results


async def ddg_search_async(
    query: str,
    max_results: int = 5,
    is_financial: bool = False,
    retries: int = 2,
    telemetry: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """
    Asynchronously search DuckDuckGo with retry backoff, source scoring, and telemetry logging.
    """
    start_t = time.time()
    if telemetry:
        telemetry.log("SEARCH", f"DuckDuckGo Query: '{query}' (limit={max_results})")

    for attempt in range(retries):
        try:
            results = await asyncio.to_thread(_sync_ddg_text, query, max_results)
            if is_financial:
                results.sort(key=lambda r: score_source_relevance(r.get("href", ""), is_financial=True), reverse=True)

            dur_ms = (time.time() - start_t) * 1000.0
            if telemetry:
                telemetry.record_tool(
                    tool_type="DuckDuckGo Search",
                    target=query,
                    status="SUCCESS",
                    duration_ms=dur_ms,
                    details=f"Retrieved {len(results)} results",
                )
                telemetry.log("INFO", f"DuckDuckGo returned {len(results)} results for '{query}' in {dur_ms:.0f}ms")
            return results

        except Exception as e:
            if attempt == retries - 1:
                dur_ms = (time.time() - start_t) * 1000.0
                if telemetry:
                    telemetry.record_tool(
                        tool_type="DuckDuckGo Search",
                        target=query,
                        status="FAILED",
                        duration_ms=dur_ms,
                        details=str(e),
                    )
                    telemetry.log("ERROR", f"DuckDuckGo query '{query}' failed: {e}")
                return []
            await asyncio.sleep(1.0 * (attempt + 1))
    return []


async def fetch_page_markdown_async(
    url: str,
    timeout: float = 6.0,
    max_chars: int = 3500,
    telemetry: Optional[Any] = None,
) -> Optional[str]:
    """
    Fetch a webpage and convert readable HTML to clean markdown with telemetry auditing.
    """
    start_t = time.time()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            dur_ms = (time.time() - start_t) * 1000.0

            if resp.status_code != 200:
                if telemetry:
                    telemetry.record_tool(
                        tool_type="Web Scraper",
                        target=url,
                        status="FAILED",
                        duration_ms=dur_ms,
                        details=f"HTTP {resp.status_code}",
                    )
                return None

            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                if telemetry:
                    telemetry.record_tool(
                        tool_type="Web Scraper",
                        target=url,
                        status="SKIPPED",
                        duration_ms=dur_ms,
                        details=f"Non-HTML content: {content_type}",
                    )
                return None

            html = resp.text
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form", "aside"]):
                tag.decompose()

            main_element = soup.find("main") or soup.find("article") or soup.body
            if not main_element:
                if telemetry:
                    telemetry.record_tool(
                        tool_type="Web Scraper",
                        target=url,
                        status="SKIPPED",
                        duration_ms=dur_ms,
                        details="No body/main content container found",
                    )
                return None

            cleaned_html = str(main_element)
            converted = md(cleaned_html, heading_style="ATX", strip=["img"])
            cleaned_md = re.sub(r"\n{3,}", "\n\n", converted).strip()

            if len(cleaned_md) > max_chars:
                result_text = cleaned_md[:max_chars] + "\n\n...[Content truncated for brevity]..."
            else:
                result_text = cleaned_md if len(cleaned_md) > 100 else None

            if result_text:
                if telemetry:
                    telemetry.record_tool(
                        tool_type="Web Scraper",
                        target=url,
                        status="SUCCESS",
                        duration_ms=dur_ms,
                        details=f"Extracted {len(result_text)} markdown chars",
                    )
                    telemetry.log("SCRAPE", f"Successfully scraped {url[:50]}... ({len(result_text)} chars)")
                return result_text
            else:
                if telemetry:
                    telemetry.record_tool(
                        tool_type="Web Scraper",
                        target=url,
                        status="SNIPPET_FALLBACK",
                        duration_ms=dur_ms,
                        details="Content too short; using snippet",
                    )
                return None

    except Exception as e:
        dur_ms = (time.time() - start_t) * 1000.0
        if telemetry:
            telemetry.record_tool(
                tool_type="Web Scraper",
                target=url,
                status="SNIPPET_FALLBACK",
                duration_ms=dur_ms,
                details=f"Timeout or anti-bot: {e}",
            )
            telemetry.log("INFO", f"Scraper fallback on {url[:45]}... (relying on DDG snippet)")
        return None
