"""Phase 2 — Scraping the documents.

Fetches **exactly the five Tier A registry URLs** and nothing else: no
anchors are followed, no "related funds", no internal links, no sitemaps
(depth 1, five GETs). One request per page, polite delay between pages,
robots.txt honored, redirects off the allowlist rejected.

Outputs:
- artifacts/01_raw/{slug}.html          raw HTML snapshot (audit trail)
- artifacts/01_raw/{slug}.meta.json     source_url, status, fetched_at, ...
- artifacts/01_raw/fetch_report.json    per-page summary, failures named
"""

from __future__ import annotations

import logging
import re
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from groww_faq.config import AppConfig
from groww_faq.models import RegistryEntry
from groww_faq.registry import TIER_A
from groww_faq.utils import sha256_text, utc_now_iso, write_json

logger = logging.getLogger(__name__)

# Below this much visible text the saved HTML is a JS shell, not content.
MIN_VISIBLE_TEXT_CHARS = 500

_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style)[^>]*>.*?</\1>")
_TAG_RE = re.compile(r"<[^>]+>")


def visible_text_length(html: str) -> int:
    """Rough visible-text length: script/style bodies and tags removed."""
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    return len(text.strip())


@dataclass
class FetchOutcome:
    slug: str
    url: str
    ok: bool
    status: int | None = None
    final_url: str | None = None
    reason: str = ""
    needs_browser: bool = False
    html_path: str | None = None
    meta_path: str | None = None
    bytes_saved: int = 0


class Fetcher:
    """HTTP GET of the five allowlisted URLs, politely, once each."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg = config.ingestion
        self.allowlist = {e.url for e in TIER_A}
        self._robots: urllib.robotparser.RobotFileParser | None = None
        self._robots_loaded = False

    # ------------------------------------------------------------------ robots
    def _load_robots(self, client: httpx.Client) -> urllib.robotparser.RobotFileParser | None:
        if self._robots_loaded:
            return self._robots
        self._robots_loaded = True
        if not self.cfg.respect_robots:
            return None
        first = urlparse(TIER_A[0].url)
        robots_url = f"{first.scheme}://{first.netloc}/robots.txt"
        try:
            resp = client.get(robots_url)
            if resp.status_code == 404:
                self._robots = None  # no robots.txt -> nothing is disallowed
            elif resp.status_code >= 400:
                logger.warning("robots.txt fetch failed (%s); proceeding", resp.status_code)
                self._robots = None
            else:
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(resp.text.splitlines())
                self._robots = rp
        except httpx.HTTPError as exc:
            logger.warning("robots.txt fetch error (%s); proceeding", exc)
            self._robots = None
        return self._robots

    def _robots_allows(self, client: httpx.Client, url: str) -> tuple[bool, str]:
        if not self.cfg.respect_robots:
            return True, ""
        rp = self._load_robots(client)
        if rp is None:
            return True, ""
        if rp.can_fetch(self.cfg.user_agent, url):
            return True, ""
        return False, "robots_disallowed"

    # ---------------------------------------------------------------- fetching
    def fetch_all(self) -> list[FetchOutcome]:
        outcomes: list[FetchOutcome] = []
        with httpx.Client(
            headers={"User-Agent": self.cfg.user_agent},
            timeout=httpx.Timeout(self.cfg.timeout_s),
            follow_redirects=True,
        ) as client:
            for i, entry in enumerate(TIER_A):
                if i > 0:
                    time.sleep(self.cfg.delay_s)
                outcomes.append(self.fetch_one(client, entry))
        self._write_report(outcomes)
        return outcomes

    def fetch_one(self, client: httpx.Client, entry: RegistryEntry) -> FetchOutcome:
        url = entry.url

        allowed, reason = self._robots_allows(client, url)
        if not allowed:
            logger.error("[%s] blocked by robots.txt — nothing saved", entry.slug)
            return FetchOutcome(slug=entry.slug, url=url, ok=False, reason=reason)

        attempts = self.cfg.retries + 1
        resp: httpx.Response | None = None
        last_error = "unknown"
        for attempt in range(1, attempts + 1):
            try:
                resp = client.get(url)
                if resp.status_code < 400:
                    break
                last_error = f"http_{resp.status_code}"
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < attempts:
                        time.sleep(min(2**attempt, 8))
                        continue
                break
            except httpx.HTTPError as exc:
                resp = None
                last_error = f"network_error:{type(exc).__name__}"
                if attempt < attempts:
                    time.sleep(min(2**attempt, 8))
                    continue
                break

        if resp is None or resp.status_code >= 400:
            logger.error("[%s] fetch failed (%s)", entry.slug, last_error)
            return FetchOutcome(
                slug=entry.slug, url=url, ok=False,
                status=None if resp is None else resp.status_code, reason=last_error,
            )

        final_url = str(resp.url)
        if final_url not in self.allowlist:
            logger.error(
                "[%s] redirected off the allowlist to %s — nothing saved",
                entry.slug, final_url,
            )
            return FetchOutcome(
                slug=entry.slug, url=url, ok=False, status=resp.status_code,
                final_url=final_url, reason="redirect_off_allowlist",
            )

        html = resp.text
        needs_browser = visible_text_length(html) < MIN_VISIBLE_TEXT_CHARS
        if needs_browser:
            logger.warning(
                "[%s] almost no visible text — marked needs_browser "
                "(Playwright fallback: fetch_with_playwright)", entry.slug,
            )
        return self._save(entry, html, resp.status_code, final_url, needs_browser)

    # ------------------------------------------------------------------ saving
    def _save(
        self,
        entry: RegistryEntry,
        html: str,
        status: int,
        final_url: str,
        needs_browser: bool,
    ) -> FetchOutcome:
        raw_dir = self.config.paths.raw
        raw_dir.mkdir(parents=True, exist_ok=True)
        html_path = raw_dir / f"{entry.slug}.html"
        html_path.write_text(html, encoding="utf-8")
        meta = {
            "slug": entry.slug,
            "source_url": entry.url,
            "final_url": final_url,
            "status": status,
            "fetched_at": utc_now_iso(),
            "sha256": sha256_text(html),
            "bytes": len(html.encode("utf-8")),
            "needs_browser": needs_browser,
        }
        meta_path = raw_dir / f"{entry.slug}.meta.json"
        write_json(meta_path, meta)
        logger.info("[%s] saved %s bytes", entry.slug, meta["bytes"])
        return FetchOutcome(
            slug=entry.slug, url=entry.url, ok=True, status=status,
            final_url=final_url, needs_browser=needs_browser,
            html_path=str(html_path), meta_path=str(meta_path),
            bytes_saved=meta["bytes"],
        )

    def _write_report(self, outcomes: list[FetchOutcome]) -> None:
        report = {
            "fetched_at": utc_now_iso(),
            "user_agent": self.cfg.user_agent,
            "pages": [
                {
                    "slug": o.slug,
                    "url": o.url,
                    "ok": o.ok,
                    "status": o.status,
                    "final_url": o.final_url,
                    "reason": o.reason,
                    "needs_browser": o.needs_browser,
                    "bytes_saved": o.bytes_saved,
                }
                for o in outcomes
            ],
            "ok_count": sum(1 for o in outcomes if o.ok),
            "failures": [
                {"slug": o.slug, "reason": o.reason} for o in outcomes if not o.ok
            ],
        }
        write_json(self.config.paths.raw / "fetch_report.json", report)


def fetch_with_playwright(url: str, user_agent: str, timeout_s: float) -> str:
    """Rendered fetch of the *same* URL — the documented fallback for pages
    flagged ``needs_browser``. Still no link following: goto, read content.

    Requires ``playwright`` (pip install playwright && playwright install chromium).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "playwright is not installed. To use the browser fallback run: "
            "pip install playwright && playwright install chromium"
        ) from exc
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(user_agent=user_agent)
            page.goto(url, timeout=timeout_s * 1000, wait_until="networkidle")
            return page.content()
        finally:
            browser.close()
