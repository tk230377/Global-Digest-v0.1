#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Scraper Agent (Production, robust decode)

- 入力: Validatorを通過した「最終記事URL」の List[str]
- 出力: 入力と同じ長さ・順序の List[str]（title + "\\n\\n" + body）。失敗は ""。
- return_meta=True の場合は、各URLごとの詳細メタ(dict)を返す:
    {
      "input_url", "resolved_url", "status_code", "content_type",
      "extractor", "used_amp", "ok", "reason",
      "title", "body", "text", "chars", "elapsed_ms"
    }

特徴:
- 抽出器フォールバック: trafilatura -> readability -> BeautifulSoup
- AMPは通常抽出が失敗した時のみ試行（オプション）
- HTTPは Session + Retry + 接続/読み取りタイムアウト / Accept-Language / User-Agent
- bytesからの自前デコード + HTMLエンティティ解除 + Unicode正規化 + 不可視文字除去
- 入力順・長さを維持、例外は安全に "" 扱い（metaでは理由を記録）

依存:
  pip install requests beautifulsoup4 python-dotenv
  pip install charset-normalizer
  （精度向上オプション）
  pip install trafilatura
  pip install readability-lxml lxml
"""

from __future__ import annotations
import os
import re
import time
import logging
from typing import List, Optional, Tuple, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# 追加: 堅牢デコード用
from charset_normalizer import from_bytes
import unicodedata, html as ihtml

# Optional deps
_HAS_TRAFILATURA = False
_HAS_READABILITY = False
try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:
    pass

try:
    from readability import Document  # type: ignore
    _HAS_READABILITY = True
except Exception:
    pass

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ========= 設定 =========
LOG_LEVEL = os.getenv("SCRAPER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] scraper: %(message)s"
)
logger = logging.getLogger("scraper")

TIMEOUT_CONNECT = float(os.getenv("SCRAPER_TIMEOUT_CONNECT_SEC", "5.0"))
TIMEOUT_READ    = float(os.getenv("SCRAPER_TIMEOUT_READ_SEC", "15.0"))
CONCURRENCY     = int(os.getenv("SCRAPER_CONCURRENCY", "8"))
MIN_CHARS       = int(os.getenv("SCRAPER_MIN_TEXT_CHARS", "0"))   # 既存設定を踏襲
MAX_CHARS       = int(os.getenv("SCRAPER_MAX_TEXT_CHARS", "12000"))
FOLLOW_AMP      = os.getenv("SCRAPER_FOLLOW_AMP", "true").strip().lower() in ("1", "true", "yes", "y")

# UAはASCII固定（latin-1関係の例外を避けるため）
USER_AGENT = os.getenv(
    "SCRAPER_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) GlobalDigest/Scraper"
)

ACCEPT_LANGUAGE = os.getenv(
    "SCRAPER_ACCEPT_LANGUAGE",
    "zh-CN,zh;q=0.9,ja;q=0.8,en;q=0.7"
)

RETRY_TOTAL       = int(os.getenv("SCRAPER_RETRY_TOTAL", "2"))
RETRY_BACKOFF     = float(os.getenv("SCRAPER_RETRY_BACKOFF_SEC", "0.5"))
RETRY_STATUS_LIST = tuple(int(x) for x in os.getenv("SCRAPER_RETRY_STATUS", "429,500,502,503,504").split(","))

# ========= HTTPセッション（スレッド毎に） =========
def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=RETRY_STATUS_LIST,
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": ACCEPT_LANGUAGE,
    })
    return s

# ========= ユーティリティ =========
def _decode_html_bytes(content: bytes, headers_ct: str) -> str:
    """HTTPヘッダ/メタ/自動推定の順で堅牢にデコードし、整形して返す。"""
    # 1) HTTPヘッダのcharset
    m = re.search(r"charset=([A-Za-z0-9_\-]+)", headers_ct or "", re.I)
    if m:
        enc = m.group(1).strip().lower()
        try:
            html = content.decode(enc, errors="replace")
            return _normalize_html(html)
        except Exception:
            pass
    # 2) <meta charset=...> をbytesから探索
    head = content[:4096].decode("ascii", errors="ignore").lower()
    m2 = re.search(r'<meta[^>]+charset=["\']?([a-z0-9_\-]+)', head)
    if m2:
        enc2 = m2.group(1).strip().lower()
        try:
            html = content.decode(enc2, errors="replace")
            return _normalize_html(html)
        except Exception:
            pass
    # 3) 自動推定（charset-normalizer）
    try:
        best = from_bytes(content).best()
        if best and best.encoding:
            html = best.output()
            return _normalize_html(html)
    except Exception:
        pass
    # 4) 最後の砦
    html = content.decode("utf-8", errors="replace")
    return _normalize_html(html)

_WS_RE = re.compile(r"[ \t\u3000]+")
_NL_RE = re.compile(r"\n{3,}")

def _normalize_html(html: str) -> str:
    """HTMLエンティティ解除 + Unicode正規化 + 不可視文字除去 + 改行整形"""
    html = ihtml.unescape(html)
    html = unicodedata.normalize("NFKC", html)
    html = re.sub(r"[\u200B-\u200F\uFEFF]", "", html)  # ゼロ幅系除去
    # 軽整形（本文抽出器の前処理には影響しない程度）
    html = html.replace("\r", "")
    html = _WS_RE.sub(" ", html)
    html = re.sub(r" +\n", "\n", html)
    html = _NL_RE.sub("\n\n", html)
    return html

def _fetch_html(session: requests.Session, url: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    """HTML文字列, 最終URL, ステータスコード, Content-Type を返す。非HTMLは html=None。"""
    try:
        r = session.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=True)
        status = r.status_code
        ct = (r.headers.get("Content-Type") or "").lower()
        final_url = r.url
        if ("text/html" not in ct) and ("application/xhtml" not in ct):
            return None, final_url, status, ct
        # ★ bytes -> 自前デコード（response.textは使わない）
        html = _decode_html_bytes(r.content, ct)
        return html, final_url, status, ct
    except Exception:
        return None, None, None, None

def _maybe_find_amp(html: str) -> Optional[str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    link = soup.find("link", rel=lambda v: v and "amphtml" in v)
    href = link.get("href") if link else None
    if href and href.startswith(("http://", "https://")):
        return href
    return None

def _clean_text(s: str) -> str:
    s = s.replace("\r", "")
    s = _WS_RE.sub(" ", s)
    s = re.sub(r" +\n", "\n", s)
    s = _NL_RE.sub("\n\n", s).strip()
    return s

def _strip_title_dup(title: str, body: str) -> str:
    head = body[: len(title) + 5]
    if title and title in head:
        body_lines = body.splitlines()
        if body_lines and title.strip() in body_lines[0]:
            body = "\n".join(body_lines[1:]).lstrip()
    return body

# ========= 抽出器 =========
def _extract_with_trafilatura(html: str, url: str) -> Optional[Tuple[str, str]]:
    if not _HAS_TRAFILATURA:
        return None
    try:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        title = None
        try:
            md = trafilatura.metadata.extract_metadata(html, url=url)
            title = md.title if md else None
        except Exception:
            pass
        if text:
            return (title or "").strip(), _clean_text(text)
    except Exception:
        return None
    return None

def _extract_with_readability(html: str, url: str) -> Optional[Tuple[str, str]]:
    if not _HAS_READABILITY:
        return None
    try:
        doc = Document(html)
        title = (doc.short_title() or "").strip()
        article_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(article_html, "lxml")
        for bad in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
            bad.decompose()
        text = _clean_text(soup.get_text("\n"))
        if text:
            return title, text
        return None
    except Exception:
        return None

def _extract_with_bs4(html: str, url: str) -> Optional[Tuple[str, str]]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # タイトル候補（<title> / og:title / twitter:title）
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True) or ""
    if not title:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            title = og["content"].strip()
    if not title:
        tw = soup.find("meta", attrs={"name": "twitter:title"})
        if tw and tw.get("content"):
            title = tw["content"].strip()

    # ノイズ除去
    for bad in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        bad.decompose()

    candidates = []
    for sel in ["article", "main", "[role=main]", "#content", ".content", ".article", ".post"]:
        for node in soup.select(sel):
            txt = _clean_text(node.get_text("\n", strip=False))
            if txt:
                candidates.append(txt)

    if not candidates:
        body = soup.find("body") or soup
        txt = _clean_text(body.get_text("\n", strip=False))
        if txt:
            candidates.append(txt)

    if not candidates:
        return None

    candidates.sort(key=len, reverse=True)
    text = candidates[0]
    return (title.strip(), text) if text else None

def _try_extract(html: str, page_url: str) -> Optional[Tuple[str, str, str]]:
    """抽出に成功したら (title, body, extractor_name) を返す。"""
    extractors = (
        ("trafilatura",  _extract_with_trafilatura),
        ("readability",  _extract_with_readability),
        ("bs4",          _extract_with_bs4),
    )
    for name, extractor in extractors:
        try:
            got = extractor(html, page_url)
        except TypeError:
            got = extractor(html)
        if got:
            title, body = got
            return title, body, name
    return None

# ========= 1URL処理 =========
def _scrape_one(url: str) -> Tuple[str, Dict[str, Any]]:
    t0 = time.time()
    meta: Dict[str, Any] = {
        "input_url": url,
        "resolved_url": None,
        "status_code": None,
        "content_type": None,
        "extractor": None,
        "used_amp": False,
        "ok": False,
        "reason": "init",
        "title": None,
        "body": None,
        "text": None,
        "chars": 0,
        "elapsed_ms": None,
    }

    s = _build_session()
    try:
        html, final_url, status_code, content_type = _fetch_html(s, url)
        meta["resolved_url"] = final_url
        meta["status_code"] = status_code
        meta["content_type"] = content_type

        if not html:
            meta["reason"] = "non_html_or_fetch_error"
            return "", _finish_meta(meta, t0)

        # 通常ページで試す
        got = _try_extract(html, final_url or url)

        # 失敗時のみ AMP 試行
        if not got and FOLLOW_AMP:
            amp_url = _maybe_find_amp(html)
            if amp_url:
                amp_html, amp_final, amp_status, amp_ct = _fetch_html(s, amp_url)
                if amp_html:
                    meta["used_amp"] = True
                    meta["resolved_url"] = amp_final or amp_url
                    meta["status_code"]  = amp_status
                    meta["content_type"] = amp_ct
                    got = _try_extract(amp_html, amp_url)

        if not got:
            meta["reason"] = "no_extractor_matched"
            return "", _finish_meta(meta, t0)

        title, body, extractor_name = got
        body = _clean_text(_strip_title_dup(title, body))
        text = (title + "\n\n" + body).strip() if title else body
        text = _clean_text(text)

        if len(text) < MIN_CHARS:
            meta["title"] = title or ""
            meta["body"] = body or ""
            meta["text"] = text
            meta["chars"] = len(body or "")
            meta["extractor"] = extractor_name
            meta["reason"] = "too_short"
            return "", _finish_meta(meta, t0)

        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS].rstrip()

        meta["title"] = title or ""
        meta["body"] = body or ""
        meta["text"] = text
        meta["chars"] = len(body or "")
        meta["extractor"] = extractor_name
        meta["ok"] = True
        meta["reason"] = "ok"
        return text, _finish_meta(meta, t0)

    except Exception as e:
        meta["reason"] = f"exception:{type(e).__name__}"
        return "", _finish_meta(meta, t0)
    finally:
        try:
            s.close()
        except Exception:
            pass

def _finish_meta(meta: Dict[str, Any], t0: float) -> Dict[str, Any]:
    meta["elapsed_ms"] = round((time.time() - t0) * 1000.0, 2)
    return meta

# ========= 公開関数（既存互換） =========
def run_scraping(valid_urls: List[str], return_meta: bool = False) -> Union[List[str], List[Dict[str, Any]]]:
    """
    仕様:
      - 入力URL順の長さと順序を維持
      - 失敗は ""（return_meta=True の場合は ok=False のメタを含む）
      - return_meta=True のメタには title/body/text/chars/extractor などを格納
    """
    if not valid_urls:
        return []

    results_text: List[str] = [""] * len(valid_urls)
    results_meta: List[Dict[str, Any]] = [{} for _ in range(len(valid_urls))]

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        future_map = {ex.submit(_scrape_one, u): i for i, u in enumerate(valid_urls)}
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                text, meta = fut.result()
            except Exception as e:
                text, meta = "", {"input_url": valid_urls[idx], "ok": False, "reason": f"exception:{type(e).__name__}"}
            results_text[idx] = text
            results_meta[idx] = meta

    return results_meta if return_meta else results_text

# ========= 互換エイリアス（呼び出し側の関数名差異を吸収） =========
def run_scraper(valid_urls: List[str], return_meta: bool = False):
    return run_scraping(valid_urls, return_meta=return_meta)

def run_scrape(valid_urls: List[str], return_meta: bool = False):
    return run_scraping(valid_urls, return_meta=return_meta)

def scrape_urls(valid_urls: List[str], return_meta: bool = False):
    return run_scraping(valid_urls, return_meta=return_meta)

def scrape(valid_urls: List[str], return_meta: bool = False):
    return run_scraping(valid_urls, return_meta=return_meta)
