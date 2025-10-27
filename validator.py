#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Validator Agent (No-GN, Content-aware)

責務:
- フルURL（Discover保証）を対象に「スクレイプに向くURLだけ」を抽出
- HTML系のみ許可（CT不明時は先頭チャンクをスニッフ）
- 簡易コンテンツ判定（本文量/テキスト比率/記事シグナル）
- トラッキング除去 / https化 / fragment削除 / 条件付きAMP解除 / 条件付きcanonical
- 入力順を維持しつつ、最大件数（既定=5）とドメイン上限を適用
- JSONL監査ログ（decision/reason/elapsed_ms/metrics）

依存:
  pip install requests
  (任意) pip install python-dotenv
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
import threading
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, urlunparse, urlsplit, urlunsplit, parse_qsl, urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- .env -------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== 設定（ENVで上書き可） =====
CONNECT_TIMEOUT       = float(os.getenv("VALIDATOR_CONNECT_TIMEOUT_SEC", "3.0"))
READ_TIMEOUT          = float(os.getenv("VALIDATOR_READ_TIMEOUT_SEC", "7.0"))
DEFAULT_TIMEOUT_T     = (CONNECT_TIMEOUT, READ_TIMEOUT)

DEFAULT_CONCURRENCY   = int(os.getenv("VALIDATOR_CONCURRENCY", "8"))
DEFAULT_RETRIES       = int(os.getenv("VALIDATOR_RETRIES", "2"))
DEFAULT_BACKOFF       = float(os.getenv("VALIDATOR_BACKOFF_SEC", "0.5"))
MAX_REDIRECTS         = int(os.getenv("VALIDATOR_MAX_REDIRECTS", "10"))

# ★ 上限件数（既定=5：要件)何件の記事--------------------------------------------
MAX_RESULTS           = int(os.getenv("VALIDATOR_MAX_RESULTS", "2"))
DOMAIN_CAP            = int(os.getenv("VALIDATOR_DOMAIN_CAP", "2"))

# コンテンツしきい値（軽量）
HEAD_BYTES_LIMIT      = int(os.getenv("VALIDATOR_HEAD_BYTES", "131072"))   # 先頭 ~128KB
MIN_TEXT_CHARS        = int(os.getenv("VALIDATOR_MIN_TEXT_CHARS", "400"))  # 最低本文文字数
MIN_TEXT_RATIO        = float(os.getenv("VALIDATOR_MIN_TEXT_RATIO", "0.05"))  # テキスト/HTML比
ALLOW_ROOT_PATH       = os.getenv("VALIDATOR_ALLOW_ROOT_PATH", "false").lower() in ("1","true","yes","y")

# 機能トグル
ENABLE_CANONICAL      = os.getenv("VALIDATOR_ENABLE_CANONICAL", "true").lower() in ("1","true","yes","y")
ENABLE_DEAMP          = os.getenv("VALIDATOR_ENABLE_DEAMP", "true").lower() in ("1","true","yes","y")

USER_AGENT = os.getenv(
    "VALIDATOR_USER_AGENT",
    "GlobalDigest/Validator (+https://example.com)"
)
ACCEPT_LANGUAGE = os.getenv(
    "VALIDATOR_ACCEPT_LANGUAGE",
    "zh-CN,zh;q=0.9,ja;q=0.85,en;q=0.8"
)

HTML_CT_PREFIXES = ("text/html", "application/xhtml+xml")

# utm_* は prefix、それ以外は完全一致
TRACKING_PREFIXES = ("utm_",)
TRACKING_EQUALS   = (
    "fbclid","gclid","mc_cid","mc_eid","igshid","ved","si","oc","spm","_hsenc"
)

LOG_LEVEL = os.getenv("VALIDATOR_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] validator: %(message)s"
)
logger = logging.getLogger("validator")

# JSONL 監査ログ
ENABLE_JSONL_LOG   = os.getenv("VALIDATOR_ENABLE_JSONL_LOG", "true").lower() in ("1","true","yes","y")
JSONL_PATH         = os.getenv("VALIDATOR_JSONL_PATH", "./validator_audit.jsonl")

# ===== 正規表現ユーティリティ =====
_CANONICAL_RE = re.compile(
    r'<link[^>]+rel=["\']canonical["\'][^>]*href=["\']([^"\']+)["\']',
    re.IGNORECASE
)
_JSONLD_NEWS_RE = re.compile(r'"@type"\s*:\s*"(NewsArticle|Article)"', re.I)
_TIME_TAG_RE    = re.compile(r"<time\b", re.I)
_ARTICLE_TAG_RE = re.compile(r"<article\b", re.I)
_TITLE_TAG_RE   = re.compile(r"<title\b", re.I)

# AMP解除パターン
_AMP_PATTERNS = (
    (re.compile(r"/amp($|[/?#])", re.I), r"/\1"),
    (re.compile(r"(\?|&)amp=1(&|$)", re.I), r"\1"),
    (re.compile(r"(\?|&)output=amp(&|$)", re.I), r"\1"),
)

# ===== スレッドローカル Session =====
_tls = threading.local()

def _get_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is not None:
        return s
    s = requests.Session()
    s.max_redirects = MAX_REDIRECTS
    retries = Retry(
        total=DEFAULT_RETRIES,
        backoff_factor=DEFAULT_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=128, pool_maxsize=128)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": ACCEPT_LANGUAGE,
    })
    _tls.session = s
    return s

# ===== URLユーティリティ =====
def _is_absolute_http_url(u: str) -> bool:
    try:
        p = urlsplit(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _normalize_scheme_netloc(path_url: str) -> str:
    p = urlparse(path_url)
    if not p.scheme:
        p = p._replace(scheme="https")
    if not p.netloc and p.path:
        if "://" in p.path:
            p = urlparse(p.path)
        else:
            parts = p.path.split("/", 1)
            netloc = parts[0]
            path = "/" + parts[1] if len(parts) > 1 else ""
            p = p._replace(netloc=netloc, path=path)
    p = p._replace(fragment="")
    return urlunparse(p)

def _strip_tracking_params(url: str) -> str:
    sp = urlsplit(url)
    kept = []
    for k, v in parse_qsl(sp.query, keep_blank_values=True):
        kl = k.lower()
        if any(kl.startswith(p) for p in TRACKING_PREFIXES):
            continue
        if kl in TRACKING_EQUALS:
            continue
        kept.append((k, v))
    new_q = urlencode(kept, doseq=True)
    return urlunsplit((sp.scheme, sp.netloc, sp.path, new_q, ""))

def _deamp(url: str) -> str:
    out = url
    for pat, repl in _AMP_PATTERNS:
        out = pat.sub(repl, out)
    return out

def _same_reg_domain(base_host: str, can_host: str) -> bool:
    # サブドメイン許容（www.差など）※tldextract未使用の安全緩和
    base = base_host.lower()
    can  = can_host.lower()
    return (can == base) or can.endswith("." + base) or base.endswith("." + can)

def _safe_apply_canonical(base_url: str, canonical_url: str) -> Optional[str]:
    """canonicalの安全採用: サブドメイン許容 / https優先 / 露骨トップ誘導回避"""
    try:
        base = urlsplit(base_url)
        can  = urlsplit(_finalize_url(canonical_url))

        if not _same_reg_domain(base.netloc, can.netloc):
            return None
        if base.scheme == "https" and can.scheme != "https":
            return None
        if can.path in ("", "/"):
            return None
        return urlunsplit(can)
    except Exception:
        return None

def _finalize_url(url: str) -> str:
    url = _normalize_scheme_netloc(url)
    url = _strip_tracking_params(url)
    # de-AMP は候補生成段階（採用は後段で判定）
    url = _deamp(url) if ENABLE_DEAMP else url
    return url

# ===== HTML/コンテンツ判定 =====
def _is_html_content_type(content_type: Optional[str]) -> bool:
    if not content_type:
        return False
    ct = content_type.split(";", 1)[0].strip().lower()
    return any(ct.startswith(prefix) for prefix in HTML_CT_PREFIXES)

def _looks_like_html(head_bytes: bytes) -> bool:
    head = head_bytes[:8192].lower()  # 8KB程度
    return (b"<!doctype html" in head) or (b"<html" in head)

_TAG_RE = re.compile(rb"<[^>]+>")
_WS_RE  = re.compile(rb"\s+")

def _content_metrics(head_bytes: bytes) -> Dict[str, Any]:
    """軽量メトリクス: テキスト長・比率・記事シグナル有無"""
    # 粗いタグ剥がし
    text_bytes = _TAG_RE.sub(b" ", head_bytes)
    text_bytes = _WS_RE.sub(b" ", text_bytes).strip()
    text_len = len(text_bytes)

    html_len = max(len(head_bytes), 1)
    text_ratio = text_len / html_len

    # 文字列検索は decode してから（失敗時は ignore）
    head = head_bytes.decode("utf-8", errors="ignore")
    has_title    = bool(_TITLE_TAG_RE.search(head))
    has_article  = bool(_ARTICLE_TAG_RE.search(head))
    has_time     = bool(_TIME_TAG_RE.search(head))
    has_jsonld   = bool(_JSONLD_NEWS_RE.search(head))

    # 仮スコア（監査用）：記事シグナルに重み
    score = 0
    if text_len >= MIN_TEXT_CHARS: score += 2
    if text_ratio >= MIN_TEXT_RATIO: score += 1
    if has_jsonld: score += 3
    if has_article: score += 2
    if has_time: score += 1
    if has_title: score += 1

    return {
        "text_len": text_len,
        "text_ratio": round(text_ratio, 4),
        "has_title": has_title,
        "has_article": has_article,
        "has_time": has_time,
        "has_jsonld": has_jsonld,
        "score_hint": score
    }

# ===== 重複・上限制御 =====
def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out

def _apply_domain_cap_preserve_order(urls: List[str], cap: int) -> List[str]:
    count: Dict[str, int] = {}
    out: List[str] = []
    for u in urls:
        host = urlsplit(u).netloc.lower()
        if count.get(host, 0) >= cap:
            continue
        count[host] = count.get(host, 0) + 1
        out.append(u)
    return out

# ===== コア：1URL検証 =====
def _resolve_one(url: str, index: int) -> Tuple[int, bool, Dict[str, Any]]:
    session = _get_session()
    t0 = time.time()
    meta: Dict[str, Any] = {
        "input_url": url,
        "normalized_url": None,
        "final_url": None,
        "status_code": None,
        "content_type": None,
        "decision": "filtered",
        "reason": "",
        "elapsed_ms": None,
        "metrics": None,
    }

    try:
        # 正規化
        norm_url = _finalize_url(url)
        meta["normalized_url"] = norm_url

        # 絶対URLのみ許可
        if not _is_absolute_http_url(norm_url):
            meta["reason"] = "invalid_url_scheme"
            return index, False, _finalize_meta(meta, t0)

        # ルート/カテゴリっぽいパスは原則除外（許可したい場合は ENV）
        sp = urlsplit(norm_url)
        if not ALLOW_ROOT_PATH and (sp.path in ("", "/")):
            meta["reason"] = "root_like_path"
            return index, False, _finalize_meta(meta, t0)

        # 最終GET（HTML判定とコンテンツ先頭抽出）
        r = session.get(norm_url, allow_redirects=True, timeout=DEFAULT_TIMEOUT_T, stream=True)
        meta["status_code"] = r.status_code
        meta["content_type"] = r.headers.get("Content-Type", "").split(";")[0].strip() if r.headers.get("Content-Type") else None

        # HTML判定（CT or スニッフ）
        is_html = _is_html_content_type(meta["content_type"])
        buf = b""
        looked = False
        if not is_html:
            try:
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk: break
                    buf += chunk
                    if len(buf) >= 8192: break
                looked = True
                is_html = _looks_like_html(buf)
            except Exception:
                is_html = False

        if not is_html:
            try: r.close()
            except Exception: pass
            meta["reason"] = "non_html_content"
            return index, False, _finalize_meta(meta, t0)

        # de-AMP 採用（候補は正規化時に生成済み。ここでは r.url と比較）
        final_url = r.url
        if ENABLE_DEAMP:
            deamp_candidate = _deamp(norm_url)
            if deamp_candidate != norm_url:
                rr = None
                try:
                    rr = session.get(deamp_candidate, allow_redirects=True, timeout=DEFAULT_TIMEOUT_T)
                    if rr.status_code == 200 and _is_html_content_type(rr.headers.get("Content-Type", "")):
                        final_url = rr.url
                except Exception:
                    pass
                finally:
                    try: rr and rr.close()
                    except Exception: pass

        # canonical 条件付き採用
        canonical_url = None
        rr = None
        try:
            if ENABLE_CANONICAL:
                if looked:
                    rr = session.get(final_url, allow_redirects=True, timeout=DEFAULT_TIMEOUT_T)
                    content = rr.content[:HEAD_BYTES_LIMIT]
                else:
                    # r から追加で読み取り（最大 ~128KB）
                    for chunk in r.iter_content(chunk_size=8192):
                        if not chunk: break
                        buf += chunk
                        if len(buf) >= HEAD_BYTES_LIMIT: break
                    content = buf or r.content[:HEAD_BYTES_LIMIT]
                m = _CANONICAL_RE.search(content.decode("utf-8", errors="ignore"))
                if m:
                    href = m.group(1).strip()
                    can = _safe_apply_canonical(final_url, href)
                    if can:
                        canonical_url = can
        finally:
            try: r.close()
            except Exception: pass
            try: rr and rr.close()
            except Exception: pass

        final_url = _finalize_url(canonical_url or final_url)

        # コンテンツ判定（先頭 HEAD_BYTES_LIMIT）
        # すでに content が無ければ、軽量GETで補う
        head_bytes = b""
        if 'content' in locals() and isinstance(content, (bytes, bytearray)):
            head_bytes = content
        else:
            r2 = None
            try:
                r2 = session.get(final_url, allow_redirects=True, timeout=DEFAULT_TIMEOUT_T, stream=True)
                for chunk in r2.iter_content(chunk_size=8192):
                    if not chunk: break
                    head_bytes += chunk
                    if len(head_bytes) >= HEAD_BYTES_LIMIT: break
            except Exception:
                pass
            finally:
                try: r2 and r2.close()
                except Exception: pass

        metrics = _content_metrics(head_bytes)
        meta["metrics"] = metrics

        # しきい値判定
        if metrics["text_len"] < MIN_TEXT_CHARS:
            meta["reason"] = "too_short_text"
            return index, False, _finalize_meta(meta, t0)

        if metrics["text_ratio"] < MIN_TEXT_RATIO:
            meta["reason"] = "too_low_text_ratio"
            return index, False, _finalize_meta(meta, t0)

        # 記事シグナルが一つも無い場合は弱い（弾く）
        if not (metrics["has_jsonld"] or metrics["has_article"] or metrics["has_time"] or metrics["has_title"]):
            meta["reason"] = "no_article_signals"
            return index, False, _finalize_meta(meta, t0)

        # OK
        meta["final_url"] = final_url
        meta["decision"] = "accepted"
        meta["reason"] = "accepted_content_ready"
        return index, True, _finalize_meta(meta, t0)

    except Exception as e:
        meta["reason"] = f"exception:{e.__class__.__name__}"
        return index, False, _finalize_meta(meta, t0)

def _finalize_meta(meta: Dict[str, Any], t0: float) -> Dict[str, Any]:
    meta["elapsed_ms"] = int((time.time() - t0) * 1000)
    return meta

# ===== エクスポート関数 =====
def run_validation(urls: List[str], return_meta: bool = False) -> List[str] | List[Dict[str, Any]]:  # noqa: E701
    """
    Discover（フルURL）から30件程度来る前提で、スクレイプに向くURLだけを抽出。
    - 入力順を保持
    - HTML/コンテンツしきい値で使用可否を判定
    - 正規化後フルURLで重複除去
    - ドメイン上限 → MAX_RESULTS（既定=5）
    - JSONL監査ログ出力
    """
    if not urls:
        return [] if not return_meta else []

    metas: List[Optional[Dict[str, Any]]] = [None] * len(urls)
    with ThreadPoolExecutor(max_workers=DEFAULT_CONCURRENCY) as ex:
        fut_map = {ex.submit(_resolve_one, u, i): i for i, u in enumerate(urls)}
        for fut in as_completed(fut_map):
            i, ok, meta = fut.result()
            metas[i] = meta

    # JSONL 監査ログ
    if ENABLE_JSONL_LOG:
        try:
            os.makedirs(os.path.dirname(JSONL_PATH) or ".", exist_ok=True)
            with open(JSONL_PATH, "a", encoding="utf-8") as f:
                for m in metas:
                    if m is None:
                        continue
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"JSONL 書き込みに失敗: {e}")

    # OKのみ
    ok_urls = [m["final_url"] for m in metas if m and m.get("decision") == "accepted" and m.get("final_url")]

    # 重複 → ドメイン上限 → MAX_RESULTS
    ok_urls = _dedupe_preserve_order(ok_urls)
    ok_urls = _apply_domain_cap_preserve_order(ok_urls, DOMAIN_CAP)
    ok_urls = ok_urls[:MAX_RESULTS]

    if return_meta:
        return [m for m in metas if m is not None]
    return ok_urls


if __name__ == "__main__":
    import sys
    test_urls = sys.argv[1:]
    if not test_urls:
        print("Usage: python validator.py <url1> <url2> ...")
        sys.exit(0)
    result = run_validation(test_urls, return_meta=False)
    for u in result:
        print(u)