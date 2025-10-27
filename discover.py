#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Discovery Agent
- Gemini 2.5 Flash に中国語ニュースURL候補(JSON)を出させる
- 軽量HTTP検証(到達/リダイレクト追従/HTML系Content-Type)
- 既存パイプライン互換API: run_discovery(interests: str, max_results: int) -> List[str]

.env:
  GEMINI_API=xxxxxxxxxxxxxxxxxxxxxxxx         # 推奨（GEMINI_API_KEY も自動フォールバック）
  DISCOVERY_GEMINI_MODEL=gemini-2.5-flash     # 既定: gemini-2.5-flash
  DISCOVERY_GEMINI_URLS_PER_INTEREST=8
  DISCOVERY_TIMEOUT_SEC=12
  DISCOVERY_CONCURRENCY=8
  DISCOVERY_MAX_REDIRECTS=5
  DISCOVERY_UA=GlobalDigest/1.1 (+https://example.com) Python-Requests
  DISCOVERY_DOMAIN_WHITELIST=...,(カンマ区切り。空で無効)
"""

from __future__ import annotations
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.utils import requote_uri

# .env 読み込み（任意）
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ===== ログ =====
logger = logging.getLogger("discover")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ===== 設定 =====
UA = os.getenv("DISCOVERY_UA", "GlobalDigest/1.1 (+https://example.com) Python-Requests")  # ASCIIのみ
TIMEOUT = float(os.getenv("DISCOVERY_TIMEOUT_SEC", "12"))
CONCURRENCY = max(1, int(os.getenv("DISCOVERY_CONCURRENCY", "8")))
MAX_REDIRECTS = int(os.getenv("DISCOVERY_MAX_REDIRECTS", "5"))
URLS_PER_INTEREST = max(1, int(os.getenv("DISCOVERY_GEMINI_URLS_PER_INTEREST", "8")))
MODEL_ID = os.getenv("DISCOVERY_GEMINI_MODEL", "gemini-2.5-flash")  # ご指定どおり

# 中国語ニュース寄りのホワイトリスト（任意・空で無効）
DOMAIN_WHITELIST = os.getenv("DISCOVERY_DOMAIN_WHITELIST", """
xinhuanet.com, people.com.cn, thepaper.cn, 36kr.com, sspai.com,
ifeng.com, caixin.com, qq.com, sohu.com, sina.com.cn, cctv.com,
chinadaily.com.cn, jiemian.com, yicai.com, nbd.com.cn, scmp.com
""").replace("\n", "").replace(" ", "")
DOMAIN_WHITELIST = [d for d in DOMAIN_WHITELIST.split(",") if d]

# ===== 依存（Gemini SDK）=====
try:
    import google.generativeai as genai  # pip install -U google-generativeai
except Exception as e:
    raise SystemExit("google-generativeai が未インストールです。`pip install -U google-generativeai`") from e


# =============================================================================
# 内部ユーティリティ
# =============================================================================

def _host(u: str) -> str:
    try:
        return urlparse(u).hostname or ""
    except Exception:
        return ""

def _host_allowed(u: str) -> bool:
    if not DOMAIN_WHITELIST:
        return True
    h = (_host(u) or "").lower()
    return any(h == d or h.endswith("." + d) for d in DOMAIN_WHITELIST)

def _looks_like_article_url(u: str) -> bool:
    # 日付パスや記事っぽいパスを好む簡易ヒューリスティック（緩め）
    return (
        bool(re.search(r"/20\d{2}/\d{1,2}/\d{1,2}/", u))  # /YYYY/MM/DD/
        or "/news" in u
        or "/article" in u
        or "/story" in u
    )

def _dedup_keep_order(urls: List[str]) -> List[str]:
    seen, out = set(), []
    for u in urls:
        if u and u not in seen:
            out.append(u); seen.add(u)
    return out


# =============================================================================
# Gemini → URL候補（JSONを強制）
# =============================================================================

def _gemini_propose_urls(api_key: str, interests: List[str], per_interest: int) -> List[Dict[str, Any]]:
    # GEMINI_API（推奨）/ GEMINI_API_KEY（互換）に対応
    genai.configure(api_key=api_key)

    domain_clause = ""
    if DOMAIN_WHITELIST:
        domain_clause = "必ず以下のドメインまたはそのサブドメインからのみ選んでください:\n" + ", ".join(DOMAIN_WHITELIST) + "\n"

    # ★ プロンプト
    prompt = f"""
あなたは「{interests}」に基づいたニュース記事を収集するプロフェッショナルです。以下のタスクを実行してください：
各興味キーワード「{interests}」について、**実際に存在する、最新の中国語ニュース記事**のURLを特定してください。
要件：
- 出力は**厳密にJSON配列**である必要があります。各要素は {{ "title": "...", "url": "https://...", "source": "...", "date_iso": "YYYY-MM-DD" }} の構造を持つこと。
- URLは http(s) で始まり、**必ず記事本文ページ**（検索結果や一覧ページではない）のものであること。
- **【最重要】URLを生成する前に、検索エンジンで最新の記事タイトルを検索し、その検索結果に表示される実際のURLを基に提案してください。推測は厳禁です。**
- **URLに含まれる記事ID（末尾の数字部分）は、ランダムに生成せず、有効であることが確認できた実際のIDを使用してください。**
- 各キーワード「{interests}」につき**少なくとも {per_interest} 件**を提案してください。重複は避けること。可能であればそれぞれのキーワード「{interests}」同士の関連したニュースも収集してください。
- {domain_clause}
- URL中に /YYYY/MM/DD/ や /20YY/ の日付情報、または10桁以上の記事IDを含むURLを優先してください。
- `date_iso` の日付は、必ず**現在から過去1年以内**のものを設定すること。
- 中国で放送されている中国語ニュースを取得してください。

興味キーワード「{interests}」: {", ".join(interests)}
"""

    model = genai.GenerativeModel(MODEL_ID)
    res = model.generate_content(prompt, safety_settings=None)
    text = (res.text or "").strip()

    # 応答がJSONブロックでない場合の救済（```json ... ```）
    if not text.startswith("["):
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Gemini 出力がJSONとして解釈できません: {e}\n---RAW---\n{text}")

    if not isinstance(data, list):
        raise RuntimeError("Gemini出力はJSON配列ではありません。")

    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if DOMAIN_WHITELIST and not _host_allowed(url):
            continue
        if not _looks_like_article_url(url):
            # 緩めに許可（厳しくするならここで continue）
            pass
        out.append({
            "title": item.get("title", ""),
            "url": url,
            "source": item.get("source", ""),
            "date_iso": item.get("date_iso"),
        })

    logger.info("[Discovery] Gemini候補 %d 件 (model=%s)", len(out), MODEL_ID)
    return out


# =============================================================================
# HTTP 検証（到達性・最終URL・Content-Type=HTML系）
# =============================================================================

def _probe_one(url: str, session: requests.Session) -> Optional[str]:
    try:
        safe_url = requote_uri(url)  # 非ASCIIをRFC準拠でエンコード
        r = session.head(safe_url, timeout=TIMEOUT, allow_redirects=True)
        final = r.url or safe_url
        if r.is_redirect and len(r.history) > MAX_REDIRECTS:
            return None
        ct = r.headers.get("Content-Type", "")
        if r.status_code >= 400 or ("text/html" not in ct and "application/xhtml+xml" not in ct):
            g = session.get(safe_url, timeout=TIMEOUT, allow_redirects=True)
            if g.status_code >= 400:
                return None
            final = g.url or safe_url
            ct = g.headers.get("Content-Type", "")
            if "text/html" not in ct and "application/xhtml+xml" not in ct:
                return None
        return final
    except Exception:
        return None

def _verify_urls(urls: List[str]) -> List[str]:
    urls = _dedup_keep_order(urls)
    results: List[Optional[str]] = [None] * len(urls)
    with requests.Session() as session:
        session.headers.update({"User-Agent": UA})
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futmap = {ex.submit(_probe_one, u, session): i for i, u in enumerate(urls)}
            for fut in as_completed(futmap):
                i = futmap[fut]
                try:
                    results[i] = fut.result()
                except Exception:
                    results[i] = None
    verified = [r for r in results if r]
    return _dedup_keep_order(verified)


# =============================================================================
# 公開API（既存パイプライン互換）
# =============================================================================

def run_discovery(interests: str, max_results: int = 10) -> List[str]:
    """
    入力: 日本語カンマ区切りの興味キーワード（例: "生成AI, ウェアラブル"）
    出力: HTTP検証済みのフルURL list[str]（順序維持）
    """
    # .env は GEMINI_API（推奨）。互換で GEMINI_API_KEY も見る
    api_key = os.getenv("GEMINI_API", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.warning("GEMINI_API / GEMINI_API_KEY が未設定です。空リストを返します。")
        return []

    kws = [re.sub(r"\s+", "", p) for p in re.split(r"[,\n;、]+", interests) if p.strip()]
    if not kws:
        return []

    candidates_struct = _gemini_propose_urls(api_key, kws, URLS_PER_INTEREST)
    candidates = [c["url"] for c in candidates_struct if isinstance(c, dict) and c.get("url")]
    verified = _verify_urls(candidates)
    final = verified[:max_results]
    logger.info("[Discovery] 候補 %d → 検証後 %d → 返却 %d 件", len(candidates), len(verified), len(final))
    return final

