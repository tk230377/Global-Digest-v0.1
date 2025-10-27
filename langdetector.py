#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Language Detector (Production, hardened)
- 入力テキストの言語を推定
- デフォルトは「中国語のみを通す」(ZH_ONLY)。.env で切替
    LANGDET_ZH_ONLY=true | false
- 判定は 文字種ヒューリスティック + langdetect(任意) の投票
- langdetect が無い環境でも劣化運転で継続

依存（必須）:
  python-dotenv（任意推奨）

依存（任意）:
  langdetect  (pip install langdetect)

公開関数（互換）:
  run_language_detection(texts: List[str]) -> List[Tuple[str, str]]
  run_language_detection_aligned(texts: List[str]) -> List[Tuple[str, str]]
"""

from __future__ import annotations
import os
import re
import logging
from typing import List, Tuple, Optional
from functools import lru_cache

# ---- .env 読み込み ----------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---- langdetect の存在確認 --------------------------------------
_HAS_LANGDETECT = False
try:
    from langdetect import detect, DetectorFactory  # type: ignore
    DetectorFactory.seed = 42  # 再現性
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

logger = logging.getLogger("langdetector")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LANGDET_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# ---- 設定（環境変数で調整可能） ---------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")

ZH_ONLY: bool = _env_bool("LANGDET_ZH_ONLY", False)
MIN_LEN_FOR_DETECT = max(16, int(os.getenv("LANGDET_MIN_LEN", "80")))          # 短文誤判定を抑制
VOTE_REQUIRE = max(1, int(os.getenv("LANGDET_VOTE_REQUIRE", "2")))            # 何票以上で採用
VOTE_TRIALS  = max(VOTE_REQUIRE, int(os.getenv("LANGDET_VOTE_TRIALS", "3")))  # 試行回数 >= 閾値
SAMPLE_MAX_CHARS = max(256, int(os.getenv("LANGDET_SAMPLE_MAX_CHARS", "3000")))
MAX_CACHE = max(256, int(os.getenv("LANGDET_CACHE_SIZE", "4096")))            # LRU キャッシュ

# =============================================================================
# 文字種ヒューリスティック & 前処理
# =============================================================================
RE_HIRAGANA = re.compile(r"[\u3040-\u309F]")
RE_KATAKANA = re.compile(r"[\u30A0-\u30FF\u31F0-\u31FF]")
RE_HAN = re.compile(r"[\u4E00-\u9FFF]")
RE_CN_PUNCT = re.compile(r"[，。、《》「」！？；：]")
RE_LATIN = re.compile(r"[A-Za-z]")
RE_URL = re.compile(r"https?://\S+")
RE_MULTI_NL = re.compile(r"\n{3,}")
RE_WS = re.compile(r"[ \t\u3000]+")

def _has_kana(s: str) -> bool:
    return bool(RE_HIRAGANA.search(s) or RE_KATAKANA.search(s))

def _han_ratio(s: str) -> float:
    if not s:
        return 0.0
    han = len(RE_HAN.findall(s))
    return han / max(1, len(s))

def _looks_chinese_by_script(s: str) -> bool:
    """
    - かなが無い
    - 中国語句読点がある、または漢字比率が一定以上
    """
    if _has_kana(s):
        return False
    if RE_CN_PUNCT.search(s):
        return True
    return _han_ratio(s) >= 0.10  # 10%以上が漢字なら中文の可能性が高い

def _normalize_for_detection(s: str) -> str:
    """
    検出前クリーニング：
      - URL除去
      - 見出し/タイトル行っぽい先頭1行（全角/半角記号だらけ）を弱める
      - 連続改行/空白の圧縮
    """
    if not s:
        return ""
    s = s.replace("\r", "")
    # "タイトル\\n\\n本文" 想定：先頭行に極端な記号が多いなら落とす
    parts = s.split("\n", 2)
    if parts:
        head = parts[0]
        # 記号比率が高い見出しを落とす（広告見出し等の誤判定回避）
        sym_ratio = sum(ch in "~!@#$%^&*()_+-=[]{},.<>?/\\|\"'`：；・…—―※★☆◆■" for ch in head) / max(1, len(head))
        if sym_ratio >= 0.35:
            s = parts[1] + ("\n" + parts[2] if len(parts) > 2 else "")
    # URL除去
    s = RE_URL.sub(" ", s)
    # 空白整形
    s = RE_WS.sub(" ", s)
    s = re.sub(r" +\n", "\n", s)
    s = RE_MULTI_NL.sub("\n\n", s)
    return s.strip()

def _sample_for_detection(s: str) -> str:
    if len(s) <= SAMPLE_MAX_CHARS:
        return s
    return s[:SAMPLE_MAX_CHARS]

# =============================================================================
# langdetect 投票（任意依存）
# =============================================================================
def _vote_langdetect_is_prefix(s: str, prefix: str, trials: int) -> bool:
    if not _HAS_LANGDETECT:
        return False
    s = _sample_for_detection(s)
    votes = 0
    for _ in range(trials):
        try:
            lang = detect(s)  # e.g., 'en', 'ja', 'zh-cn', 'zh-tw'
            if lang.startswith(prefix):
                votes += 1
        except Exception:
            # 例外はノーカウント
            pass
    return votes >= VOTE_REQUIRE

# =============================================================================
# 言語分類（キャッシュ付）
# =============================================================================
@lru_cache(maxsize=MAX_CACHE)
def _classify_lang_cached(raw: str) -> str:
    """
    'zh' / 'ja' / 'en' / 'other'
    """
    ss = _normalize_for_detection(raw or "")
    if not ss:
        return "other"

    # まず文字種
    if _has_kana(ss):
        return "ja"
    if _looks_chinese_by_script(ss):
        if len(ss) >= MIN_LEN_FOR_DETECT and _vote_langdetect_is_prefix(ss, "zh", VOTE_TRIALS):
            return "zh"
        # langdetect 無 or 票不足でも安全側で zh
        return "zh"

    # 英語の簡易当て
    if RE_LATIN.search(ss) and not RE_HAN.search(ss):
        # 可能なら裏取り
        if len(ss) >= MIN_LEN_FOR_DETECT and _vote_langdetect_is_prefix(ss, "en", VOTE_TRIALS):
            return "en"
        return "en"

    # その他は可能なら langdetect へ
    if _HAS_LANGDETECT and len(ss) >= MIN_LEN_FOR_DETECT:
        try:
            lang = detect(_sample_for_detection(ss))
            if lang.startswith("zh"):
                return "zh"
            if lang.startswith("ja"):
                return "ja"
            if lang.startswith("en"):
                return "en"
        except Exception:
            pass
    return "other"

def _classify_lang(s: str) -> str:
    return _classify_lang_cached(s)

# =============================================================================
# 公開API
# =============================================================================
def run_language_detection(texts: List[str]) -> List[Tuple[str, str]]:
    """
    既定(ZH_ONLY=True): 中国語テキストのみ [( 'zh', text ), ...] を返す。
    ZH_ONLY=False: すべてのテキストに対して (lang, text) を返す。
    ※ zh-only の場合は出力件数が入力より少なくなり得る（順序も崩れる）。
    """
    if not texts:
        logger.info("[LangDetector] 入力テキストが空です")
        return []

    out: List[Tuple[str, str]] = []
    zh_count = 0
    for t in texts:
        if not t or not t.strip():
            continue
        lang = _classify_lang(t)
        if ZH_ONLY:
            if lang == "zh":
                out.append(("zh", t))
                zh_count += 1
        else:
            out.append((lang, t))

    if ZH_ONLY:
        logger.info(f"[LangDetector] zh-only: 中国語 {zh_count} 件 / 入力 {len(texts)} 件")
    else:
        logger.info(f"[LangDetector] multi: 出力 {len(out)} 件 / 入力 {len(texts)} 件（言語混在）")
    return out

# =============================================================================
# 公開API（パイプライン向け、順序・長さを維持）
# =============================================================================
def run_language_detection_aligned(texts: List[str]) -> List[Tuple[str, str]]:
    """
    入力と同じ長さの List[Tuple[lang, text]] を返す（順序維持）。
    - ZH_ONLY=True: 非 zh は ("", "") を入れて穴埋め
    - ZH_ONLY=False: 全件 (lang, text) を返す
    """
    if not texts:
        logger.info("[LangDetector] 入力テキストが空です")
        return []

    out: List[Tuple[str, str]] = []
    for t in texts:
        if not t or not t.strip():
            out.append(("", ""))  # 空は空で保持
            continue
        lang = _classify_lang(t)
        if ZH_ONLY:
            out.append(("zh", t) if lang == "zh" else ("", ""))
        else:
            out.append((lang, t))
    return out


#add 10/22
def run_language_detection_with_meta(items: List[dict], zh_only: bool = False) -> List[dict]:
    """
    スクレイパーの出力（List[dict]）を受け取り、各要素に 'lang' と 'lang_reason' を追加して返す。
    zh_only=True の場合は中国語以外を除外する。
    """
    results = []
    for item in items:
        text = item.get("text", "").strip()
        if not text:
            item["lang"] = "unknown"
            item["lang_reason"] = "empty"
            if not zh_only:
                results.append(item)
            continue

        lang = _classify_lang(text)
        item["lang"] = lang
        item["lang_reason"] = "ok"

        if zh_only and lang != "zh":
            continue
        results.append(item)
    return results

"""
# ---- スタンドアロン実行（手動テスト） ---------------------------------------
if __name__ == "__main__":
    samples = [
        "OpenAI launches a new model today. It's optimized for reasoning.",
        "OpenAIが新しいモデルを発表。推論に最適化されているという。",
        "OpenAI 今日发布了一个新模型，据称在推理方面进行了优化。",
        "臺灣今天天氣很好，適合出門走走。",
        "今日は良い天気ですね。散歩に行きましょう。",
        "生成AI 在产业界的应用愈发广泛，特别是在制造与物流领域。",
        "",
    ]

    print(f"langdetect_available={_HAS_LANGDETECT}, ZH_ONLY={ZH_ONLY}")
    print("=== run_language_detection ===")
    print(run_language_detection(samples))
    print("=== run_language_detection_aligned ===")
    print(run_language_detection_aligned(samples))
   """