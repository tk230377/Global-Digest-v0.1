#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Translator (ZH → JA, Production, env-driven)

- .env を厳密に読み、Gemini設定を自己診断してから翻訳を実行
- 必須環境変数:
    GEMINI_API_KEY               # 例: AI Studio のキー
    TRANSLATOR_USE_GEMINI=true   # true/1/yes/on で有効化
    TRANSLATOR_MODEL=gemini-1.5-flash  # 推奨: gemini-1.5-flash / gemini-1.5-pro
    TRANSLATOR_BATCH_SIZE=6
    TRANSLATOR_TONE=polite
    TRANSLATOR_MAX_INPUT_CHARS=6000
    TRANSLATOR_LOG_LEVEL=INFO
"""

from __future__ import annotations
import os
import re
import logging
from typing import List, Dict, Any

# ---- .env -------------------------------------------------------------------
def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        # カレントディレクトリ優先で .env をロード
        load_dotenv(override=False)
    except Exception:
        pass

_load_env()

# ---- ロガー -----------------------------------------------------------------
logger = logging.getLogger("translator")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("TRANSLATOR_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# ---- 環境変数ユーティリティ ---------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "y")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _mask_key(k: str) -> str:
    if not k:
        return "(empty)"
    return (k[:6] + "..." + k[-4:]) if len(k) >= 14 else "(short)"

# ---- 設定（.env で上書き） ---------------------------------------------------
USE_GEMINI: bool = _env_bool("TRANSLATOR_USE_GEMINI", True)
MODEL_NAME: str = os.getenv("TRANSLATOR_MODEL", "gemini-1.5-flash").strip()
BATCH_SIZE: int = max(1, _env_int("TRANSLATOR_BATCH_SIZE", 6))
MAX_INPUT_CHARS: int = _env_int("TRANSLATOR_MAX_INPUT_CHARS", 6000)
TONE: str = os.getenv("TRANSLATOR_TONE", "polite").strip().lower()  # "polite" | "plain"
LOG_LEVEL = os.getenv("TRANSLATOR_LOG_LEVEL", "INFO").upper()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# 診断ログ（APIキーは伏字）
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.info(
    "[Translator] cfg: use_gemini=%s model=%s batch=%d max_chars=%d tone=%s api_key=%s",
    USE_GEMINI, MODEL_NAME, BATCH_SIZE, MAX_INPUT_CHARS, TONE, _mask_key(GEMINI_API_KEY)
)

# ---- Gemini 初期化 -----------------------------------------------------------
_USE_GEMINI = False
_api_error_logged = False
_genai_model = None

if USE_GEMINI:
    if not GEMINI_API_KEY:
        logger.warning("[Translator] GEMINI_API_KEY が未設定です。フォールバックに切り替えます。")
    else:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=GEMINI_API_KEY)
            _genai_model = genai.GenerativeModel(MODEL_NAME)
            # 疎通チェック（失敗しても本処理は継続可能）
            try:
                _ = _genai_model.generate_content("ok")
            except Exception as e:
                logger.warning("[Translator] Gemini 疎通時に警告: %s（本処理は継続）", e)
            _USE_GEMINI = True
            logger.info("[Translator] Gemini 有効: model=%s", MODEL_NAME)
        except Exception as e:
            logger.warning("[Translator] Gemini 初期化に失敗。フォールバックへ: %s", e)

# =============================================================================
# 文字種ヒューリスティック（zh_only 用の軽量判定）
# =============================================================================
RE_HIRAGANA = re.compile(r"[\u3040-\u309F]")
RE_KATAKANA = re.compile(r"[\u30A0-\u30FF\u31F0-\u31FF]")
RE_HAN = re.compile(r"[\u4E00-\u9FFF]")
RE_CN_PUNCT = re.compile(r"[，。、《》「」！？；：]")

def _looks_chinese_like(s: str) -> bool:
    if not s:
        return False
    if RE_HIRAGANA.search(s) or RE_KATAKANA.search(s):
        return False
    if RE_CN_PUNCT.search(s):
        return True
    han = len(RE_HAN.findall(s))
    return han >= max(10, int(len(s) * 0.08))

# =============================================================================
# LLM プロンプト
# =============================================================================
PROMPT_ZH_TO_JA = """あなたはプロの日本語テクニカルエディターです。以下のテキストを自然で読みやすい日本語に翻訳してください。

要件:
- 箇条書きの記号（•, -, 1., 2. など）や番号付きリストはそのまま維持
- 固有名詞（企業名・人名・製品名・規制名・地名・日付・金額）は正確に保持
- 事実ベースで、主張と見解を混同しない
- 文体は「{tone}」で統一
- 出力はプレーンテキストのみ（前置きや説明・引用符は不要）

--- 翻訳対象 ---
"""

def _tone_label() -> str:
    return "です・ます調" if TONE == "polite" else "常体"

# =============================================================================
# 内部実装
# =============================================================================
def _clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit].rstrip() + "…"

def _translate_gemini_batch(texts: List[str]) -> List[str]:
    """Gemini に 1件ずつ投げる（箇条書き保持のため逐次が安定）"""
    global _api_error_logged   # ← ここを関数先頭に置く（SyntaxError 対策）
    outs: List[str] = []
    try:
        model = _genai_model  # 初期化済みインスタンスを再利用
        if model is None:
            raise RuntimeError("Gemini model is not initialized")
        for t in texts:
            content = PROMPT_ZH_TO_JA.format(tone=_tone_label()) + _clip(t, MAX_INPUT_CHARS)
            try:
                res = model.generate_content(content)
                ja = (getattr(res, "text", "") or "").strip()
                outs.append(ja if ja else f"【翻訳未実施】{_clip(t, 200)}")
            except Exception as e:
                if not _api_error_logged:
                    logger.warning("[Translator] Gemini 呼び出し失敗。フォールバック返却に切替: %s", e)
                outs.append(f"【翻訳未実施】{_clip(t, 200)}")
        return outs
    except Exception as e:
        if not _api_error_logged:
            logger.warning("[Translator] Gemini 処理前失敗。以降フォールバックへ: %s", e)
            _api_error_logged = True
        return [f"【翻訳未実施】{_clip(t, 200)}" for t in texts]

def _translate_fallback_batch(texts: List[str]) -> List[str]:
    """APIなし環境向けフォールバック：翻訳せずパイプライン継続"""
    return [f"【翻訳未実施】{_clip(t, 200)}" if t else "" for t in texts]

# =============================================================================
# 公開 API（gdscheduler 互換）
# =============================================================================
def run_translation(summaries: List[str], *, return_meta: bool = False, zh_only: bool = False):
    """
    入力: List[str]（原言語の要約; 空文字含み得る）
    出力:
      - return_meta=False -> List[str]（順序維持・同長。空/失敗は "" or 「翻訳未実施...」）
      - return_meta=True  -> List[dict]（src/ja/ok/engine/reason を含む）
    オプション:
      - zh_only=True のとき、中文らしくない要素は翻訳をスキップ
    """
    if not summaries:
        logger.info("[Translator] 入力が空です")
        return [] if not return_meta else []

    n = len(summaries)
    out_texts: List[str] = [""] * n
    out_meta: List[Dict[str, Any]] = [{} for _ in range(n)]

    batch_idx: List[int] = []
    batch_txt: List[str] = []

    def _flush():
        nonlocal batch_idx, batch_txt
        if not batch_idx:
            return
        if _USE_GEMINI:
            outs = _translate_gemini_batch(batch_txt)
            engine = "gemini"
        else:
            outs = _translate_fallback_batch(batch_txt)
            engine = "fallback"
        for i, ja in zip(batch_idx, outs):
            src = summaries[i] or ""
            out_texts[i] = ja or ""
            out_meta[i] = {
                "index": i, "src": src, "ja": out_texts[i], "ok": bool(out_texts[i]),
                "engine": engine, "reason": "ok" if out_texts[i] else "empty"
            }
        batch_idx = []
        batch_txt = []

    for i, s in enumerate(summaries):
        src = (s or "").strip()
        if not src:
            out_texts[i] = ""
            out_meta[i] = {"index": i, "src": "", "ja": "", "ok": False, "engine": None, "reason": "empty"}
            continue
        if zh_only and not _looks_chinese_like(src):
            out_texts[i] = f"【翻訳対象外（zh_only）】{_clip(src, 200)}"
            out_meta[i] = {"index": i, "src": src, "ja": out_texts[i], "ok": True, "engine": "skip", "reason": "zh_only_skip"}
            continue
        batch_idx.append(i)
        batch_txt.append(src)
        if len(batch_idx) >= BATCH_SIZE:
            _flush()

    _flush()

    logger.info(
        "[Translator] 翻訳完了: %d/%d 件 (engine=%s)",
        sum(1 for t in out_texts if t), n, "gemini" if _USE_GEMINI else "fallback"
    )
    return out_texts if not return_meta else out_meta

# 互換用エイリアス
run_translate_aligned = run_translation

# ---- スタンドアロン実行（手動テスト） ---------------------------------------
if __name__ == "__main__":
    samples = [
        "• OpenAI 推出新模型，強調推理能力與成本效率。\n• 於多項基準測試表現優於前代。\n• 製造、物流、醫療等產業可受益。",
        "1. 蘋果在加州發表新處理器與筆電。\n2. 能效與圖形性能顯著提升，強調本地 AI 體驗。",
        "これは日本語の箇条書きです。\n• 翻訳の必要はありません。",
        "",
    ]
    outs = run_translation(samples, return_meta=True, zh_only=True)
    for m in outs:
        print(f"\n--- 翻訳 index={m['index']} engine={m['engine']} reason={m['reason']} ---\n{m['ja']}\n")
