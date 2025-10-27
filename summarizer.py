#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Summarizer (Production, order-preserving)

- 入力: List[Tuple[lang, text]]
- 出力: List[str]（入力と同じ長さ・順序。失敗は ""）
- 既定は zh（中国語）を主対象。他言語が混ざっても落とさず要約を返す
- Gemini が使えない/失敗時は抽出型に自動フォールバック

依存(必須):
  pip install python-dotenv

依存(任意: LLM要約を使う場合):
  pip install google-generativeai

環境変数(.env; すべて任意):
  GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
  SUMMARIZER_USE_GEMINI=true
  SUMMARIZER_MODEL=gemini-1.5-flash
  SUMMARIZER_MAX_INPUT_CHARS=6000
  SUMMARIZER_TARGET_BULLETS=5
  SUMMARIZER_BATCH_SIZE=4
  SUMMARIZER_LOG_LEVEL=INFO
"""

from __future__ import annotations
import os
import re
import logging
from typing import List, Tuple, Optional

# ---- .env -------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("summarizer")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("SUMMARIZER_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# ---- 設定 -------------------------------------------------------------------
USE_GEMINI = os.getenv("SUMMARIZER_USE_GEMINI", "true").strip().lower() in ("1", "true", "yes", "on")
MODEL_NAME = os.getenv("SUMMARIZER_MODEL", "gemini-2.5-flash").strip()
MAX_INPUT_CHARS = int(os.getenv("SUMMARIZER_MAX_INPUT_CHARS", "6000"))  # 1記事あたりの入力上限
TARGET_BULLETS = int(os.getenv("SUMMARIZER_TARGET_BULLETS", "5"))
BATCH_SIZE = max(1, int(os.getenv("SUMMARIZER_BATCH_SIZE", "4")))

# ---- Gemini 初期化（任意） ---------------------------------------------------
_USE_GEMINI = False
_api_error_logged = False
try:
    import google.generativeai as genai  # type: ignore
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if USE_GEMINI and api_key:
        genai.configure(api_key=api_key)
        # モデル存在チェック（早期失敗）
        _ = genai.GenerativeModel(MODEL_NAME)
        _USE_GEMINI = True
        logger.info(f"[Summarizer] Gemini 有効: model={MODEL_NAME}")
    else:
        logger.info("[Summarizer] Gemini 無効（APIキー未設定または USE_GEMINI=false）")
except Exception as e:
    logger.info(f"[Summarizer] Gemini 初期化スキップ: {e}")
    _USE_GEMINI = False

# =============================================================================
# 抽出型サマリー（フォールバック）
# =============================================================================

# 中国語の文区切り（かなり大まか）
_RE_CN_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")
# CJK漢字（簡易）
_RE_CJK = re.compile(r"[\u4E00-\u9FFF]")

def _clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit].rstrip() + "…"

def _split_sentences_cn(text: str) -> List[str]:
    """中華圏ニュース向けの素朴な文分割"""
    parts = _RE_CN_SENT_SPLIT.split(text)
    out: List[str] = []
    buf = ""
    for p in parts:
        buf += p
        if p.endswith(("\n", "。", "！", "？", "!", "?")):
            s = buf.strip()
            if s:
                out.append(s)
            buf = ""
    if buf.strip():
        out.append(buf.strip())

    # 極端な短文を前文とマージ
    merged: List[str] = []
    for s in out:
        if merged and len(s) < 12:
            merged[-1] = (merged[-1] + s).strip()
        else:
            merged.append(s)
    return merged

def _rank_sentences_cn(text: str, topk: int = 5) -> List[str]:
    """文字頻度ベース（簡易TF）で文ランキング"""
    sents = _split_sentences_cn(text)
    if not sents:
        return []
    # 全体頻度
    freq = {}
    for ch in _RE_CJK.findall(text):
        freq[ch] = freq.get(ch, 0) + 1
    scored = []
    for i, s in enumerate(sents):
        score = 0
        for ch in _RE_CJK.findall(s):
            score += freq.get(ch, 0)
        # 長さペナルティ
        L = len(s)
        if L < 15:
            score *= 0.6
        elif L > 120:
            score *= 0.8
        scored.append((score, i, s))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored[:topk]]

def _fallback_summarize_one(text: str, bullets: int = TARGET_BULLETS) -> str:
    """抽出型（見出し + 箇条書き）"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    title = None
    if lines and len(lines[0]) <= 80:
        title = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
    else:
        body = "\n".join(lines)
    body = _clip(body, MAX_INPUT_CHARS)
    picks = _rank_sentences_cn(body, topk=bullets)
    if not picks:
        # 最後の手段：先頭から bullets 文
        picks = _split_sentences_cn(body)[:bullets]
    picks = [p.strip() for p in picks if p.strip()]
    bullets_cn = "\n".join([f"• {p}" for p in picks])
    if title:
        return f"{title}\n{bullets_cn}"
    return bullets_cn or _clip(text, 400)

# =============================================================================
# Gemini 要約
# =============================================================================

_PROMPT_ZH = """你是一名專業的新聞編輯。請用中文對輸入文章進行「事実ベースの要約」。
要求：
- 箇条書き {bullets} 點（簡潔で正確に、推測や誇張はしない）
- 固有名詞（人名・企業名・規制名・地名・日付・金額）を保持
- 可能なら発表主体・発表日・対象地域を明記
- ソースの主張と記者の見解を混同しない
- 1点あたり140字程度
出力はプレーンテキスト（箇条書き）のみ。前置きや説明は不要。

入力が「タイトル+本文」形式の場合、タイトルは1行目、本文はそれ以降です。
"""

def _gemini_summarize_batch(texts: List[str], bullets: int) -> List[str]:
    """texts は既に MAX_INPUT_CHARS でクリップ済み想定。失敗時は空文字を返さずフォールバック結果を返す。"""
    if not _USE_GEMINI or not texts:
        return [_fallback_summarize_one(t, bullets) for t in texts]
    outs: List[str] = []
    try:
        model = genai.GenerativeModel(MODEL_NAME)  # type: ignore
        for t in texts:
            prompt = _PROMPT_ZH.format(bullets=bullets)
            content = f"{prompt}\n\n=== 文書開始 ===\n{t}\n=== 文書終了 ==="
            try:
                res = model.generate_content(content)
                out = (res.text or "").strip()
                if not out:
                    outs.append(_fallback_summarize_one(t, bullets))
                else:
                    outs.append(out)
            except Exception as e:
                # 個別失敗はその要素のみフォールバック
                outs.append(_fallback_summarize_one(t, bullets))
        return outs
    except Exception as e:
        global _api_error_logged
        if not _api_error_logged:
            logger.warning(f"[Summarizer] Gemini バッチ失敗。以降フォールバックを使用: {e}")
            _api_error_logged = True
        return [_fallback_summarize_one(t, bullets) for t in texts]

# =============================================================================
# 公開 API（順序維持）
# =============================================================================

def run_summarization(lang_texts: List[Tuple[str, str]]) -> List[str]:
    """
    入力: [(lang, text), ...]
    出力: List[str]（入力と同じ長さ・順序。空や失敗は ""）
    方針:
      - zh はバッチ処理（LLM/抽出）
      - 非 zh も捨てずに要約（LLMがあればLLM、なければ抽出）
      - どの言語でも MAX_INPUT_CHARS でクリップ
    """
    n = len(lang_texts)
    if n == 0:
        logger.info("[Summarizer] 入力が空です")
        return []

    results: List[str] = [""] * n  # 順序維持
    # zh をまとめる（インデックス付きでバッチ処理 → 同じ位置に書き戻す）
    zh_batch_idx: List[int] = []
    zh_batch_txt: List[str] = []

    def _flush_zh_batch():
        nonlocal zh_batch_idx, zh_batch_txt, results
        if not zh_batch_idx:
            return
        outs = _gemini_summarize_batch(zh_batch_txt, TARGET_BULLETS)
        for i, out in zip(zh_batch_idx, outs):
            results[i] = out or ""
        zh_batch_idx = []
        zh_batch_txt = []

    for i, (lang, text) in enumerate(lang_texts):
        t = (text or "").strip()
        if not t:
            results[i] = ""
            continue
        # 入力上限をクリップ
        if len(t) > MAX_INPUT_CHARS:
            t = t[:MAX_INPUT_CHARS].rstrip() + "…"

        if (lang or "").startswith("zh"):
            # zh はバッチに貯める
            zh_batch_idx.append(i)
            zh_batch_txt.append(t)
            if len(zh_batch_idx) >= BATCH_SIZE:
                _flush_zh_batch()
        else:
            # 非 zh は即時処理（順序維持のため、結果をその場で埋める）
            if _USE_GEMINI:
                out = _gemini_summarize_batch([t], TARGET_BULLETS)[0]
                results[i] = out or ""
            else:
                results[i] = _fallback_summarize_one(t, TARGET_BULLETS)

    # zh の残りを処理
    _flush_zh_batch()

    logger.info(f"[Summarizer] 要約完了: {sum(1 for s in results if s):d}/{n} 件")
    return results

# ---- スタンドアロン実行（手動テスト） ---------------------------------------
if __name__ == "__main__":
    samples: List[Tuple[str, str]] = [
        ("zh", "OpenAI推出了新模型。這個模型強調在推理上的效能提升，"
               "適用於多步驟思考場景。在基準測試中表現提升，並且計算成本更低。"),
        ("zh", "蘋果公司在加州舉辦發表會，公布最新處理器與筆電。能效比與圖形性能顯著提升，"
               "並強調本地端AI體驗。"),
        ("ja", "これは日本語のテキストです。テストとして数文を入れます。生成AIの応用が進んでいます。"),
        ("en", "NVIDIA announced a new GPU architecture with better efficiency and inference throughput."),
        ("",   ""),  # 空要素（順序維持で空文字を返す）
    ]
    outs = run_summarization(samples)
    for i, s in enumerate(outs, 1):
        print(f"\n--- Summary {i} ---\n{s}\n")
