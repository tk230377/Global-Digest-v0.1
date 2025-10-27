#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scraper以降のパイプライン単体テスト
- Discover/Validator をスキップして手動URLを指定
- URLリストを貼り替えるだけでOK
"""

from scraper import run_scraping
from langdetector import run_language_detection_with_meta
from summarizer import run_summarization
from translator import run_translate_aligned as run_translation
from deliver import run_delivery

import os, json, time

TRACE_DIR = os.getenv("GD_TRACE_DIR", "./trace_runs")
TRACE_ENABLED = os.getenv("GD_TRACE", "1").strip() in ("1", "true", "yes")
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join(TRACE_DIR, RUN_ID)

if TRACE_ENABLED:
    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"[TRACE] 出力先: {RUN_DIR}")


# ======== ここに自分でURLを貼る ========
urls = [
    "http://www.nbd.com.cn/articles/2024-06-03/1234567.html",
    "https://wallstreetcn.com/articles/3754603",
]

# ======== Scraper ========
print("\n=== [C] Scraper 実行 ===")
scraped = run_scraping(urls, return_meta=True)
for i, meta in enumerate(scraped, 1):
    print(f"{i:02d}. ok={meta.get('ok')}, chars={meta.get('chars')}, url={meta.get('resolved_url')}")
    if not meta.get("ok"):
        print("  reason:", meta.get("reason"))

    if TRACE_ENABLED:
        with open(os.path.join(RUN_DIR, "C_scraper_meta.jsonl"), "w", encoding="utf-8") as f:
            for m in scraped:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")


# ======== LangDetector ========
print("\n=== [D] Language Detection ===")
ld_input = []
for item in scraped:
    text = item.get("text") or item.get("body") or ""
    if text.strip():
        ld_input.append({"text": text})
langtagged = run_language_detection_with_meta(ld_input)
print("Detected languages:", [x.get("lang") for x in langtagged])

# ======== Summarizer ========
print("\n=== [E] Summarizer ===")
pairs = [(x.get("lang", "zh"), x.get("text", "")) for x in langtagged]
summaries = run_summarization(pairs)
for s in summaries:
    print("----\n", s[:200], "\n----")

if TRACE_ENABLED:
    with open(os.path.join(RUN_DIR, "E_summaries_native.txt"), "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(s.strip() + "\n\n" + ("="*80) + "\n\n")


# ======== Translator ========
print("\n=== [F] Translator ===")
jp_summaries = run_translation(summaries)
for s in jp_summaries:
    print("----\n", s[:200], "\n----")

# ======== Delivery（メール送信 or ローカル保存） ========
print("\n=== [G] Delivery ===")
run_delivery(jp_summaries)
