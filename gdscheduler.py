#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Scheduler (Production)
- CLIエントリ & 周期実行
- Discovery -> Validator -> Scraper -> LangDetector -> Summarizer -> Translator -> Delivery
- .env を読み込んで設定を利用

本版の主な変更点:
- Discovery は max_articles の倍率で“多めに”取得（GDS_DISCOVERY_MULTIPLIER）
- Scraper は return_meta=True で受け、title/body/text/chars などを保持
- LangDetector 用テキストは本文中心にサンプリング（LANGDETECT_* 環境変数）
- Summarizer へは (lang, body) を優先的に渡す
"""

from __future__ import annotations
import argparse
import logging
import os
import signal
import sys
import time
from typing import List, Tuple, Dict, Any

# ---- .env 読み込み ----------------------------------------------------------
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ---- エージェントのインポート -----------------------------------------------
from discover import run_discovery
from validator import run_validation
from scraper import run_scraping
from langdetector import run_language_detection_with_meta  # メタ付きI/F
from summarizer import run_summarization
from translator import run_translate_aligned as run_translation
from deliver import run_delivery

# ---- ロギング設定 -----------------------------------------------------------
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "global_digest.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("gdscheduler")

# ---- 環境変数（本スケジューラ用）-------------------------------------------
DISCOVERY_MULTIPLIER = int(os.getenv("GDS_DISCOVERY_MULTIPLIER", "3"))  # Discoveryへ渡す倍率
LD_SAMPLE_CHARS      = int(os.getenv("LANGDETECT_SAMPLE_CHARS", "2000"))  # 検出に使う本文上限
LD_MIN_BODY_CHARS    = int(os.getenv("LANGDETECT_MIN_BODY_CHARS", "300"))  # 短文判定しきい値

# =============================================================================
# ヘルパ
# =============================================================================
def _prepare_langdetect_items(scraped_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scraperメタ（title/body/text/...）から、LangDetector用の text を本文中心に再構成。
    - 本文(body)が LD_MIN_BODY_CHARS 未満なら title を前置して補強
    - 先頭 LD_SAMPLE_CHARS までに制限
    - ここでは配列長・順序を Scraper のまま維持（ok=False のスロットは text="" で渡す）
    """
    out: List[Dict[str, Any]] = []
    for item in scraped_items:
        # デフォルト
        sample_text = ""
        if item and item.get("ok"):
            body = (item.get("body") or "").strip()
            title = (item.get("title") or "").strip()
            if len(body) < LD_MIN_BODY_CHARS:
                sample_text = (title + "\n" + body).strip()[:LD_SAMPLE_CHARS]
            else:
                sample_text = body[:LD_SAMPLE_CHARS]
        # 元メタを浅いコピーして text を差し替え
        new_item = dict(item or {})
        new_item["text"] = sample_text
        out.append(new_item)
    return out

# =============================================================================
# パイプライン
# =============================================================================
def run_pipeline(interests: str, max_articles: int = 10) -> None:
    t0 = time.time()
    logger.info("=== Global Digest パイプライン開始 ===")

    # A. Discovery（多めに取る → Validatorで10件に絞る）
    disc_max = max(1, max_articles) * max(1, DISCOVERY_MULTIPLIER)
    urls: List[str] = run_discovery(interests, max_results=disc_max)
    logger.info(f"[A] Discovery: {len(urls)} URLs (requested up to {disc_max})")

    # B. Validator
    valid_urls: List[str] = run_validation(urls, return_meta=False)
    cap = int(os.getenv("VALIDATOR_MAX_RESULTS", "5"))
    logger.info(f"[B] Validator: {len(valid_urls)} valid URLs (<= {cap} by policy)")

    if not valid_urls:
        logger.warning("[B] 有効URLが0件のため、このサイクルは終了します。")
        logger.info(f"=== パイプライン完了（{time.time()-t0:.2f}s）===\n")
        return

    """
    for i, u in enumerate(valid_urls, start=1):
        logger.debug(f"[B→C] #{i:02d} URL to scrape: {u}")
    URLの表示

        """
    # C. Scraper（return_meta=True: title/body/text/chars/... を保持）
    scraped_items = run_scraping(valid_urls, return_meta=True)  

    if not scraped_items:
        logger.warning("スクレイピング結果が0件のため、終了します。")
        logger.info(f"=== パイプライン完了（{time.time()-t0:.2f}s）===\n")
        return

    ok_count = sum(1 for it in scraped_items if it.get("ok"))
    min_chars_gate = os.getenv("SCRAPER_MIN_TEXT_CHARS", "120")
    logger.info(f"[C] ok={ok_count}/{len(scraped_items)} (min_chars gate={min_chars_gate})")
   
    if not scraped_items:
        logger.warning("スクレイピング結果が0件のため、終了します。")
        logger.info(f"=== パイプライン完了（{time.time()-t0:.2f}s）===\n")
        return

    # C' スクレイピング統計（任意の簡易ログ）
    ok_count = sum(1 for it in scraped_items if it.get("ok"))
    logger.info(f"[C] ok={ok_count}/{len(scraped_items)} (min_chars gate={os.getenv('SCRAPER_MIN_TEXT_CHARS', '120')})")

    # D. Language Detector（本文中心に text を差し替えてから実行）
    ld_input_items = _prepare_langdetect_items(scraped_items)
    langtagged_items = run_language_detection_with_meta(ld_input_items, zh_only=False)
    logger.info(f"[D] LangDetect: {len(langtagged_items)} lang-tagged items")

    if not langtagged_items:
        logger.warning("対象言語テキストが0件のため、終了します。")
        logger.info(f"=== パイプライン完了（{time.time()-t0:.2f}s）===\n")
        return

    # D' 簡易統計
    nonempty = sum(1 for it in langtagged_items if (it.get("text","") or "").strip())
    zh_count = sum(1 for it in langtagged_items if (it.get("lang","") or "").startswith("zh"))
    logger.info(f"[D→E] texts_nonempty={nonempty}/{len(langtagged_items)} | zh_like={zh_count}")

    # E. Summarizer（原言語：本文優先、なければ text をフォールバック）
    #    -> (lang, text_for_summary) の配列を作る
    lang_pairs: List[Tuple[str, str]] = []
    for item in langtagged_items:
        lang = item.get("lang", "")
        # 要約は本文中心で。本文が無ければ検出用textを使う。
        src_text = (item.get("body") or "").strip() or (item.get("text") or "")
        lang_pairs.append((lang, src_text))
    native_summaries: List[str] = run_summarization(lang_pairs)
    logger.info(f"[E] Summarizer: {len(native_summaries)} summaries")

    if not native_summaries:
        logger.warning("要約が0件のため、終了します。")
        logger.info(f"=== パイプライン完了（{time.time()-t0:.2f}s）===\n")
        return

    # F. Translator（日本語化：順序維持・失敗は空文字のまま）
    jp_summaries: List[str] = run_translation(native_summaries, return_meta=False, zh_only=False)
    logger.info(f"[F] Translator: {len(jp_summaries)} JP summaries")

    if not jp_summaries:
        logger.warning("日本語要約が0件のため、終了します。")
        logger.info(f"=== パイプライン完了（{time.time()-t0:.2f}s）===\n")
        return

    # G. Delivery（既存I/Fを維持：必要なら後日 title/URL も渡すI/Fに拡張可能）
    run_delivery(jp_summaries)
    logger.info("[G] Delivery: done")

    elapsed = time.time() - t0
    logger.info(f"=== パイプライン完了（{elapsed:.2f}s）===\n")

# =============================================================================
# CLI / スケジューラループ
# =============================================================================

_shutdown = False

def _signal_handler(sig, frame):
    global _shutdown
    logger.info(f"受信シグナル: {sig}. 次回サイクル前に停止します。")
    _shutdown = True

# Ctrl+CやSIGTERMで優雅に停止
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global Digest Scheduler")
    p.add_argument("--interests", type=str, required=True,
                   help="日本語カンマ区切りの興味キーワード（例: '生成AI, ウェアラブル'）")
    p.add_argument("--interval", type=int, default=86_400,
                   help="次回実行までの秒数（デフォルト=86400=1日）")
    p.add_argument("--max-articles", type=int, default=10,
                   help="1サイクルで処理する記事の上限（Validator側の上限とも整合）")
    p.add_argument("--once", action="store_true",
                   help="1回だけ実行して終了（スケジュール無し）")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="ログレベル")
    return p.parse_args()

def main():
    args = parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    logger.info("=== Global Digest Scheduler 起動 ===")
    logger.info(f"interests={args.interests} | interval={args.interval}s | once={args.once} | max_articles={args.max_articles} | discovery_x{DISCOVERY_MULTIPLIER}")

    try:
        # 1回実行モード
        if args.once:
            run_pipeline(args.interests, max_articles=args.max_articles)
            return

        # 周期実行モード
        while not _shutdown:
            cycle_start = time.time()
            run_pipeline(args.interests, max_articles=args.max_articles)

            # 次回までスリープ（シグナルで中断可）
            wait = args.interval - (time.time() - cycle_start)
            if wait > 0:
                logger.info(f"次の実行まで {int(wait)} 秒待機します（Ctrl+Cで停止）")
                slept = 0
                while slept < wait and not _shutdown:
                    step = min(5, wait - slept)
                    time.sleep(step)
                    slept += step
            else:
                logger.info("処理に時間がかかったため、即座に次サイクルを開始します。")

    except Exception as e:
        logger.exception(f"致命的なエラーでスケジューラを終了します: {e}")
        sys.exit(1)
    finally:
        logger.info("=== Scheduler 終了 ===")

if __name__ == "__main__":
    main()
