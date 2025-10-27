#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Digest - Delivery Agent
- 日本語要約（配列）をメールで配信（テキスト＋HTML）
- 既定: Gmail SMTP/STARTTLS
- .env 未設定や送信失敗時は ./outbox/ に .eml を保存してフォールバック

依存:
  pip install python-dotenv

環境変数(.env):
  EMAIL_USER=your.email@gmail.com          # 送信元（Gmail推奨: アプリパスワード）
  EMAIL_PASS=xxxxxxxxxxxxxxxx              # アプリパスワード
  RECIPIENT_EMAIL=foo@example.com,bar@ex.com  # カンマ区切りで複数可
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587

  # 追加オプション（任意）
  DELIVER_SENDER_NAME=Global Digest Bot
  DELIVER_SUBJECT_PREFIX=Global Digest
  DELIVER_TIMEZONE=Asia/Tokyo
  DELIVER_MAX_ITEMS=20             # 1通あたりの要約件数上限（多すぎる場合は分割送信）
  DELIVER_DRY_RUN=false            # trueなら送信せずに .eml 保存のみ
  DELIVER_SAVE_PREVIEW=true        # 送信成功時も .eml を保存するか
  DELIVER_OUTBOX_DIR=outbox        # プレビュー保存先
"""

from __future__ import annotations
import os
import re
import ssl
import smtplib
import logging
from typing import List, Tuple
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from html import escape as html_escape
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# ---- .env 読み込み ----------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger("deliver")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---- 既定値/設定 -------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_PASS = os.getenv("EMAIL_PASS", "").strip()
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "").strip()

SENDER_NAME = os.getenv("DELIVER_SENDER_NAME", "Global Digest Bot").strip()
SUBJECT_PREFIX = os.getenv("DELIVER_SUBJECT_PREFIX", "Global Digest").strip()
TIMEZONE_NAME = os.getenv("DELIVER_TIMEZONE", "Asia/Tokyo").strip()
MAX_ITEMS = int(os.getenv("DELIVER_MAX_ITEMS", "20"))
DRY_RUN = os.getenv("DELIVER_DRY_RUN", "false").lower() in ("1", "true", "yes", "on")
SAVE_PREVIEW = os.getenv("DELIVER_SAVE_PREVIEW", "true").lower() in ("1", "true", "yes", "on")
OUTBOX_DIR = os.getenv("DELIVER_OUTBOX_DIR", "outbox").strip()

UA_FOOTER = "Global Digest – Agentic RPA Newsletter System"

# ---- ユーティリティ ----------------------------------------------------------

def _now_tz() -> datetime:
    if ZoneInfo and TIMEZONE_NAME:
        try:
            return datetime.now(ZoneInfo(TIMEZONE_NAME))
        except Exception:
            pass
    return datetime.now(timezone.utc)

def _split_recipients(s: str) -> List[str]:
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[,\s;]+", s) if p.strip()]
    # 簡易メール形式チェック（@を含む程度）
    return [p for p in parts if "@" in p]

def _chunk(lst: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [lst]
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def _build_subject(idx: int, total_parts: int, dt: datetime) -> str:
    date_str = dt.strftime("%Y-%m-%d")
    if total_parts > 1:
        return f"{SUBJECT_PREFIX} {date_str} ({idx}/{total_parts})"
    return f"{SUBJECT_PREFIX} {date_str}"

def _build_plain_text(items: List[str]) -> str:
    lines = []
    for i, s in enumerate(items, 1):
        s = (s or "").strip()
        lines.append(f"{i:02d}. {s}")
    lines.append("")
    lines.append(f"-- {UA_FOOTER}")
    return "\n".join(lines)

def _build_html(items: List[str]) -> str:
    lis = []
    for s in items:
        # 最低限のエスケープ
        safe = html_escape(s or "").replace("\n", "<br>")
        lis.append(f"<li style='margin-bottom:8px; line-height:1.6;'>{safe}</li>")
    html = f"""\
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{html_escape(SUBJECT_PREFIX)}</title>
  </head>
  <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans JP', Helvetica, Arial, 'Hiragino Kaku Gothic ProN', 'Meiryo', sans-serif; color:#222; background:#fafafa; padding:24px;">
    <div style="max-width:720px; margin:auto; background:#fff; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.06); padding:24px;">
      <h1 style="margin-top:0; font-size:20px;">{html_escape(SUBJECT_PREFIX)}</h1>
      <ol style="padding-left:20px; margin-top:16px;">
        {''.join(lis)}
      </ol>
      <hr style="margin:24px 0; border:none; border-top:1px solid #eee;">
      <p style="font-size:12px; color:#777;">{html_escape(UA_FOOTER)}</p>
    </div>
  </body>
</html>
"""
    return html

def _ensure_outbox() -> None:
    try:
        os.makedirs(OUTBOX_DIR, exist_ok=True)
    except Exception:
        pass

def _save_eml(msg: EmailMessage, prefix: str = "digest") -> str:
    _ensure_outbox()
    dt = _now_tz()
    fname = f"{prefix}_{dt.strftime('%Y%m%d_%H%M%S_%f')}.eml"
    path = os.path.join(OUTBOX_DIR, fname)
    with open(path, "wb") as f:
        f.write(bytes(msg))
    return path

def _send_smtp(msg: EmailMessage) -> None:
    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
        server.ehlo()
        server.starttls(context=ctx)
        server.ehlo()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# ---- 公開API ---------------------------------------------------------------

def run_delivery(jp_summaries: List[str]) -> None:
    """
    日本語要約（配列）をメール送信する。
    - 件数が多い場合は DELIVER_MAX_ITEMS ごとに分割して複数通送信
    - 送信失敗やドライラン時は ./outbox/ に .eml を保存
    """
    if not jp_summaries:
        logger.info("[Delivery] 入力が空のため送信しません。")
        return

    recipients = _split_recipients(RECIPIENT_EMAIL)
    if not recipients and not DRY_RUN:
        logger.warning("[Delivery] RECIPIENT_EMAIL が未設定です。DRY_RUN としてプレビュー保存に切替えます。")
        # 強制ドライラン
        dry = True
    else:
        dry = DRY_RUN

    parts = _chunk(jp_summaries, MAX_ITEMS if MAX_ITEMS > 0 else len(jp_summaries))
    total_parts = len(parts)
    now = _now_tz()

    sent_count = 0
    saved_paths: List[str] = []

    for idx, items in enumerate(parts, 1):
        subject = _build_subject(idx, total_parts, now)

        msg = EmailMessage()
        msg["Subject"] = subject
        from_disp = formataddr((SENDER_NAME, EMAIL_USER or "no-reply@example.com"))
        msg["From"] = from_disp
        if recipients:
            msg["To"] = ", ".join(recipients)
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain="global-digest.local")

        plain = _build_plain_text(items)
        html = _build_html(items)
        msg.set_content(plain)
        msg.add_alternative(html, subtype="html")

        # 送信 or プレビュー保存
        must_save = SAVE_PREVIEW or dry
        try:
            if dry:
                logger.info(f"[Delivery] DRY_RUN: 実送信せず .eml 保存のみ: subject='{subject}'")
            else:
                if not EMAIL_USER or not EMAIL_PASS:
                    raise RuntimeError("EMAIL_USER/EMAIL_PASS が未設定です。")
                _send_smtp(msg)
                sent_count += 1
                logger.info(f"[Delivery] 送信完了: subject='{subject}' 宛先={len(recipients)}件")
        except Exception as e:
            logger.warning(f"[Delivery] 送信失敗。プレビューにフォールバック: {e}")
            must_save = True

        if must_save:
            path = _save_eml(msg, prefix="digest")
            saved_paths.append(path)
            logger.info(f"[Delivery] プレビュー保存: {path}")

    logger.info(f"[Delivery] 完了: 送信 {sent_count} 通 / 分割 {total_parts} 通, プレビュー保存 {len(saved_paths)} 件")
    # 戻り値は不要（ログで確認）
    return


# スタンドアロン実行（手動テスト）
if __name__ == "__main__":
    samples = [
        "• OpenAIの新モデルが公開。推論最適化とコスト効率を両立。\n• 主要ベンチマークで前世代を上回る結果。",
        "1. Appleが新SoCとノートPCを発表。\n2. 省電力とGPU性能が大幅向上。オンデバイスAIを強調。",
        "・中国のEV市場で価格競争が再燃。補助金の動向が焦点に。"
    ]
    run_delivery(samples)
