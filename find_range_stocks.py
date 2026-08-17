"""
find_range_stocks.py
=====================
One-off / occasional screener (run manually via GitHub Actions
workflow_dispatch) that looks across the S&P 500 for stocks that behave
like a "range trader's" dream: they oscillate between a fairly stable
low and high for months at a time, without a strong long-term up/down
trend, and with a wide-enough gap between the low and high to make
buying near the bottom and selling near the top worthwhile.

METHOD (per ticker, ~9 months of daily closes):
  1. low  = 10th percentile of closing prices
     high = 90th percentile of closing prices
     (percentiles instead of absolute min/max so one freak spike/crash
     day doesn't distort the range)
  2. range_pct = (high - low) / low * 100
     -> how wide the channel is, in %. We want this reasonably large
        (a bigger gap = more profit potential per round-trip).
  3. trend_pct = slope of a linear regression of price vs. time,
     expressed as %-change-per-day relative to the average price,
     then multiplied by the number of trading days in the window.
     -> We want this close to zero: a flat/sideways stock, not one
        that is simply drifting up or down over the period (that's a
        trend, not a range).
  4. cycles = number of times the price swings from the lower band
     back up to the upper band and back down again over the window.
     -> We want at least 2 full cycles: proof the stock actually
        revisits both ends of the range repeatedly, not just once.
  5. composite score = range_pct * (cycles) / (1 + abs(trend_pct))
     -> rewards wide + oscillating + flat stocks; penalizes trending
        ones even if their range_pct looks big.

The top 10 stocks by composite score are:
  - appended to watchlist.json (target_high = high, target_low = low),
    skipping any ticker already present in the watchlist
  - summarized in a single Telegram message

IMPORTANT CAVEATS (also sent in the Telegram message):
  - This is a backward-looking statistical pattern, not a prediction.
    A stock that has traded in a range for 9 months can break out of
    it (up or down) at any time - nothing here guarantees it won't.
  - This is not financial advice. Review each candidate yourself
    before trusting the range for real buy/sell decisions.
"""

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

WATCHLIST_FILE = "watchlist.json"
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
LOOKBACK_PERIOD = "9mo"
LOW_PCTILE = 10
HIGH_PCTILE = 90
MIN_RANGE_PCT = 15.0       # ignore stocks whose channel is too narrow to bother with
MAX_TREND_PCT = 12.0       # ignore stocks that are mostly just trending, not ranging
MIN_CYCLES = 2
TOP_N = 10
BATCH_SIZE = 60            # tickers per yf.download batch call

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def send_telegram_message(text):
    import os
    token = os.environ.get(TELEGRAM_BOT_TOKEN_ENV)
    chat_id = os.environ.get(TELEGRAM_CHAT_ID_ENV)
    if not token or not chat_id:
        print("Telegram credentials missing; printing message instead:\n", text)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


def get_sp500_tickers():
    resp = requests.get(SP500_CSV_URL, timeout=20)
    resp.raise_for_status()
    lines = resp.text.splitlines()
    header = lines[0].split(",")
    symbol_idx = header.index("Symbol")
    tickers = []
    for line in lines[1:]:
        if not line.strip():
            continue
        symbol = line.split(",")[symbol_idx].strip()
        tickers.append(symbol.replace(".", "-"))  # BRK.B -> BRK-B for yfinance
    return tickers


def count_cycles(closes, low, high):
    """Count full lower-band -> upper-band -> lower-band round trips."""
    lower_zone = low + (high - low) * 0.15
    upper_zone = high - (high - low) * 0.15
    state = None  # "low" or "high"
    cycles = 0
    for price in closes:
        if price <= lower_zone:
            if state == "high":
                cycles += 1
            state = "low"
        elif price >= upper_zone:
            state = "high"
    return cycles


def analyze_ticker(symbol, closes):
    closes = closes.dropna()
    if len(closes) < 60:
        return None

    low = float(np.percentile(closes, LOW_PCTILE))
    high = float(np.percentile(closes, HIGH_PCTILE))
    if low <= 0:
        return None
    range_pct = (high - low) / low * 100

    x = np.arange(len(closes))
    slope, _ = np.polyfit(x, closes.values, 1)
    mean_price = float(closes.mean())
    trend_pct = (slope * len(closes) / mean_price) * 100

    cycles = count_cycles(closes.values, low, high)

    if range_pct < MIN_RANGE_PCT or abs(trend_pct) > MAX_TREND_PCT or cycles < MIN_CYCLES:
        return None

    score = range_pct * cycles / (1 + abs(trend_pct))
    return {
        "ticker": symbol,
        "low": round(low, 2),
        "high": round(high, 2),
        "range_pct": round(range_pct, 1),
        "trend_pct": round(trend_pct, 1),
        "cycles": cycles,
        "score": round(score, 1),
        "last_price": round(float(closes.iloc[-1]), 2),
    }


def scan_universe(tickers):
    results = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"Scanning batch {i // BATCH_SIZE + 1} ({len(batch)} tickers)...")
        try:
            data = yf.download(
                tickers=" ".join(batch), period=LOOKBACK_PERIOD, group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"Batch download error: {e}")
            continue

        for symbol in batch:
            try:
                closes = data[symbol]["Close"] if len(batch) > 1 else data["Close"]
            except Exception:
                continue
            try:
                result = analyze_ticker(symbol, closes)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
        time.sleep(1)  # be polite to Yahoo Finance between batches
    return results


def main():
    print("Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    print(f"Scanning {len(tickers)} tickers for range-bound behavior...")

    candidates = scan_universe(tickers)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[:TOP_N]

    if not top:
        send_telegram_message("🔍 סריקת מניות טווח: לא נמצאו מועמדות מתאימות הפעם.")
        return

    watchlist = load_json(WATCHLIST_FILE, [])
    existing_tickers = {row["ticker"] for row in watchlist}

    added = []
    for c in top:
        if c["ticker"] in existing_tickers:
            continue
        watchlist.append({
            "ticker": c["ticker"],
            "name": c["ticker"],
            "target_high": c["high"],
            "target_low": c["low"],
        })
        added.append(c)

    save_json(WATCHLIST_FILE, watchlist)

    lines = [
        "📊 סריקת מניות טווח (S&P 500) הושלמה",
        f"נבדקו {len(tickers)} מניות, נמצאו {len(candidates)} מועמדות שעונות על הקריטריונים.",
        "",
        f"עשרת המניות המובילות (מבין {len(candidates)}):",
        "",
    ]
    for c in top:
        status = "✅ נוספה למעקב" if c in added else "⏭ כבר ברשימה"
        lines.append(
            f"{c['ticker']}: טווח {c['low']}–{c['high']} "
            f"(פער {c['range_pct']}%, {c['cycles']} מחזורים, מגמה {c['trend_pct']}%) — {status}"
        )
    lines.append("")
    lines.append("⚠️ זה דפוס סטטיסטי לאחור, לא תחזית. מניה יכולה לפרוץ מהטווח בכל רגע. "
                  "בדוק כל מועמדת בעצמך לפני החלטת קנייה/מכירה.")

    send_telegram_message("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
