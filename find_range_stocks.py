"""
find_range_stocks.py
=====================
One-off / occasional screener (run manually via GitHub Actions
workflow_dispatch) that looks across the S&P 500 for stocks that behave
like a "range trader's" dream: they oscillate between a fairly stable
low and high for months at a time, without a strong long-term up/down
trend, and with a wide-enough gap between the low and high to make
buying near the bottom and selling near the top worthwhile.

METHOD (per ticker, 2 years of daily closes):
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
     -> We want at least MIN_CYCLES round trips (default 4 over 2
        years, i.e. roughly 2/year): proof the stock actually revisits
        both ends of the range repeatedly, not just once or twice.
  5. half_drift_pct = how much the channel's midpoint has moved
     between the first and second half of the 2-year window.
     -> A real range should look similar in both halves. If the
        midpoint has drifted a lot, the "range" may just be an
        average of two different regimes (e.g. a range that only
        formed recently, or one that's dissolving) rather than a
        genuinely stable channel. We require this drift to stay small.
  6. lower_touches / upper_touches = how many SEPARATE times the price
     entered the bottom zone / top zone (not just total cycles, which
     could hide a stock that only really visited one side a lot and
     the other side barely). We require at least MIN_TOUCHES_EACH_BAND
     (default 3) touches of EACH side.
  7. time_in_zones_pct = % of all trading days spent near either
     extreme (within the outer 15% of the range on either side).
     -> This is the key "genuinely moves, isn't stuck" check: a stock
        could technically cross both bands a few times while spending
        95% of its days parked in the dead middle. We require a
        minimum share of days actively near an extreme.
  8. recent_touch = whether the price has actually touched the lower
     or upper band at least once in the last ~90 trading days.
     -> Without this, a stock could have ranged beautifully 18 months
        ago and be doing something completely different today.
  9. composite score = range_pct * cycles / (1 + abs(trend_pct)) /
     (1 + half_drift_pct / 10) * (1 + time_in_zones_pct / 100)
     -> rewards wide + oscillating + flat + stable + actively-moving
        stocks; penalizes trending, drifting, or "stuck" ones even if
        their range_pct looks big.

The top 10 stocks by composite score are:
  - appended to watchlist.json (target_high = high, target_low = low),
    skipping any ticker already present in the watchlist
  - summarized in a single Telegram message, including the cycle count,
    half-window drift, and trend, so you can see exactly why each one
    passed before trusting it

IMPORTANT CAVEATS (also sent in the Telegram message):
  - This is a backward-looking statistical pattern, not a prediction.
    A stock that has traded in a range for 2 years can break out of
    it (up or down) at any time - nothing here guarantees it won't.
    The extra checks (longer history, half-window stability, recency)
    raise confidence that the pattern is real and still active, but
    they cannot eliminate the risk of a future breakout.
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
LOOKBACK_PERIOD = "2y"      # longer history = more chances to prove the range repeats
LOW_PCTILE = 10
HIGH_PCTILE = 90
MIN_RANGE_PCT = 15.0        # ignore stocks whose channel is too narrow to bother with
MAX_TREND_PCT = 12.0        # ignore stocks that are mostly just trending, not ranging
MIN_CYCLES = 4              # over 2 years: at least ~2 full round-trips per year
MAX_HALF_DRIFT_PCT = 20.0   # channel midpoint can't drift more than this between the two halves
MIN_TRADING_DAYS = 300      # need close to a full 2 years of data to trust the pattern
RECENT_WINDOW_DAYS = 90     # must have touched a band recently - proves the range is still "live"
MIN_TOUCHES_EACH_BAND = 3   # must genuinely revisit BOTH the top and the bottom, not just once
MIN_TIME_IN_ZONES_PCT = 12.0  # % of days spent near an extreme - proof it actively moves, not stuck
TOP_N = 10
BATCH_SIZE = 60             # tickers per yf.download batch call

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


def analyze_touches(closes, low, high):
    """Count distinct touches of the lower/upper band separately, and how
    much of the whole window was spent near either extreme.

    A stock could technically pass the cycle count by drifting slowly
    from one end to the other just a handful of times. This looks
    instead at how often it actually revisits EACH side, and how much
    of its time is spent "working" the range rather than sitting in the
    dead middle doing nothing - i.e. proof it genuinely moves, not that
    it's stuck."""
    lower_zone = low + (high - low) * 0.15
    upper_zone = high - (high - low) * 0.15
    lower_touches = 0
    upper_touches = 0
    days_in_zone = 0
    in_lower = False
    in_upper = False
    for price in closes:
        if price <= lower_zone:
            days_in_zone += 1
            if not in_lower:
                lower_touches += 1
            in_lower, in_upper = True, False
        elif price >= upper_zone:
            days_in_zone += 1
            if not in_upper:
                upper_touches += 1
            in_lower, in_upper = False, True
        else:
            in_lower, in_upper = False, False
    return lower_touches, upper_touches, days_in_zone


def analyze_ticker(symbol, closes):
    closes = closes.dropna()
    n = len(closes)
    if n < MIN_TRADING_DAYS:
        return None

    low = float(np.percentile(closes, LOW_PCTILE))
    high = float(np.percentile(closes, HIGH_PCTILE))
    if low <= 0:
        return None
    range_pct = (high - low) / low * 100

    x = np.arange(n)
    slope, _ = np.polyfit(x, closes.values, 1)
    mean_price = float(closes.mean())
    trend_pct = (slope * n / mean_price) * 100

    cycles = count_cycles(closes.values, low, high)
    years = n / 252
    cycles_per_year = cycles / years

    lower_touches, upper_touches, days_in_zone = analyze_touches(closes.values, low, high)
    time_in_zones_pct = days_in_zone / n * 100

    # --- stability check: split the window in half and compare the two channels.
    # A real, tradeable range should look roughly the same in both halves, not
    # just in the period as a whole (which could be an average of two very
    # different regimes, or a range that only just formed).
    half = n // 2
    first_half = closes.iloc[:half]
    second_half = closes.iloc[half:]
    low1, high1 = np.percentile(first_half, LOW_PCTILE), np.percentile(first_half, HIGH_PCTILE)
    low2, high2 = np.percentile(second_half, LOW_PCTILE), np.percentile(second_half, HIGH_PCTILE)
    mid_full = (low + high) / 2
    mid1, mid2 = (low1 + high1) / 2, (low2 + high2) / 2
    half_drift_pct = abs(mid1 - mid2) / mid_full * 100

    # --- recency check: has the price actually touched a band recently?
    # Without this, a stock could have ranged beautifully 18 months ago and
    # then be doing something completely different now.
    recent = closes.iloc[-RECENT_WINDOW_DAYS:] if n >= RECENT_WINDOW_DAYS else closes
    lower_zone = low + (high - low) * 0.15
    upper_zone = high - (high - low) * 0.15
    recent_touch = bool((recent <= lower_zone).any() or (recent >= upper_zone).any())

    if (range_pct < MIN_RANGE_PCT or abs(trend_pct) > MAX_TREND_PCT or
            cycles < MIN_CYCLES or half_drift_pct > MAX_HALF_DRIFT_PCT or
            not recent_touch or
            lower_touches < MIN_TOUCHES_EACH_BAND or upper_touches < MIN_TOUCHES_EACH_BAND or
            time_in_zones_pct < MIN_TIME_IN_ZONES_PCT):
        return None

    # score rewards wide + oscillating + flat + stable-across-time stocks,
    # and further rewards stocks that spend more time actively working the range
    score = (range_pct * cycles) / (1 + abs(trend_pct)) / (1 + half_drift_pct / 10)
    score *= (1 + time_in_zones_pct / 100)

    return {
        "ticker": symbol,
        "low": round(low, 2),
        "high": round(high, 2),
        "range_pct": round(range_pct, 1),
        "trend_pct": round(trend_pct, 1),
        "cycles": cycles,
        "cycles_per_year": round(cycles_per_year, 1),
        "lower_touches": lower_touches,
        "upper_touches": upper_touches,
        "time_in_zones_pct": round(time_in_zones_pct, 1),
        "half_drift_pct": round(half_drift_pct, 1),
        "score": round(score, 1),
        "last_price": round(float(closes.iloc[-1]), 2),
        "days_analyzed": n,
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
        f"נבדקו {len(tickers)} מניות על פני 2 שנות מסחר, נמצאו {len(candidates)} מועמדות "
        f"שעונות על קריטריונים מחמירים (טווח יציב בשני חצאי התקופה + פעילות עדכנית).",
        "",
        f"עשרת המניות המובילות (מבין {len(candidates)}):",
        "",
    ]
    for c in top:
        status = "✅ נוספה למעקב" if c in added else "⏭ כבר ברשימה"
        lines.append(
            f"{c['ticker']}: טווח {c['low']}–{c['high']} (פער {c['range_pct']}%)\n"
            f"   {c['cycles']} מחזורים ({c['cycles_per_year']}/שנה) | "
            f"נגיעות: {c['lower_touches']} תחתונות / {c['upper_touches']} עליונות | "
            f"{c['time_in_zones_pct']}% מהזמן בקצוות\n"
            f"   סטיית יציבות {c['half_drift_pct']}%, מגמה {c['trend_pct']}% — {status}"
        )
    lines.append("")
    lines.append("⚠️ זה דפוס סטטיסטי לאחור, לא תחזית. מניה יכולה לפרוץ מהטווח בכל רגע. "
                  "בדוק כל מועמדת בעצמך לפני החלטת קנייה/מכירה.")

    send_telegram_message("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
