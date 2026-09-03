"""
Stock price alert checker.

Two independent features:
1. Target-price alerts: reads watchlist.json, checks each stock against
   its target price range, sends a Telegram alert when crossed.
2. Market-wide big-move alerts: scans the WHOLE US market (via Yahoo
   Finance's day_gainers/day_losers screeners) and a curated list of
   major Israeli (TASE) stocks (ta_tickers.json), and reports any stock
   that moved more than MOVE_THRESHOLD_PCT in a day, grouped into a
   separate summary message per market. Sent at most once per day per
   stock/market. Each mover is cross-referenced against the prediction
   engine's most recent prior call for that ticker (direction + score),
   so you can see whether the big move matches what was predicted.

PREDICTION ENGINE (runs once/day): scores a broad universe (full S&P 500
+ TASE watchlist tickers + today's biggest US movers) on a mix of real
technical indicators (RSI, MACD, moving-average trend, volume trend),
fundamentals (analyst target upside, 52-week range position, short
interest), a 30-day run-up penalty, and an overall market-regime nudge
(is the S&P 500 itself in an uptrend or downtrend). This is NOT a news
or sentiment feed - there's no free, reliable way to score "what's in
the news" without a paid API, so that part is intentionally left out
rather than faked.

State (already-sent alerts) is kept in state.json so re-runs don't spam.
"""

import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).parent
WATCHLIST_FILE = BASE_DIR / "watchlist.json"
TA_TICKERS_FILE = BASE_DIR / "ta_tickers.json"
STATE_FILE = BASE_DIR / "state.json"
CURRENT_PRICES_FILE = BASE_DIR / "current_prices.json"
PREDICTIONS_FILE = BASE_DIR / "predictions.json"
STARRED_FILE = BASE_DIR / "starred.json"
MY_PORTFOLIO_FILE = BASE_DIR / "my_portfolio.json"

SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

MOVE_THRESHOLD_PCT = 10.0
US_SCREENER_COUNT = 250  # how many top gainers/losers to pull from Yahoo
PREDICTION_SCORE_THRESHOLD = 1.5  # "directionally significant" - used for market-breadth awareness
PREDICTION_STRONG_THRESHOLD = 3.0  # kept for reference/backwards compatibility, no longer drives selection
PREFILTER_THRESHOLD = 1.0   # only fetch fundamentals (slow) for tickers past this technical-only score
DAILY_TOP_PICKS_LIMIT = 25  # curated "strong" list is capped here - see run_predictions for why
PRICE_HISTORY_PERIOD = "1y"
BATCH_SIZE = 60

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def load_json(path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def send_telegram_message(text, parse_mode=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials, skipping send. Message was:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    resp = requests.post(url, data=payload)
    if not resp.ok:
        print(f"Failed to send Telegram message: {resp.status_code} {resp.text}")


TELEGRAM_MAX_CHARS = 3500  # keep well under Telegram's 4096-char hard limit


def send_telegram_message_chunked(header, lines, parse_mode=None, sep="\n"):
    """Sends `header` + `lines` (joined by `sep`) as one message, or as
    several numbered messages if it would exceed Telegram's length limit -
    otherwise a long list (e.g. 100+ US stocks) gets silently rejected
    (400 'message is too long') and nothing is sent at all."""
    chunks = []
    current = []
    current_len = len(header)
    for line in lines:
        added_len = len(line) + len(sep)
        if current and current_len + added_len > TELEGRAM_MAX_CHARS:
            chunks.append(current)
            current = []
            current_len = len(header)
        current.append(line)
        current_len += added_len
    if current:
        chunks.append(current)

    total = len(chunks)
    for i, chunk_lines in enumerate(chunks, start=1):
        part_header = header if total == 1 else f"{header} (חלק {i}/{total})"
        send_telegram_message(part_header + "\n\n" + sep.join(chunk_lines), parse_mode=parse_mode)


# ---------- target-price watchlist alerts ----------

def get_price_and_prev_close(ticker):
    stock = yf.Ticker(ticker)
    price = stock.fast_info.get("last_price")
    prev_close = stock.fast_info.get("previous_close")
    if price is None or prev_close is None:
        hist = stock.history(period="5d")
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None, None
        price = float(closes.iloc[-1]) if price is None else price
        prev_close = float(closes.iloc[-2]) if prev_close is None else prev_close
    return float(price), float(prev_close)


def price_crossed(price, target, direction):
    if direction == "above":
        return price >= target
    return price <= target


MARKET_INDICES = {
    "sp500": {"symbol": "^GSPC", "label": "S&P 500"},
    "ta125": {"symbol": "^TA125.TA", "label": 'מדד ת"א 125'},
    "usdils": {"symbol": "ILS=X", "label": "דולר/שקל"},
    "btc": {"symbol": "BTC-USD", "label": "ביטקוין"},
}


def get_market_indices():
    """Snapshot of key benchmarks, refreshed every 15 min alongside the
    watchlist prices - gives quick market/macro context at a glance."""
    result = {}
    for key, meta in MARKET_INDICES.items():
        try:
            price, prev_close = get_price_and_prev_close(meta["symbol"])
        except Exception as e:
            print(f"Error fetching market index {key} ({meta['symbol']}): {e}")
            continue
        if price is None:
            continue
        pct = ((price - prev_close) / prev_close * 100) if prev_close else None
        result[key] = {
            "label": meta["label"],
            "price": price,
            "pct_change": round(pct, 2) if pct is not None else None,
        }
    return result


def run_watchlist_alerts(state, prediction_store=None):
    watchlist = load_json(WATCHLIST_FILE, [])
    watchlist_tickers = {item["ticker"] for item in watchlist}

    starred = load_json(STARRED_FILE, [])
    curated_tickers = []
    if prediction_store:
        curated_tickers = (prediction_store.get("top_picks") or {}).get("tickers", [])
    # keeps the dynamic predictions list's price/% line fresh every 15 min,
    # same as the manual watchlist - without re-running the full daily engine
    extra_tickers = [
        t for t in dict.fromkeys(list(curated_tickers) + list(starred))
        if t not in watchlist_tickers
    ]

    current_prices = {}
    for item in watchlist:
        ticker = item["ticker"]
        name = item.get("name", ticker)
        target_high = item.get("target_high") or None
        target_low = item.get("target_low") or None

        try:
            price, prev_close = get_price_and_prev_close(ticker)
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            continue
        if price is None:
            print(f"No price data for {ticker}")
            continue

        current_prices[ticker] = {
            "price": price,
            "prev_close": prev_close,
            "pct_change": ((price - prev_close) / prev_close * 100) if prev_close else None,
        }

        for target, direction, key_suffix, label in (
            (target_high, "above", "above", "עלה מעל"),
            (target_low, "below", "below", "ירד מתחת ל"),
        ):
            if target is None:
                continue
            key = f"{ticker}_{key_suffix}_{target}"
            triggered_before = state.get(key, False)
            condition_now = price_crossed(price, target, direction)
            print(f"{name} ({ticker}): price={price:.2f}, {key_suffix}_target={target}, "
                  f"met={condition_now}, already_alerted={triggered_before}")

            if condition_now and not triggered_before:
                msg = (f"🔔 התראת מניה\n{name} ({ticker})\n"
                       f"{label} {target}\nמחיר נוכחי: {price:.2f}")
                send_telegram_message(msg)
                state[key] = True
            elif not condition_now and triggered_before:
                state[key] = False

    for ticker in extra_tickers:
        try:
            price, prev_close = get_price_and_prev_close(ticker)
        except Exception as e:
            print(f"Error fetching live price for {ticker}: {e}")
            continue
        if price is None:
            continue
        current_prices[ticker] = {
            "price": price,
            "prev_close": prev_close,
            "pct_change": ((price - prev_close) / prev_close * 100) if prev_close else None,
        }

    save_json(CURRENT_PRICES_FILE, {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prices": current_prices,
        "indices": get_market_indices(),
    })


# ---------- "התיק שלי" (my real holdings, manually maintained) ----------

def compute_my_portfolio_snapshot(holdings):
    """For each holding, try to fetch a live price. A ticker that can't be
    resolved at all (e.g. an Israeli mutual/index fund with no tradable
    ticker on Yahoo Finance) is marked ok=False and excluded from the
    value/weight/return math entirely - it is never silently included with
    missing or wrong data, per the agreed rule."""
    usdils_rate = None
    try:
        usdils_rate, _ = get_price_and_prev_close("ILS=X")
    except Exception as e:
        print(f"Error fetching USD/ILS rate for my-portfolio: {e}")

    rows = []
    for h in holdings:
        ticker = (h.get("ticker") or "").strip()
        qty = h.get("quantity")
        if not ticker or not qty:
            continue
        try:
            price, prev_close = get_price_and_prev_close(ticker)
        except Exception as e:
            print(f"my_portfolio: no price for {ticker}: {type(e).__name__}: {e}")
            price, prev_close = None, None

        if price is None:
            rows.append({"ticker": ticker, "name": h.get("name") or ticker, "ok": False})
            continue

        # TASE tickers (.TA) are quoted by Yahoo in Agorot, not Shekels
        # (1 Shekel = 100 Agorot) - divide by 100. Everything else is
        # assumed USD and converted at the live USD/ILS rate.
        israeli = is_israeli(ticker)
        unit_price_ils = (price / 100.0) if israeli else price * (usdils_rate or 1.0)
        value_ils = unit_price_ils * qty
        pct_change = ((price - prev_close) / prev_close * 100) if prev_close else None

        rows.append({
            "ticker": ticker, "name": h.get("name") or ticker, "ok": True,
            "price_native": round(price, 2),
            "value_ils": round(value_ils, 2),
            "pct_change": round(pct_change, 2) if pct_change is not None else None,
        })

    priced = [r for r in rows if r.get("ok") and r.get("pct_change") is not None]
    total_value = sum(r["value_ils"] for r in priced)
    for r in priced:
        r["weight_pct"] = round(r["value_ils"] / total_value * 100, 1) if total_value else None

    weighted_return = None
    if total_value:
        weighted_return = sum(r["value_ils"] * r["pct_change"] for r in priced) / total_value

    return {
        "rows": rows,
        "total_value_ils": round(total_value, 2) if total_value else None,
        "weighted_return_pct": round(weighted_return, 3) if weighted_return is not None else None,
        "usdils_rate": usdils_rate,
    }


def update_my_portfolio(store):
    """Tomer's real, manually-entered holdings (edited from the app or by
    sending me a screenshot - see MY_PORTFOLIO_FILE), refreshed every run
    (~15 min) alongside the watchlist. Tracks per-holding daily P&L plus a
    compounding index ('my_portfolio_sim') so the running total is never
    erased when a holding is sold and replaced with a new one - the index
    just keeps compounding forward, the same way portfolio_sim does for
    Top 10, and the new holding's returns simply join it going forward."""
    holdings = load_json(MY_PORTFOLIO_FILE, [])
    snapshot = compute_my_portfolio_snapshot(holdings)
    store["my_portfolio_snapshot"] = {
        **snapshot,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    sim = store.setdefault("my_portfolio_sim", {
        "value": 100.0, "pending_date": None, "pending_return_pct": None, "daily_log": [],
    })

    if snapshot["weighted_return_pct"] is None:
        return  # nothing priced right now (e.g. holdings list is empty) - leave the index untouched

    today_str = date.today().isoformat()
    pending_date = sim.get("pending_date")

    if pending_date is None or pending_date == today_str:
        # first run ever, or still the same day - keep refreshing the "live"
        # return for today without compounding it into the index yet.
        sim["pending_date"] = today_str
        sim["pending_return_pct"] = snapshot["weighted_return_pct"]
    else:
        # a new day has started - lock in the last-seen return for the
        # previous day, exactly once, then start tracking today fresh.
        sim["value"] = sim["value"] * (1 + (sim["pending_return_pct"] or 0) / 100)
        sim["daily_log"].append({
            "date": pending_date, "return_pct": sim["pending_return_pct"], "value": round(sim["value"], 3),
        })
        sim["daily_log"] = sim["daily_log"][-90:]
        sim["pending_date"] = today_str
        sim["pending_return_pct"] = snapshot["weighted_return_pct"]

    sim["total_return_pct"] = round((sim["value"] / 100 - 1) * 100, 2)


# ---------- market-wide big-move alerts ----------

def get_us_movers(threshold, today, state, prediction_store):
    movers = []
    seen_symbols = set()
    for screen_name in ["day_gainers", "day_losers"]:
        try:
            result = yf.screen(screen_name, count=US_SCREENER_COUNT)
            quotes = result.get("quotes", [])
        except Exception as e:
            print(f"Error running US screener '{screen_name}': {e}")
            continue
        for q in quotes:
            symbol = q.get("symbol")
            pct = q.get("regularMarketChangePercent")
            price = q.get("regularMarketPrice")
            name = q.get("shortName") or symbol
            if not symbol or pct is None or symbol in seen_symbols:
                continue
            if abs(pct) >= threshold:
                move_key = f"US_{symbol}_bigmove_{today}"
                if not state.get(move_key, False):
                    line = f"{name} ({symbol}): {pct:+.1f}% (מחיר: {price})"
                    line += "\n" + format_prediction_match(prediction_store, symbol, today, pct)
                    movers.append(line)
                    state[move_key] = True
                seen_symbols.add(symbol)
    return movers


def get_il_movers(threshold, today, state, prediction_store):
    tickers = load_json(TA_TICKERS_FILE, [])
    if not tickers:
        return []
    movers = []
    try:
        data = yf.download(
            tickers=" ".join(tickers), period="5d", group_by="ticker",
            threads=True, progress=False, auto_adjust=False,
        )
    except Exception as e:
        print(f"Error batch-downloading TASE tickers: {e}")
        return []

    for ticker in tickers:
        try:
            closes = data[ticker]["Close"].dropna() if len(tickers) > 1 else data["Close"].dropna()
            if len(closes) < 2:
                continue
            price = float(closes.iloc[-1])
            prev_close = float(closes.iloc[-2])
        except Exception as e:
            print(f"No usable data for {ticker}: {e}")
            continue

        pct = (price - prev_close) / prev_close * 100
        if abs(pct) >= threshold:
            move_key = f"IL_{ticker}_bigmove_{today}"
            if not state.get(move_key, False):
                line = f"{ticker}: {pct:+.1f}% (מ-{prev_close:.2f} ל-{price:.2f})"
                line += "\n" + format_prediction_match(prediction_store, ticker, today, pct)
                movers.append(line)
                state[move_key] = True
    return movers


def run_market_wide_alerts(state, prediction_store):
    today = date.today().isoformat()

    us_movers = get_us_movers(MOVE_THRESHOLD_PCT, today, state, prediction_store)
    if us_movers:
        send_telegram_message_chunked(
            f"📈📉 תנודה חדה - שוק ארה\"ב (מעל {MOVE_THRESHOLD_PCT:.0f}%)", us_movers, sep="\n\n",
        )

    il_movers = get_il_movers(MOVE_THRESHOLD_PCT, today, state, prediction_store)
    if il_movers:
        send_telegram_message_chunked(
            f"📈📉 תנודה חדה - בורסת תל אביב (מעל {MOVE_THRESHOLD_PCT:.0f}%)", il_movers, sep="\n\n",
        )


def is_israeli(ticker):
    return ticker.upper().endswith(".TA")


def find_last_prediction(prediction_store, ticker, today):
    """Most recent prediction made for this ticker before today, if any."""
    candidates = [
        e for e in prediction_store.get("history", [])
        if e.get("ticker") == ticker and e.get("date") and e["date"] < today
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda e: e["date"])
    return candidates[-1]


def format_prediction_match(prediction_store, ticker, today, actual_pct):
    """Short line noting whether a big move today matches a prior prediction."""
    pred = find_last_prediction(prediction_store, ticker, today)
    if pred is None:
        return "   🔮 לא נמצאה תחזית קודמת למניה זו"
    actual_direction = "up" if actual_pct >= 0 else "down"
    matched = pred["predicted"] == actual_direction
    direction_he = "עלייה" if pred["predicted"] == "up" else "ירידה"
    mark = "✅ תואם" if matched else "❌ לא תואם"
    strength = " 🔥 חזקה" if pred.get("strong") else ""
    return f"   🔮 נחזה ב-{pred['date']}: {direction_he}{strength} (ציון {pred['score']}) {mark}"


def classify_starred_status(entry):
    """Buy/sell status label for a starred stock, based on the same
    support/resistance logic used for the visual range bar in the app."""
    price = entry.get("price")
    support = entry.get("support")
    resistance = entry.get("resistance")

    if price is None or (support is None and resistance is None):
        return "⚠️ אין מספיק נתונים לניתוח טווח כרגע"

    if support is not None and resistance is not None and resistance > support:
        pos = (price - support) / (resistance - support)  # 0 = at support, 1 = at resistance
        if price <= support:
            return f"🟢 מומלצת לקנייה - מתחת לתמיכה ({support})"
        elif pos <= 0.2:
            return f"🟡 קרובה לקנייה - ליד תמיכה ({support})"
        elif price >= resistance:
            return f"🔴 מומלצת למכירה/שורט - מעל ההתנגדות ({resistance})"
        elif pos >= 0.8:
            return f"🟠 קרובה למכירה - ליד ההתנגדות ({resistance})"
        else:
            return "⚪ באמצע הטווח, אין איתות ברור כרגע"

    if resistance is not None:
        if price >= resistance:
            return f"🔴 מומלצת למכירה/שורט - מעל ההתנגדות ({resistance})"
        pct_away = (resistance - price) / price * 100
        if pct_away <= 3:
            return f"🟠 קרובה למכירה - ליד ההתנגדות ({resistance})"
        return f"⚪ מתחת להתנגדות ({resistance})"

    # only support is known
    if price <= support:
        return f"🟢 מומלצת לקנייה - מתחת לתמיכה ({support})"
    pct_away = (price - support) / price * 100
    if pct_away <= 3:
        return f"🟡 קרובה לקנייה - ליד תמיכה ({support})"
    return f"⚪ מעל התמיכה ({support})"


def send_starred_report(today_entries):
    """Special daily Telegram digest for ⭐ starred stocks - independent of
    the curated top-picks list, so a starred stock always gets a status
    update even if it doesn't make today's top 25."""
    starred = load_json(STARRED_FILE, [])
    if not starred:
        return
    entries_by_ticker = {e["ticker"]: e for e in today_entries}
    lines = []
    for ticker in starred:
        entry = entries_by_ticker.get(ticker)
        if not entry:
            lines.append(f"*{ticker}*\n⚠️ אין נתונים היום (בעיית הורדת נתונים)")
            continue
        price = entry.get("price")
        price_txt = f"{price:.2f}" if price is not None else "—"
        status = classify_starred_status(entry)
        lines.append(f"*{ticker}* (מחיר: {price_txt})\n{status}")
    if lines:
        send_telegram_message_chunked(
            "⭐ עדכון יומי - מניות במעקב", lines, parse_mode="Markdown", sep="\n\n",
        )


# ---------- next-day prediction engine (with self-grading / learning) ----------
# Uses only signals knowable BEFORE a move happens (not post-event facts like
# "beat earnings" or "M&A rumor" - those are only known in hindsight).

def compute_rsi_series(closes, period=14):
    """Full RSI series (not just the latest value) - needed so divergence
    checks below can compare RSI at two different past points in time."""
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_rsi(closes, period=14):
    valid = compute_rsi_series(closes, period).dropna()
    return float(valid.iloc[-1]) if len(valid) else None


def compute_macd_series(closes):
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line


def compute_macd_bullish(closes):
    if len(closes) < 26:
        return None
    macd_line, signal_line = compute_macd_series(closes)
    return bool(macd_line.iloc[-1] > signal_line.iloc[-1])


def compute_obv(closes, volumes):
    """On-Balance Volume: running total of volume, added on up days and
    subtracted on down days - a proxy for whether real participation is
    confirming a price move or not."""
    vol = volumes.reindex(closes.index).fillna(0) if volumes is not None else pd.Series(0, index=closes.index)
    direction = np.sign(closes.diff().fillna(0))
    return (direction * vol).cumsum()


def compute_divergences(closes, rsi_series, macd_line, obv_series, pivot_highs_idx, pivot_lows_idx):
    """LEADING (not lagging) indicators: compares the two most recent swing
    pivots already found for support/resistance. If price sets a new higher
    high but RSI/MACD/OBV do NOT confirm it with their own higher high, that's
    a classic early warning that the up-move is losing strength BEFORE price
    itself turns over (bearish divergence) - mirror logic for lower lows
    (bullish divergence). This is a leading complement to the stop-loss /
    support-break checks elsewhere, which only fire after the fact."""
    result = {
        "rsi_bearish_div": False, "rsi_bullish_div": False,
        "macd_bearish_div": False, "macd_bullish_div": False,
        "obv_bearish_div": False, "obv_bullish_div": False,
    }
    n = len(closes)

    def valid_pair(idx_list):
        cands = [i for i in idx_list if 0 <= i < n]
        return cands[-2:] if len(cands) >= 2 else None

    highs2 = valid_pair(pivot_highs_idx)
    if highs2:
        i1, i2 = highs2
        if closes.iloc[i2] > closes.iloc[i1]:  # price: higher high
            if pd.notna(rsi_series.iloc[i1]) and pd.notna(rsi_series.iloc[i2]) and rsi_series.iloc[i2] < rsi_series.iloc[i1]:
                result["rsi_bearish_div"] = True
            if pd.notna(macd_line.iloc[i1]) and pd.notna(macd_line.iloc[i2]) and macd_line.iloc[i2] < macd_line.iloc[i1]:
                result["macd_bearish_div"] = True
            if pd.notna(obv_series.iloc[i1]) and pd.notna(obv_series.iloc[i2]) and obv_series.iloc[i2] < obv_series.iloc[i1]:
                result["obv_bearish_div"] = True

    lows2 = valid_pair(pivot_lows_idx)
    if lows2:
        i1, i2 = lows2
        if closes.iloc[i2] < closes.iloc[i1]:  # price: lower low
            if pd.notna(rsi_series.iloc[i1]) and pd.notna(rsi_series.iloc[i2]) and rsi_series.iloc[i2] > rsi_series.iloc[i1]:
                result["rsi_bullish_div"] = True
            if pd.notna(macd_line.iloc[i1]) and pd.notna(macd_line.iloc[i2]) and macd_line.iloc[i2] > macd_line.iloc[i1]:
                result["macd_bullish_div"] = True
            if pd.notna(obv_series.iloc[i1]) and pd.notna(obv_series.iloc[i2]) and obv_series.iloc[i2] > obv_series.iloc[i1]:
                result["obv_bullish_div"] = True

    return result


def compute_adx(highs, lows, closes, period=14):
    """Wilder's ADX - how STRONG the current trend is (not its direction).
    A declining ADX means the trend (up or down) is losing steam even while
    price hasn't reversed yet - another leading signal, distinct from the
    divergence checks above."""
    try:
        highs = highs.dropna()
        lows = lows.dropna()
        closes_local = closes.dropna()
        n = min(len(highs), len(lows), len(closes_local))
        if n < period * 3:
            return None, None
        highs = highs.iloc[-n:].reset_index(drop=True)
        lows = lows.iloc[-n:].reset_index(drop=True)
        c = closes_local.iloc[-n:].reset_index(drop=True)

        prev_close = c.shift(1)
        tr = pd.concat([highs - lows, (highs - prev_close).abs(), (lows - prev_close).abs()], axis=1).max(axis=1)
        up_move = highs.diff()
        down_move = -lows.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx_series = dx.ewm(alpha=1 / period, adjust=False).mean().dropna()

        if len(adx_series) < 2:
            return None, None
        adx_now = float(adx_series.iloc[-1])
        if not np.isfinite(adx_now):
            return None, None
        lookback = 6 if len(adx_series) >= 6 else len(adx_series) - 1
        adx_prev = float(adx_series.iloc[-1 - lookback])
        weakening = bool(np.isfinite(adx_prev) and adx_now < adx_prev)
        return round(adx_now, 1), weakening
    except Exception as e:
        print(f"ADX calc failed: {type(e).__name__}: {e}")
        return None, None


def compute_support_resistance(closes, price):
    """Finds the nearest meaningful swing-low (support - a buy zone, since
    a bounce is more likely there) and swing-high (resistance - a sell/short
    zone, since a pullback is more likely there) using local pivots over the
    trailing ~1 year of daily closes.

    A stock breaking out above every recent pivot high (new highs) has no
    historical ceiling left to reference; the old logic of "sell near the
    top" stops applying. In that case we don't just leave it blank - we
    project a level using recent volatility, and flag it as projected so
    the dashboard can show it differently (e.g. "no ceiling yet" instead of
    a hard number). Same idea in reverse for support below all recent lows.
    """
    vals = closes.values
    n = len(vals)
    pivot_window = 5  # a local extreme must beat +/- this many days on both sides
    pivot_highs_idx, pivot_lows_idx = [], []
    for i in range(pivot_window, n - pivot_window):
        seg = vals[i - pivot_window:i + pivot_window + 1]
        if vals[i] >= seg.max() and vals[i] > vals[i - pivot_window] and vals[i] > vals[i + pivot_window]:
            pivot_highs_idx.append(i)
        if vals[i] <= seg.min() and vals[i] < vals[i - pivot_window] and vals[i] < vals[i + pivot_window]:
            pivot_lows_idx.append(i)
    pivot_highs = [float(vals[i]) for i in pivot_highs_idx]
    pivot_lows = [float(vals[i]) for i in pivot_lows_idx]

    std_raw = pd.Series(vals).pct_change().tail(60).std()
    recent_std = float(std_raw) if pd.notna(std_raw) and np.isfinite(std_raw) else 0.0
    fallback_move = max(recent_std * 8, 0.08)  # at least 8%, or 8x the recent daily volatility

    above = [p for p in pivot_highs if p > price]
    resistance_projected = not above
    resistance = min(above) if above else price * (1 + fallback_move)

    below = [p for p in pivot_lows if p < price]
    support_projected = not below
    support = max(below) if below else price * (1 - fallback_move)

    # guard against any leftover non-finite value - round() raises
    # OverflowError on inf, which the caller's try/except was silently
    # swallowing, meaning support/resistance quietly went missing entirely.
    if not np.isfinite(resistance):
        resistance = price * 1.08
        resistance_projected = True
    if not np.isfinite(support):
        support = price * 0.92
        support_projected = True

    return {
        "resistance": round(float(resistance), 2),
        "resistance_projected": bool(resistance_projected),
        "support": round(float(support), 2),
        "support_projected": bool(support_projected),
        "_pivot_highs_idx": pivot_highs_idx,
        "_pivot_lows_idx": pivot_lows_idx,
    }


CHART_MAX_POINTS = 140


def build_chart_payload(closes, factors):
    """Compact chart data (recent closes + which of them are the pivot
    points behind the support/resistance levels) - only built for tickers
    we'll actually attach it to (curated + starred), to keep the file small."""
    vals = closes.values
    n = len(vals)
    start = max(0, n - CHART_MAX_POINTS)
    trimmed = vals[start:]
    highs_idx = [i - start for i in factors.get("_pivot_highs_idx", []) if i >= start]
    lows_idx = [i - start for i in factors.get("_pivot_lows_idx", []) if i >= start]
    return {
        "closes": [round(float(v), 2) for v in trimmed],
        "pivot_highs": highs_idx,
        "pivot_lows": lows_idx,
        "resistance": factors.get("resistance"),
        "support": factors.get("support"),
        "resistance_projected": factors.get("resistance_projected"),
        "support_projected": factors.get("support_projected"),
    }


def compute_technical_factors(closes, volumes, highs=None, lows=None):
    """Everything derivable from price/volume history alone - no network
    calls per ticker, so this is cheap enough to run on the whole universe.
    highs/lows are optional (only needed for ADX) - they come from data
    already downloaded for closes/volumes, no extra network cost."""
    closes = closes.dropna()
    n = len(closes)
    if n < 60:
        return None

    price = float(closes.iloc[-1])
    window = closes.iloc[-252:] if n >= 252 else closes
    year_high, year_low = float(window.max()), float(window.min())
    range_pos = ((price - year_low) / (year_high - year_low) * 100) if year_high > year_low else None

    run_up_30d = None
    if n >= 22:
        run_up_30d = float((closes.iloc[-1] - closes.iloc[-22]) / closes.iloc[-22] * 100)

    rsi = compute_rsi(closes)
    macd_bullish = compute_macd_bullish(closes)

    ma_trend = None
    if n >= 200:
        sma50 = float(closes.rolling(50).mean().iloc[-1])
        sma200 = float(closes.rolling(200).mean().iloc[-1])
        if price > sma50 > sma200:
            ma_trend = "golden"
        elif price < sma50 < sma200:
            ma_trend = "death"
        else:
            ma_trend = "mixed"

    vol_ratio = None
    if volumes is not None:
        volumes = volumes.dropna()
        if len(volumes) >= 65:
            recent_vol = float(volumes.iloc[-5:].mean())
            base_vol = float(volumes.iloc[-65:-5].mean())
            if base_vol > 0:
                vol_ratio = recent_vol / base_vol

    factors = {
        "price": price,
        "range_pos": range_pos,
        "run_up_30d": run_up_30d,
        "rsi": rsi,
        "macd_bullish": macd_bullish,
        "ma_trend": ma_trend,
        "vol_ratio": vol_ratio,
    }

    try:
        factors.update(compute_support_resistance(closes, price))
    except Exception as e:
        print(f"Support/resistance calc failed for this ticker: {type(e).__name__}: {e}")

    # --- leading-indicator layer (divergences + ADX) - see compute_divergences
    # and compute_adx docstrings. Wrapped defensively so a failure here never
    # takes down the whole technical-factors calc for a ticker. ---
    try:
        rsi_series = compute_rsi_series(closes)
        macd_line, _ = compute_macd_series(closes)
        obv_series = compute_obv(closes, volumes)
        factors.update(compute_divergences(
            closes, rsi_series, macd_line, obv_series,
            factors.get("_pivot_highs_idx", []), factors.get("_pivot_lows_idx", []),
        ))
    except Exception as e:
        print(f"Divergence calc failed for this ticker: {type(e).__name__}: {e}")

    if highs is not None and lows is not None:
        try:
            adx, adx_weakening = compute_adx(highs, lows, closes)
            factors["adx"] = adx
            factors["adx_weakening"] = adx_weakening
        except Exception as e:
            print(f"ADX attach failed for this ticker: {type(e).__name__}: {e}")

    return factors


def get_fundamental_factors(ticker):
    """Analyst target + short interest - only fetched for tickers that
    already look interesting technically, since .info calls are slow."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    target_mean = info.get("targetMeanPrice")
    upside_pct = None
    if target_mean and price:
        upside_pct = (target_mean - price) / price * 100

    short_pct = info.get("shortPercentOfFloat")
    if short_pct is not None:
        short_pct = short_pct * 100

    # analyst consensus: recommendationMean is 1 (Strong Buy) .. 5 (Strong Sell)
    recommendation_mean = info.get("recommendationMean")
    analyst_count = info.get("numberOfAnalystOpinions")

    # company name/summary come along for free from the same .info call -
    # underscore-prefixed so the bulk per-ticker entry copy skips them (only
    # curated/starred tickers get these attached, same as chart data, to
    # keep the file small)
    name = info.get("longName") or info.get("shortName")
    summary = info.get("longBusinessSummary")
    if summary and len(summary) > 700:
        summary = summary[:697].rsplit(" ", 1)[0] + "…"

    return {
        "upside_pct": upside_pct,
        "short_pct": short_pct,
        "recommendation_mean": recommendation_mean,
        "analyst_count": analyst_count,
        "sector": info.get("sector"),
        "_company_name": name,
        "_business_summary": summary,
    }


def get_latest_news(ticker):
    """Most recent headline + link for this ticker, so the dashboard can
    show a real, current report a person can read themselves - this is
    NOT a sentiment score (see module docstring: no reliable free way to
    score news), just a pointer to what to go read."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    if not items:
        return None, None
    top = items[0]
    content = top.get("content", top)  # yfinance news schema has varied across versions
    title = content.get("title") or top.get("title")
    link = (
        (content.get("canonicalUrl") or {}).get("url")
        or (content.get("clickThroughUrl") or {}).get("url")
        or top.get("link")
    )
    if not title or not link:
        return None, None
    return title, link


def _flatten_close_series(closes):
    """yfinance sometimes returns a single-column DataFrame (MultiIndex
    columns) even for a single ticker, instead of a plain Series. Always
    normalize to a 1-D Series so downstream .rolling()/.iloc[-1] work."""
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    return closes


def get_market_regime():
    """Is the overall market (S&P 500) trending up or down right now?
    Used as a small nudge on every score, not an override - a stock can
    still score bullish in a down market and vice versa."""
    try:
        data = yf.download("SPY", period="1y", progress=False, auto_adjust=True)
        spy = _flatten_close_series(data["Close"]).dropna()
        if len(spy) < 60:
            return {"bullish": None, "recent_10d_pct": None}
        sma50 = float(spy.rolling(50).mean().iloc[-1])
        sma200 = float(spy.rolling(200).mean().iloc[-1]) if len(spy) >= 200 else float(spy.mean())
        recent_pct = float((spy.iloc[-1] - spy.iloc[-10]) / spy.iloc[-10] * 100) if len(spy) >= 10 else 0.0
        return {"bullish": bool(sma50 > sma200), "recent_10d_pct": round(recent_pct, 2)}
    except Exception as e:
        print(f"Market regime check failed: {e}")
        return {"bullish": None, "recent_10d_pct": None}


def compute_prediction_score(factors, market_regime):
    # split into two clusters that pull in opposite directions, so they can
    # be reconciled instead of just cancelling each other out below
    trend_score = 0.0
    reversion_score = 0.0

    # --- trend / momentum: bets WITH the direction the stock is already moving ---
    if factors.get("ma_trend") == "golden":
        trend_score += 1.5
    elif factors.get("ma_trend") == "death":
        trend_score -= 1.5

    if factors.get("macd_bullish") is not None:
        trend_score += 1.2 if factors["macd_bullish"] else -1.2

    if factors.get("vol_ratio") is not None and factors.get("run_up_30d") is not None:
        if factors["vol_ratio"] >= 1.5:
            # a volume surge CONFIRMS whatever direction the stock is already moving in
            trend_score += 1.0 if factors["run_up_30d"] > 0 else -1.0

    # --- mean-reversion: bets AGAINST an overextended move ---
    if factors.get("rsi") is not None:
        if factors["rsi"] <= 30:
            reversion_score += 1.5  # oversold
        elif factors["rsi"] >= 70:
            reversion_score -= 1.5  # overbought

    # "near a 52-week extreme" is only treated as a reversion signal when the
    # trend cluster ISN'T already confirming a continuation in that same
    # direction - otherwise a healthy uptrend at new highs gets unfairly
    # marked down for the very thing that makes it strong.
    if factors.get("range_pos") is not None:
        if factors["range_pos"] >= 80 and trend_score <= 0:
            reversion_score -= 2
        elif factors["range_pos"] <= 20 and trend_score >= 0:
            reversion_score += 2

    if factors.get("run_up_30d") is not None:
        # gentler and capped, so one big prior move can't erase a genuinely
        # strong trend reading above
        capped_runup = max(-40, min(40, factors["run_up_30d"]))
        reversion_score -= capped_runup * 0.03

    # proximity to support/resistance: being close to a floor tilts bullish
    # (bounce more likely there), close to a ceiling tilts bearish (pullback
    # more likely) - a smaller, complementary nudge to the range_pos signal
    # above, using the actual pivot-based levels instead of the simple
    # 52-week high/low.
    price = factors.get("price")
    support, resistance = factors.get("support"), factors.get("resistance")
    if price and support and resistance and resistance > support:
        pos_in_channel = (price - support) / (resistance - support)  # 0 = at support, 1 = at resistance
        if pos_in_channel <= 0.1 and not factors.get("support_projected"):
            reversion_score += 1.0
        elif pos_in_channel >= 0.9 and not factors.get("resistance_projected"):
            reversion_score -= 1.0

    score = trend_score + reversion_score

    # --- fundamentals ---
    if factors.get("upside_pct") is not None:
        if factors["upside_pct"] >= 15:
            score += 2  # analysts see a lot of room above current price -> bullish
        elif factors["upside_pct"] <= -10:
            score -= 2  # already trading above target -> priced for perfection

    if factors.get("short_pct") is not None:
        score += min(factors["short_pct"], 30) * 0.05  # short-squeeze potential, capped

    if factors.get("recommendation_mean") is not None and (factors.get("analyst_count") or 0) >= 5:
        # recommendationMean: 1 = Strong Buy ... 5 = Strong Sell. Only trust this
        # when enough analysts actually cover the stock (>=5), otherwise a single
        # analyst's opinion could swing it unreliably.
        rec = factors["recommendation_mean"]
        if rec <= 2.0:
            score += 1.5
        elif rec >= 4.0:
            score -= 1.5

    # --- market regime: nudges the whole score, and additionally scales the
    # trend cluster specifically, since momentum strategies tend to work
    # better when the broader market is confirming the same direction ---
    if market_regime.get("bullish") is not None:
        regime_sign = 1 if market_regime["bullish"] else -1
        score += trend_score * regime_sign * 0.15
        score += 0.3 * regime_sign

    return round(score, 2)


def get_sp500_tickers():
    try:
        resp = requests.get(SP500_CSV_URL, timeout=20)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        header = lines[0].split(",")
        symbol_idx = header.index("Symbol")
        return [line.split(",")[symbol_idx].strip().replace(".", "-")
                for line in lines[1:] if line.strip()]
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return []


def build_prediction_universe():
    """A STABLE core universe (full S&P 500 + TASE list + watchlist) so
    that yesterday's predictions actually overlap with today's movers,
    plus today's biggest movers added on top for extra same-day coverage."""
    tickers = set(get_sp500_tickers())
    for item in load_json(WATCHLIST_FILE, []):
        tickers.add(item["ticker"])
    for t in load_json(TA_TICKERS_FILE, []):
        tickers.add(t)
    for t in load_json(STARRED_FILE, []):
        tickers.add(t)
    try:
        existing_store = load_json(PREDICTIONS_FILE, {})
        for h in (existing_store.get("monthly_portfolio") or {}).get("holdings", []):
            tickers.add(h["ticker"])
    except Exception:
        pass
    for screen_name in ["day_gainers", "day_losers", "most_actives"]:
        try:
            result = yf.screen(screen_name, count=100)
            for q in result.get("quotes", []):
                if q.get("symbol"):
                    tickers.add(q["symbol"])
        except Exception as e:
            print(f"Error screening {screen_name} for prediction universe: {e}")
    return sorted(tickers)


def load_prediction_store():
    return load_json(PREDICTIONS_FILE, {"history": [], "accuracy": {}})


def _pos_in_channel(e):
    price, support, resistance = e.get("price"), e.get("support"), e.get("resistance")
    if price and support and resistance and resistance > support:
        return (price - support) / (resistance - support)
    return None


# each trigger checked against the ENTIRE graded universe (all US + Israeli
# tickers analyzed each day, not just the curated top 25/10) - this is the
# same "did the final predicted direction turn out correct" measure as
# overall accuracy, just conditioned on one signal being present. It's a
# simple, transparent diagnostic (not a controlled experiment - triggers
# overlap and aren't independent), meant to surface which signals correlate
# with better/worse outcomes so the formula's weights can be reviewed.
FACTOR_DEFINITIONS = [
    ("ma_golden", "מגמת ממוצעים חיובית (Golden Cross)", lambda e: e.get("ma_trend") == "golden"),
    ("ma_death", "מגמת ממוצעים שלילית (Death Cross)", lambda e: e.get("ma_trend") == "death"),
    ("macd_bullish", "MACD חיובי", lambda e: e.get("macd_bullish") is True),
    ("macd_bearish", "MACD שלילי", lambda e: e.get("macd_bullish") is False),
    ("rsi_oversold", "RSI מתחת ל-30 (תשואת יתר)", lambda e: e.get("rsi") is not None and e["rsi"] <= 30),
    ("rsi_overbought", "RSI מעל 70 (קניית יתר)", lambda e: e.get("rsi") is not None and e["rsi"] >= 70),
    ("vol_surge", "נפח מסחר חריג (פי 1.5+)", lambda e: e.get("vol_ratio") is not None and e["vol_ratio"] >= 1.5),
    ("near_support", "קרוב לתמיכה (10% תחתונים בטווח)",
     lambda e: (_pos_in_channel(e) is not None and _pos_in_channel(e) <= 0.1 and not e.get("support_projected"))),
    ("near_resistance", "קרוב להתנגדות (10% עליונים בטווח)",
     lambda e: (_pos_in_channel(e) is not None and _pos_in_channel(e) >= 0.9 and not e.get("resistance_projected"))),
    ("analyst_upside_high", "אפסייד אנליסטים 15%+", lambda e: e.get("upside_pct") is not None and e["upside_pct"] >= 15),
    ("analyst_rec_positive", "המלצת אנליסטים חיובית (5+ אנליסטים)",
     lambda e: (e.get("recommendation_mean") is not None and (e.get("analyst_count") or 0) >= 5 and e["recommendation_mean"] <= 2.0)),
    ("short_high", "שורט גבוה (15%+)", lambda e: e.get("short_pct") is not None and e["short_pct"] >= 15),
]
MIN_FACTOR_SAMPLE = 20  # ignore a trigger's stats until it has enough graded occurrences to mean something


def analyze_factor_performance(store):
    graded = [e for e in store["history"] if e.get("graded")]
    baseline = round(sum(1 for e in graded if e["correct"]) / len(graded) * 100, 1) if graded else None

    results = []
    for key, label, cond in FACTOR_DEFINITIONS:
        try:
            subset = [e for e in graded if cond(e)]
        except Exception:
            continue
        if len(subset) < MIN_FACTOR_SAMPLE:
            continue
        hits = sum(1 for e in subset if e["correct"])
        hit_rate = round(hits / len(subset) * 100, 1)
        edge = round(hit_rate - baseline, 1) if baseline is not None else None
        results.append({"key": key, "label": label, "n": len(subset), "hit_rate": hit_rate, "edge": edge})

    results.sort(key=lambda r: (r["edge"] if r["edge"] is not None else 0), reverse=True)
    store["factor_analysis"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "baseline_n": len(graded),
        "min_sample": MIN_FACTOR_SAMPLE,
        "factors": results,
    }
    return store["factor_analysis"]


def send_factor_analysis_report(analysis):
    """Once-daily Telegram digest of which triggers are pulling their
    weight and which aren't - purely diagnostic, doesn't touch the formula
    by itself."""
    if not analysis or not analysis.get("factors"):
        return
    baseline = analysis.get("baseline")
    baseline_n = analysis.get("baseline_n")
    factors = analysis["factors"]

    lines = [f"בסיס השוואה: {baseline}% הצלחה על {baseline_n} תחזיות שנבדקו בסה\"כ (כל השוק, ארה\"ב+ישראל יחד)."]

    top = factors[:3]
    bottom = list(reversed(factors[-3:])) if len(factors) > 3 else []
    if top:
        lines.append("🟢 הטריגרים החזקים ביותר היום:")
        for f in top:
            sign = "+" if f["edge"] >= 0 else ""
            lines.append(f"• {f['label']}: {f['hit_rate']}% ({sign}{f['edge']} מהבסיס, n={f['n']})")
    if bottom:
        lines.append("🔴 הטריגרים החלשים ביותר היום:")
        for f in bottom:
            sign = "+" if f["edge"] >= 0 else ""
            lines.append(f"• {f['label']}: {f['hit_rate']}% ({sign}{f['edge']} מהבסיס, n={f['n']})")
    lines.append("המלצה לעדכון משקלים בנוסחה תישלח בנפרד כשיצטבר מספיק היסטוריה (בדרך כלל כמה שבועות).")

    send_telegram_message_chunked("🧪 ניתוח טריגרים יומי", lines, sep="\n")


def recompute_accuracy(store):
    graded = [e for e in store["history"] if e.get("graded")]
    strong_graded = [e for e in graded if e.get("strong")]
    top10_graded = [e for e in graded if e.get("top10")]
    # "up only" = exactly the population the ₪100,000 portfolio simulation
    # trades (buy recommendations only, not the sell/short ones), so this
    # number answers "if I always bought the top 10 recommended, what's my
    # hit rate" - which is different from top10_accuracy above, which also
    # includes correctness of sell/short calls within the top 10.
    top10_up_graded = [e for e in top10_graded if e.get("predicted") == "up"]

    # EXPERIMENTAL comparison group (see compute_risk_reward_score) - tracked
    # in parallel, never affects the real top10/portfolio numbers above.
    top10_exp_graded = [e for e in graded if e.get("top10_experimental")]
    top10_exp_up_graded = [e for e in top10_exp_graded if e.get("predicted") == "up"]

    # BASELINE comparison group - pure original momentum-score ranking (see
    # run_predictions). This is the "other end" of the formula_blend alpha
    # from top10_experimental (risk/reward) - calibrate_formula_blend
    # compares these two groups against each other, not against the real
    # (blended) top10 numbers above, since the real numbers already mix them.
    top10_orig_graded = [e for e in graded if e.get("top10_original")]
    top10_orig_up_graded = [e for e in top10_orig_graded if e.get("predicted") == "up"]

    # EXPERIMENTAL comparison group C (see compute_leading_adjusted_score).
    top10_leading_graded = [e for e in graded if e.get("top10_leading")]
    top10_leading_up_graded = [e for e in top10_leading_graded if e.get("predicted") == "up"]

    # DAILY (not cumulative) Top 10 accuracy - only the most recently graded
    # prediction-date's entries, so the headline tile reflects "how did
    # yesterday's Top 10 do", not an all-time average. The cumulative number
    # is kept separately (top10_accuracy above/below) for the boxes that are
    # meant to show the running track record.
    top10_daily_date = max((e["date"] for e in top10_graded), default=None)
    top10_graded_daily = [e for e in top10_graded if e["date"] == top10_daily_date] if top10_daily_date else []

    # "SINCE FORMULA CHANGE" (v4.8, formula_blend.live_since) - a clean
    # slice of the real top10_up numbers that excludes everything graded
    # before the blend went live, so the headline "current formula" numbers
    # aren't diluted by the old (worse-performing) formula's history. Kept
    # ALONGSIDE top10_up_accuracy above, never replacing it.
    formula_live_since = (store.get("formula_blend") or {}).get("live_since")
    top10_up_since_change = (
        [e for e in top10_up_graded if e["date"] >= formula_live_since] if formula_live_since else []
    )

    def calc(subset):
        if not subset:
            return None
        hits = sum(1 for e in subset if e["correct"])
        return round(hits / len(subset) * 100, 1)

    store["accuracy"] = {
        "overall": calc(graded),
        "total_graded": len(graded),
        "strong_only": calc(strong_graded),
        "strong_hits": sum(1 for e in strong_graded if e["correct"]),
        "strong_misses": sum(1 for e in strong_graded if not e["correct"]),
        "strong_total": len(strong_graded),
        "top10_accuracy": calc(top10_graded),
        "top10_hits": sum(1 for e in top10_graded if e["correct"]),
        "top10_misses": sum(1 for e in top10_graded if not e["correct"]),
        "top10_total": len(top10_graded),
        "top10_up_accuracy": calc(top10_up_graded),
        "top10_up_hits": sum(1 for e in top10_up_graded if e["correct"]),
        "top10_up_total": len(top10_up_graded),
        "top10_experimental_accuracy": calc(top10_exp_graded),
        "top10_experimental_hits": sum(1 for e in top10_exp_graded if e["correct"]),
        "top10_experimental_total": len(top10_exp_graded),
        "top10_experimental_up_accuracy": calc(top10_exp_up_graded),
        "top10_experimental_up_total": len(top10_exp_up_graded),
        "top10_original_accuracy": calc(top10_orig_graded),
        "top10_original_hits": sum(1 for e in top10_orig_graded if e["correct"]),
        "top10_original_total": len(top10_orig_graded),
        "top10_original_up_accuracy": calc(top10_orig_up_graded),
        "top10_original_up_total": len(top10_orig_up_graded),
        "top10_leading_accuracy": calc(top10_leading_graded),
        "top10_leading_hits": sum(1 for e in top10_leading_graded if e["correct"]),
        "top10_leading_total": len(top10_leading_graded),
        "top10_leading_up_accuracy": calc(top10_leading_up_graded),
        "top10_leading_up_total": len(top10_leading_up_graded),
        "top10_accuracy_daily": calc(top10_graded_daily),
        "top10_daily_hits": sum(1 for e in top10_graded_daily if e["correct"]),
        "top10_daily_total": len(top10_graded_daily),
        "top10_daily_date": top10_daily_date,
        "top10_up_accuracy_since_formula_change": calc(top10_up_since_change),
        "top10_up_total_since_formula_change": len(top10_up_since_change),
        "formula_live_since": formula_live_since,
        "us": calc([e for e in graded if not is_israeli(e["ticker"])]),
        "il": calc([e for e in graded if is_israeli(e["ticker"])]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def grade_pending_predictions(store):
    """Check yesterday-or-earlier predictions against the actual price now,
    mark them correct/incorrect, so we can measure and improve the formula.
    Uses batched downloads (same approach as the main engine) instead of one
    API call per ticker - hundreds of individual yf.Ticker() calls in a tight
    loop were getting rate-limited by Yahoo and silently failing every time,
    which is why accuracy stayed empty.

    An entry stays "live" (re-graded with the current price on every run)
    for the rest of the day it was FIRST graded on - so a prediction that
    looked wrong at 10am can flip to correct by 2pm if the stock recovers,
    matching what's actually happening in the market right now. Once a new
    day starts, whatever it landed on is finalized and never touched again -
    otherwise we'd be endlessly relitigating old predictions forever."""
    today = date.today().isoformat()
    pending = [
        e for e in store["history"]
        if e.get("date") != today and (not e.get("graded") or e.get("graded_date") == today)
    ]
    if not pending:
        return False

    tickers = sorted({e["ticker"] for e in pending})
    print(f"Grading {len(pending)} pending/live predictions across {len(tickers)} tickers...")

    current_price_by_ticker = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            data = yf.download(
                tickers=" ".join(batch), period="5d", group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"Grading batch download error: {e}")
            continue
        for symbol in batch:
            try:
                closes = data[symbol]["Close"] if len(batch) > 1 else _flatten_close_series(data["Close"])
                closes = closes.dropna()
                if len(closes):
                    current_price_by_ticker[symbol] = float(closes.iloc[-1])
            except Exception:
                continue
        time.sleep(1)

    print(f"Got current prices for {len(current_price_by_ticker)}/{len(tickers)} tickers")

    changed = False
    for entry in pending:
        current_price = current_price_by_ticker.get(entry["ticker"])
        if current_price is None or not entry.get("price"):
            continue
        actual_pct = (current_price - entry["price"]) / entry["price"] * 100
        actual_direction = "up" if actual_pct >= 0 else "down"
        entry["actual_price"] = round(current_price, 4)
        entry["actual_pct_change"] = round(actual_pct, 2)
        entry["actual_direction"] = actual_direction
        entry["correct"] = actual_direction == entry["predicted"]
        entry["graded"] = True
        entry["graded_date"] = today
        changed = True
    if changed:
        recompute_accuracy(store)
    return changed


def _compound_period_return(daily_log, days):
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    relevant = [e for e in daily_log if e["date"] >= cutoff]
    if not relevant:
        return None
    total = 1.0
    for e in relevant:
        total *= (1 + e["return_pct"] / 100)
    return round((total - 1) * 100, 2)


MONTHLY_HOLD_DAYS = 30
MONTHLY_STOP_LOSS_PCT = -8.0  # trigger an urgent sell warning if a holding drops this much from entry


def manage_monthly_portfolio(store, today_entries):
    """A slower, buy-and-hold alternative to the daily-rebalanced ₪100,000
    simulation: picks today's real Top 10 once, holds them for ~30 days
    (no daily trading costs eating the return), and refreshes the list at
    the end of the cycle. In between, watches each holding daily and fires
    an urgent, separate Telegram alert the moment one breaks down - a stop
    loss, a support breakdown, or the technical signal flipping negative -
    so a bad holding doesn't just get silently ridden out for a month."""
    print(f"manage_monthly_portfolio: starting, {len(today_entries)} entries for today")
    today = date.today().isoformat()
    mp = store.setdefault("monthly_portfolio", {
        "start_date": None,
        "next_refresh_date": None,
        "holdings": [],
        "history": [],
    })
    entries_by_ticker = {e["ticker"]: e for e in today_entries}

    needs_refresh = mp["next_refresh_date"] is None or today >= mp["next_refresh_date"]
    print(f"manage_monthly_portfolio: needs_refresh={needs_refresh}, "
          f"next_refresh_date={mp['next_refresh_date']}, current_holdings={len(mp['holdings'])}")

    if needs_refresh:
        if mp["holdings"]:
            closed = []
            for h in mp["holdings"]:
                current = entries_by_ticker.get(h["ticker"])
                exit_price = current["price"] if current and current.get("price") else h["entry_price"]
                pct = (exit_price - h["entry_price"]) / h["entry_price"] * 100 if h.get("entry_price") else None
                closed.append({**h, "exit_date": today, "exit_price": exit_price, "return_pct": round(pct, 2) if pct is not None else None})
            valid_returns = [c["return_pct"] for c in closed if c["return_pct"] is not None]
            avg_return = round(sum(valid_returns) / len(valid_returns), 2) if valid_returns else None
            mp["history"].append({
                "cycle_start": mp["start_date"], "cycle_end": today,
                "holdings": closed, "avg_return_pct": avg_return,
            })
            mp["history"] = mp["history"][-24:]  # keep ~2 years of monthly cycles

        new_top10 = [e for e in today_entries if e.get("top10") and e.get("predicted") == "up"]
        print(f"manage_monthly_portfolio: found {len(new_top10)} top10 entries among today's {len(today_entries)}")
        mp["holdings"] = [
            {
                "ticker": e["ticker"], "entry_date": today, "entry_price": e.get("price"),
                "entry_score": e["score"], "warned": False, "early_warned": False,
            }
            for e in new_top10 if e.get("price")
        ]
        mp["start_date"] = today
        mp["next_refresh_date"] = (date.today() + timedelta(days=MONTHLY_HOLD_DAYS)).isoformat()
        print(f"manage_monthly_portfolio: created new cycle with {len(mp['holdings'])} holdings, "
              f"next refresh {mp['next_refresh_date']}")

        if mp["holdings"]:
            lines = [f"{h['ticker']}: מחיר כניסה {h['entry_price']}" for h in mp["holdings"]]
            send_telegram_message_chunked(
                f"📅 תיק חודשי חדש - הרשימה עודכנה (מחזור הבא ב-{mp['next_refresh_date']})",
                lines, sep="\n",
            )
        return

    for h in mp["holdings"]:
        if h.get("warned") or not h.get("entry_price"):
            continue
        current = entries_by_ticker.get(h["ticker"])
        if not current or current.get("price") is None:
            continue
        price = current["price"]
        pct_from_entry = (price - h["entry_price"]) / h["entry_price"] * 100

        # --- leading (early) warning layer: momentum divergences / ADX
        # weakening. Fires BEFORE any hard technical break, so it's advisory
        # only - doesn't set h["warned"] (which would stop the hard checks
        # below) and fires at most once per holding via its own flag. This
        # is the "alert before the fall, not just after" layer that was
        # missing until now. ---
        if not h.get("early_warned"):
            leading_signal = None
            if current.get("rsi_bearish_div"):
                leading_signal = "דיברגנס שלילי ב-RSI - המחיר עשה שיא גבוה יותר בלי אישור מ-RSI (איתות מקדים להיחלשות מומנטום)"
            elif current.get("macd_bearish_div"):
                leading_signal = "דיברגנס שלילי ב-MACD - שיא במחיר בלי אישור מ-MACD"
            elif current.get("obv_bearish_div"):
                leading_signal = "דיברגנס שלילי ב-OBV - שיא במחיר בלי אישור בנפח המסחר"
            elif current.get("adx_weakening") and current.get("adx") is not None and current["adx"] < 25:
                leading_signal = f"עוצמת המגמה נחלשת (ADX={current['adx']})"
            if leading_signal:
                h["early_warned"] = True
                send_telegram_message_chunked(
                    f"⚠️ איתות מקדים - תיק חודשי: {h['ticker']}",
                    [f"{leading_signal}\nמחיר כניסה: {h['entry_price']} | מחיר נוכחי: {price} ({pct_from_entry:+.1f}%)\n"
                     f"התרעה מוקדמת בלבד - אין עדיין שבירה טכנית מלאה, רק היחלשות מומנטום. לא בהכרח למכור, אבל שווה לשים לב."],
                    sep="\n",
                )

        warning_reason = None
        if pct_from_entry <= MONTHLY_STOP_LOSS_PCT:
            warning_reason = f"ירידה של {pct_from_entry:.1f}% מהכניסה (מתחת לסף העצירה)"
        elif current.get("support") and not current.get("support_projected") and price <= current["support"]:
            warning_reason = f"המחיר שבר כלפי מטה את רמת התמיכה ({current['support']})"
        elif current.get("macd_bullish") is False and current.get("score", 0) < 0:
            warning_reason = "האיתות הטכני התהפך לשלילי (MACD שלילי + ציון שלילי)"

        if warning_reason:
            h["warned"] = True
            send_telegram_message_chunked(
                f"🚨 אזהרה - תיק חודשי: {h['ticker']}",
                [f"{warning_reason}\nמחיר כניסה: {h['entry_price']} | מחיר נוכחי: {price} ({pct_from_entry:+.1f}%)\nשקול למכור."],
                sep="\n",
            )


def update_portfolio_simulation(store, flag_key="top10", sim_key="portfolio_sim"):
    """Simulates a ₪100,000 portfolio, equal-weighted daily across that
    day's Top 10 'predicted up' picks - i.e. the buy recommendations only,
    not the sell/short ones. Compounds one day at a time as predictions get
    graded. Approximate: no fees/slippage/spread modeled, and "next day"
    here means "whenever this ticker's prediction next got graded" (usually
    within hours of the next trading close, but can lag if a run was
    missed). A ticker recommended on consecutive days is mathematically
    equivalent to "holding" it rather than selling/rebuying, since no
    transaction costs are modeled - so nothing extra needs to be tracked
    for that case specifically.

    flag_key/sim_key let this same logic drive a second, independent
    simulation for the experimental risk/reward-weighted top10, stored
    under its own key so it never mixes with the real numbers."""
    sim = store.setdefault(sim_key, {
        "start_value": 100000,
        "currency": "ILS",
        "value": 100000.0,
        "last_processed_date": None,
        "daily_log": [],
    })

    graded_top10 = [
        e for e in store["history"]
        if e.get(flag_key) and e.get("graded") and e.get("predicted") == "up"
    ]
    if not graded_top10:
        return

    by_date = {}
    for e in graded_top10:
        by_date.setdefault(e["date"], []).append(e)

    today = date.today().isoformat()
    last = sim.get("last_processed_date")
    # ">=" (not just ">") so the most recent day can be reprocessed with a
    # fresh return while its entries are still "live" (graded_date == today) -
    # otherwise the daily P&L would freeze at whatever it was on the first
    # check of the day instead of tracking the market in real time.
    dates_to_process = sorted(d for d in by_date if last is None or d >= last)

    # if we're about to redo 'last', roll the running value back to what it
    # was BEFORE that day's return was applied, so it doesn't get compounded
    # twice - pulled from the existing log entry, then that entry is dropped
    # and replaced fresh below.
    running_value = sim["value"]
    if sim["daily_log"] and last is not None and dates_to_process and dates_to_process[0] == last:
        if sim["daily_log"][-1]["date"] == last:
            running_value = sim["daily_log"][-1]["value_start"]
            sim["daily_log"] = sim["daily_log"][:-1]

    for d in dates_to_process:
        entries = by_date[d]
        returns = [e["actual_pct_change"] for e in entries if e.get("actual_pct_change") is not None]
        if not returns:
            continue
        avg_return = sum(returns) / len(returns)
        value_start = running_value
        value_end = value_start * (1 + avg_return / 100)
        running_value = value_end
        sim["daily_log"].append({
            "date": d,
            "value_start": round(value_start, 2),
            "value_end": round(value_end, 2),
            "return_pct": round(avg_return, 2),
            "tickers": [
                {"ticker": e["ticker"], "pct_change": e.get("actual_pct_change")}
                for e in entries
            ],
        })
        sim["last_processed_date"] = d

    sim["value"] = running_value
    sim["daily_log"] = sim["daily_log"][-400:]  # keep the file from growing forever
    sim["total_return_pct"] = round((sim["value"] / sim["start_value"] - 1) * 100, 2)
    sim["monthly_return_pct"] = _compound_period_return(sim["daily_log"], 30)
    sim["annual_return_pct"] = _compound_period_return(sim["daily_log"], 365)


def build_formula_comparison(store):
    """Side-by-side report of every formula being tracked: the real (live)
    selection - now a calibrated blend, see calibrate_formula_blend - plus
    the two pure baseline endpoints it's blended between, plus the
    leading-indicators experiment. Purely informational; calibrate_formula_
    blend is what actually acts on this, automatically, on its own
    schedule and guardrails."""
    acc = store.get("accuracy") or {}
    alpha = (store.get("formula_blend") or {}).get("alpha", DEFAULT_FORMULA_ALPHA)
    blend = store.get("formula_blend") or {}
    live_since = blend.get("live_since")

    def sim_stats(key):
        sim = store.get(key) or {}
        return {
            "total_return_pct": sim.get("total_return_pct"),
            "monthly_return_pct": sim.get("monthly_return_pct"),
            "value": sim.get("value"),
        }

    # "since formula change" - a clean slice that excludes everything from
    # before the blend went live (live_since), so the real formula's
    # numbers aren't stuck being diluted by the old formula's worse
    # historical track record. Kept ALONGSIDE the all-time cumulative
    # numbers above, never replacing them - see chat: nothing gets deleted.
    since_change = None
    cutover_value = blend.get("portfolio_sim_value_at_cutover")
    current_sim_value = (store.get("portfolio_sim") or {}).get("value")
    portfolio_return_since_change = None
    if cutover_value and current_sim_value is not None:
        portfolio_return_since_change = round((current_sim_value / cutover_value - 1) * 100, 2)
    if live_since:
        since_change = {
            "live_since": live_since,
            "accuracy": acc.get("top10_up_accuracy_since_formula_change"),
            "total": acc.get("top10_up_total_since_formula_change"),
            "portfolio_return_pct": portfolio_return_since_change,
        }

    store["formula_comparison"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "note": f"הבחירה האמיתית היא עירוב בין שתי הבסיסיות (עוצמת חיזוי טהורה מול יחס סיכוי/סיכון טהור), במשקל נוכחי alpha={alpha:.2f} לטובת יחס סיכוי/סיכון. העירוב מכויל אוטומטית פעם בחודש בערך, בצעדים מוגבלים ורק כשיש הבדל מובהק וגדול מספיק - לא בתגובה לתנודה של יום-יומיים.",
        "since_change": since_change,
        "formulas": [
            {
                "key": "current", "label": f"הנוסחה האמיתית (עירוב, alpha={alpha:.2f})",
                "accuracy": acc.get("top10_up_accuracy"), "total": acc.get("top10_up_total"),
                **sim_stats("portfolio_sim"),
            },
            {
                "key": "original", "label": "בסיס - עוצמת חיזוי טהורה (ללא עירוב)",
                "accuracy": acc.get("top10_original_up_accuracy"), "total": acc.get("top10_original_up_total"),
                **sim_stats("portfolio_sim_original"),
            },
            {
                "key": "risk_reward", "label": "בסיס - יחס סיכוי/סיכון טהור (ללא עירוב)",
                "accuracy": acc.get("top10_experimental_up_accuracy"), "total": acc.get("top10_experimental_up_total"),
                **sim_stats("portfolio_sim_experimental"),
            },
            {
                "key": "leading", "label": "ניסיונית - מותאמת אינדיקטורים מקדימים",
                "accuracy": acc.get("top10_leading_up_accuracy"), "total": acc.get("top10_leading_up_total"),
                **sim_stats("portfolio_sim_leading"),
            },
        ],
    }
    return store["formula_comparison"]


def compute_risk_metrics(daily_log, value_field="value_end"):
    """Max drawdown (%, worst peak-to-trough drop) and volatility (%, std
    dev of daily returns) computed from any sim's daily_log value history.
    Shared by Top10, my-portfolio, and the index benchmark sims below, so
    the return-% comparisons everywhere can be read alongside how bumpy
    the ride was to get there - not just the destination. Returns Nones
    if there isn't enough history yet (need at least 2 data points)."""
    values = [d.get(value_field) for d in daily_log if d.get(value_field) is not None]
    if len(values) < 2:
        return {"max_drawdown_pct": None, "volatility_pct": None}

    peak = values[0]
    max_dd = 0.0
    daily_returns = []
    for i, v in enumerate(values):
        if v > peak:
            peak = v
        if peak:
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd
        if i > 0 and values[i - 1]:
            daily_returns.append((v - values[i - 1]) / values[i - 1] * 100)

    if len(daily_returns) >= 2:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        vol = variance ** 0.5
    else:
        vol = None

    return {
        "max_drawdown_pct": round(max_dd, 2),
        "volatility_pct": round(vol, 2) if vol is not None else None,
    }


INDEX_BENCHMARKS = {
    "index_sim_sp500": {"symbol": "^GSPC", "label": "S&P 500"},
    "index_sim_ta125": {"symbol": "^TA125.TA", "label": 'מדד ת"א 125'},
}


def update_index_benchmark_sim(store, sim_key, symbol):
    """Buy-and-hold benchmark: a nominal ₪100,000 invested once, on the
    same day the Top10 ₪100,000 simulation (portfolio_sim) started, and
    left untouched since - so it can be compared directly (in ₪, not just
    %) against portfolio_sim. Like portfolio_sim itself, this tracks pure
    % price movement of the index and applies it to a nominal ₪ figure -
    it does not model real FX conversion, exactly the same simplification
    already used for the Top10/my-portfolio sims (they mix ILS and USD
    tickers and compound % returns only, not real currency amounts).
    Rebuilt from full daily history each run rather than compounded
    incrementally like portfolio_sim, since a single yfinance history call
    is cheap and this avoids any drift from missed daily runs."""
    portfolio_log = (store.get("portfolio_sim") or {}).get("daily_log") or []
    if not portfolio_log:
        return  # nothing to anchor the start date to yet
    start_date_str = portfolio_log[0]["date"]

    try:
        hist = yf.Ticker(symbol).history(start=start_date_str)
        closes = hist["Close"].dropna()
    except Exception as e:
        print(f"Error fetching index history for {symbol}: {e}")
        return
    if closes.empty:
        return

    start_price = float(closes.iloc[0])
    if not start_price:
        return

    daily_log = []
    for ts, close in closes.items():
        value = 100000.0 * (float(close) / start_price)
        daily_log.append({"date": ts.strftime("%Y-%m-%d"), "value_end": round(value, 2)})

    sim = store.setdefault(sim_key, {})
    sim["symbol"] = symbol
    sim["start_value"] = 100000
    sim["start_date"] = daily_log[0]["date"]
    sim["value"] = daily_log[-1]["value_end"]
    sim["daily_log"] = daily_log[-400:]
    sim["total_return_pct"] = round((sim["value"] / 100000 - 1) * 100, 2)


def build_benchmark_comparison(store):
    """Direct comparison of Top10 vs my real portfolio vs the major market
    indices, in both return-% and risk terms - the thing that actually
    answers 'is the formula winning?'. Both time windows shown: since
    tracking started, and since the formula change (see build_formula_
    comparison for why that split matters), plus max drawdown/volatility
    for each so a higher return that came with a much rougher ride is
    visible, not hidden behind the headline %."""
    portfolio_sim = store.get("portfolio_sim") or {}
    my_sim = store.get("my_portfolio_sim") or {}
    live_since = (store.get("formula_blend") or {}).get("live_since")

    def value_at_or_after(daily_log, target_date, field):
        if not target_date:
            return None
        for entry in daily_log:
            if entry["date"] >= target_date:
                return entry.get(field)
        return None

    entries = {}

    top10_log = portfolio_sim.get("daily_log") or []
    entries["top10"] = {
        "label": "Top 10 (סימולציית ₪100,000)",
        "value": portfolio_sim.get("value"),
        "total_return_pct": portfolio_sim.get("total_return_pct"),
        **compute_risk_metrics(top10_log, "value_end"),
    }
    cutover_top10 = value_at_or_after(top10_log, live_since, "value_end")
    entries["top10"]["return_since_formula_change_pct"] = (
        round((portfolio_sim["value"] / cutover_top10 - 1) * 100, 2)
        if cutover_top10 and portfolio_sim.get("value") is not None else None
    )

    my_log = my_sim.get("daily_log") or []
    # my_portfolio_sim tracks a base-100 index internally (not ₪100,000 like
    # portfolio_sim/the index sims), so it's scaled by 1000 here to express
    # it on the same ₪100,000-nominal basis as everything else being
    # compared - otherwise this would show "₪102" instead of "₪102,000".
    my_value_ils = my_sim.get("value") * 1000 if my_sim.get("value") is not None else None
    entries["my_portfolio"] = {
        "label": "התיק שלי",
        "value": my_value_ils,
        "total_return_pct": my_sim.get("total_return_pct"),
        **compute_risk_metrics(my_log, "value"),
    }
    cutover_my = value_at_or_after(my_log, live_since, "value")
    entries["my_portfolio"]["return_since_formula_change_pct"] = (
        round((my_sim["value"] / cutover_my - 1) * 100, 2)
        if cutover_my and my_sim.get("value") is not None else None
    )

    for sim_key, meta in INDEX_BENCHMARKS.items():
        sim = store.get(sim_key) or {}
        log = sim.get("daily_log") or []
        entry = {
            "label": meta["label"],
            "value": sim.get("value"),
            "total_return_pct": sim.get("total_return_pct"),
            **compute_risk_metrics(log, "value_end"),
        }
        cutover_val = value_at_or_after(log, live_since, "value_end")
        entry["return_since_formula_change_pct"] = (
            round((sim["value"] / cutover_val - 1) * 100, 2)
            if cutover_val and sim.get("value") is not None else None
        )
        entries[sim_key] = entry

    store["benchmark_comparison"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "since_date": top10_log[0]["date"] if top10_log else None,
        "since_formula_change": live_since,
        "entries": entries,
    }
    return store["benchmark_comparison"]


def ensure_formula_blend_initialized(store):
    """Makes sure formula_blend and its cutover markers exist as early as
    possible in a run, so recompute_accuracy's 'since formula change'
    numbers are always available (not just after calibrate_formula_blend
    runs later). Cheap and idempotent - safe to call multiple times."""
    blend = store.setdefault("formula_blend", {
        "alpha": DEFAULT_FORMULA_ALPHA, "last_calibrated": None, "history": [],
    })
    blend.setdefault("live_since", date.today().isoformat())
    if "portfolio_sim_value_at_cutover" not in blend:
        blend["portfolio_sim_value_at_cutover"] = (store.get("portfolio_sim") or {}).get("value", 100000.0)
    return blend


def calibrate_formula_blend(store):
    """Bounded, evidence-gated, automatic monthly calibration of the real
    Top10 formula's blend weight (alpha) between the two pure baselines
    (original momentum-score vs risk/reward). This is deliberately NOT
    unconstrained self-tuning:
      - runs at most about once a month (CALIBRATION_INTERVAL_DAYS)
      - requires a minimum sample size on BOTH baselines before trusting
        the comparison at all (MIN_SAMPLE_FOR_CALIBRATION)
      - requires a minimum accuracy-percentage-point gap before treating it
        as real signal rather than noise (MIN_EFFECT_SIZE_PCT)
      - moves alpha by at most MAX_MONTHLY_ALPHA_STEP in either direction
        per calibration, so a single strong month can't swing the real
        formula all the way to one extreme
      - every calibration (or explicit no-op) is logged with the numbers
        behind it, and a Telegram notice is sent either way."""
    blend = ensure_formula_blend_initialized(store)

    today = date.today()
    last = blend.get("last_calibrated")
    if last:
        try:
            days_since = (today - date.fromisoformat(last)).days
        except ValueError:
            days_since = CALIBRATION_INTERVAL_DAYS
        if days_since < CALIBRATION_INTERVAL_DAYS:
            return

    acc = store.get("accuracy") or {}
    orig_acc = acc.get("top10_original_up_accuracy")
    orig_n = acc.get("top10_original_up_total") or 0
    rr_acc = acc.get("top10_experimental_up_accuracy")
    rr_n = acc.get("top10_experimental_up_total") or 0

    if orig_acc is None or rr_acc is None or orig_n < MIN_SAMPLE_FOR_CALIBRATION or rr_n < MIN_SAMPLE_FOR_CALIBRATION:
        return  # not enough data yet this cycle - try again next cycle, no changes, no log entry

    old_alpha = blend["alpha"]
    gap = rr_acc - orig_acc  # positive = risk/reward pulling ahead

    if abs(gap) < MIN_EFFECT_SIZE_PCT:
        new_alpha = old_alpha
        note = (
            f"אין הבדל מובהק החודש (יחס סיכוי/סיכון {rr_acc}% מול עוצמת חיזוי {orig_acc}%, "
            f"פער {gap:+.1f} נק' - מתחת לסף {MIN_EFFECT_SIZE_PCT} נק') - המשקל נשאר {old_alpha:.2f}."
        )
    else:
        direction = 1 if gap > 0 else -1
        step = min(MAX_MONTHLY_ALPHA_STEP, abs(gap) / 100)
        new_alpha = round(max(0.0, min(1.0, old_alpha + direction * step)), 3)
        winner = "יחס סיכוי/סיכון" if gap > 0 else "עוצמת חיזוי טהורה"
        note = (
            f"{winner} ניצח החודש (יחס סיכוי/סיכון {rr_acc}% מול עוצמת חיזוי {orig_acc}%, n={rr_n}/{orig_n}) - "
            f"המשקל (alpha) זז מ-{old_alpha:.2f} ל-{new_alpha:.2f} לטובת יחס סיכוי/סיכון."
            if gap > 0 else
            f"{winner} ניצח החודש (עוצמת חיזוי {orig_acc}% מול יחס סיכוי/סיכון {rr_acc}%, n={orig_n}/{rr_n}) - "
            f"המשקל (alpha) זז מ-{old_alpha:.2f} ל-{new_alpha:.2f} לטובת עוצמת חיזוי טהורה."
        )

    blend["alpha"] = new_alpha
    blend["last_calibrated"] = today.isoformat()
    blend["history"].append({
        "date": today.isoformat(), "old_alpha": old_alpha, "new_alpha": new_alpha,
        "original_accuracy": orig_acc, "original_n": orig_n,
        "risk_reward_accuracy": rr_acc, "risk_reward_n": rr_n, "note": note,
    })
    blend["history"] = blend["history"][-24:]

    send_telegram_message_chunked("⚙️ כיול חודשי אוטומטי - נוסחת ה-Top 10", [note], sep="\n")
    return blend


def compute_risk_reward_score(entry):
    """EXPERIMENTAL (parallel comparison only - see EXPERIMENT_TOP10_RISK_REWARD
    and the top10_experimental flag below). Weights conviction by how much
    bigger the potential move is than the potential downside, instead of
    conviction alone. A stock with 3x more room to its target than to its
    stop gets ~3x the weight (capped both ways so one extreme case can't
    dominate); a stock with a cramped, unfavorable risk/reward gets
    down-weighted even if its raw conviction score is high."""
    score = entry.get("score", 0)
    price, support, resistance = entry.get("price"), entry.get("support"), entry.get("resistance")
    if not price or not support or not resistance or resistance <= support:
        return abs(score)

    upside_room = max((resistance - price) / price * 100, 0)
    downside_room = max((price - support) / price * 100, 0)
    if entry.get("predicted") == "up":
        reward, risk = upside_room, downside_room
    else:
        reward, risk = downside_room, upside_room

    rr_ratio = reward / max(risk, 0.5)  # floor the denominator so a near-zero stop distance doesn't blow up
    rr_ratio = max(0.2, min(rr_ratio, 5.0))  # cap both directions
    return abs(score) * rr_ratio


def compute_leading_adjusted_score(entry):
    """EXPERIMENTAL (comparison formula C - tracked in parallel exactly like
    compute_risk_reward_score above, under its own top10_leading flag/
    accuracy/portfolio-sim). Same base conviction score, but discounted when
    a leading-indicator divergence CONTRADICTS the predicted direction (an
    early sign the move may already be running out of steam) or when ADX
    shows the trend actively weakening. The bet: fewer false positives on
    picks that look strong on the surface but are already quietly losing
    momentum underneath."""
    score = abs(entry.get("score", 0))
    predicted = entry.get("predicted")
    contradicting_div = False
    if predicted == "up" and (entry.get("rsi_bearish_div") or entry.get("macd_bearish_div") or entry.get("obv_bearish_div")):
        contradicting_div = True
    elif predicted == "down" and (entry.get("rsi_bullish_div") or entry.get("macd_bullish_div") or entry.get("obv_bullish_div")):
        contradicting_div = True
    if contradicting_div:
        score *= 0.5
    if entry.get("adx_weakening") and entry.get("adx") is not None and entry["adx"] < 20:
        score *= 0.85
    return score


MAX_PICKS_PER_SECTOR = 5  # 50% of a 10-pick list - keeps the experiment from concentrating in one sector

# --- real Top10 formula blend (see calibrate_formula_blend) ---
DEFAULT_FORMULA_ALPHA = 1.0  # 0 = pure original momentum-score, 1 = pure risk/reward.
# Starting at 1.0 (Aug 2026): risk/reward has clearly outperformed the original
# formula on tracked data (51.7% vs 40% accuracy, +2.18% vs -7.42% simulated
# portfolio) - see FORMULA_BLEND_FILE / formula_comparison for the live numbers.
MIN_SAMPLE_FOR_CALIBRATION = 30  # per formula - below this, a month's comparison isn't trusted at all
MIN_EFFECT_SIZE_PCT = 5.0  # minimum accuracy-percentage-point gap to act on - anything smaller is treated as noise
MAX_MONTHLY_ALPHA_STEP = 0.15  # rate cap - alpha can move at most this much in a single calibration
CALIBRATION_INTERVAL_DAYS = 28  # roughly monthly, deliberately not more often (no chasing the latest winner)


def compute_blended_top10_score(entry, alpha, rank_a, rank_b):
    """Higher-is-better score for the REAL Top10 selection, blending two
    rankings by alpha (0 = pure original momentum-score, 1 = pure
    risk/reward). Blending at the RANK level (not raw score values) avoids
    scale-mismatch between the two formulas' very different score ranges.
    alpha itself is set by calibrate_formula_blend, not by this function."""
    ticker = entry["ticker"]
    worst_rank = max(len(rank_a), len(rank_b), 1)
    ra = rank_a.get(ticker, worst_rank)
    rb = rank_b.get(ticker, worst_rank)
    blended_rank = (1 - alpha) * ra + alpha * rb
    return -blended_rank


def select_diversified_top10(pool, key_fn, max_per_sector=MAX_PICKS_PER_SECTOR):
    """Ranks by key_fn (descending) same as before, but skips a candidate
    once its sector already has max_per_sector picks - so a day where e.g.
    half the market's movers are all tech names doesn't turn into a
    10-for-10 tech bet."""
    ranked = sorted(pool, key=key_fn, reverse=True)
    selected = []
    sector_counts = {}
    for e in ranked:
        sector = e.get("sector") or "לא ידוע"
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected.append(e)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected) == 10:
            break
    return selected


def analyst_score_0_100(recommendation_mean, upside_pct):
    """Converts whatever analyst data is available into a 0-100 score, same
    direction as the other sub-scores used in the 'בדוק מניה' on-demand
    check below (higher = more bullish case). recommendationMean (1=Strong
    Buy..5=Strong Sell) is the primary signal when present; upside_pct
    (mean target price vs current price) is used as a fallback for tickers
    analysts cover with a price target but no formal rating. Returns None
    if neither is available, so the caller can drop this component from
    the blend rather than fabricate a neutral 50."""
    if recommendation_mean is not None:
        return round(max(0, min((5 - recommendation_mean) / 4 * 100, 100)), 1)
    if upside_pct is not None:
        return round(max(0, min(50 + upside_pct * 1.5, 100)), 1)
    return None


def compute_single_ticker_score(ticker, technical_factors, fundamental_factors, market_regime):
    """The 'בדוק מניה' on-demand analysis (see check_stock.py): reuses the
    exact same scoring engines the daily Top10 pipeline uses
    (compute_prediction_score, compute_risk_reward_score,
    compute_leading_adjusted_score), adds a new analyst-rating component,
    and blends all four into one 1-100 score - plus returns each sub-score
    separately for the breakdown view.

    Unlike the real Top10 selection (compute_blended_top10_score), this
    can't blend at the RANK level - there's no universe to rank an ad-hoc
    single ticker against. So each sub-score is independently normalized
    to a 0-100 scale using a fixed ceiling (chosen from the formulas' own
    typical ranges, not calibrated against tracked history the way the
    real Top10 alpha is - there isn't a population of graded ad-hoc
    lookups to calibrate against). Weights are fixed and equal (25% each)
    for the same reason: this is a judgment call to revisit once there's
    real usage data, not an evidence-gated calibration like the Top10
    blend elsewhere in this file."""
    factors = dict(technical_factors)
    factors.update(fundamental_factors)
    score = compute_prediction_score(factors, market_regime)
    predicted = "up" if score >= 0 else "down"
    entry = {"ticker": ticker, "score": score, "predicted": predicted, **factors}

    rr_raw = compute_risk_reward_score(entry)
    leading_raw = compute_leading_adjusted_score(entry)
    analyst_raw = analyst_score_0_100(
        fundamental_factors.get("recommendation_mean"),
        fundamental_factors.get("upside_pct"),
    )

    original_norm = round(max(0, min(abs(score) / 10 * 100, 100)), 1)
    rr_norm = round(max(0, min(rr_raw / 25 * 100, 100)), 1)
    leading_norm = round(max(0, min(leading_raw / 10 * 100, 100)), 1)

    components = {
        "original": original_norm,
        "risk_reward": rr_norm,
        "leading_adjusted": leading_norm,
        "analyst": analyst_raw,  # may be None - excluded from blend below if so
    }
    available = {k: v for k, v in components.items() if v is not None}
    overall = round(sum(available.values()) / len(available), 1) if available else None

    return {
        "ticker": ticker,
        "predicted_direction": predicted,
        "raw_score": round(score, 2),
        "overall_score": overall,
        "components": components,
        "excluded_from_blend": [k for k, v in components.items() if v is None],
        "price": factors.get("price"),
        "support": factors.get("support"),
        "resistance": factors.get("resistance"),
        "analyst_count": fundamental_factors.get("analyst_count"),
        "sector": fundamental_factors.get("sector"),
    }


def analyze_single_ticker(ticker):
    """Entry point for check_stock.py (the on-demand 'בדוק מניה' workflow).
    Downloads fresh data for just this one ticker - independent of the
    daily universe scan - and runs it through compute_single_ticker_score."""
    ticker = ticker.strip().upper()
    try:
        data = yf.download(tickers=ticker, period=PRICE_HISTORY_PERIOD,
                            group_by="ticker", threads=False, progress=False, auto_adjust=True)
        closes = _flatten_close_series(data["Close"] if "Close" in data else data[ticker]["Close"])
        volumes = _flatten_close_series(data["Volume"] if "Volume" in data else data[ticker]["Volume"])
        highs = _flatten_close_series(data["High"] if "High" in data else data[ticker]["High"])
        lows = _flatten_close_series(data["Low"] if "Low" in data else data[ticker]["Low"])
    except Exception as e:
        return {"ticker": ticker, "error": f"לא הצלחתי למשוך נתוני מחיר: {e}"}

    tf = compute_technical_factors(closes, volumes, highs, lows)
    if not tf:
        return {"ticker": ticker, "error": "אין מספיק היסטוריית מחיר לניתוח (טיקר חדש/לא סחיר?)"}

    fund = get_fundamental_factors(ticker)
    market_regime = get_market_regime()
    result = compute_single_ticker_score(ticker, tf, fund, market_regime)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    return result


def run_predictions(store):
    today = date.today().isoformat()
    already_predicted_today = any(e.get("date") == today for e in store["history"])
    if already_predicted_today:
        print("Predictions already run today, skipping.")
        return

    universe = build_prediction_universe()
    print(f"Prediction universe: {len(universe)} tickers")
    market_regime = get_market_regime()
    print(f"Market regime: {market_regime}")

    technical = {}
    chart_data_by_ticker = {}
    for i in range(0, len(universe), BATCH_SIZE):
        batch = universe[i:i + BATCH_SIZE]
        print(f"Downloading batch {i // BATCH_SIZE + 1}/{-(-len(universe)//BATCH_SIZE)} "
              f"({len(batch)} tickers)...")
        try:
            data = yf.download(
                tickers=" ".join(batch), period=PRICE_HISTORY_PERIOD, group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"Batch download error: {e}")
            continue

        for symbol in batch:
            try:
                closes = data[symbol]["Close"] if len(batch) > 1 else _flatten_close_series(data["Close"])
                volumes = data[symbol]["Volume"] if len(batch) > 1 else _flatten_close_series(data["Volume"])
                highs = data[symbol]["High"] if len(batch) > 1 else _flatten_close_series(data["High"])
                lows = data[symbol]["Low"] if len(batch) > 1 else _flatten_close_series(data["Low"])
            except Exception:
                continue
            try:
                tf = compute_technical_factors(closes, volumes, highs, lows)
                if tf:
                    technical[symbol] = tf
                    chart_data_by_ticker[symbol] = build_chart_payload(closes.dropna(), tf)
            except Exception as e:
                print(f"Technical analysis error for {symbol}: {e}")
        time.sleep(1)

    print(f"Technical factors computed for {len(technical)} tickers")

    # stage 1: cheap technical-only score to decide who's worth the slow .info() call
    prelim_scores = {}
    for symbol, tf in technical.items():
        prelim_scores[symbol] = compute_prediction_score(tf, market_regime)

    candidates = {s for s, sc in prelim_scores.items() if abs(sc) >= PREFILTER_THRESHOLD}
    # a starred ticker should always get its company info/fundamentals
    # refreshed, even on a day its score happens to be too weak to clear
    # the pre-filter on its own - otherwise it'd silently lose its company
    # name/summary that day while still showing a chart.
    candidates |= (set(load_json(STARRED_FILE, [])) & set(technical.keys()))
    monthly_tickers = {h["ticker"] for h in (store.get("monthly_portfolio") or {}).get("holdings", [])}
    candidates |= (monthly_tickers & set(technical.keys()))
    print(f"{len(candidates)} tickers passed the pre-filter (or are starred), fetching fundamentals for those...")

    us_strong = []
    il_strong = []
    company_info_by_ticker = {}

    for symbol in technical:
        factors = dict(technical[symbol])
        if symbol in candidates:
            try:
                fund = get_fundamental_factors(symbol)
                factors.update(fund)
                if fund.get("_company_name") or fund.get("_business_summary"):
                    company_info_by_ticker[symbol] = {
                        "name": fund.get("_company_name"),
                        "summary": fund.get("_business_summary"),
                    }
            except Exception as e:
                print(f"Fundamentals error for {symbol}: {e}")

        score = compute_prediction_score(factors, market_regime)
        predicted = "up" if score >= 0 else "down"

        entry = {
            "date": today, "ticker": symbol, "score": score, "predicted": predicted,
            "strong": False, "top10": False, "graded": False,  # "strong"/"top10" decided below, once every ticker has a score
        }
        for k, v in factors.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            entry[k] = v

        store["history"].append(entry)

    # --- market breadth: how many stocks crossed the "directionally
    # significant" threshold today, regardless of how many we actually
    # highlight. If this is very high (200+), that's really a statement
    # about the whole market's direction, not about any specific stock -
    # useful to know, but not something that helps pick individual names,
    # so it's reported separately from the curated list below. ---
    today_entries = [e for e in store["history"] if e["date"] == today]
    breadth = [e for e in today_entries if abs(e["score"]) >= PREDICTION_SCORE_THRESHOLD]
    breadth_up = sum(1 for e in breadth if e["predicted"] == "up")
    breadth_down = len(breadth) - breadth_up

    # --- curated picks: no matter how many stocks cross the threshold, only
    # the top DAILY_TOP_PICKS_LIMIT by conviction get highlighted/alerted/
    # fetched news for - the point is to focus on a shortlist, not to relist
    # everything that happens to be bullish or bearish today. ---
    curated = sorted(breadth, key=lambda e: abs(e["score"]), reverse=True)[:DAILY_TOP_PICKS_LIMIT]
    curated_tickers = {e["ticker"] for e in curated}
    starred_tickers = set(load_json(STARRED_FILE, []))

    # --- REAL Top10 selection: blended between the original momentum-score
    # ranking and the risk/reward-weighted ranking, per formula_blend's
    # alpha (see calibrate_formula_blend - adjusted monthly, gradually,
    # only on clear evidence). This replaced a plain "top 10 by raw
    # conviction" cut in Aug 2026 once tracked data showed risk/reward
    # clearly winning; diversification (select_diversified_top10) now
    # applies to the real picks too, same as it already did for the
    # experimental ones below. ---
    formula_alpha = (store.get("formula_blend") or {}).get("alpha", DEFAULT_FORMULA_ALPHA)
    rank_a = {e["ticker"]: i for i, e in enumerate(sorted(breadth, key=lambda e: abs(e["score"]), reverse=True))}
    rank_b = {e["ticker"]: i for i, e in enumerate(sorted(breadth, key=compute_risk_reward_score, reverse=True))}
    real_top10 = select_diversified_top10(
        breadth, lambda e: compute_blended_top10_score(e, formula_alpha, rank_a, rank_b),
    )
    real_top10_tickers = {e["ticker"] for e in real_top10}
    for entry in today_entries:
        entry["top10"] = entry["ticker"] in real_top10_tickers

    # A blended-formula Top10 pick can in principle fall outside the top-25
    # raw-conviction cut above (that's the whole point of blending toward
    # risk/reward) - make sure it still gets "strong" treatment (news fetch,
    # Telegram alert, chart data) rather than silently missing out.
    missing_top10 = [e for e in real_top10 if e["ticker"] not in curated_tickers]
    if missing_top10:
        curated = curated + missing_top10
        curated_tickers = curated_tickers | {e["ticker"] for e in missing_top10}

    chart_eligible = curated_tickers | starred_tickers | monthly_tickers | real_top10_tickers

    # --- baseline experiment: pure original momentum-score ranking (what
    # "top10" used to mean before Aug 2026), kept as its own tracked line
    # purely so calibrate_formula_blend always has a clean A/B comparison
    # to calibrate against, even after the real formula becomes a blend. ---
    original_top10 = select_diversified_top10(breadth, lambda e: abs(e["score"]))
    original_tickers = {e["ticker"] for e in original_top10}
    for entry in today_entries:
        entry["top10_original"] = entry["ticker"] in original_tickers

    # --- EXPERIMENT (still tracked in full, parallel to the real picks
    # above): pure risk/reward ranking, no diversification cap difference,
    # kept as its own tracked line so the blend's alpha can keep being
    # calibrated against a clean, undiluted risk/reward baseline. ---
    experimental_top10 = select_diversified_top10(breadth, compute_risk_reward_score)
    experimental_tickers = {e["ticker"] for e in experimental_top10}
    for entry in today_entries:
        entry["top10_experimental"] = entry["ticker"] in experimental_tickers

    # --- EXPERIMENT C (also parallel, also doesn't touch the real picks):
    # same breadth pool, ranked by the leading-indicator-adjusted score. ---
    leading_top10 = select_diversified_top10(breadth, compute_leading_adjusted_score)
    leading_tickers = {e["ticker"] for e in leading_top10}
    for entry in today_entries:
        entry["top10_leading"] = entry["ticker"] in leading_tickers

    for entry in today_entries:
        ticker = entry["ticker"]
        if ticker in chart_eligible and ticker in chart_data_by_ticker:
            entry["chart"] = chart_data_by_ticker[ticker]
        info = company_info_by_ticker.get(ticker)
        if info:
            if info.get("name"):
                entry["company_name"] = info["name"]  # cheap (short string) - fine for any candidate
            if ticker in chart_eligible and info.get("summary"):
                entry["business_summary"] = info["summary"]  # longer text - only for curated/starred

    for idx, entry in enumerate(curated):
        entry["strong"] = True
        symbol = entry["ticker"]
        try:
            news_title, news_link = get_latest_news(symbol)
            if news_title:
                entry["news_title"] = news_title
                entry["news_link"] = news_link
        except Exception as e:
            print(f"News fetch error for {symbol}: {e}")

        direction = "עלייה" if entry["predicted"] == "up" else "ירידה"
        line = f"*{symbol}: {direction} צפויה (ציון {entry['score']})* 🔥"
        if is_israeli(symbol):
            il_strong.append(line)
        else:
            us_strong.append(line)

    store["top_picks"] = {"date": today, "tickers": [e["ticker"] for e in curated]}

    breadth_line = (
        f"רוחב שוק היום: {len(breadth)} מניות חצו סף מובהקות "
        f"({breadth_up} כלפי מעלה, {breadth_down} כלפי מטה)."
    )
    if len(breadth) >= 200:
        breadth_line += " זהו סימן לתנועה כללית של כל השוק, לא איתות על מניה ספציפית."

    if us_strong:
        send_telegram_message_chunked(
            f"🔮 חיזוי ליום המסחר הבא - שוק ארה\"ב\n{breadth_line}", us_strong, parse_mode="Markdown",
        )
    if il_strong:
        send_telegram_message_chunked(
            f"🔮 חיזוי ליום המסחר הבא - בורסת ת\"א\n{breadth_line}", il_strong, parse_mode="Markdown",
        )

    send_starred_report(today_entries)

    store["market_regime"] = market_regime
    store["market_breadth"] = {
        "date": today,
        "total": len(breadth),
        "up": breadth_up,
        "down": breadth_down,
        "universe_size": len(technical),
    }


US_MARKET_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}


def is_market_trading_day():
    """Weekend/US-market-holiday check. (Previously this checked SPY's
    latest daily bar date against today, on the assumption that yfinance
    live-updates today's bar during market hours - that assumption turned
    out to be unreliable: it was returning False all day even on normal
    trading days, silently skipping grading/predictions/monthly-portfolio
    every single run. A plain calendar check is simpler and, unlike that
    approach, doesn't depend on uncertain intraday data timing.)"""
    today = date.today()
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if today.isoformat() in US_MARKET_HOLIDAYS_2026:
        return False
    return True


def main():
    state = load_json(STATE_FILE, {})
    prediction_store = load_prediction_store()
    ensure_formula_blend_initialized(prediction_store)

    run_watchlist_alerts(state, prediction_store)

    try:
        update_my_portfolio(prediction_store)
    except Exception as e:
        print(f"My-portfolio update failed, continuing without it: {type(e).__name__}: {e}")

    if not is_market_trading_day():
        print("Market hasn't traded today yet (weekend/holiday/pre-open) - "
              "skipping mover alerts, grading, and new predictions.")
        save_json(PREDICTIONS_FILE, prediction_store)
        save_json(STATE_FILE, state)
        return

    run_market_wide_alerts(state, prediction_store)  # cross-references yesterday's predictions

    today_str = date.today().isoformat()

    try:
        grade_pending_predictions(prediction_store)
        update_portfolio_simulation(prediction_store)
        update_portfolio_simulation(prediction_store, "top10_experimental", "portfolio_sim_experimental")
        update_portfolio_simulation(prediction_store, "top10_leading", "portfolio_sim_leading")
        update_portfolio_simulation(prediction_store, "top10_original", "portfolio_sim_original")
        calibrate_formula_blend(prediction_store)
        build_formula_comparison(prediction_store)

        try:
            update_index_benchmark_sim(prediction_store, "index_sim_sp500", "^GSPC")
            update_index_benchmark_sim(prediction_store, "index_sim_ta125", "^TA125.TA")
            build_benchmark_comparison(prediction_store)
        except Exception as e:
            print(f"Benchmark comparison update failed, continuing without it: {type(e).__name__}: {e}")

        last_factor_run = (prediction_store.get("factor_analysis") or {}).get("updated_at", "")[:10]
        if last_factor_run != today_str:
            analysis = analyze_factor_performance(prediction_store)
            send_factor_analysis_report(analysis)

        run_predictions(prediction_store)
    except Exception as e:
        # Never let a prediction-engine bug wipe out the rest of the run -
        # watchlist alerts and price data must still get saved below.
        print(f"Prediction engine failed, continuing without it: {e}")

    try:
        today_entries_for_mp = [e for e in prediction_store["history"] if e["date"] == today_str]
        manage_monthly_portfolio(prediction_store, today_entries_for_mp)
    except Exception as e:
        print(f"Monthly portfolio management failed: {type(e).__name__}: {e}")

    save_json(PREDICTIONS_FILE, prediction_store)
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
