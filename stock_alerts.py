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

# Bump this every time stock_alerts.py's PREDICTION LOGIC changes (not for
# comment-only or cosmetic edits) - stamped onto every prediction entry and
# into last_run_status, so it's possible to tell exactly which backend
# version actually produced a given day's Top10/predictions data, instead
# of having to infer it after the fact from which fields happen to be
# present (see the v5.4.3-era "why is overall_score missing" investigation
# this was added to prevent having to repeat).
BACKEND_VERSION = "5.16.0"

WATCHLIST_FILE = BASE_DIR / "watchlist.json"
TA_TICKERS_FILE = BASE_DIR / "ta_tickers.json"
STATE_FILE = BASE_DIR / "state.json"
CURRENT_PRICES_FILE = BASE_DIR / "current_prices.json"
PREDICTIONS_FILE = BASE_DIR / "predictions.json"
STARRED_FILE = BASE_DIR / "starred.json"
MY_PORTFOLIO_FILE = BASE_DIR / "my_portfolio.json"
# Fixed research universe (item: "מניות חשופות-קריפטו") - regular equities
# with heavy crypto exposure/correlation, scored by the exact same formula
# as everything else. Deliberately NOT direct crypto (BTC-USD etc.) - see
# v5.5.0 CHANGELOG for why that was rejected (24/7 trading, no trading-day
# calendar, no analyst/RS-Rating universe - a different architecture).
CRYPTO_EXPOSED_FILE = BASE_DIR / "crypto_exposed.json"

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
    # TASE (.TA) tickers skip fast_info entirely and go straight to
    # the multi-day history fallback below. fast_info was returning the same
    # value for last_price and previous_close for at least some TASE names
    # (FTAL.TA, AZRG.TA - always showing 0% change in "נהל אחזקות") -
    # suspected cause: TASE's trading day (Sun-Thu, ends ~17:30 IL time)
    # is already over by the time this runs relative to the US-market-hours
    # schedule, and fast_info's "current session" logic doesn't handle that
    # correctly for this exchange. .history() is more reliable for a plain
    # close-to-close diff regardless of session boundaries.
    if is_israeli(ticker):
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        closes = hist["Close"].dropna()
        if len(closes) < 2:
            return None, None
        return float(closes.iloc[-1]), float(closes.iloc[-2])

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


def build_extra_price_tickers(watchlist_tickers, curated_tickers, starred, monthly_tickers, crypto_exposed=()):
    """Which tickers besides the watchlist need a live price fetched every
    15 min for the frontend's price/% line - Top10/forecast picks, starred
    tickers, monthly-portfolio holdings, AND the crypto-exposed-stocks
    section (added v5.5.0 - same reasoning as monthly_tickers below: a
    ticker source silently missing from this set only shows up as a flat
    0.0% in the app days later, so it's listed explicitly here). Pulled out
    as its own pure function (no network, no I/O) so this is unit-testable."""
    return [
        t for t in dict.fromkeys(list(curated_tickers) + list(starred) + list(monthly_tickers) + list(crypto_exposed))
        if t not in watchlist_tickers
    ]


def run_watchlist_alerts(state, prediction_store=None):
    watchlist = load_json(WATCHLIST_FILE, [])
    watchlist_tickers = {item["ticker"] for item in watchlist}

    starred = load_json(STARRED_FILE, [])
    curated_tickers = []
    monthly_tickers = []
    if prediction_store:
        curated_tickers = (prediction_store.get("top_picks") or {}).get("tickers", [])
        monthly_tickers = [h["ticker"] for h in (prediction_store.get("monthly_portfolio") or {}).get("holdings", [])]
        # v5.10.0: the new daily-return tiles average the live % of each list,
        # so every list shown needs a live price - "מומלצות עכשיו" and the
        # long-term list weren't in this set before, and the real Top10 isn't
        # guaranteed to be a subset of the curated shortlist.
        today_iso = date.today().isoformat()
        curated_tickers = list(curated_tickers) + [
            e["ticker"] for e in prediction_store.get("history", []) if e.get("date") == today_iso and e.get("top10")
        ] + [p["ticker"] for p in ((prediction_store.get("tomorrow_forecast") or {}).get("picks") or [])] + [
            p["ticker"] for p in ((prediction_store.get("long_term_picks") or {}).get("picks") or [])
        ] + [h["ticker"] for h in ((prediction_store.get("live_portfolio") or {}).get("holdings") or [])] + [
            h["ticker"] for k in ("A", "B") for h in (((prediction_store.get("strategy_book") or {}).get(k) or {}).get("holdings") or [])
        ]  # v5.15.0: unusual-move alerts need a live price for every list the user follows
    crypto_exposed_tickers = load_json(CRYPTO_EXPOSED_FILE, [])
    # keeps the dynamic predictions list's price/% line fresh every 15 min,
    # same as the manual watchlist - without re-running the full daily engine
    extra_tickers = build_extra_price_tickers(
        watchlist_tickers, curated_tickers, starred, monthly_tickers, crypto_exposed_tickers
    )

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


def compute_trend_template(price, sma50, sma150, sma200, sma200_rising, year_high, year_low):
    """Minervini Trend Template ('Stage 2 uptrend' check) - a trend-STRUCTURE
    filter, not a directional signal like everything else in this file. The
    other indicators (RSI/MACD/OBV/ADX) read short-term momentum; this reads
    whether the stock is even in a healthy long-term uptrend to begin with.
    Returns how many of the 7 classic criteria are met, so the caller can
    apply it as a penalty on the overall score - a stock can look great on
    short-term momentum while still being in a broken long-term trend
    (Stage 4), where that momentum is much less trustworthy."""
    criteria = []
    if None not in (price, sma50, sma150, sma200):
        criteria.append(price > sma150 and price > sma200)
        criteria.append(sma150 > sma200)
        criteria.append(price > sma50)
        criteria.append(sma50 > sma150 and sma50 > sma200)
    if sma200_rising is not None:
        criteria.append(bool(sma200_rising))
    if None not in (price, year_low):
        criteria.append(price >= year_low * 1.25)
    if None not in (price, year_high):
        criteria.append(price >= year_high * 0.75)

    if not criteria:
        return {"criteria_met": None, "criteria_total": 0, "passes": None, "multiplier": 1.0}

    met = sum(1 for c in criteria if c)
    total = len(criteria)
    frac = met / total
    # full marks -> no discount at all; every criterion missed pulls the
    # multiplier down toward 0.6, same shape as the other penalty-style
    # adjustments in this file (compute_leading_adjusted_score above) -
    # a nudge, not a hard veto, since Stage 2 is a useful lens but not
    # infallible on its own.
    multiplier = 0.6 + 0.4 * frac
    return {"criteria_met": met, "criteria_total": total, "passes": met == total, "multiplier": round(multiplier, 3)}


def detect_vcp_pattern(closes, volumes, pivot_highs_idx, pivot_lows_idx, lookback_days=150):
    """EXPERIMENTAL / informational only - NOT folded into the score.
    Rough heuristic for a Volatility Contraction Pattern: a sequence of
    pullbacks (peak-to-trough) that get progressively shallower, ideally
    alongside declining volume (buyers absorbing supply, sellers drying
    up). This is a simplified read of the pattern (real VCP analysis looks
    at more than just pullback depth) - treated as a flag to note and
    watch, not a validated signal to size a formula weight around, unlike
    the other factors here which have tracked accuracy behind them."""
    events = sorted(
        [(i, "high") for i in pivot_highs_idx] + [(i, "low") for i in pivot_lows_idx]
    )
    if closes is not None and len(closes) > 0:
        cutoff = len(closes) - lookback_days
        events = [(i, kind) for i, kind in events if i >= cutoff]

    pullbacks = []  # (depth_pct, high_idx, low_idx)
    last_high = None
    for i, kind in events:
        if kind == "high":
            last_high = i
        elif kind == "low" and last_high is not None:
            high_price = float(closes.iloc[last_high])
            low_price = float(closes.iloc[i])
            if high_price > 0:
                pullbacks.append((max(0.0, (high_price - low_price) / high_price * 100), last_high, i))
            last_high = None

    if len(pullbacks) < 2:
        return {"vcp_detected": False, "contractions": 0}

    depths = [p[0] for p in pullbacks]
    contractions = sum(1 for i in range(1, len(depths)) if depths[i] < depths[i - 1])
    is_contracting = contractions >= len(depths) - 1 and len(depths) >= 2  # every step shrinks

    vol_ok = True
    if volumes is not None and len(pullbacks) >= 2:
        try:
            first_vol = float(volumes.iloc[pullbacks[0][1]:pullbacks[0][2] + 1].mean())
            last_vol = float(volumes.iloc[pullbacks[-1][1]:pullbacks[-1][2] + 1].mean())
            vol_ok = last_vol < first_vol
        except Exception:
            vol_ok = True  # don't let a volume-alignment hiccup block detection on price alone

    return {"vcp_detected": bool(is_contracting and vol_ok), "contractions": len(depths)}


def compute_technical_factors(closes, volumes, highs=None, lows=None):
    """Everything derivable from price/volume history alone - no network
    calls per ticker, so this is cheap enough to run on the whole universe.
    highs/lows are optional (only needed for ADX) - they come from data
    already downloaded for closes/volumes, no extra network cost."""
    closes = closes.dropna()
    n = len(closes)
    if n < 60:
        return None

    # Data-sanity check (added v5.5.0, after the NFE incident): a single-day
    # move this extreme is essentially never a real trading day for an
    # operating company - almost always a stock split, reverse split, or
    # debt-restructuring recapitalization whose unadjusted price history got
    # mixed with the pre-event prices. Left undetected, that discontinuity
    # contaminates every rolling average/support-resistance calc below with
    # garbage, which is exactly what let NFE score high enough for Top10 on
    # a bogus ~3770% "move". Flag it here instead of silently scoring it;
    # the actual exclusion from Top10/strong happens where breadth is built.
    data_suspect = False
    data_suspect_reason = None
    daily_returns = closes.pct_change().dropna()
    if not daily_returns.empty:
        max_move = float(daily_returns.abs().max())
        if max_move > 2.0:  # >200% in a single day
            data_suspect = True
            bad_date = daily_returns.abs().idxmax()
            try:
                bad_date_str = bad_date.strftime("%Y-%m-%d")
            except Exception:
                bad_date_str = str(bad_date)
            data_suspect_reason = (
                f"תנועה יומית קיצונית ({max_move * 100:.0f}%) ב-{bad_date_str} - "
                f"כנראה split/פעולה קונצרנית, לא תנועת מסחר אמיתית"
            )

    price = float(closes.iloc[-1])
    window = closes.iloc[-252:] if n >= 252 else closes
    year_high, year_low = float(window.max()), float(window.min())
    range_pos = ((price - year_low) / (year_high - year_low) * 100) if year_high > year_low else None

    # Dual Momentum's "absolute momentum" leg (v5.6.0) - feeds ONLY the
    # parallel/experimental compute_dual_momentum_lowvol_score below, never
    # the real score above. True = this ticker's trailing ~12-month return
    # (same window as year_high/year_low above) is below a constant
    # risk-free proxy - the classic Dual Momentum signal to step aside from
    # this name, per the offline VectorBT research this formula is from
    # (research/vectorbt_regime_momentum_research.py).
    mom_negative = None
    mom_window = window.dropna()
    if len(mom_window) >= 200 and mom_window.iloc[0] > 0:
        trailing_return_pct = float((mom_window.iloc[-1] - mom_window.iloc[0]) / mom_window.iloc[0] * 100)
        mom_negative = trailing_return_pct < DUAL_MOMENTUM_RISK_FREE_ANNUAL_PCT

    run_up_30d = None
    if n >= 22:
        run_up_30d = float((closes.iloc[-1] - closes.iloc[-22]) / closes.iloc[-22] * 100)

    rsi = compute_rsi(closes)
    macd_bullish = compute_macd_bullish(closes)

    ma_trend = None
    sma50 = sma150 = sma200 = None
    sma200_rising = None
    if n >= 200:
        sma50 = float(closes.rolling(50).mean().iloc[-1])
        sma150 = float(closes.rolling(150).mean().iloc[-1]) if n >= 150 else None
        sma200 = float(closes.rolling(200).mean().iloc[-1])
        if price > sma50 > sma200:
            ma_trend = "golden"
        elif price < sma50 < sma200:
            ma_trend = "death"
        else:
            ma_trend = "mixed"
        sma200_series = closes.rolling(200).mean()
        if len(sma200_series.dropna()) >= 21:
            sma200_rising = bool(sma200_series.iloc[-1] > sma200_series.iloc[-21])

    run_up_180d = None
    if n >= 127:
        run_up_180d = float((closes.iloc[-1] - closes.iloc[-127]) / closes.iloc[-127] * 100)

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
        "year_high": year_high,
        "year_low": year_low,
        "range_pos": range_pos,
        "run_up_30d": run_up_30d,
        "run_up_180d": run_up_180d,
        "rsi": rsi,
        "macd_bullish": macd_bullish,
        "ma_trend": ma_trend,
        "sma50": sma50,
        "sma150": sma150,
        "sma200": sma200,
        "sma200_rising": sma200_rising,
        "vol_ratio": vol_ratio,
        "data_suspect": data_suspect,
        "mom_negative": mom_negative,
    }
    if data_suspect_reason:
        factors["data_suspect_reason"] = data_suspect_reason
    factors["trend_template"] = compute_trend_template(
        price, sma50, sma150, sma200, sma200_rising, year_high, year_low
    )

    try:
        factors.update(compute_support_resistance(closes, price))
    except Exception as e:
        print(f"Support/resistance calc failed for this ticker: {type(e).__name__}: {e}")

    try:
        factors["vcp"] = detect_vcp_pattern(
            closes, volumes, factors.get("_pivot_highs_idx", []), factors.get("_pivot_lows_idx", [])
        )
    except Exception as e:
        print(f"VCP detection failed for this ticker: {type(e).__name__}: {e}")

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


QUALITY_MIN_METRICS = 3  # fewer than this many fundamentals available -> no quality score (not a fake neutral 50)


def _lin_score(value, zero_at, full_at):
    """Linear 0-100 mapping, clamped. zero_at > full_at is allowed (inverted
    metrics such as debt-to-equity, where lower is better)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    frac = (v - zero_at) / (full_at - zero_at)
    return round(max(0.0, min(frac, 1.0)) * 100, 1)


def compute_quality_score(info):
    """v5.8.0 - long-horizon 'is this a good business' lens, deliberately
    independent of the short-term timing score: a strong, steady company
    with no momentum right now (the LMT case, 29.9.2026) should read as
    'quality, wait for timing', not as 'bad stock'.

    Uses only fields yfinance .info already returns (no new network call):
    returnOnEquity, profitMargins, revenueGrowth, debtToEquity (yfinance
    reports it in percent, e.g. 150 = 1.5x), freeCashflow. Each is mapped to
    0-100 with fixed, explainable ceilings - not a percentile against the
    universe, so the score for a given ticker never shifts just because
    other tickers moved. Negative equity makes debtToEquity meaningless, so
    a negative value is skipped rather than scored.

    Returns (score or None, parts dict)."""
    info = info or {}
    parts = {
        "roe": _lin_score(info.get("returnOnEquity"), 0.0, 0.25),
        "profit_margin": _lin_score(info.get("profitMargins"), 0.0, 0.25),
        "revenue_growth": _lin_score(info.get("revenueGrowth"), -0.10, 0.20),
        "debt_to_equity": None,
        "free_cash_flow": None,
    }
    de = info.get("debtToEquity")
    if isinstance(de, (int, float)) and np.isfinite(de) and de >= 0:
        parts["debt_to_equity"] = _lin_score(de, 300.0, 50.0)
    fcf = info.get("freeCashflow")
    if isinstance(fcf, (int, float)) and np.isfinite(fcf):
        parts["free_cash_flow"] = 100.0 if fcf > 0 else 0.0
    available = [v for v in parts.values() if v is not None]
    if len(available) < QUALITY_MIN_METRICS:
        return None, parts
    return round(sum(available) / len(available), 1), parts


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

    # v5.8.0: long-term quality lens (profitability / growth / leverage /
    # cash flow) - comes free from the same .info call. Reduced to ONE
    # number here (quality_score) so the per-ticker history entry grows by
    # a single field, not six; the per-metric parts are underscore-prefixed
    # (skipped by the bulk entry copy) and only surface on 'בדוק מניה'.
    quality_score, quality_parts = compute_quality_score(info)

    return {
        "upside_pct": upside_pct,
        "short_pct": short_pct,
        "recommendation_mean": recommendation_mean,
        "quality_score": quality_score,
        "_quality_parts": quality_parts,
        "analyst_count": analyst_count,
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
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

    # --- RS Rating: percentile rank (0-100) of this ticker's 6-month price
    # performance against the full universe scanned that day (see
    # run_predictions, where it's attached before this function runs).
    # Gentle and capped like the other nudges above, not a dominant term -
    # a stock outperforming everything else gets a modest tailwind, one
    # near the bottom a modest headwind. Absent for the check_stock.py
    # single-ticker path, which has no "today's universe" to rank against. ---
    if factors.get("rs_rating") is not None:
        score += (factors["rs_rating"] - 50) * 0.03

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
    for t in load_json(CRYPTO_EXPOSED_FILE, []):
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
    graded = [e for e in store["history"] if e.get("graded") and not e.get("stale_snapshot")]  # v5.14.1: copied-snapshot days excluded
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
    graded = [e for e in store["history"] if e.get("graded") and not e.get("stale_snapshot")]  # v5.14.1: copied-snapshot days excluded
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

    # EXPERIMENTAL comparison group D (v5.6.0, see compute_fast_rs_score).
    top10_fast_rs_graded = [e for e in graded if e.get("top10_fast_rs")]
    top10_fast_rs_up_graded = [e for e in top10_fast_rs_graded if e.get("predicted") == "up"]

    # EXPERIMENTAL comparison group E (v5.6.0, see compute_dual_momentum_lowvol_score).
    top10_dml_graded = [e for e in graded if e.get("top10_dual_momentum_lowvol")]
    top10_dml_up_graded = [e for e in top10_dml_graded if e.get("predicted") == "up"]

    # EXPERIMENTAL comparison group F (v5.7.0, see compute_analyst_momentum_score).
    top10_am_graded = [e for e in graded if e.get("top10_analyst_momentum")]
    top10_am_up_graded = [e for e in top10_am_graded if e.get("predicted") == "up"]

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
        "top10_fast_rs_accuracy": calc(top10_fast_rs_graded),
        "top10_fast_rs_hits": sum(1 for e in top10_fast_rs_graded if e["correct"]),
        "top10_fast_rs_total": len(top10_fast_rs_graded),
        "top10_fast_rs_up_accuracy": calc(top10_fast_rs_up_graded),
        "top10_fast_rs_up_total": len(top10_fast_rs_up_graded),
        "top10_dual_momentum_lowvol_accuracy": calc(top10_dml_graded),
        "top10_dual_momentum_lowvol_hits": sum(1 for e in top10_dml_graded if e["correct"]),
        "top10_dual_momentum_lowvol_total": len(top10_dml_graded),
        "top10_dual_momentum_lowvol_up_accuracy": calc(top10_dml_up_graded),
        "top10_dual_momentum_lowvol_up_total": len(top10_dml_up_graded),
        "top10_analyst_momentum_accuracy": calc(top10_am_graded),
        "top10_analyst_momentum_hits": sum(1 for e in top10_am_graded if e["correct"]),
        "top10_analyst_momentum_total": len(top10_am_graded),
        "top10_analyst_momentum_up_accuracy": calc(top10_am_up_graded),
        "top10_analyst_momentum_up_total": len(top10_am_up_graded),
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


def mark_stale_snapshot_days(store):
    """Idempotent: flag history days whose snapshot was a copy of the
    previous day (>= STALE_SNAPSHOT_SHARE identical US prices). Their entry
    price is really the previous session's, so their graded move spans two
    sessions - accuracy stats leave them out. (The previous day itself is
    fine: grading uses live prices, not the next snapshot.)"""
    by = {}
    for e in store.get("history", []):
        if e.get("price") is not None and not e["ticker"].endswith(".TA"):
            by.setdefault(e["date"], {})[e["ticker"]] = e["price"]
    days = sorted(by)
    bad = set()
    for a, b in zip(days, days[1:]):
        common = set(by[a]) & set(by[b])
        if len(common) >= 50 and sum(1 for t in common if by[a][t] == by[b][t]) / len(common) >= STALE_SNAPSHOT_SHARE:
            bad.add(b)   # b's entry price is really a's price, so b's graded move spans two sessions
    for e in store.get("history", []):
        if e["date"] in bad:
            e["stale_snapshot"] = True
    store["stale_snapshot_days"] = sorted(bad)
    return sorted(bad)


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


LARGE_CAP_MIN_MARKET_CAP = 50_000_000_000  # $50B floor - on TOP of S&P 500 membership itself, per Tomer's "both" answer, not either/or
MONTHLY_MIN_BELOW_HIGH_PCT = 15   # must be at least 15% below its 52-week high
MONTHLY_MAX_ABOVE_LOW_PCT = 25    # AND within 25% of its 52-week low - both conditions together, not either


def _monthly_portfolio_meets_criteria(e, sp500_tickers):
    """Shared qualification check - large-cap S&P500, predicted up, trading
    low enough relative to its own 52-week range. Single source of truth
    used both to pick NEW monthly-portfolio candidates (select_monthly_
    portfolio_candidates below) and to re-check an EXISTING holding at a
    rotation checkpoint (see manage_monthly_portfolio) - so a holding is
    judged by the exact same bar it was originally picked with."""
    if e.get("ticker") not in sp500_tickers or e.get("predicted") != "up":
        return False
    if not (e.get("year_high") and e.get("year_low") and e.get("price")):
        return False
    if (e.get("market_cap") or 0) < LARGE_CAP_MIN_MARKET_CAP:
        return False
    below_high_pct = (e["year_high"] - e["price"]) / e["year_high"] * 100
    above_low_pct = (e["price"] - e["year_low"]) / e["year_low"] * 100
    return below_high_pct >= MONTHLY_MIN_BELOW_HIGH_PCT and above_low_pct <= MONTHLY_MAX_ABOVE_LOW_PCT


def select_monthly_portfolio_candidates(today_entries, sp500_tickers, limit=10, exclude_tickers=None):
    """Monthly-portfolio-specific selection - deliberately NOT just today's
    Top 10 (that's a short-term-momentum pick, which is exactly why almost
    every monthly holding was showing an early-warning divergence flag -
    wrong tool for a 30-day hold). Looks instead for large, stable S&P 500
    names trading low relative to their own 52-week range but still
    directionally positive - "buy quality on a dip", not "buy what's
    hottest today".

    Two-stage like the main pipeline: cheap technical prefilter first
    (S&P 500 membership + 52-week range position + predicted direction -
    all already computed for every scanned ticker regardless of the
    momentum prefilter elsewhere), THEN an extra fundamentals fetch only
    for that short list to check market cap - deliberately bypasses the
    momentum-based PREFILTER_THRESHOLD candidates gate used for the daily
    Top10, since a stock near its 52-week low often has a weak *momentum*
    score and would never reach that gate on its own, even though it's
    exactly the kind of name this selection is looking for.

    limit/exclude_tickers (v5.7.0): lets manage_monthly_portfolio ask for
    just enough NEW candidates to refill the slots that actually turned
    over at a rotation checkpoint (or a single replacement mid-cycle),
    excluding tickers already held so a kept holding is never "replaced"
    with itself. Defaults (limit=10, no exclusions) reproduce the original
    full-list behavior.

    Fallback (per Tomer): the $50B/15%/25% bar is strict and some months
    may not produce 10 qualifiers on its own - rather than ship a
    half-empty or empty monthly portfolio, unfilled slots are backfilled
    with the large-cap S&P 500 names CLOSEST to meeting both price
    conditions (never relaxing the market-cap bar itself - that one stays
    non-negotiable). Each returned entry carries "backup_pick": True/False
    so the frontend can mark backfilled picks with a subtle, non-market-cap
    -relaxing indicator.
    """
    exclude_tickers = exclude_tickers or set()
    pool = []
    for e in today_entries:
        if e["ticker"] in exclude_tickers:
            continue
        if e["ticker"] not in sp500_tickers or e.get("predicted") != "up":
            continue
        if e.get("year_high") and e.get("year_low") and e.get("price"):
            pool.append(e)

    for e in pool:
        if e.get("market_cap") is None:  # already fetched if it happened to pass the daily momentum prefilter too
            try:
                e.update(get_fundamental_factors(e["ticker"]))
            except Exception as ex:
                print(f"Monthly-portfolio fundamentals fetch failed for {e['ticker']}: {ex}")

    large_cap_pool = [e for e in pool if (e.get("market_cap") or 0) >= LARGE_CAP_MIN_MARKET_CAP]

    def below_high_pct(e):
        return (e["year_high"] - e["price"]) / e["year_high"] * 100

    def above_low_pct(e):
        return (e["price"] - e["year_low"]) / e["year_low"] * 100

    def meets_price_conditions(e):
        return below_high_pct(e) >= MONTHLY_MIN_BELOW_HIGH_PCT and above_low_pct(e) <= MONTHLY_MAX_ABOVE_LOW_PCT

    qualified = [e for e in large_cap_pool if meets_price_conditions(e)]
    qualified.sort(key=lambda e: abs(e["score"]), reverse=True)
    for e in qualified:
        e["backup_pick"] = False

    result = qualified[:limit]

    if len(result) < limit:
        chosen = {e["ticker"] for e in result}
        runner_ups = [e for e in large_cap_pool if e["ticker"] not in chosen]

        def distance_from_qualifying(e):
            # 0 on a dimension already satisfied, otherwise how far short,
            # normalized so both dimensions are comparable
            high_shortfall = max(0.0, MONTHLY_MIN_BELOW_HIGH_PCT - below_high_pct(e)) / MONTHLY_MIN_BELOW_HIGH_PCT
            low_shortfall = max(0.0, above_low_pct(e) - MONTHLY_MAX_ABOVE_LOW_PCT) / MONTHLY_MAX_ABOVE_LOW_PCT
            return high_shortfall + low_shortfall

        runner_ups.sort(key=distance_from_qualifying)
        for e in runner_ups[:limit - len(result)]:
            e["backup_pick"] = True
            result.append(e)

    return result


def compound_monthly_portfolio_sim(store, closed_holding):
    """Realizes one closed monthly-portfolio holding's return into a single
    running ₪100,000-equivalent index (10 always-equal 10%-slots) that
    compounds forward across rotation checkpoints AND mid-cycle
    replacements alike - never resets, the same "decisions compound
    forward" pattern as my_portfolio_sim/portfolio_sim. A closed holding's
    contribution is always 10% of the total (one of the 10 slots),
    regardless of how many days it was actually held - kept simple, and
    consistent with the equal-weighted-₪100,000 framing already used
    elsewhere for this portfolio. Good exits compound the index up, bad
    exits compound it down, in direct proportion to the replace/keep
    decisions actually made - not to overall market movement, since a KEPT
    holding contributes nothing here until it eventually closes."""
    if closed_holding.get("return_pct") is None:
        return
    sim = store.setdefault("monthly_portfolio_sim", {
        "start_value": 100000, "currency": "USD", "value": 100000.0, "trade_log": [],
    })
    sim["value"] = sim["value"] * (1 + closed_holding["return_pct"] / 100 * 0.10)
    sim["trade_log"].append({
        "ticker": closed_holding["ticker"], "exit_date": closed_holding["exit_date"],
        "return_pct": closed_holding["return_pct"], "exit_reason": closed_holding.get("exit_reason"),
        "value_after": round(sim["value"], 2),
    })
    sim["trade_log"] = sim["trade_log"][-200:]
    sim["total_return_pct"] = round((sim["value"] / sim["start_value"] - 1) * 100, 2)


def manage_monthly_portfolio(store, today_entries):
    """A slower, buy-and-hold alternative to the daily-rebalanced ₪100,000
    simulation: picks large-cap S&P 500 names trading low relative to
    their own 52-week range (see select_monthly_portfolio_candidates -
    deliberately NOT the same short-term-momentum Top10 picks), and aims
    to hold each one for as long as it keeps qualifying - not a forced
    full reshuffle every 30 days (2026-09-23 redesign, per Tomer: buy/sell
    costs mean turnover should be minimized, and which stocks stay or go
    should be decided by each stock's own strength/warnings, never by
    overall market direction).

    Two separate turnover triggers, both close a holding into
    compound_monthly_portfolio_sim (a single running ₪100,000-equivalent
    index that never resets - see that function) and both get reported in
    the rotation-checkpoint Telegram summary:
      1. Urgent warning, ANY day: a holding that breaks down (stop loss /
         support break / technical flip) is replaced immediately with the
         next-best qualifying candidate not already held - see the warned-
         holdings loop below. If no replacement candidate is available
         that day, it's retried on subsequent days until one is, or the
         next rotation checkpoint forces a close either way.
      2. Rotation checkpoint (every MONTHLY_HOLD_DAYS days): every holding
         still standing is re-checked against the SAME bar it was
         originally picked with (_monthly_portfolio_meets_criteria) - one
         that still qualifies is KEPT AS-IS (entry_date/entry_price
         untouched, so its running return stays continuous, exactly like
         "holding, not re-buying"). Only holdings that no longer qualify
         are closed and refilled."""
    print(f"manage_monthly_portfolio: starting, {len(today_entries)} entries for today")
    today = date.today().isoformat()
    mp = store.setdefault("monthly_portfolio", {
        "start_date": None,
        "next_refresh_date": None,
        "holdings": [],
        "mid_cycle_trades": [],  # urgent-warning replacements since the last checkpoint, flushed into history there
        "history": [],
    })
    mp.setdefault("mid_cycle_trades", [])
    entries_by_ticker = {e["ticker"]: e for e in today_entries}
    sp500_tickers = set(get_sp500_tickers())

    needs_refresh = mp["next_refresh_date"] is None or today >= mp["next_refresh_date"]
    print(f"manage_monthly_portfolio: needs_refresh={needs_refresh}, "
          f"next_refresh_date={mp['next_refresh_date']}, current_holdings={len(mp['holdings'])}")

    if needs_refresh:
        kept, closed_now = [], []
        for h in mp["holdings"]:
            current = entries_by_ticker.get(h["ticker"])
            if current is not None and current.get("market_cap") is None:
                try:
                    current.update(get_fundamental_factors(h["ticker"]))
                except Exception as ex:
                    print(f"Monthly-portfolio re-qualify fetch failed for {h['ticker']}: {ex}")
            still_qualifies = (
                not h.get("warned") and current is not None
                and _monthly_portfolio_meets_criteria(current, sp500_tickers)
            )
            if still_qualifies:
                kept.append({**h, "early_warned": False})  # fresh evaluation window starts now
                continue
            exit_price = current["price"] if current and current.get("price") else h["entry_price"]
            pct = (exit_price - h["entry_price"]) / h["entry_price"] * 100 if h.get("entry_price") else None
            reason = "warned" if h.get("warned") else ("no_longer_qualifies" if current else "no_price_today")
            closed_h = {**h, "exit_date": today, "exit_price": exit_price,
                        "return_pct": round(pct, 2) if pct is not None else None, "exit_reason": reason}
            closed_now.append(closed_h)
            compound_monthly_portfolio_sim(store, closed_h)

        all_closed_this_period = mp["mid_cycle_trades"] + closed_now
        valid_returns = [c["return_pct"] for c in all_closed_this_period if c["return_pct"] is not None]
        avg_return = round(sum(valid_returns) / len(valid_returns), 2) if valid_returns else None
        mp["history"].append({
            "period_start": mp["start_date"], "period_end": today,
            "kept_tickers": [h["ticker"] for h in kept],
            "closed": all_closed_this_period, "avg_return_pct": avg_return,
        })
        mp["history"] = mp["history"][-24:]  # keep ~2 years of rotation checkpoints

        slots_to_fill = 10 - len(kept)
        new_candidates = []
        if slots_to_fill > 0:
            # exclude not just kept tickers, but everything held BEFORE this
            # checkpoint (including what was just closed above) - a ticker
            # closed this run for no longer qualifying must not immediately
            # get re-bought as a backup-fill in the same breath
            previously_held = {h["ticker"] for h in mp["holdings"]}
            new_candidates = select_monthly_portfolio_candidates(
                today_entries, sp500_tickers, limit=slots_to_fill,
                exclude_tickers=previously_held,
            )
        print(f"manage_monthly_portfolio: kept {len(kept)}, closed {len(all_closed_this_period)} "
              f"({len(mp['mid_cycle_trades'])} mid-cycle + {len(closed_now)} at checkpoint), "
              f"filling {slots_to_fill} slot(s) with {len(new_candidates)} new candidate(s)")

        new_holdings = [
            {
                "ticker": e["ticker"], "entry_date": today, "entry_price": e.get("price"),
                "entry_score": e["score"], "warned": False, "early_warned": False,
                "backup_pick": e.get("backup_pick", False),
            }
            for e in new_candidates if e.get("price")
        ]
        mp["holdings"] = kept + new_holdings
        mp["mid_cycle_trades"] = []
        mp["start_date"] = today
        mp["next_refresh_date"] = (date.today() + timedelta(days=MONTHLY_HOLD_DAYS)).isoformat()
        print(f"manage_monthly_portfolio: cycle checkpoint done, {len(mp['holdings'])} holdings, "
              f"next refresh {mp['next_refresh_date']}")

        mid_cycle_count = sum(1 for c in all_closed_this_period if c.get("exit_reason") == "warned")
        other_count = len(all_closed_this_period) - mid_cycle_count
        lines = [
            f"נשארו ללא שינוי: {len(kept)}",
            f"נסגרו/הוחלפו במחזור: {len(all_closed_this_period)}",
            f"  מתוכן עקב אזהרה דחופה: {mid_cycle_count}",
            f"  מתוכן בסוף המחזור (לא עומדות יותר בקריטריונים): {other_count}",
        ]
        if avg_return is not None:
            lines.append(f"תשואה ממוצעת על מה שנסגר במחזור: {avg_return:+.2f}%")
        sim = store.get("monthly_portfolio_sim")
        if sim:
            lines.append(f"מדד מצטבר (₪100,000, לא מתאפס בין מחזורים): {sim['total_return_pct']:+.2f}%")
        if new_holdings:
            lines.append("")
            lines.append("החזקות חדשות שנכנסו:")
            lines += [f"{h['ticker']}: מחיר כניסה {h['entry_price']}" for h in new_holdings]
        send_telegram_message_chunked(
            f"📅 תיק חודשי - סיכום מחזור (הבא ב-{mp['next_refresh_date']})", lines, sep="\n",
        )
        return

    # --- daily monitoring between checkpoints ---
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
                [f"{warning_reason}\nמחיר כניסה: {h['entry_price']} | מחיר נוכחי: {price} ({pct_from_entry:+.1f}%)\n"
                 f"מחפש תחליף איכותי להחלפה מיידית..."],
                sep="\n",
            )

    # --- immediate mid-cycle replacement for ANY currently-warned holding
    # (whether it just got warned above, or was warned on an earlier day
    # and no replacement candidate was available yet - retried here every
    # run until one is found or the next checkpoint closes it anyway) ---
    warned_holdings = [h for h in mp["holdings"] if h.get("warned")]
    if warned_holdings:
        held_tickers = {h["ticker"] for h in mp["holdings"]}
        # deliberately STRICT here (no backup_pick/runner-up fallback, unlike
        # the checkpoint refill above) - an urgent replacement should only
        # swap into a name that genuinely qualifies ("מניה טובה"), never into
        # a "closest we could find" backup just to fill the slot immediately.
        # If nothing genuinely qualifies today, the warned holding stays put
        # (already alerted) and this is retried again next run.
        strict_pool = [
            e for e in today_entries
            if e["ticker"] not in held_tickers and _monthly_portfolio_meets_criteria(e, sp500_tickers)
        ]
        for e in strict_pool:
            if e.get("market_cap") is None:
                try:
                    e.update(get_fundamental_factors(e["ticker"]))
                except Exception as ex:
                    print(f"Monthly-portfolio replacement fetch failed for {e['ticker']}: {ex}")
        strict_pool = [e for e in strict_pool if _monthly_portfolio_meets_criteria(e, sp500_tickers)]
        strict_pool.sort(key=lambda e: abs(e["score"]), reverse=True)
        replacements = [e for e in strict_pool[:len(warned_holdings)] if e.get("price")]
        for h, new_e in zip(warned_holdings, replacements):
            current = entries_by_ticker.get(h["ticker"])
            exit_price = current["price"] if current and current.get("price") else h["entry_price"]
            pct = (exit_price - h["entry_price"]) / h["entry_price"] * 100 if h.get("entry_price") else None
            closed_h = {**h, "exit_date": today, "exit_price": exit_price,
                        "return_pct": round(pct, 2) if pct is not None else None, "exit_reason": "warned"}
            mp["mid_cycle_trades"].append(closed_h)
            compound_monthly_portfolio_sim(store, closed_h)

            idx = next(i for i, x in enumerate(mp["holdings"]) if x["ticker"] == h["ticker"])
            mp["holdings"][idx] = {
                "ticker": new_e["ticker"], "entry_date": today, "entry_price": new_e.get("price"),
                "entry_score": new_e["score"], "warned": False, "early_warned": False,
                "backup_pick": new_e.get("backup_pick", False),
            }
            return_txt = f"{closed_h['return_pct']:+.1f}%" if closed_h['return_pct'] is not None else "—"
            send_telegram_message_chunked(
                f"🔄 הוחלפה - תיק חודשי: {h['ticker']} → {new_e['ticker']}",
                [f"{h['ticker']} נסגרה ({return_txt})\n"
                 f"הוחלפה ב-{new_e['ticker']} (מחיר כניסה {new_e.get('price')})"],
                sep="\n",
            )
        still_unmatched = len(warned_holdings) - len(replacements)
        if still_unmatched > 0:
            print(f"manage_monthly_portfolio: {still_unmatched} warned holding(s) still without a "
                  f"replacement candidate today - will retry next run")


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
        "currency": "USD",
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
            {
                "key": "fast_rs", "label": "ניסיונית - RS מהיר (מומנטום יחסי 20-30 יום)",
                "accuracy": acc.get("top10_fast_rs_up_accuracy"), "total": acc.get("top10_fast_rs_up_total"),
                **sim_stats("portfolio_sim_fast_rs"),
            },
            {
                "key": "dual_momentum_lowvol", "label": "ניסיונית - Dual Momentum + Low-Vol (מחקר VectorBT)",
                "accuracy": acc.get("top10_dual_momentum_lowvol_up_accuracy"),
                "total": acc.get("top10_dual_momentum_lowvol_up_total"),
                **sim_stats("portfolio_sim_dual_momentum_lowvol"),
            },
            {
                "key": "analyst_momentum", "label": "ניסיונית - מומנטום ציון-אנליסטים (30 יום)",
                "accuracy": acc.get("top10_analyst_momentum_up_accuracy"),
                "total": acc.get("top10_analyst_momentum_up_total"),
                **sim_stats("portfolio_sim_analyst_momentum"),
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


# --- v5.7.0 parallel-tracked experiment: analyst-score momentum ("lazy"
# Zacks-style, per Tomer 2026-09-23) - tracks the CHANGE in the app's own
# analyst_score_0_100 over the trailing ~30 days, instead of a new external
# earnings-estimate-revision data source (OpenBB/Zacks). Cheap: no new
# network calls, since recommendation_mean/upside_pct are already fetched
# for the whole PREFILTER_THRESHOLD candidate pool (see run_predictions) -
# only the delta-vs-history computation is new. ---
ANALYST_MOMENTUM_LOOKBACK_DAYS = 30
ANALYST_MOMENTUM_SCALE = 0.02  # additive-score-multiplier per point of analyst_score change


def build_analyst_score_history_index(history, today_str, lookback_days=ANALYST_MOMENTUM_LOOKBACK_DAYS):
    """One pass over store['history'] (excluding today) collecting, per
    ticker, the EARLIEST analyst_score seen within the trailing
    lookback_days - the baseline compute_analyst_score_delta compares
    today's analyst_score against. O(n) once per run rather than a
    per-ticker rescan."""
    cutoff = (date.fromisoformat(today_str) - timedelta(days=lookback_days)).isoformat()
    earliest = {}
    for e in history:
        d = e.get("date")
        if d is None or d >= today_str or d < cutoff:
            continue
        score = e.get("analyst_score")
        if score is None:
            continue
        ticker = e["ticker"]
        if ticker not in earliest or d < earliest[ticker][0]:
            earliest[ticker] = (d, score)
    return {ticker: score for ticker, (d, score) in earliest.items()}


def compute_analyst_momentum_score(entry):
    """EXPERIMENTAL (comparison formula F - tracked in parallel exactly like
    compute_fast_rs_score/compute_dual_momentum_lowvol_score, under its own
    top10_analyst_momentum flag/accuracy/portfolio-sim - see
    recompute_accuracy, build_formula_comparison, main). Same base
    conviction score, weighted by how much this ticker's analyst_score has
    RISEN over the trailing ~30 days (analyst_score_delta) - the bet: a
    rising analyst-sentiment trend is a better signal than the existing
    formula's static analyst-score LEVEL (which can saturate to 100 from a
    thin/stale rating with no information about direction, a known
    weakness - see the 2026-09 research writeup)."""
    score = abs(entry.get("score", 0))
    delta = entry.get("analyst_score_delta")
    if delta is not None:
        score *= max(0.0, 1 + delta * ANALYST_MOMENTUM_SCALE)
    return score



# this is lifted from: research/vectorbt_regime_momentum_research.py).
# Neither touches the real Top10 selection - both tracked exactly like the
# existing top10_experimental/top10_leading comparison formulas above,
# under their own top10_fast_rs/top10_dual_momentum_lowvol flags. ---
DUAL_MOMENTUM_RISK_FREE_ANNUAL_PCT = 4.0  # simplified constant risk-free proxy (T-bill-ish)
LOW_BETA_THRESHOLD = 0.8
LOW_BETA_BONUS = 1.5   # same order of magnitude as the other +/- nudges in compute_prediction_score
FAST_RS_SCALE = 0.01   # steeper than the existing slow-RS nudge inside compute_prediction_score
                        # (0.03/point, additive) - this one multiplies, and its whole point is
                        # to react fast, so tracked in parallel to see empirically if it's too much


def compute_beta_vs_spy(closes, spy_closes):
    """Rolling beta of this ticker vs SPY over the trailing window they
    overlap on - feeds ONLY compute_dual_momentum_lowvol_score's Low-Vol
    tilt below, never the real score. Deliberately computed off a SEPARATE
    SPY download (get_spy_close_series_for_beta, called once per
    run_predictions run) rather than sharing get_market_regime()'s own SPY
    fetch, so this new experimental plumbing can never accidentally affect
    the tested, already-live market-regime calculation."""
    s = closes.pct_change().dropna()
    m = spy_closes.pct_change().dropna()
    joined = pd.concat([s, m], axis=1, join="inner")
    if len(joined) < 60:
        return None
    joined.columns = ["stock", "mkt"]
    var = joined["mkt"].var()
    if not var or (isinstance(var, float) and var != var):  # NaN check without importing math here
        return None
    cov = joined["stock"].cov(joined["mkt"])
    return float(cov / var)


def get_spy_close_series_for_beta():
    """A dedicated SPY download for compute_beta_vs_spy above - deliberately
    separate from get_market_regime()'s own SPY download (see that
    function's docstring reasoning) rather than sharing it."""
    try:
        data = yf.download("SPY", period=PRICE_HISTORY_PERIOD, progress=False, auto_adjust=True)
        s = _flatten_close_series(data["Close"]).dropna()
        return s if len(s) >= 60 else None
    except Exception as e:
        print(f"SPY download for beta calc failed: {e}")
        return None


def compute_fast_rs_score(entry):
    """EXPERIMENTAL (comparison formula D - tracked in parallel exactly like
    compute_risk_reward_score/compute_leading_adjusted_score, under its own
    top10_fast_rs flag/accuracy/portfolio-sim - see recompute_accuracy,
    build_formula_comparison, main). Same base conviction score, but
    weighted by fast_rs_rating - a ~20-30 trading day relative-strength
    percentile (see run_predictions) instead of the existing RS Rating's
    6-month one. The bet: rotate into whatever's leading RIGHT NOW faster
    than the slow RS Rating or the SMA50/200 regime tag can react."""
    score = abs(entry.get("score", 0))
    fast_rs = entry.get("fast_rs_rating")
    if fast_rs is not None:
        score *= max(0.0, 1 + (fast_rs - 50) * FAST_RS_SCALE)
    return score


def compute_dual_momentum_lowvol_score(entry, market_regime):
    """EXPERIMENTAL (comparison formula E - same tracking pattern as above,
    under top10_dual_momentum_lowvol). Two independent pieces, both lifted
    from the offline VectorBT research that motivated them, kept here as a
    BINARY GATE plus a REGIME-GATED TILT rather than blended at the rank
    level - deliberately, per that research's design:
      - absolute momentum negative (mom_negative, see compute_technical_
        factors) on a 'predicted up' entry excludes it from this formula's
        Top10 entirely
      - a low-beta bonus (compute_beta_vs_spy) only applies while the
        existing SMA50/200 regime tag is bearish - never during a bullish
        regime."""
    if entry.get("predicted") == "up" and entry.get("mom_negative") is True:
        return 0.0
    score = abs(entry.get("score", 0))
    beta = entry.get("beta_vs_spy")
    if market_regime.get("bullish") is False and beta is not None and beta < LOW_BETA_THRESHOLD:
        score += LOW_BETA_BONUS
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


TIMING_CONVICTION_CEILING = 10.0   # |score| at/above this = full-strength timing signal
TIMING_RR_MAX_POINTS = 10.0        # risk/reward can move the timing score by at most this much
VERDICT_BUY_TIMING = 70            # timing at/above this (and long-term not weak) -> buy. ~top 10% of the scanned universe; every Top10 pick on 29.9.2026 was >= 70
VERDICT_AVOID_TIMING = 40          # timing below this (and quality not strong) -> avoid
VERDICT_QUALITY_STRONG = 65        # quality at/above this -> 'quality, wait for timing' instead of avoid
VERDICT_QUALITY_WEAK = 40          # quality below this blocks a buy verdict
OVERALL_TIMING_WEIGHT = 0.6        # overall = 60% timing + 40% long-term (quality/analyst) when available


def compute_timing_score(entry):
    """v5.8.0 - DIRECTIONAL short-term timing score, 0-100 with 50 = no
    signal. Replaces the pre-5.8.0 trio (original / risk_reward /
    leading_adjusted), which had two real problems found 29.9.2026:
      1. all three used abs(score) - conviction MAGNITUDE, not direction -
         so a stock the engine strongly expects to FALL scored high, and a
         neutral one (LMT, raw -0.39) scored ~4 instead of ~50;
      2. all three derive from the same base score, so short-term momentum
         was effectively counted three times.
    Now: one conviction term (the leading-indicator-adjusted score, which
    already discounts contradicting divergences), signed by the predicted
    direction, plus a bounded risk/reward nudge that is itself scaled by
    conviction (so a no-signal stock can't be pushed far from 50 by its
    support/resistance geometry alone)."""
    raw = entry.get("score") or 0
    direction = 1 if raw >= 0 else -1
    conviction = min(compute_leading_adjusted_score(entry), TIMING_CONVICTION_CEILING)
    base = conviction / TIMING_CONVICTION_CEILING * 50  # 0..50

    rr_points = 0.0
    abs_score = abs(raw)
    if abs_score > 0:
        rr_ratio = compute_risk_reward_score(entry) / abs_score  # 0.2..5 by construction
        if rr_ratio > 0:
            rr_points = max(-TIMING_RR_MAX_POINTS, min(float(np.log2(rr_ratio)) * 5, TIMING_RR_MAX_POINTS))
            rr_points *= min(conviction / 3.0, 1.0)

    timing = 50 + direction * (base + rr_points)
    return round(max(0.0, min(timing, 100.0)), 1)


def compute_verdict(timing, long_term, has_fundamentals=True):
    """One explicit recommendation label, so the ❌/✅ tags stop depending
    on a single blended number. long_term may be None (no data at all).
    'quality_wait' requires real fundamentals (has_fundamentals): analyst
    ratings alone skew bullish across almost the whole market, so on their
    own they are not enough to call a company high-quality."""
    if timing is None:
        return None
    if timing >= VERDICT_BUY_TIMING and (long_term is None or long_term >= VERDICT_QUALITY_WEAK):
        return "buy"
    if (timing < VERDICT_BUY_TIMING and has_fundamentals and long_term is not None
            and long_term >= VERDICT_QUALITY_STRONG):
        return "quality_wait"
    if timing < VERDICT_AVOID_TIMING:
        return "avoid"
    return "neutral"


def build_score_breakdown(entry, recommendation_mean, upside_pct):
    """Shared by compute_single_ticker_score ('בדוק מניה') and the daily
    Top10/watchlist breakdown attached in run_predictions below - one
    formula, one place, so the badge shown on a Top10/watchlist card and
    the result of running the same ticker through 'בדוק מניה' can never
    silently drift apart.

    v5.8.0: two separate lenses instead of one blend of four same-signal
    parts - see compute_timing_score (short term, directional) and
    compute_quality_score (long term, business quality). The analyst rating
    is the other long-term input. overall_score is kept (sorting, history,
    old UI paths) as 60% timing + 40% long-term, falling back to timing
    alone when no long-term data exists. The Top10 SELECTION itself does
    not use this at all (it uses compute_blended_top10_score), so the
    running experiments are unaffected."""
    timing = compute_timing_score(entry)
    analyst_raw = analyst_score_0_100(recommendation_mean, upside_pct)
    quality_raw = entry.get("quality_score")

    long_parts = [v for v in (quality_raw, analyst_raw) if v is not None]
    long_term = round(sum(long_parts) / len(long_parts), 1) if long_parts else None
    overall = timing if long_term is None else round(
        OVERALL_TIMING_WEIGHT * timing + (1 - OVERALL_TIMING_WEIGHT) * long_term, 1
    )

    components = {
        "timing": timing,
        "quality": quality_raw,   # may be None
        "analyst": analyst_raw,   # may be None - kept under this key: analyst_score/analyst_momentum read it
    }
    return {
        "overall_score": overall,
        "timing_score": timing,
        "quality_score": quality_raw,
        "long_term_score": long_term,
        "verdict": compute_verdict(timing, long_term, has_fundamentals=quality_raw is not None),
        "components": components,
        "excluded_from_blend": [k for k, v in components.items() if v is None],
    }


def _round_price(v, digits=2):
    """v5.8.0: prices shown to the user are rounded at the source (the
    'בדוק מניה' card showed 518.0999755859375 - a float32 artifact from
    yfinance). Non-numeric/None passes through unchanged."""
    try:
        return round(float(v), digits) if v is not None else None
    except (TypeError, ValueError):
        return v


def _add_sessions(d_iso, n):
    d, k = date.fromisoformat(d_iso), 0
    while k < n:
        d += timedelta(days=1)
        k += d.weekday() < 5
    return d.isoformat()


def strategy_view_for(ticker, book):
    """v5.13.1 - the one line 'בדוק מניה' shows about the research-validated
    strategies: buy / hold / sell / nothing, and until when. Read from the
    strategy book (rebuilt after every US close for all S&P 500 names), so
    the answer is exactly what the strategy portfolios are doing."""
    if not book or not book.get("session"):
        return {"action": "unknown"}
    tickers = set(book.get("tickers") or [])
    if tickers and ticker not in tickers:
        return {"action": "not_applicable", "session": book["session"]}
    session = book["session"]
    for key in ("A", "B"):
        b = book.get(key) or {}
        for h in b.get("holdings") or []:
            if h["ticker"] == ticker:
                left = int(h.get("sessions_left") or 0)
                return {"action": "sell" if left <= 1 else "hold", "strategy": key, "label": b.get("label"),
                        "entry_date": h.get("entry_date"), "sessions_left": left,
                        "exit_date": _add_sessions(session, max(left, 1)), "session": session}
    for key in ("A", "B"):
        b = book.get(key) or {}
        for pnd in b.get("pending") or []:
            if pnd["ticker"] == ticker:
                hold = BOOK_STRATEGIES[key]["hold"]
                return {"action": "buy", "strategy": key, "label": b.get("label"),
                        "entry_date": _add_sessions(session, 1), "exit_date": _add_sessions(session, hold),
                        "session": session}
    return {"action": "none", "session": session}


def compute_single_ticker_score(ticker, technical_factors, fundamental_factors, market_regime, rs_reference=None):
    """The 'בדוק מניה' on-demand analysis (see check_stock.py): reuses the
    exact same scoring engines the daily Top10 pipeline uses
    (compute_prediction_score, compute_risk_reward_score,
    compute_leading_adjusted_score), adds a new analyst-rating component,
    and blends all four into one 1-100 score via build_score_breakdown -
    plus returns each sub-score separately for the breakdown view.

    rs_reference, if provided, is yesterday's/today's full-universe list of
    6-month performances (see run_predictions) - lets this one-off lookup
    get an RS Rating against a real recent market snapshot instead of
    skipping the factor entirely."""
    factors = dict(technical_factors)
    factors.update(fundamental_factors)
    if rs_reference and factors.get("run_up_180d") is not None:
        below = sum(1 for p in rs_reference if p <= factors["run_up_180d"])
        factors["rs_rating"] = round(below / len(rs_reference) * 100, 1)

    score = compute_prediction_score(factors, market_regime)
    trend_multiplier = (factors.get("trend_template") or {}).get("multiplier", 1.0)
    score = round(score * trend_multiplier, 2)
    predicted = "up" if score >= 0 else "down"
    entry = {"ticker": ticker, "score": score, "predicted": predicted, **factors}

    breakdown = build_score_breakdown(
        entry, fundamental_factors.get("recommendation_mean"), fundamental_factors.get("upside_pct")
    )

    return {
        "ticker": ticker,
        "predicted_direction": predicted,
        "raw_score": round(score, 2),
        "overall_score": breakdown["overall_score"],
        "timing_score": breakdown["timing_score"],
        "quality_score": breakdown["quality_score"],
        "long_term_score": breakdown["long_term_score"],
        "verdict": breakdown["verdict"],
        "quality_parts": fundamental_factors.get("_quality_parts"),
        "components": breakdown["components"],
        "excluded_from_blend": breakdown["excluded_from_blend"],
        "price": _round_price(factors.get("price")),
        "support": _round_price(factors.get("support")),
        "resistance": _round_price(factors.get("resistance")),
        "analyst_count": fundamental_factors.get("analyst_count"),
        "sector": fundamental_factors.get("sector"),
        "short_pct": factors.get("short_pct"),
        "rs_rating": factors.get("rs_rating"),
        "trend_template": factors.get("trend_template"),
        "vcp": factors.get("vcp"),
    }


def get_upcoming_earnings_date(ticker):
    """Days until the next known earnings report, if yfinance has one on
    file - shown as a standalone timing-risk flag on the 'בדוק מניה' card,
    not folded into the score (a different kind of information: WHEN a
    big, formula-independent price jump could happen, not which direction
    the stock is leaning). Returns None on any failure/no data - this is a
    nice-to-have, never worth failing the whole check over."""
    try:
        cal = yf.Ticker(ticker).get_earnings_dates(limit=4)
        if cal is None or cal.empty:
            return None
        today = pd.Timestamp.now(tz=cal.index.tz)
        future = cal[cal.index >= today]
        if future.empty:
            return None
        next_date = future.index.min()
        return {"date": next_date.strftime("%Y-%m-%d"), "days_away": (next_date - today).days}
    except Exception:
        return None


def analyze_single_ticker(ticker):
    """Entry point for check_stock.py (the on-demand 'בדוק מניה' workflow).
    Downloads fresh data for just this one ticker - independent of the
    daily universe scan - and runs it through compute_single_ticker_score."""
    ticker_input = ticker.strip().upper()
    ticker = ticker_input
    tase_fallback_used = False

    def _fetch(tk):
        data = yf.download(tickers=tk, period=PRICE_HISTORY_PERIOD,
                            group_by="ticker", threads=False, progress=False, auto_adjust=True)
        c = _flatten_close_series(data["Close"] if "Close" in data else data[tk]["Close"]).dropna()
        if c.empty:
            raise ValueError("no price data returned")
        v = _flatten_close_series(data["Volume"] if "Volume" in data else data[tk]["Volume"])
        h = _flatten_close_series(data["High"] if "High" in data else data[tk]["High"])
        l = _flatten_close_series(data["Low"] if "Low" in data else data[tk]["Low"])
        return c, v, h, l

    try:
        closes, volumes, highs, lows = _fetch(ticker)
    except Exception as first_error:
        # v5.6.0: a bare TASE ticker (e.g. "SAE" for Shufersal) fails here
        # silently, because Yahoo Finance needs the ".TA" suffix. Before
        # giving up, retry once with ".TA" appended - covers the common
        # case without requiring the user to already know Yahoo's naming
        # quirk. Only attempted when the input doesn't already end in .TA.
        if not ticker.endswith(".TA"):
            try:
                ta_ticker = ticker + ".TA"
                closes, volumes, highs, lows = _fetch(ta_ticker)
                ticker = ta_ticker
                tase_fallback_used = True
            except Exception:
                return {
                    "ticker": ticker_input,
                    "error": (
                        f"לא הצלחתי למשוך נתוני מחיר עבור '{ticker_input}'. "
                        f"אם זו מניה מבורסת תל-אביב, נסה להוסיף בעצמך את הסיומת .TA (למשל SAE.TA)."
                    ),
                }
        else:
            return {"ticker": ticker_input, "error": f"לא הצלחתי למשוך נתוני מחיר: {first_error}"}

    tf = compute_technical_factors(closes, volumes, highs, lows)
    if not tf:
        return {"ticker": ticker, "error": "אין מספיק היסטוריית מחיר לניתוח (טיקר חדש/לא סחיר?)"}

    fund = get_fundamental_factors(ticker)
    market_regime = get_market_regime()
    rs_ref = (load_json(STATE_FILE, {}).get("rs_rating_reference") or {}).get("performances")
    result = compute_single_ticker_score(ticker, tf, fund, market_regime, rs_reference=rs_ref)
    result["earnings"] = get_upcoming_earnings_date(ticker)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    if tase_fallback_used:
        # v5.6.0: let the frontend tell the user "SAE not found, showing SAE.TA"
        # instead of silently swapping tickers with no explanation.
        result["tase_fallback_used"] = True
        result["original_input"] = ticker_input

    # market-status awareness (item: "נתונים אמיתיים ונכונים") - which
    # date does this price actually reflect, and is that market even open
    # today? Scoped to "בדוק מניה" only, not the shared daily engine - see
    # is_market_trading_day's docstring for why.
    try:
        result["price_as_of_date"] = closes.dropna().index[-1].strftime("%Y-%m-%d")
    except Exception:
        result["price_as_of_date"] = None
    result["market_status"] = get_market_status_for_ticker(ticker)
    # v5.8.0: price chart for 'בדוק מניה' (1D/1M/1Y/2Y/5Y + tap-for-price).
    # Separate download from the 1-year technical-analysis fetch above, so
    # the scoring inputs are byte-for-byte unchanged; any failure here only
    # drops the chart, never the whole check.
    result["chart"] = get_price_chart_data(ticker)
    try:
        result["strategy_view"] = strategy_view_for(ticker, load_json(PREDICTIONS_FILE, {}).get("strategy_book"))
    except Exception as e:
        print(f"Strategy view failed for {ticker}: {e}")
    return result


CHART_DAILY_PERIOD = "5y"
CHART_INTRADAY_INTERVAL = "5m"


def get_price_chart_data(ticker):
    """{"daily": [[YYYY-MM-DD, close], ...] (up to 5 years),
        "intraday": {"date": YYYY-MM-DD, "points": [[HH:MM, close], ...]}
                    - the LAST trading session only, in the exchange's own
                    local time}. Either part may be missing/None.
    Prices rounded to 2 decimals (see _round_price). Returns None if
    nothing at all could be fetched."""
    out = {"daily": None, "intraday": None,
           "generated_at": datetime.now(timezone.utc).isoformat()}
    try:
        data = yf.download(tickers=ticker, period=CHART_DAILY_PERIOD, interval="1d",
                           group_by="ticker", threads=False, progress=False, auto_adjust=True)
        closes = _flatten_close_series(data["Close"] if "Close" in data else data[ticker]["Close"]).dropna()
        if not closes.empty:
            out["daily"] = [[idx.strftime("%Y-%m-%d"), _round_price(v)] for idx, v in closes.items()]
    except Exception as e:
        print(f"Chart daily fetch failed for {ticker}: {e}")
    try:
        data = yf.download(tickers=ticker, period="5d", interval=CHART_INTRADAY_INTERVAL,
                           group_by="ticker", threads=False, progress=False, auto_adjust=True)
        closes = _flatten_close_series(data["Close"] if "Close" in data else data[ticker]["Close"]).dropna()
        if not closes.empty:
            last_day = closes.index[-1].date()
            session = closes[[ts.date() == last_day for ts in closes.index]]
            out["intraday"] = {
                "date": last_day.isoformat(),
                "points": [[ts.strftime("%H:%M"), _round_price(v)] for ts, v in session.items()],
            }
    except Exception as e:
        print(f"Chart intraday fetch failed for {ticker}: {e}")
    if not out["daily"] and not out["intraday"]:
        return None
    return out



def run_tomorrow_forecast(store):
    """Item 19: a genuinely SEPARATE forecast for the next trading session,
    independent of the official once-a-day Top10 (run_predictions), which
    locks in whenever it first runs each day and is tracked/graded against
    the following close - see the already_predicted_today guard there.
    This can run again the same day (e.g. near market close, on fresher
    data) or on demand, and deliberately never touches store["history"] or
    predictions.json's tracked/graded fields - it writes its own separate
    file (tomorrow_forecast.json) so a forecast run can never contaminate
    the accuracy record or interfere with grading.

    Intentionally a self-contained scan (not sharing run_predictions'
    download loop) even though that duplicates some logic - the tracked
    pipeline behind the real Top10/accuracy numbers is the one thing in
    this app that must stay untouched; a shared refactor there risks
    breaking it for the sake of a much lower-stakes feature."""
    universe = build_prediction_universe()
    market_regime = get_market_regime()
    print(f"Tomorrow-forecast universe: {len(universe)} tickers")

    technical = {}
    for i in range(0, len(universe), BATCH_SIZE):
        batch = universe[i:i + BATCH_SIZE]
        try:
            data = yf.download(
                tickers=" ".join(batch), period=PRICE_HISTORY_PERIOD, group_by="ticker",
                threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            print(f"Tomorrow-forecast batch download error: {e}")
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
            except Exception as e:
                print(f"Tomorrow-forecast technical error for {symbol}: {e}")
        time.sleep(1)

    print(f"Tomorrow-forecast: technical factors computed for {len(technical)} tickers")

    # RS Rating against THIS scan's own snapshot (may differ slightly from
    # this morning's official run, since prices moved during the day)
    perf_pairs = [(s, tf["run_up_180d"]) for s, tf in technical.items() if tf.get("run_up_180d") is not None]
    if perf_pairs:
        ranked = sorted(perf_pairs, key=lambda p: p[1])
        total = len(ranked)
        for rank, (symbol, _) in enumerate(ranked):
            technical[symbol]["rs_rating"] = round(rank / max(total - 1, 1) * 100, 1)

    prelim_scores = {}
    for symbol, tf in technical.items():
        prelim_score = compute_prediction_score(tf, market_regime)
        prelim_multiplier = (tf.get("trend_template") or {}).get("multiplier", 1.0)
        prelim_scores[symbol] = prelim_score * prelim_multiplier

    candidates = {s for s, sc in prelim_scores.items() if abs(sc) >= PREFILTER_THRESHOLD}
    # match run_predictions' candidate pool exactly (starred + monthly
    # portfolio tickers always included) - without this, a ticker that's
    # only in Top10 because it's starred/held could vanish entirely from
    # the forecast's candidate pool even though its price barely moved,
    # making the two lists look far more different than they really are.
    candidates |= (set(load_json(STARRED_FILE, [])) & set(technical.keys()))
    monthly_tickers = {h["ticker"] for h in (store.get("monthly_portfolio") or {}).get("holdings", [])}
    candidates |= (monthly_tickers & set(technical.keys()))
    print(f"Tomorrow-forecast: {len(candidates)} tickers passed the pre-filter, fetching fundamentals...")

    entries = []
    for symbol in technical:
        factors = dict(technical[symbol])
        if symbol in candidates:
            try:
                factors.update(get_fundamental_factors(symbol))
            except Exception as e:
                print(f"Tomorrow-forecast fundamentals error for {symbol}: {e}")

        score = compute_prediction_score(factors, market_regime)
        trend_multiplier = (factors.get("trend_template") or {}).get("multiplier", 1.0)
        score = round(score * trend_multiplier, 2)
        predicted = "up" if score >= 0 else "down"
        entry = {"ticker": symbol, "score": score, "predicted": predicted, **factors}
        entries.append(entry)

    breadth = [e for e in entries if abs(e["score"]) >= PREDICTION_SCORE_THRESHOLD]
    if not breadth:
        store["tomorrow_forecast"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "picks": [],
            "note": "אין מספיק מניות עם אות משמעותי כרגע לתחזית למחר.",
        }
        return store["tomorrow_forecast"]

    formula_alpha = (store.get("formula_blend") or {}).get("alpha", DEFAULT_FORMULA_ALPHA)
    rank_a = {e["ticker"]: i for i, e in enumerate(sorted(breadth, key=lambda e: abs(e["score"]), reverse=True))}
    rank_b = {e["ticker"]: i for i, e in enumerate(sorted(breadth, key=compute_risk_reward_score, reverse=True))}
    picks = select_diversified_top10(
        breadth, lambda e: compute_blended_top10_score(e, formula_alpha, rank_a, rank_b),
    )

    full_picks = []
    for entry in picks:
        breakdown = build_score_breakdown(entry, entry.get("recommendation_mean"), entry.get("upside_pct"))
        try:
            earnings = get_upcoming_earnings_date(entry["ticker"])
        except Exception as e:
            print(f"Tomorrow-forecast earnings lookup failed for {entry['ticker']}: {e}")
            earnings = None
        # start from the full entry (same shape predictionCardHtml already
        # knows how to render - rsi/macd/ma_trend/analysts/short_pct/etc.),
        # normalizing numpy scalar types to native Python (same conversion
        # run_predictions applies before anything gets JSON-saved) and
        # layering the breakdown + earnings on top.
        full_pick = {}
        for k, v in entry.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            full_pick[k] = v
        full_pick.update({
            "overall_score": breakdown["overall_score"],
            "timing_score": breakdown["timing_score"],
            "long_term_score": breakdown["long_term_score"],
            "verdict": breakdown["verdict"],
            "score_components": breakdown["components"],
            "earnings": earnings,
        })
        full_picks.append(full_pick)

    store["tomorrow_forecast"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "picks": full_picks,
    }
    return store["tomorrow_forecast"]


SELL_QUEUE_LOOKBACK_DAYS = 30
SELL_TARGET_PROXIMITY = 0.985      # within 1.5% of the resistance level recorded at pick time = "target reached"
SELL_STOP_BUFFER = 0.98           # close must be 2%+ below the pick-day support (pivot supports can sit right under the price - a 0.3% dip is noise, not a broken stop)
SELL_FLIP_MIN_SCORE = 1.5          # a flip to "down" must be at least this strong to count (not a coin-flip day)
SELL_TREND_DROP = 2                # Minervini criteria lost since the pick


def compute_position_queue(history, today_entries, real_top10_tickers, today_str,
                           lookback_days=SELL_QUEUE_LOOKBACK_DAYS):
    """v5.8.0 - every ticker picked for the Top10 within the last
    lookback_days that is NOT in today's Top10 lands in exactly one of two
    lists: sell (a concrete exit reason fired) or hold (no exit reason yet).
    Nothing silently disappears, which was the confusing part before.

    Why the rewrite (found 29.9.2026): the old rule required
    price >= resistance, but compute_support_resistance defines resistance
    as the nearest pivot ABOVE the current price (or price * 1.08 when there
    is none), so that condition could never be true and the sell list was
    permanently empty. Same trap for support. The fix is to compare today's
    price against the levels RECORDED ON THE PICK DAY - those are fixed
    numbers that the price can actually cross. It also only ever looked for
    "take profit at the top"; the most important exit - the pick is failing
    - wasn't covered at all.

    Sell triggers (any one is enough; each is a real, observable event):
      stop_broken    - price closed 2%+ below the support recorded when picked
      target_reached - price within 1.5% of (or above) the resistance
                       recorded when picked, AND a confirming reversal
                       sign (RSI > 70 or MACD turned bearish)
      direction_flip - the engine now predicts DOWN with |score| >= 1.5
      trend_broken   - lost >= 2 Minervini criteria since the pick AND is
                       below the pick price

    Returns (hold_items, sell_items), both with everything the UI needs to
    explain the row on its own (pick date/price, change since pick,
    current levels, and a plain-Hebrew reason line)."""
    cutoff = (date.fromisoformat(today_str) - timedelta(days=lookback_days)).isoformat()
    first_pick, last_pick = {}, {}
    for e in history:
        if not e.get("top10") or e.get("date", "") < cutoff or e.get("date", "") >= today_str:
            continue
        t = e["ticker"]
        if t not in first_pick or e["date"] < first_pick[t]["date"]:
            first_pick[t] = e
        if t not in last_pick or e["date"] > last_pick[t]["date"]:
            last_pick[t] = e

    today_by_ticker = {e["ticker"]: e for e in today_entries}
    hold_items, sell_items = [], []
    for ticker, lp in last_pick.items():
        if ticker in real_top10_tickers:
            continue  # still an active buy today
        fp = first_pick[ticker]
        current = today_by_ticker.get(ticker)
        if not current or current.get("price") is None:
            continue  # left the scanned universe - no fresh data to judge; ages out of the window

        price = float(current["price"])
        pick_price = fp.get("price")
        change_pct = round((price - pick_price) / pick_price * 100, 2) if pick_price else None
        pick_support, pick_resistance = lp.get("support"), lp.get("resistance")

        reasons = []
        if pick_support and price < pick_support * SELL_STOP_BUFFER:
            reasons.append(("stop_broken", f"שברה את התמיכה {pick_support:.2f} שנקבעה ביום ההמלצה"))
        if pick_resistance and price >= pick_resistance * SELL_TARGET_PROXIMITY:
            rsi_hot = (current.get("rsi") or 0) > 70
            macd_bear = current.get("macd_bullish") is False
            if rsi_hot or macd_bear:
                sign = "RSI מעל 70" if rsi_hot else "MACD התהפך לשלילי"
                reasons.append(("target_reached", f"הגיעה ליעד {pick_resistance:.2f} + {sign}"))
        if current.get("predicted") == "down" and abs(current.get("score") or 0) >= SELL_FLIP_MIN_SCORE:
            reasons.append(("direction_flip", "התחזית התהפכה לירידה"))
        picked_tt = (lp.get("trend_template") or {}).get("criteria_met")
        current_tt = (current.get("trend_template") or {}).get("criteria_met")
        if (picked_tt is not None and current_tt is not None and current_tt <= picked_tt - SELL_TREND_DROP
                and pick_price and price < pick_price):
            reasons.append(("trend_broken", f"המגמה נחלשה ({picked_tt}→{current_tt} קריטריונים) ומתחת למחיר ההמלצה"))

        breakdown = build_score_breakdown(current, current.get("recommendation_mean"), current.get("upside_pct"))
        item = {
            "ticker": ticker,
            "predicted": current.get("predicted"),
            "score": current.get("score"),
            "price": _round_price(price),
            "support": _round_price(current.get("support")),
            "resistance": _round_price(current.get("resistance")),
            "picked_on": fp.get("date"),
            "last_picked_on": lp.get("date"),
            "pick_price": _round_price(pick_price),
            "pick_support": _round_price(pick_support),
            "pick_resistance": _round_price(pick_resistance),
            "change_since_pick_pct": change_pct,
            "short_pct": current.get("short_pct"), "rs_rating": current.get("rs_rating"),
            "trend_template": current.get("trend_template"), "vcp": current.get("vcp"),
            "reasons": {code: True for code, _ in reasons},
            "reason_texts": [text for _, text in reasons],
            "overall_score": breakdown["overall_score"],
            "timing_score": breakdown["timing_score"],
            "long_term_score": breakdown["long_term_score"],
            "verdict": breakdown["verdict"],
            "score_components": breakdown["components"],
        }
        if reasons:
            sell_items.append(item)
        else:
            dist_stop = round((price - pick_support) / price * 100, 1) if pick_support else None
            item["reason_texts"] = [
                "אין איתות יציאה" + (f" · {dist_stop}% מעל התמיכה של יום ההמלצה" if dist_stop is not None else "")
            ]
            hold_items.append(item)

    sell_items.sort(key=lambda x: (x["change_since_pick_pct"] if x["change_since_pick_pct"] is not None else 0))
    hold_items.sort(key=lambda x: (x["change_since_pick_pct"] if x["change_since_pick_pct"] is not None else 0), reverse=True)
    return hold_items, sell_items


STALE_SNAPSHOT_SHARE = 0.5   # this share of US prices identical to the previous snapshot = Yahoo served stale data


def expected_last_session():
    """The last US session that should be COMPLETE right now (same rule as
    the strategy book): before 16:15 New York time it's the previous
    weekday, otherwise today (weekends roll back to Friday)."""
    ny = _ny_now()
    d = ny.date()
    if (ny.hour, ny.minute) < US_CLOSE_NY:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def yahoo_daily_is_fresh():
    """v5.14.1 (the WDC case, 3.10.2026): the daily snapshot is taken by the
    first run after 00:00 UTC - exactly when Yahoo sometimes still serves
    the previous session's daily bars. On 7 of the last 19 trading days the
    whole snapshot was a copy of the day before. Returns (fresh, last_bar,
    expected) - the snapshot is only taken when SPY's latest daily bar is
    the last completed session; otherwise the next 15-minute run retries."""
    try:
        bars = yf.Ticker("SPY").history(period="10d")["Close"].dropna()
        last = bars.index[-1].date()
    except Exception as e:
        print(f"Freshness check failed ({e}) - not blocking the run")
        return True, None, None
    exp = expected_last_session()
    return last >= exp, last, exp


def snapshot_looks_stale(store, today):
    """Second guard, after the snapshot is built: if most US prices equal
    the previous snapshot's prices exactly, Yahoo served stale data."""
    dates = sorted({e["date"] for e in store["history"] if e["date"] < today})
    if not dates:
        return False, 0.0
    prev = {e["ticker"]: e.get("price") for e in store["history"] if e["date"] == dates[-1]}
    cur = [e for e in store["history"] if e["date"] == today and not e["ticker"].endswith(".TA")]
    pairs = [(e.get("price"), prev.get(e["ticker"])) for e in cur if prev.get(e["ticker"]) is not None and e.get("price") is not None]
    if len(pairs) < 50:
        return False, 0.0
    share = sum(1 for a, b in pairs if a == b) / len(pairs)
    return share >= STALE_SNAPSHOT_SHARE, share


def run_predictions(store):
    today = date.today().isoformat()
    already_predicted_today = any(e.get("date") == today for e in store["history"])
    if already_predicted_today:
        print("Predictions already run today, skipping.")
        return
    fresh, last_bar, expected = yahoo_daily_is_fresh()
    if not fresh:
        print(f"Yahoo daily data not fresh yet (SPY last bar {last_bar}, expected {expected}) - "
              f"skipping the daily snapshot, will retry next run.")
        store["snapshot_gate"] = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "waiting",
                                  "last_bar": str(last_bar), "expected": str(expected)}
        return

    universe = build_prediction_universe()
    print(f"Prediction universe: {len(universe)} tickers")
    market_regime = get_market_regime()
    print(f"Market regime: {market_regime}")
    spy_close_for_beta = get_spy_close_series_for_beta()  # v5.6.0, see compute_beta_vs_spy

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
                    if spy_close_for_beta is not None:
                        tf["beta_vs_spy"] = compute_beta_vs_spy(closes, spy_close_for_beta)
                    technical[symbol] = tf
                    chart_data_by_ticker[symbol] = build_chart_payload(closes.dropna(), tf)
            except Exception as e:
                print(f"Technical analysis error for {symbol}: {e}")
        time.sleep(1)

    print(f"Technical factors computed for {len(technical)} tickers")

    # --- RS Rating: percentile rank (0-100) of each ticker's 6-month price
    # performance against the full universe scanned today - attached here,
    # before any scoring, so it flows into compute_prediction_score exactly
    # like every other factor. Tickers without enough history for
    # run_up_180d (n < 127 trading days) simply don't get a rating. ---
    perf_pairs = [(s, tf["run_up_180d"]) for s, tf in technical.items() if tf.get("run_up_180d") is not None]
    if perf_pairs:
        ranked = sorted(perf_pairs, key=lambda p: p[1])
        total = len(ranked)
        for rank, (symbol, _) in enumerate(ranked):
            technical[symbol]["rs_rating"] = round(rank / max(total - 1, 1) * 100, 1)
        # snapshot of the distribution so check_stock.py can rate a single
        # ad-hoc ticker against "today's market" without re-scanning the
        # whole universe itself
        store["rs_rating_reference"] = {
            "date": today,
            "performances": [p[1] for p in perf_pairs],
        }
    print(f"RS Rating attached for {len(perf_pairs)} tickers")

    # fast_rs_rating (v5.6.0): same idea as RS Rating above, but over a
    # ~20-30 trading day window (run_up_30d) instead of 6 months - "who's
    # leading right now". Feeds compute_fast_rs_score's parallel-tracked
    # comparison formula only - never the real Top10 selection.
    fast_perf_pairs = [(s, tf["run_up_30d"]) for s, tf in technical.items() if tf.get("run_up_30d") is not None]
    if fast_perf_pairs:
        fast_ranked = sorted(fast_perf_pairs, key=lambda p: p[1])
        fast_total = len(fast_ranked)
        for rank, (symbol, _) in enumerate(fast_ranked):
            technical[symbol]["fast_rs_rating"] = round(rank / max(fast_total - 1, 1) * 100, 1)
    print(f"Fast RS Rating attached for {len(fast_perf_pairs)} tickers")

    # stage 1: cheap technical-only score to decide who's worth the slow .info() call
    prelim_scores = {}
    for symbol, tf in technical.items():
        prelim_score = compute_prediction_score(tf, market_regime)
        prelim_multiplier = (tf.get("trend_template") or {}).get("multiplier", 1.0)
        prelim_scores[symbol] = prelim_score * prelim_multiplier

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
        trend_multiplier = (factors.get("trend_template") or {}).get("multiplier", 1.0)
        score = round(score * trend_multiplier, 2)
        predicted = "up" if score >= 0 else "down"

        entry = {
            "date": today, "ticker": symbol, "score": score, "predicted": predicted,
            "strong": False, "top10": False, "graded": False,  # "strong"/"top10" decided below, once every ticker has a score
            "engine_version": BACKEND_VERSION,
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

    stale, share = snapshot_looks_stale(store, today)
    if stale:
        store["history"] = [e for e in store["history"] if e["date"] != today]
        print(f"Daily snapshot rejected: {share:.0%} of US prices identical to the previous snapshot - will retry next run.")
        store["snapshot_gate"] = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "rejected_stale",
                                  "identical_share": round(share, 3)}
        return
    store["snapshot_gate"] = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "ok",
                              "session": str(expected_last_session()), "identical_share": round(share, 3)}

    # --- market breadth: how many stocks crossed the "directionally
    # significant" threshold today, regardless of how many we actually
    # highlight. If this is very high (200+), that's really a statement
    # about the whole market's direction, not about any specific stock -
    # useful to know, but not something that helps pick individual names,
    # so it's reported separately from the curated list below. ---
    today_entries = [e for e in store["history"] if e["date"] == today]
    # data_suspect entries (v5.5.0, see compute_technical_factors) are kept
    # in history for visibility/audit but never eligible for curated/Top10 -
    # their score is computed from a contaminated price series (see NFE
    # incident, 2026-09) so it can't be trusted for ranking.
    breadth = [
        e for e in today_entries
        if abs(e["score"]) >= PREDICTION_SCORE_THRESHOLD and not e.get("data_suspect")
    ]
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

    # --- full breakdown (the same overall 1-100 score + component breakdown
    # "בדוק מניה" computes) for the actual Top 10 AND the manually-tracked
    # watchlist ("המניות שלי") - shown as a clickable badge on those cards.
    # Cheap to do for this small set (10 + a handful of watchlist tickers),
    # not attempted for the whole scanned universe. Uses build_score_breakdown
    # so this can never drift from what "בדוק מניה" would compute for the
    # same ticker on the same day. ---
    watchlist_tickers = {item["ticker"] for item in load_json(WATCHLIST_FILE, [])}
    crypto_exposed_tickers = set(load_json(CRYPTO_EXPOSED_FILE, []))
    breakdown_tickers = real_top10_tickers | watchlist_tickers | crypto_exposed_tickers
    for entry in today_entries:
        if entry["ticker"] not in breakdown_tickers:
            continue
        breakdown = build_score_breakdown(
            entry, entry.get("recommendation_mean"), entry.get("upside_pct")
        )
        entry["overall_score"] = breakdown["overall_score"]
        entry["timing_score"] = breakdown["timing_score"]
        entry["long_term_score"] = breakdown["long_term_score"]
        entry["verdict"] = breakdown["verdict"]
        entry["score_components"] = breakdown["components"]
        entry["analyst_score"] = breakdown["components"]["analyst"]
        try:
            entry["earnings"] = get_upcoming_earnings_date(entry["ticker"])
        except Exception as e:
            print(f"Earnings lookup failed for {entry['ticker']}: {e}")
            entry["earnings"] = None

    # --- analyst-score momentum (v5.7.0, see compute_analyst_momentum_score):
    # widen analyst_score beyond just breakdown_tickers above to EVERY entry
    # that already has the underlying recommendation_mean/upside_pct (fetched
    # for the whole PREFILTER_THRESHOLD candidate pool earlier - no new
    # network calls here), so the parallel comparison formula has a real
    # breadth pool to rank, not just today's already-curated Top10/watchlist. ---
    for entry in today_entries:
        if entry.get("analyst_score") is None:
            entry["analyst_score"] = analyst_score_0_100(
                entry.get("recommendation_mean"), entry.get("upside_pct")
            )
    baseline_by_ticker = build_analyst_score_history_index(store["history"], today)
    for entry in today_entries:
        if entry.get("analyst_score") is not None and entry["ticker"] in baseline_by_ticker:
            entry["analyst_score_delta"] = round(entry["analyst_score"] - baseline_by_ticker[entry["ticker"]], 1)
    print(f"Analyst-score momentum: {len(baseline_by_ticker)} tickers had a usable "
          f"{ANALYST_MOMENTUM_LOOKBACK_DAYS}-day-old baseline today")

    # --- buy / hold / sell queue (v5.8.0 rewrite of item 22, see
    # compute_position_queue for the full reasoning) ---
    hold_queue, sell_queue = compute_position_queue(
        store["history"], today_entries, real_top10_tickers, today
    )
    store["sell_queue"] = {"date": today, "lookback_days": SELL_QUEUE_LOOKBACK_DAYS, "items": sell_queue}
    store["hold_queue"] = {"date": today, "lookback_days": SELL_QUEUE_LOOKBACK_DAYS, "items": hold_queue}

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

    # --- EXPERIMENT D (v5.6.0, parallel, doesn't touch the real picks): same
    # breadth pool, ranked by short-term (~20-30d) relative strength instead
    # of the existing 6-month RS Rating - see compute_fast_rs_score. ---
    fast_rs_top10 = select_diversified_top10(breadth, compute_fast_rs_score)
    fast_rs_tickers = {e["ticker"] for e in fast_rs_top10}
    for entry in today_entries:
        entry["top10_fast_rs"] = entry["ticker"] in fast_rs_tickers

    # --- EXPERIMENT E (v5.6.0, parallel, doesn't touch the real picks): Dual
    # Momentum binary gate + Low-Vol tilt gated to a bearish regime - see
    # compute_dual_momentum_lowvol_score and the VectorBT research it's from. ---
    dual_mom_lowvol_top10 = select_diversified_top10(
        breadth, lambda e: compute_dual_momentum_lowvol_score(e, market_regime),
    )
    dual_mom_lowvol_tickers = {e["ticker"] for e in dual_mom_lowvol_top10}
    for entry in today_entries:
        entry["top10_dual_momentum_lowvol"] = entry["ticker"] in dual_mom_lowvol_tickers

    # --- EXPERIMENT F (v5.7.0, parallel, doesn't touch the real picks):
    # analyst-score momentum - see compute_analyst_momentum_score. ---
    analyst_momentum_top10 = select_diversified_top10(breadth, compute_analyst_momentum_score)
    analyst_momentum_tickers = {e["ticker"] for e in analyst_momentum_top10}
    for entry in today_entries:
        entry["top10_analyst_momentum"] = entry["ticker"] in analyst_momentum_tickers

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

    # exact timestamp the prediction engine actually finished running -
    # NOT just "when the workflow last completed" (benchmark_comparison's
    # timestamp updates every run regardless of this once-a-day guard
    # above, which was the whole source of the "why don't I see new data"
    # confusion). The frontend uses this specifically to show honestly
    # whether today's Top10 predates a given code/formula change.
    store["predictions_generated_at"] = datetime.now(timezone.utc).isoformat()


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
    approach, doesn't depend on uncertain intraday data timing.)

    NOTE: this gates the whole daily engine (Top10/grading/monthly
    portfolio) using the US calendar only, for ALL tickers including .TA -
    left as-is deliberately (see ISRAEL_MARKET_HOLIDAYS_2026 below, which
    is only used for "בדוק מניה"). Touching the shared daily pipeline's
    market-day logic risks skewing the accuracy/grading numbers that are
    the whole point of the app right now, for the sake of ~6-8 days/year
    where TASE is closed but the US market isn't - Tomer explicitly chose
    to scope this fix to the on-demand single-ticker check only."""
    today = date.today()
    if today.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if today.isoformat() in US_MARKET_HOLIDAYS_2026:
        return False
    return True


# TASE full-closure holidays for 2026, from Bank of Israel's official
# Markets Division calendar (ימי פעילות חטיבת השווקים לשנת 2026) - as of
# the Jan-2026 schedule change, TASE trades Monday-Friday (not Sunday-
# Thursday anymore), Friday on shortened hours; the dates below are the
# additional Israeli-holiday closures on top of that Mon-Fri week.
ISRAEL_MARKET_HOLIDAYS_2026 = {
    "2026-03-03", "2026-03-04",  # Purim, Shushan Purim
    "2026-04-02", "2026-04-08",  # Passover (1st and last day)
    "2026-04-22",  # Independence Day
    "2026-05-22",  # Shavuot
    "2026-07-23",  # Tisha B'Av
    "2026-09-13",  # 2nd day Rosh Hashana (falls on a Sunday, already off under the new Mon-Fri week - listed for completeness)
    "2026-09-21",  # Yom Kippur
}


def get_market_status_for_ticker(ticker):
    """For "בדוק מניה" only (see is_market_trading_day's note above for why
    this isn't wired into the shared daily engine): is TODAY a trading day
    on the specific market this ticker belongs to (TASE for .TA tickers,
    US markets otherwise)? Used to warn the user the check ran on a day
    that market was closed, rather than silently showing a stale price
    with no explanation."""
    today = date.today()
    is_weekend_il = today.weekday() >= 5  # TASE is Mon-Fri since Jan 2026, same weekday numbering as US
    if is_israeli(ticker):
        market = "ת\"א (TASE)"
        is_trading_day = not is_weekend_il and today.isoformat() not in ISRAEL_MARKET_HOLIDAYS_2026
    else:
        market = "ארה\"ב (US)"
        is_trading_day = not is_weekend_il and today.isoformat() not in US_MARKET_HOLIDAYS_2026
    return {"market": market, "is_trading_day": is_trading_day}


# ===========================================================================
# v5.9.0 - three-layer portfolio: exposure rule (core), long-term quality
# list, and benchmarks. Background: the 26-year VectorBT v3 study
# (29.9.2026) found the Top10 formula adds no statistically significant
# edge over simply holding the same universe (+0.9%/yr, t=0.59), while a
# plain "index below its 200-day average -> cut exposure" rule halved the
# worst drawdown (-55% -> -26%/-31%). So the app now leads with exposure,
# measures every list against honest benchmarks, and treats the daily
# Top10 as an experiment.
# ===========================================================================
EXPOSURE_INDEX_SYMBOL = "^GSPC"
EXPOSURE_SMA_DAYS = 200
EXPOSURE_REDUCED_PCT = 50          # recommended equity exposure while the index is below its 200-day average
CASH_ANNUAL_PCT = 4.0              # simplified cash yield for the cash part of the exposure sims (same constant as the research)
CASH_DAILY = (1 + CASH_ANNUAL_PCT / 100) ** (1 / 252) - 1
EW_BENCHMARK_MAX_ABS_PCT = 50.0    # a >50% one-day move in the equal-weight benchmark is a data error (split etc.), skip it

LONG_TERM_PICKS = 10
LONG_TERM_MAX_PER_SECTOR = 3      # a long-term core list should not be half one sector
LONG_TERM_REFRESH_DAYS = 30
LONG_TERM_ENTRY_TIMING = 55        # timing at/above this -> "אפשר להיכנס עכשיו", else "להמתין לתזמון"
LONG_TERM_ENTRY_MIN_TREND = 4      # v5.11.0: Minervini criteria needed for a plain "enter"; below it a timing signal means "quality in a correction" (dip)
SHARE_CLASS_DUPLICATES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "BRK-A": "BRK-B"}  # second share class of the same company - never list both
LONG_TERM_DIP_FIRST_TRANCHE = 1 / 3  # dip entries are staged: this share of the position now
FUND_CACHE_MAX_AGE_DAYS = 30
FUND_CACHE_FETCH_PER_RUN = 60      # fundamentals are slow (.info) - spread the S&P 500 fetch over several 15-min runs
FUND_CACHE_MIN_COVERAGE = 0.9      # refresh the long-term list only once this share of S&P 500 names has fresh fundamentals


def compute_exposure_signal(store):
    """Daily exposure recommendation from the S&P 500 vs its 200-day
    average, plus the full per-date series (used by the exposure sims below
    - each day's return uses the state known at the PREVIOUS close, so there
    is no look-ahead). Rebuilt from ~2 years of index history every run."""
    try:
        closes = yf.Ticker(EXPOSURE_INDEX_SYMBOL).history(period="2y")["Close"].dropna()
    except Exception as e:
        print(f"Exposure signal fetch failed: {e}")
        return
    if len(closes) < EXPOSURE_SMA_DAYS + 1:
        return
    sma = closes.rolling(EXPOSURE_SMA_DAYS).mean()
    series = []
    for ts, c in closes.items():
        m = sma.loc[ts]
        if pd.isna(m):
            continue
        series.append([ts.strftime("%Y-%m-%d"), bool(c >= m)])
    last_close, last_sma = float(closes.iloc[-1]), float(sma.iloc[-1])
    above = last_close >= last_sma
    last_flip = None
    for i in range(len(series) - 1, 0, -1):
        if series[i][1] != series[i - 1][1]:
            last_flip = series[i][0]
            break
    store["exposure"] = {
        "as_of": series[-1][0],
        "symbol": EXPOSURE_INDEX_SYMBOL,
        "index_close": round(last_close, 2),
        "sma200": round(last_sma, 2),
        "pct_vs_sma": round((last_close / last_sma - 1) * 100, 2),
        "above_sma": bool(above),
        "recommended_exposure_pct": 100 if above else EXPOSURE_REDUCED_PCT,
        "last_flip_date": last_flip,
        "series": series[-400:],
    }


def _exposure_state_before(series_map, sorted_dates, d):
    """Exposure state (True = above SMA) as of the last index close
    strictly BEFORE date d; None if unknown."""
    import bisect
    i = bisect.bisect_left(sorted_dates, d) - 1
    return series_map[sorted_dates[i]] if i >= 0 else None


def build_exposure_sim(store, base_returns, sim_key, label, start_date=None):
    """Wraps a list of (date, daily_return_pct) with the exposure rule:
    full exposure while the index was above its 200-day average at the
    previous close, otherwise EXPOSURE_REDUCED_PCT in the strategy and the
    rest in cash. Rebuilt from scratch every run (idempotent)."""
    series = (store.get("exposure") or {}).get("series") or []
    if not base_returns or not series:
        return
    series_map = {d: a for d, a in series}
    sorted_dates = sorted(series_map)
    value, log = 100000.0, []
    for d, r in base_returns:
        above = _exposure_state_before(series_map, sorted_dates, d)
        if above is None or above:
            eff = r / 100
            exp_pct = 100
        else:
            w = EXPOSURE_REDUCED_PCT / 100
            eff = w * r / 100 + (1 - w) * CASH_DAILY
            exp_pct = EXPOSURE_REDUCED_PCT
        value *= 1 + eff
        log.append({"date": d, "value_end": round(value, 2), "exposure_pct": exp_pct})
    store[sim_key] = {
        "label": label, "start_value": 100000, "start_date": start_date or log[0]["date"],
        "value": log[-1]["value_end"], "total_return_pct": round((log[-1]["value_end"] / 100000 - 1) * 100, 2),
        "daily_log": log[-400:],
    }


def _returns_from_value_log(log, value_key="value_end"):
    """[(date, pct)] from a log of cumulative values (first entry = start)."""
    out = []
    for prev, cur in zip(log, log[1:]):
        a, b = prev.get(value_key), cur.get(value_key)
        if a and b:
            out.append((cur["date"], (b / a - 1) * 100))
    return out


def build_equal_weight_benchmark(store):
    """Equal-weight benchmark over the WHOLE scanned universe, on exactly
    the same grading schedule as the Top10 sim: for every date the Top10
    sim traded, the average graded next-day move of every scanned ticker
    (both predicted directions, data_suspect rows and >50% moves excluded).
    Answers the question the research raised: does picking 10 beat just
    holding everything we scan? Rebuilt every run."""
    top_log = (store.get("portfolio_sim") or {}).get("daily_log") or []
    if not top_log:
        return
    dates = {r["date"] for r in top_log}
    sums, counts = {}, {}
    for e in store.get("history", []):
        d = e.get("date")
        if d not in dates or not e.get("graded") or e.get("data_suspect"):
            continue
        r = e.get("actual_pct_change")
        if r is None or abs(r) > EW_BENCHMARK_MAX_ABS_PCT:
            continue
        sums[d] = sums.get(d, 0.0) + r
        counts[d] = counts.get(d, 0) + 1
    value, log = 100000.0, []
    for d in sorted(dates):
        if not counts.get(d):
            continue
        r = sums[d] / counts[d]
        value *= 1 + r / 100
        log.append({"date": d, "value_end": round(value, 2), "return_pct": round(r, 3), "n": counts[d]})
    if not log:
        return
    store["benchmark_equal_weight_sim"] = {
        "label": "החזקה שווה של כל היקום שנסרק", "start_value": 100000, "start_date": log[0]["date"],
        "value": log[-1]["value_end"], "total_return_pct": round((log[-1]["value_end"] / 100000 - 1) * 100, 2),
        "daily_log": log[-400:],
    }


def refresh_fundamentals_cache(store, today_entries, sp500_tickers, today_str):
    """Keeps store['fundamentals_cache'] fresh for S&P 500 members, a few
    dozen .info calls per run (see FUND_CACHE_FETCH_PER_RUN) so no single
    15-minute run gets slow. Returns the share of scanned S&P 500 names
    with a fresh entry."""
    cache = store.setdefault("fundamentals_cache", {})
    cutoff = (date.fromisoformat(today_str) - timedelta(days=FUND_CACHE_MAX_AGE_DAYS)).isoformat()
    members = sorted({e["ticker"] for e in today_entries if e["ticker"] in sp500_tickers})
    if not members:
        return 0.0
    stale = [t for t in members if (cache.get(t) or {}).get("fetched", "") < cutoff]
    for t in stale[:FUND_CACHE_FETCH_PER_RUN]:
        try:
            f = get_fundamental_factors(t)
        except Exception as ex:
            print(f"Fundamentals cache fetch failed for {t}: {ex}")
            continue
        cache[t] = {
            "fetched": today_str,
            "quality_score": f.get("quality_score"),
            "recommendation_mean": f.get("recommendation_mean"),
            "upside_pct": _round_price(f.get("upside_pct"), 1),
            "market_cap": f.get("market_cap"),
            "sector": f.get("sector"),
            "name": f.get("_company_name"),
        }
    fresh = sum(1 for t in members if (cache.get(t) or {}).get("fetched", "") >= cutoff)
    return fresh / len(members)


def manage_long_term_picks(store, today_entries):
    """v5.9.0 - the 🏛 long-term list: 10 large-cap S&P 500 companies ranked
    by business quality + analyst rating (build_score_breakdown's long-term
    lens), max LONG_TERM_MAX_PER_SECTOR per sector, refreshed every
    LONG_TERM_REFRESH_DAYS. Each pick also carries today's timing score and
    an entry tag, so a strong company that isn't moving yet reads as
    'wait for timing' rather than as a bad stock.

    Its own ₪100,000 sim (long_term_sim): equal weight at each refresh,
    marked to market every run from today's scanned prices; value is
    realized and re-split equally at the next refresh. Not backtested -
    there are no free historical fundamentals - so it is measured live
    from its first refresh."""
    today = date.today().isoformat()
    lt = store.setdefault("long_term_picks", {"refreshed_on": None, "next_refresh_date": None, "picks": [], "history": []})
    sim = store.setdefault("long_term_sim", {"start_value": 100000, "start_date": None, "value": 100000.0,
                                              "holdings": [], "daily_log": []})
    by_ticker = {e["ticker"]: e for e in today_entries}
    sp500 = set(get_sp500_tickers())
    coverage = refresh_fundamentals_cache(store, today_entries, sp500, today)
    cache = store.get("fundamentals_cache") or {}

    # mark the running sim to market first (before any rebalance)
    if sim["holdings"]:
        total = 0.0
        for h in sim["holdings"]:
            p = (by_ticker.get(h["ticker"]) or {}).get("price")
            if p:
                h["last_price"] = float(p)
            total += h["alloc"] * h["last_price"] / h["entry_price"]
        sim["value"] = round(total, 2)
        if sim["daily_log"] and sim["daily_log"][-1]["date"] == today:
            sim["daily_log"][-1]["value_end"] = sim["value"]
        else:
            sim["daily_log"].append({"date": today, "value_end": sim["value"]})
        sim["daily_log"] = sim["daily_log"][-400:]

    due = (not lt["picks"] or (lt.get("next_refresh_date") or "") <= today
           # v5.11.0 self-heal: a list built before the share-class filter existed is rebuilt once
           or any(p["ticker"] in SHARE_CLASS_DUPLICATES for p in lt["picks"]))
    if due and coverage >= FUND_CACHE_MIN_COVERAGE:
        pool = []
        for t, f in cache.items():
            e = by_ticker.get(t)
            if (t not in sp500 or e is None or not e.get("price") or f.get("quality_score") is None
                    or t in SHARE_CLASS_DUPLICATES
                    or (f.get("market_cap") or 0) < LARGE_CAP_MIN_MARKET_CAP):
                continue
            analyst = analyst_score_0_100(f.get("recommendation_mean"), f.get("upside_pct"))
            parts = [v for v in (f["quality_score"], analyst) if v is not None]
            pool.append({**e, "sector": f.get("sector") or e.get("sector"), "company_name": f.get("name"),
                         "quality_score": f["quality_score"], "analyst_score": analyst,
                         "long_term_score": round(sum(parts) / len(parts), 1)})
        chosen = select_diversified_top10(pool, lambda x: x["long_term_score"], max_per_sector=LONG_TERM_MAX_PER_SECTOR)[:LONG_TERM_PICKS]
        if chosen:
            # realize the old basket, re-split equally into the new one
            value = sim["value"] if sim["holdings"] else 100000.0
            if not sim["start_date"]:
                sim["start_date"] = today
                sim["daily_log"] = [{"date": today, "value_end": round(value, 2)}]
            alloc = value / len(chosen)
            sim["holdings"] = [{"ticker": c["ticker"], "entry_price": float(c["price"]), "last_price": float(c["price"]),
                                "alloc": alloc} for c in chosen]
            lt["history"].append({"date": today, "tickers": [c["ticker"] for c in chosen], "value_at_refresh": round(value, 2)})
            lt["history"] = lt["history"][-24:]
            lt["refreshed_on"] = today
            lt["next_refresh_date"] = (date.fromisoformat(today) + timedelta(days=LONG_TERM_REFRESH_DAYS)).isoformat()
            lt["picks"] = [{
                "ticker": c["ticker"], "company_name": c.get("company_name"), "sector": c.get("sector"),
                "quality_score": c["quality_score"], "analyst_score": c["analyst_score"],
                "long_term_score": c["long_term_score"], "entry_price": _round_price(c["price"]),
                "picked_on": today,
            } for c in chosen]
    lt["fundamentals_coverage_pct"] = round(coverage * 100, 1)

    # today's timing + entry tag for every current pick (cheap, every run)
    for p in lt["picks"]:
        e = by_ticker.get(p["ticker"])
        if not e:
            continue
        p["price"] = _round_price(e.get("price"))
        p["predicted"] = e.get("predicted")
        p["score"] = e.get("score")
        p["support"], p["resistance"] = _round_price(e.get("support")), _round_price(e.get("resistance"))
        p["trend_template"] = e.get("trend_template")
        p["timing_score"] = compute_timing_score(e)
        # v5.11.0 (the AVGO case, 30.9.2026): timing alone said "buy" while the
        # long trend was 1/7. Now: timing + a healthy trend = enter; timing
        # without the trend = dip (quality company in a correction, staged
        # entry); no timing = wait. Unknown trend counts as not confirmed.
        tt_met = (p["trend_template"] or {}).get("criteria_met")
        if p["timing_score"] < LONG_TERM_ENTRY_TIMING:
            p["entry_tag"] = "wait"
        elif tt_met is not None and tt_met >= LONG_TERM_ENTRY_MIN_TREND:
            p["entry_tag"] = "enter"
        else:
            p["entry_tag"] = "dip"
        p["change_since_pick_pct"] = round((e["price"] / p["entry_price"] - 1) * 100, 2) if p.get("entry_price") else None
        p["overall_score"] = p["long_term_score"]
        p["score_components"] = {"timing": p["timing_score"], "quality": p["quality_score"], "analyst": p["analyst_score"]}
        p["verdict"] = compute_verdict(p["timing_score"], p["long_term_score"], has_fundamentals=True)
    try:
        update_valuations(store, lt["picks"], today)
    except Exception as e:
        print(f"Valuation update failed: {e}")


def build_strategy_comparison(store):
    """One table of every tracked strategy vs its honest benchmarks, each
    with its own start date (they did not all start on the same day - the
    frontend shows that, so no row is compared against an unfair window)."""
    rows = []

    def add(key, label, layer, sim_key, value_key="value"):
        sim = store.get(sim_key) or {}
        v = sim.get(value_key)
        if v is None:
            return
        start = sim.get("start_date") or ((sim.get("daily_log") or [{}])[0].get("date"))
        if not start:
            return  # not started yet (e.g. long-term list still collecting fundamentals)
        rows.append({"key": key, "label": label, "layer": layer, "start_date": start,
                     "value": round(float(v), 2), "return_pct": round((float(v) / 100000 - 1) * 100, 2)})

    add("core_exposure", "S&P 500 + כלל חשיפה", "core", "core_sim_sp500_exposure")
    add("sp500", "S&P 500 - החזקה רגילה", "benchmark", "index_sim_sp500")
    add("long_term", "🏛 טווח ארוך - 10 מניות איכות", "long", "long_term_sim")
    add("long_term_exposure", "🏛 טווח ארוך + כלל חשיפה", "long", "long_term_sim_exposure")
    add("top10", "⚡ טווח קצר - Top10 (ניסוי)", "short", "portfolio_sim")
    add("top10_exposure", "⚡ Top10 + כלל חשיפה", "short", "portfolio_sim_exposure")
    add("recommendations", "🎯 ההמלצות לטווח קצר (טאב המלצות)", "short", "portfolio_sim_recommendations")
    add("live", "📒 התיק החי (3 מניות, כולל עמלות)", "live", "live_portfolio")
    for key in ("A", "B"):
        sb = (store.get("strategy_book") or {}).get(key)
        if sb and sb.get("start_date"):
            row = {"key": f"book_{key}", "label": f"{sb['label']} (תיק אסטרטגיה, כולל עמלות)", "layer": "short",
                   "start_date": sb["start_date"], "value": sb["value"], "return_pct": sb["total_return_pct"]}
            if sb.get("spy_same_window_pct") is not None:   # starts before our own S&P sim - use the book's own SPY window
                row["sp500_same_window_pct"] = sb["spy_same_window_pct"]
                row["vs_sp500_pct"] = round(row["return_pct"] - row["sp500_same_window_pct"], 2)
            rows.append(row)
    add("long_enter", "🏛 ✅ טווח ארוך - כניסה במגמה תקינה", "long", "portfolio_sim_long_enter")
    add("long_dip", "🏛 🟡 טווח ארוך - קנייה בתיקון", "long", "portfolio_sim_long_dip")
    add("long_value", "🏛 💎 טווח ארוך - זולה מול השווי שלה", "long", "portfolio_sim_long_value")
    add("equal_weight", "החזקה שווה של כל היקום שנסרק", "benchmark", "benchmark_equal_weight_sim")
    mv = (store.get("portfolios_view") or {}).get("monthly")
    if mv and mv.get("start_date"):
        rows.append({"key": "monthly", "label": "📅 התיק החודשי", "layer": "long", "start_date": mv["start_date"],
                     "value": mv["marked_value"], "return_pct": mv["return_pct"]})
    # v5.10.1: each row's gap vs S&P 500 is measured over THAT row's own
    # window (the long-term list started 30.9 and was being compared with
    # S&P's return since 21.8)
    sp = store.get("index_sim_sp500") or {}
    sp_log = sp.get("daily_log") or []
    if sp_log and sp.get("value") is not None:
        for r in rows:
            if r.get("sp500_same_window_pct") is not None:
                continue
            base = None
            for e in sp_log:
                if r["start_date"] and e["date"] <= r["start_date"]:
                    base = e["value_end"]
            if base is None:
                base = sp.get("start_value", 100000)
            r["sp500_same_window_pct"] = round((float(sp["value"]) / base - 1) * 100, 2)
            r["vs_sp500_pct"] = round(r["return_pct"] - r["sp500_same_window_pct"], 2)
    store["strategy_comparison"] = {"updated_at": datetime.now(timezone.utc).isoformat(), "rows": rows}


def update_three_layer_views(store):
    """Runs every cycle after the existing sims/benchmarks are updated."""
    compute_exposure_signal(store)
    build_equal_weight_benchmark(store)
    sp_log = (store.get("index_sim_sp500") or {}).get("daily_log") or []
    build_exposure_sim(store, _returns_from_value_log(sp_log), "core_sim_sp500_exposure", "S&P 500 + כלל חשיפה",
                       start_date=sp_log[0]["date"] if sp_log else None)
    top_log = (store.get("portfolio_sim") or {}).get("daily_log") or []
    build_exposure_sim(store, [(r["date"], r["return_pct"]) for r in top_log], "portfolio_sim_exposure", "Top10 + כלל חשיפה")
    lt_log = (store.get("long_term_sim") or {}).get("daily_log") or []
    build_exposure_sim(store, _returns_from_value_log(lt_log), "long_term_sim_exposure", "טווח ארוך + כלל חשיפה",
                       start_date=lt_log[0]["date"] if lt_log else None)
    build_strategy_comparison(store)


# ===========================================================================
# v5.10.0 - "🎯 סיכום והמלצות" tab: a rules-based recommendation sheet,
# rebuilt every cycle from data the app already has (no paid API). Every
# rule is explicit so the tab can say exactly WHY something is recommended,
# and the short-term recommendations are flagged on the history entries
# (rec_short) so they get graded and simulated like any other list.
# ===========================================================================
REC_SHORT_MAX = 3
REC_SHORT_MIN_RR = 1.5             # (target - price) / (price - stop) must be at least this
REC_SHORT_EARNINGS_BLACKOUT_DAYS = 7
REC_LONG_MAX = 3
REC_LAYER_SHARE = {"core": 70, "long": 20, "short": 10}   # % of the portfolio, a suggestion (see the tab's disclaimer)
REC_WARN_FLIP_SCORE = 1.5


def _short_rec_candidate(e):
    """Returns (ok, reward_risk, stop, reasons_failed) for one Top10 entry."""
    failed = []
    if e.get("verdict") is None or e.get("timing_score") is None:
        # entry written before its breakdown existed (e.g. the day a new
        # backend version ships mid-day) - compute it now, same formula
        b = build_score_breakdown(e, e.get("recommendation_mean"), e.get("upside_pct"))
        e.setdefault("timing_score", b["timing_score"])
        e.setdefault("long_term_score", b["long_term_score"])
        if e.get("verdict") is None:
            e["verdict"] = b["verdict"]
    price, sup, res = e.get("price"), e.get("support"), e.get("resistance")
    if e.get("predicted") != "up":
        failed.append("התחזית אינה לעלייה")
    if e.get("verdict") != "buy":
        failed.append("אין תג ✅")
    if not (e.get("trend_template") or {}).get("passes"):
        failed.append("מגמת Minervini לא תקינה")
    if e.get("data_suspect"):
        failed.append("נתון חשוד")
    stop = sup * SELL_STOP_BUFFER if sup else None
    rr = None
    if price and stop and res and price > stop:
        rr = (res - price) / (price - stop)
        if rr < REC_SHORT_MIN_RR:
            failed.append(f"יחס רווח/סיכון {rr:.1f} (נדרש {REC_SHORT_MIN_RR})")
    else:
        failed.append("אין יעד/סטופ")
    earn = e.get("earnings") or {}
    if earn.get("days_away") is not None and earn["days_away"] <= REC_SHORT_EARNINGS_BLACKOUT_DAYS:
        failed.append(f"דוח כספי בעוד {earn['days_away']} ימים")
    return (not failed), rr, stop, failed


def backfill_rec_flags(store):
    """One-time: apply the short-term recommendation rules to past days'
    Top10 entries (each day judged only on that day's own data - no look-
    ahead), so the recommendations sim has a track record from day one
    instead of starting empty. Marked as backfilled in the UI."""
    if store.get("rec_backfill_done"):
        return
    today = date.today().isoformat()
    by_date = {}
    for e in store.get("history", []):
        if e.get("top10") and e.get("date", "") < today:
            by_date.setdefault(e["date"], []).append(e)
    first = None
    for d in sorted(by_date):
        passing = []
        for e in by_date[d]:
            probe = dict(e)
            ok, rr, _, _ = _short_rec_candidate(probe)
            if ok:
                passing.append((probe.get("timing_score") or 0, rr or 0, e["ticker"]))
        chosen = {t for _, _, t in sorted(passing, reverse=True)[:REC_SHORT_MAX]}
        for e in by_date[d]:
            e["rec_short"] = e["ticker"] in chosen
        if chosen and first is None:
            first = d
    store["rec_backfill_done"] = {"at": today, "first_rec_date": first}


# ===========================================================================
# v5.14.0 - 💎 valuation lens for the long-term list: is this quality
# company cheap versus ITS OWN recent history? (Tomer, 2.10.2026: "buy good
# companies below their value".) Yahoo has no long fundamentals history, so
# this can't be backtested - it is measured live from today in its own sim.
#   P/E now (price / trailing EPS) vs the average P/E at the last ~4 fiscal
#   year ends (year-end price / that year's diluted EPS). 💎 when the P/E is
#   at least 15% below the company's own average and every EPS was positive
#   (a P/E on a loss year is meaningless). Free-cash-flow yield is shown as
#   a second opinion but doesn't decide the tag.
# ===========================================================================
VALUE_DISCOUNT = 0.85              # P/E now <= 85% of the company's own average -> 💎
VALUE_CACHE_DAYS = 30              # financial statements change quarterly - refetch monthly


def _row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            return df.loc[n]
    return None


def fetch_valuation_history(ticker):
    """Yearly EPS, free cash flow and the year-end price, from yfinance.
    Returns {"years": [{"date", "eps", "fcf", "price"}...], "shares"} or None."""
    tk = yf.Ticker(ticker)
    inc = tk.income_stmt
    cf = tk.cashflow
    eps = _row(inc, ["Diluted EPS", "Basic EPS"])
    fcf = _row(cf, ["Free Cash Flow"])
    if eps is None:
        return None
    hist = tk.history(period="6y")["Close"].dropna()
    if hist.empty:
        return None
    hist.index = hist.index.tz_localize(None) if getattr(hist.index, "tz", None) is not None else hist.index
    years = []
    for d, v in eps.items():
        try:
            d = pd.Timestamp(d)
        except Exception:
            continue
        before = hist[hist.index <= d]
        if before.empty or v is None or not np.isfinite(float(v)):
            continue
        f = fcf.get(d) if fcf is not None else None
        years.append({"date": d.strftime("%Y-%m-%d"), "eps": float(v), "price": round(float(before.iloc[-1]), 2),
                      "fcf": float(f) if f is not None and np.isfinite(float(f)) else None})
    years.sort(key=lambda y: y["date"])
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        pass
    return {"years": years[-5:], "shares": info.get("sharesOutstanding"), "trailing_eps": info.get("trailingEps")}


def valuation_view(hist_v, price):
    """Pure function (testable): current vs own-average P/E and FCF yield."""
    if not hist_v or not price:
        return None
    ys = hist_v.get("years") or []
    pes = [y["price"] / y["eps"] for y in ys if y.get("eps") and y["eps"] > 0]
    all_positive = bool(ys) and all((y.get("eps") or 0) > 0 for y in ys)
    t_eps = hist_v.get("trailing_eps") or (ys[-1]["eps"] if ys else None)
    out = {"years": len(ys), "all_eps_positive": all_positive}
    if t_eps and t_eps > 0 and len(pes) >= 3:
        pe_now = price / t_eps
        pe_avg = sum(pes) / len(pes)
        out.update({"pe_now": round(pe_now, 1), "pe_avg": round(pe_avg, 1),
                    "pe_vs_avg_pct": round((pe_now / pe_avg - 1) * 100, 1)})
        out["cheap"] = bool(all_positive and pe_now <= pe_avg * VALUE_DISCOUNT)
    else:
        out["cheap"] = False
    shares = hist_v.get("shares")
    fys = [y for y in ys if y.get("fcf") is not None]
    if shares and fys:
        fy_now = fys[-1]["fcf"] / (shares * price) * 100
        fy_avg = sum(y["fcf"] / (shares * y["price"]) * 100 for y in fys) / len(fys)
        out.update({"fcf_yield_now_pct": round(fy_now, 2), "fcf_yield_avg_pct": round(fy_avg, 2)})
    return out


def update_valuations(store, picks, today):
    cache = store.setdefault("valuation_cache", {})
    cutoff = (date.fromisoformat(today) - timedelta(days=VALUE_CACHE_DAYS)).isoformat()
    for p in picks:
        t = p["ticker"]
        c = cache.get(t)
        if not c or c.get("fetched", "") < cutoff:
            try:
                h = fetch_valuation_history(t)
                if h:
                    cache[t] = {"fetched": today, **h}
                    c = cache[t]
            except Exception as e:
                print(f"Valuation fetch failed for {t}: {e}")
        p["valuation"] = valuation_view(c, p.get("price")) if c else None
        p["value_tag"] = bool(p["valuation"] and p["valuation"].get("cheap"))


def build_recommendations(store):
    today = date.today().isoformat()
    exposure = store.get("exposure") or {}
    exp_pct = exposure.get("recommended_exposure_pct", 100)
    todays = [e for e in store.get("history", []) if e.get("date") == today]
    by_ticker = {e["ticker"]: e for e in todays}
    top10 = [e for e in todays if e.get("top10")]

    # --- short term (v5.13.0): the two research-validated strategies ---
    # The previous rule set (Top10 + ✅ + Minervini + reward/risk) lost money
    # in its own trade-based record and is retired; its history flags
    # (rec_short) are kept untouched so that record stays visible, frozen.
    book = store.get("strategy_book") or {}
    layer_short = REC_LAYER_SHARE["short"] * exp_pct / 100
    short = []
    for key in ("A", "B"):
        sb = book.get(key) or {}
        cfg = BOOK_STRATEGIES[key]
        for pnd in (sb.get("pending") or [])[:REC_SHORT_MAX]:
            e = by_ticker.get(pnd["ticker"]) or {}
            b = build_score_breakdown(e, e.get("recommendation_mean"), e.get("upside_pct")) if e else {}
            sig_d = date.fromisoformat(pnd["signal_date"])
            exit_d, n_bd = sig_d, 0
            while n_bd < cfg["hold"]:
                exit_d += timedelta(days=1)
                n_bd += exit_d.weekday() < 5
            if key == "A":
                score = 90 + min(max((pnd.get("rank") or 0) * 100 - 12, 0), 9)
                why = f"ירדה {round((pnd.get('rank') or 0) * 100, 1)}% ב-5 ימים, במגמה עולה"
            else:
                score = 85 + min(max((pnd.get("rank") or 0) - 3, 0), 4)
                why = f"קפיצת פתיחה בווליום פי {round(pnd.get('rank') or 0, 1)}"
            short.append({"ticker": pnd["ticker"], "strategy": key, "strategy_label": cfg["label"],
                          "price": pnd["close"], "signal_date": pnd["signal_date"], "hold_sessions": cfg["hold"],
                          "exit_estimate": exit_d.isoformat(), "why": why, "already_held": pnd.get("already_held"),
                          "target": None, "stop": None, "sector": e.get("sector"),
                          "timing_score": b.get("timing_score"), "long_term_score": b.get("long_term_score"),
                          "book_score": round(score, 1)})
    short.sort(key=lambda r: -r["book_score"])
    for r in short:
        r["weight_pct"] = round(layer_short / max(len(short), 1), 1)
    near_miss = []

    # --- long term ---
    lt = store.get("long_term_picks") or {}
    picks = lt.get("picks") or []
    sectors = {}
    for p in picks:
        sectors.setdefault(p.get("sector") or "לא ידוע", []).append(p)
    sector_rows = []
    for name, ps in sectors.items():
        scores = [p["long_term_score"] for p in ps if p.get("long_term_score") is not None]
        sector_rows.append({"sector": name, "count": len(ps),
                            "avg_long_term_score": round(sum(scores) / len(scores), 1) if scores else None,
                            "enter_count": sum(1 for p in ps if p.get("entry_tag") == "enter"),
                            "dip_count": sum(1 for p in ps if p.get("entry_tag") == "dip")})
    sector_rows.sort(key=lambda r: ((r["enter_count"] + r["dip_count"] > 0), r["avg_long_term_score"] or 0, r["count"]), reverse=True)
    lead = sector_rows[0]["sector"] if sector_rows else None
    enter = sorted([p for p in picks if p.get("entry_tag") == "enter"],
                   key=lambda p: ((p.get("sector") or "לא ידוע") == lead, p.get("long_term_score") or 0), reverse=True)
    def _long_row(p):
        tt = p.get("trend_template") or {}
        return {"ticker": p["ticker"], "company_name": p.get("company_name"), "sector": p.get("sector"),
                "price": p.get("price"), "long_term_score": p.get("long_term_score"),
                "quality_score": p.get("quality_score"), "analyst_score": p.get("analyst_score"),
                "timing_score": p.get("timing_score"), "trend_met": tt.get("criteria_met"),
                "trend_total": tt.get("criteria_total"), "entry_tag": p.get("entry_tag"),
                "value_tag": p.get("value_tag"), "valuation": p.get("valuation")}
    long_recs = [_long_row(p) for p in enter[:REC_LONG_MAX]]
    dips = sorted([p for p in picks if p.get("entry_tag") == "dip"],
                  key=lambda p: ((p.get("sector") or "לא ידוע") == lead, p.get("long_term_score") or 0), reverse=True)
    dip_recs = [_long_row(p) for p in dips[:REC_LONG_MAX]]
    layer_long = REC_LAYER_SHARE["long"] * exp_pct / 100
    n_long = max(len(long_recs) + len(dip_recs), 1)
    for r in long_recs:
        r["weight_pct"] = round(layer_long / n_long, 1)
    for r in dip_recs:
        r["weight_pct"] = round(layer_long / n_long, 1)
        r["first_tranche_pct"] = round(r["weight_pct"] * LONG_TERM_DIP_FIRST_TRANCHE, 1)
    # measured like everything else: flag today's entries so each entry type gets its own graded sim
    enter_set, dip_set = {r["ticker"] for r in long_recs}, {r["ticker"] for r in dip_recs}
    value_set = {p["ticker"] for p in picks if p.get("value_tag")}
    for e in todays:
        e["rec_long_enter"] = e["ticker"] in enter_set
        e["rec_long_dip"] = e["ticker"] in dip_set
        e["rec_long_value"] = e["ticker"] in value_set
    watch = [{"ticker": p["ticker"], "sector": p.get("sector"), "long_term_score": p.get("long_term_score"),
              "timing_score": p.get("timing_score")} for p in picks if p.get("entry_tag") == "wait"]

    # --- your holdings ---
    warnings = []
    my = load_json(MY_PORTFOLIO_FILE, [])
    holdings = [("התיק שלי", h.get("ticker")) for h in my if h.get("ticker")]
    holdings += [("תיק חודשי", h.get("ticker")) for h in (store.get("monthly_portfolio") or {}).get("holdings", [])]
    not_scanned = []
    rec_note = {r["ticker"]: "מופיעה גם בהמלצות: ✅ לקנות לטווח ארוך" for r in long_recs}
    rec_note.update({r["ticker"]: "מופיעה גם בהמלצות: 🟡 איכותית בתיקון - המגמה הארוכה שבורה, התזמון הקצר חיובי" for r in dip_recs})
    rec_note.update({r["ticker"]: "מופיעה גם בהמלצות: ⚡ קנייה לטווח קצר" for r in short})
    for src, t in holdings:
        e = by_ticker.get(t)
        if not e:
            not_scanned.append(t)
            continue
        why = []
        if e.get("predicted") == "down" and abs(e.get("score") or 0) >= REC_WARN_FLIP_SCORE:
            why.append("תחזית ירידה משמעותית")
        tt = e.get("trend_template") or {}
        # v5.11.0: not for the monthly portfolio - it buys large caps that fell
        # from their highs BY DESIGN, so a weak trend there is the strategy,
        # not a warning (it fired on 6 of 10 holdings on 30.9.2026)
        if src != "תיק חודשי" and tt.get("criteria_total") and tt.get("criteria_met", 7) <= 2:
            why.append(f"מגמה חלשה ({tt.get('criteria_met')}/{tt.get('criteria_total')})")
        earn = e.get("earnings") or {}
        if earn.get("days_away") is not None and earn["days_away"] <= REC_SHORT_EARNINGS_BLACKOUT_DAYS:
            why.append(f"דוח כספי בעוד {earn['days_away']} ימים")
        if why:
            warnings.append({"source": src, "ticker": t, "price": _round_price(e.get("price")), "reasons": why,
                             "note": rec_note.get(t)})

    # --- what's working ---
    rows = (store.get("strategy_comparison") or {}).get("rows") or []
    sp = next((r for r in rows if r["key"] == "sp500"), None)
    working = []
    for r in rows:
        if r["key"] == "sp500" or sp is None:
            continue
        working.append({"label": r["label"], "start_date": r["start_date"], "return_pct": r["return_pct"],
                        "vs_sp500_pct": r.get("vs_sp500_pct", round(r["return_pct"] - sp["return_pct"], 2))})
    working.sort(key=lambda r: r["vs_sp500_pct"], reverse=True)

    rec_sim = store.get("portfolio_sim_recommendations") or {}
    store["recommendations"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": today,
        "exposure_pct": exp_pct,
        "exposure": {k: exposure.get(k) for k in ("above_sma", "pct_vs_sma", "last_flip_date")},
        # every layer scales with the exposure rule; the rest is cash
        "layers": {k: round(v * exp_pct / 100, 1) for k, v in REC_LAYER_SHARE.items()},
        "cash_pct": round(100 - exp_pct, 1),
        "short": short,
        "short_near_miss": near_miss[:5],
        "short_rules": "🔄 ירידה חדה: מניית S&P 500 שירדה 12%+ ב-5 ימים מעל ממוצע 200 - קנייה בפתיחה הבאה, מכירה בסגירה של יום המסחר ה-10. 🚀 קפיצת חדשות: פער פתיחה 5%+, ווליום פי 3, סגירה בחצי העליון, כשה-S&P מעל ממוצע 200 - החזקה 60 ימי מסחר. בלי סטופ ובלי יעד, בדיוק כמו במחקר.",
        "book_session": book.get("session"),
        "long_lead_sector": lead,
        "long_sectors": sector_rows,
        "long": long_recs,
        "long_dip": dip_recs,
        "long_rules": f"✅ לקנות = תזמון {LONG_TERM_ENTRY_TIMING}+ ומגמת Minervini {LONG_TERM_ENTRY_MIN_TREND}/7 ומעלה · 🟡 בתיקון = תזמון {LONG_TERM_ENTRY_TIMING}+ אבל מגמה חלשה, כניסה בשלבים ({round(LONG_TERM_DIP_FIRST_TRANCHE * 100)}% מהפוזיציה עכשיו) · ⏳ להמתין = תזמון נמוך",
        "long_watch": watch,
        "long_coverage_pct": lt.get("fundamentals_coverage_pct"),
        "warnings": warnings,
        "not_scanned": sorted(set(not_scanned)),
        "working": working,
        "rec_sim": {k: rec_sim.get(k) for k in ("value", "last_processed_date")} if rec_sim else None,
        "rec_backfilled_from": (store.get("rec_backfill_done") or {}).get("first_rec_date"),
    }


# ===========================================================================
# v5.12.0 - 📒 "תיקים": a LIVE paper portfolio ($100,000) that follows the
# recommendations tab like a real investor would - 3 positions, whole
# shares, commissions and slippage, smart (hysteresis) replacement - plus a
# trade-based track record for the short-term recommendations. All values
# in USD; TASE prices (Yahoo quotes them in agorot) are converted with the
# live USD/ILS rate.
# ===========================================================================
LIVE_START_USD = 100000.0
LIVE_SLOTS = 3
LIVE_COMMISSION_PCT = 0.1          # per buy and per sell
LIVE_MIN_FEE_USD = 5.0
LIVE_SLIPPAGE_PCT = 0.05           # buys fill this much higher, sells this much lower
LIVE_OUT_DAYS_TO_SELL = 3          # consecutive days out of the recommendations before a holding may be replaced
LIVE_MIN_HOLD_DAYS = 5             # trading days; stops/targets ignore this
LIVE_REPLACE_GAP = 10              # a replacement must score at least this much higher than the holding
LIVE_TRIM_TOLERANCE = 1.10         # trim only when invested > 110% of the exposure target (avoids fee churn)
LIVE_TOPUP_BELOW = 0.80            # top a position up only when it is under 80% of its slot
REC_TRADE_MAX_DAYS = 20            # trade-based record: a short-term rec closes at target, stop, or after this many trading days


def _usdils_rate():
    ind = (load_json(CURRENT_PRICES_FILE, {}) or {}).get("indices") or {}
    r = (ind.get("usdils") or {}).get("price")
    try:
        return float(r) if r else None
    except (TypeError, ValueError):
        return None


def to_usd(ticker, native_price, usdils):
    """TASE quotes are in agorot -> shekels -> dollars. Everything else is
    treated as USD (same convention as the rest of the app's sims)."""
    if native_price is None:
        return None
    p = float(native_price)
    if ticker.endswith(".TA"):
        if not usdils:
            return None
        return p / 100.0 / usdils
    return p


def _combined_score(timing, long_term):
    if timing is None and long_term is None:
        return None
    if long_term is None:
        return timing
    if timing is None:
        return long_term
    return round(OVERALL_TIMING_WEIGHT * timing + (1 - OVERALL_TIMING_WEIGHT) * long_term, 1)


def _live_fee(notional):
    return round(max(LIVE_MIN_FEE_USD, abs(notional) * LIVE_COMMISSION_PCT / 100), 2)


def _live_candidates(store):
    """All current ✅ recommendations (short-term + long-term 'enter'),
    ranked by the combined score. Dip entries are staged by definition,
    so they are not full-size live positions."""
    rec = store.get("recommendations") or {}
    cands = {}
    for r in rec.get("short") or []:
        cands[r["ticker"]] = {"ticker": r["ticker"], "source": "short", "stop": r.get("stop"), "target": r.get("target"),
                              "max_hold": r.get("hold_sessions"), "strategy": r.get("strategy"),
                              "why": f"{r.get('strategy_label', 'טווח קצר')}: {r.get('why', '')}",
                              "score": r.get("book_score") or _combined_score(r.get("timing_score"), r.get("long_term_score"))}
    for r in rec.get("long") or []:
        if r["ticker"] in cands:
            continue
        cands[r["ticker"]] = {"ticker": r["ticker"], "source": "long", "stop": None, "target": None,
                              "score": _combined_score(r.get("timing_score"), r.get("long_term_score"))}
    return sorted(cands.values(), key=lambda c: c["score"] or 0, reverse=True)


def manage_live_portfolio(store, today_entries):
    today = date.today().isoformat()
    lp = store.setdefault("live_portfolio", {
        "start_date": today, "start_value": LIVE_START_USD, "currency": "USD", "cash": LIVE_START_USD,
        "holdings": [], "trades": [], "daily_log": [], "last_decision_date": None,
        "settings": {"commission_pct": LIVE_COMMISSION_PCT, "min_fee_usd": LIVE_MIN_FEE_USD,
                     "slippage_pct": LIVE_SLIPPAGE_PCT, "slots": LIVE_SLOTS},
    })
    usdils = _usdils_rate()
    live = (load_json(CURRENT_PRICES_FILE, {}) or {}).get("prices") or {}
    by_ticker = {e["ticker"]: e for e in today_entries}

    def native_now(t):
        v = (live.get(t) or {}).get("price")
        if v is None:
            v = (by_ticker.get(t) or {}).get("price")
        return v

    # 1. mark to market (every run)
    for h in lp["holdings"]:
        n = native_now(h["ticker"])
        u = to_usd(h["ticker"], n, usdils)
        if u:
            h["last_price_usd"], h["last_price_native"] = round(u, 4), n

    def total_value():
        return lp["cash"] + sum(h["shares"] * h["last_price_usd"] for h in lp["holdings"])

    def sell(h, reason, shares=None):
        shares = h["shares"] if shares is None else shares
        px = h["last_price_usd"] * (1 - LIVE_SLIPPAGE_PCT / 100)
        gross = shares * px
        fee = _live_fee(gross)
        lp["cash"] += gross - fee
        cost_basis = shares * h["entry_price_usd"] + h["entry_fee_usd"] * shares / h["entry_shares"]
        pnl = gross - fee - cost_basis
        lp["trades"].append({"date": today, "ticker": h["ticker"], "side": "sell", "shares": shares,
                             "price_usd": round(px, 4), "price_native": h.get("last_price_native"), "fee_usd": fee,
                             "reason": reason, "pnl_usd": round(pnl, 2),
                             "pnl_pct": round(pnl / cost_basis * 100, 2) if cost_basis else None,
                             "held_days": h.get("held_days", 0), "source": h.get("source")})
        h["shares"] -= shares
        return shares

    def buy(c, budget):
        n = native_now(c["ticker"])
        u = to_usd(c["ticker"], n, usdils)
        if not u or budget <= 0:
            return None
        px = u * (1 + LIVE_SLIPPAGE_PCT / 100)
        shares = int(budget // px)
        while shares > 0 and shares * px + _live_fee(shares * px) > min(budget, lp["cash"]):
            shares -= 1
        if shares <= 0:
            return None
        fee = _live_fee(shares * px)
        lp["cash"] -= shares * px + fee
        lp["trades"].append({"date": today, "ticker": c["ticker"], "side": "buy", "shares": shares,
                             "price_usd": round(px, 4), "price_native": n, "fee_usd": fee,
                             "reason": c.get("why") or ("המלצה לטווח קצר" if c["source"] == "short" else "המלצה לטווח ארוך"),
                             "source": c["source"]})
        return {"ticker": c["ticker"], "shares": shares, "entry_shares": shares, "entry_price_usd": round(px, 4),
                "entry_price_native": n, "entry_fee_usd": fee, "entry_date": today, "source": c["source"],
                "stop": c.get("stop"), "target": c.get("target"), "score_at_entry": c.get("score"),
                "max_hold": c.get("max_hold"), "strategy": c.get("strategy"),
                "last_price_usd": round(u, 4), "last_price_native": n, "held_days": 0, "out_days": 0}

    # 2. decisions - once a day, only after today's recommendations exist
    rec = store.get("recommendations") or {}
    if lp.get("last_decision_date") != today and rec.get("date") == today and today_entries:
        cands = _live_candidates(store)
        cand_map = {c["ticker"]: c for c in cands}
        exp_pct = rec.get("exposure_pct", 100)
        for h in lp["holdings"]:
            h["held_days"] = h.get("held_days", 0) + 1
            h["out_days"] = 0 if h["ticker"] in cand_map else h.get("out_days", 0) + 1
            e = by_ticker.get(h["ticker"]) or {}
            b = build_score_breakdown(e, e.get("recommendation_mean"), e.get("upside_pct")) if e else {}
            h["score_now"] = _combined_score(b.get("timing_score"), b.get("long_term_score")) if b else None

        # 2a. hard exits: stop / target (native prices, same units as the rec card)
        for h in list(lp["holdings"]):
            n = h.get("last_price_native")
            if n is None:
                continue
            if h.get("stop") and n <= h["stop"]:
                sell(h, "🛑 סטופ")
            elif h.get("target") and n >= h["target"]:
                sell(h, "🎯 הגיעה ליעד")
            elif h.get("max_hold") and h.get("held_days", 0) >= h["max_hold"]:
                sell(h, f"⏱ סוף תקופת ההחזקה ({h['max_hold']} ימי מסחר)")
        lp["holdings"] = [h for h in lp["holdings"] if h["shares"] > 0]

        # 2b. smart replacement: out of the recs for N days, held long enough,
        # AND a clearly better candidate is waiting
        held = {h["ticker"] for h in lp["holdings"]}
        waiting = [c for c in cands if c["ticker"] not in held]   # best first
        nxt = 0  # each waiting candidate can justify at most one replacement
        for h in sorted(lp["holdings"], key=lambda x: x.get("score_now") or 0):
            if h["out_days"] < LIVE_OUT_DAYS_TO_SELL or h["held_days"] < LIVE_MIN_HOLD_DAYS or nxt >= len(waiting):
                continue
            if h.get("max_hold"):
                continue  # research strategies are held for their full period, never swapped out early
            best = waiting[nxt]
            if (best["score"] or 0) >= (h.get("score_now") or 0) + LIVE_REPLACE_GAP:
                sell(h, f"🔄 הוחלפה ב-{best['ticker']} (מחוץ להמלצות {h['out_days']} ימים)")
                best["why"] = f"מחליפה את {h['ticker']}"
                nxt += 1
        lp["holdings"] = [h for h in lp["holdings"] if h["shares"] > 0]

        # 2c. exposure: trim when well above target, never churn small differences
        value = total_value()
        target_invested = value * exp_pct / 100
        invested = value - lp["cash"]
        if invested > target_invested * LIVE_TRIM_TOLERANCE and lp["holdings"]:
            ratio = target_invested / invested
            for h in lp["holdings"]:
                cut = int(h["shares"] * (1 - ratio))
                if cut > 0:
                    sell(h, f"🧭 כלל חשיפה ({exp_pct}%)", cut)
            lp["holdings"] = [h for h in lp["holdings"] if h["shares"] > 0]

        # 2d. fill empty slots, then top up clearly under-sized positions
        value = total_value()
        slot = value * exp_pct / 100 / LIVE_SLOTS
        held = {h["ticker"] for h in lp["holdings"]}
        for c in [c for c in waiting if c["ticker"] not in held]:
            if len(lp["holdings"]) >= LIVE_SLOTS:
                break
            invested = sum(h["shares"] * h["last_price_usd"] for h in lp["holdings"])
            room = value * exp_pct / 100 - invested
            pos = buy(c, min(slot, room, lp["cash"]))
            if pos:
                lp["holdings"].append(pos)
        for h in lp["holdings"]:
            cur = h["shares"] * h["last_price_usd"]
            if cur < slot * LIVE_TOPUP_BELOW and h["ticker"] in cand_map:
                extra = buy({**cand_map[h["ticker"]], "why": "השלמה לגודל פוזיציה"}, min(slot - cur, lp["cash"]))
                if extra:
                    tot = h["shares"] + extra["shares"]
                    h["entry_price_usd"] = round((h["entry_price_usd"] * h["shares"] + extra["entry_price_usd"] * extra["shares"]) / tot, 4)
                    h["entry_fee_usd"] += extra["entry_fee_usd"]
                    h["entry_shares"] = h.get("entry_shares", h["shares"]) + extra["shares"]
                    h["shares"] = tot
        lp["last_decision_date"] = today

    # 3. value log + stats (every run)
    value = round(total_value(), 2)
    lp["value"] = value
    lp["total_return_pct"] = round((value / lp["start_value"] - 1) * 100, 2)
    if lp["daily_log"] and lp["daily_log"][-1]["date"] == today:
        lp["daily_log"][-1]["value"] = value
    else:
        lp["daily_log"].append({"date": today, "value": value})
    lp["daily_log"] = lp["daily_log"][-500:]
    lp["trades"] = lp["trades"][-500:]
    closed = [t for t in lp["trades"] if t["side"] == "sell" and t.get("pnl_usd") is not None]
    wins = [t for t in closed if t["pnl_usd"] > 0]
    losses = [t for t in closed if t["pnl_usd"] <= 0]
    lp["stats"] = {
        "trades": len(lp["trades"]), "closed": len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else None,
        "fees_usd": round(sum(t["fee_usd"] for t in lp["trades"]), 2),
        "realized_pnl_usd": round(sum(t["pnl_usd"] for t in closed), 2),
        "invested_pct": round((value - lp["cash"]) / value * 100, 1) if value else 0,
        "usdils": usdils,
    }


def build_rec_trade_record(store):
    """Trade-based record of the short-term recommendations (rebuilt from
    history every run): every day a ticker is recommended opens a virtual
    trade at that day's price (unless the same ticker's trade is still
    open); it closes at the pick-day target, stop, or after
    REC_TRADE_MAX_DAYS trading days, judged on later days' prices only.
    No commissions here - this measures the rules; the live portfolio
    measures rules + costs."""
    hist = store.get("history", [])
    px = {}
    for e in hist:
        if e.get("price") is not None:
            px.setdefault(e["ticker"], {})[e["date"]] = float(e["price"])
    dates = sorted({e["date"] for e in hist})
    recs = sorted([e for e in hist if e.get("rec_short")], key=lambda e: e["date"])
    trades, open_until = [], {}
    for e in recs:
        t, d0 = e["ticker"], e["date"]
        if open_until.get(t, "") >= d0:
            continue
        entry = float(e["price"])
        stop = e["support"] * SELL_STOP_BUFFER if e.get("support") else None
        target = e.get("resistance")
        later = [d for d in dates if d > d0 and d in px.get(t, {})]
        exit_d = exit_px = None
        reason = "open"
        for i, d in enumerate(later[:REC_TRADE_MAX_DAYS]):
            p = px[t][d]
            if stop and p <= stop:
                exit_d, exit_px, reason = d, p, "stop"
                break
            if target and p >= target:
                exit_d, exit_px, reason = d, p, "target"
                break
            if i == REC_TRADE_MAX_DAYS - 1:
                exit_d, exit_px, reason = d, p, "time"
        last_d = exit_d or (later[-1] if later else d0)
        last_px = exit_px if exit_px is not None else (px[t][later[-1]] if later else entry)
        open_until[t] = exit_d or "9999"
        trades.append({"ticker": t, "entry_date": d0, "entry": round(entry, 2), "exit_date": exit_d,
                       "exit": round(exit_px, 2) if exit_px is not None else None, "reason": reason,
                       "last_date": last_d, "return_pct": round((last_px / entry - 1) * 100, 2)})
    # a ticker that left the scanned universe has no more prices - close it
    # at its last known price instead of leaving it "open" forever
    latest = dates[-1] if dates else None
    for x in trades:
        if x["reason"] == "open" and latest and x["last_date"] < latest and \
                (date.fromisoformat(latest) - date.fromisoformat(x["last_date"])).days > 7:
            x["reason"], x["exit_date"] = "no_data", x["last_date"]
    closed = [x for x in trades if x["reason"] != "open"]
    wins = [x for x in closed if x["return_pct"] > 0]
    losses = [x for x in closed if x["return_pct"] <= 0]
    store["rec_trade_record"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "max_days": REC_TRADE_MAX_DAYS,
        "trades": trades[-200:],
        "n": len(trades), "closed": len(closed), "open": len(trades) - len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
        "avg_win_pct": round(sum(x["return_pct"] for x in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(x["return_pct"] for x in losses) / len(losses), 2) if losses else None,
        "avg_trade_pct": round(sum(x["return_pct"] for x in closed) / len(closed), 2) if closed else None,
        "by_reason": {r: sum(1 for x in closed if x["reason"] == r) for r in ("target", "stop", "time", "no_data")},
    }


def build_portfolios_view(store):
    """Monthly portfolio marked to market with live prices, for the 📒 tab
    (its sim only realizes value when a holding closes)."""
    mp = store.get("monthly_portfolio") or {}
    sim = store.get("monthly_portfolio_sim") or {}
    live = (load_json(CURRENT_PRICES_FILE, {}) or {}).get("prices") or {}
    usdils = _usdils_rate()
    rows, rets = [], []
    for h in mp.get("holdings") or []:
        n = (live.get(h["ticker"]) or {}).get("price")
        entry = h.get("entry_price")
        ret = round((n / entry - 1) * 100, 2) if (n and entry) else None
        if ret is not None:
            rets.append(ret)
        rows.append({"ticker": h["ticker"], "entry_date": h.get("entry_date"), "entry_native": _round_price(entry),
                     "price_native": _round_price(n), "return_pct": ret, "warned": h.get("warned"),
                     "day_pct": _round_price((live.get(h["ticker"]) or {}).get("pct_change"))})
    realized = float(sim.get("value") or 100000)
    n_slots = max(len(mp.get("holdings") or []), 10)
    marked = realized * (1 + sum(rets) / 100 / n_slots) if rets else realized
    store["portfolios_view"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "monthly": {"start_date": (mp.get("history") or [{}])[0].get("cycle_start") or mp.get("start_date"),
                    "cycle_start": mp.get("start_date"), "next_refresh_date": mp.get("next_refresh_date"),
                    "realized_value": round(realized, 2), "marked_value": round(marked, 2),
                    "return_pct": round((marked / 100000 - 1) * 100, 2), "holdings": rows,
                    "closed_trades": (sim.get("trade_log") or [])[-50:]},
        "usdils": usdils,
    }


# ===========================================================================
# v5.13.0 - the two strategies that survived the 26-year theory lab
# (stages 1-3, 2.10.2026), run EXACTLY as researched:
#   A "ירידה חדה":   S&P 500 stock, close <= -12% vs 5 sessions ago, close
#                    above its 200-day average -> buy the NEXT open, sell at
#                    the close of the 10th session. No stop, no target.
#   B "קפיצת חדשות": open >= +5% vs previous close, volume >= 3x its 20-day
#                    average, close in the upper half of the day's range,
#                    S&P 500 above its 200-day average -> buy the next open,
#                    sell at the close of the 60th session.
# The "strategy book" rebuilds both paper portfolios from daily bars after
# every completed session - entries and exits use the real open/close of
# the right day, the same way the research did, plus a ~1-year backfill so
# there is a track record from day one. Isolated from the prediction
# pipeline on purpose (its own download, its own state).
# ===========================================================================
BOOK_PERIOD = "2y"                 # 200 sessions for the average + ~1 year of backfill
BOOK_COST_PCT = 0.1                # commission per side (min $5) - same as the live portfolio
BOOK_SLIPPAGE_PCT = 0.05
BOOK_STRATEGIES = {
    "A": {"label": "🔄 ירידה חדה", "hold": 10, "slots": 10, "drop": -0.12},
    "B": {"label": "🚀 קפיצת חדשות", "hold": 60, "slots": 20, "gap": 0.05, "vol_x": 3.0},
}
US_CLOSE_NY = (16, 15)             # a bar for "today" is only complete after this New York time


def _ny_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=4)


def _download_book_panels(tickers):
    O, H, L, C, V = {}, {}, {}, {}, {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            data = yf.download(tickers=" ".join(batch), period=BOOK_PERIOD, group_by="ticker",
                               threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"Strategy-book batch download error: {e}")
            continue
        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            except Exception:
                continue
            if len(df) < 210:
                continue
            O[t], H[t], L[t], C[t], V[t] = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
        time.sleep(1)
    mk = lambda d: pd.DataFrame(d).sort_index()
    return mk(O), mk(H), mk(L), mk(C), mk(V)


def compute_book_signals(O, H, L, C, V, spy_close):
    sma200 = C.rolling(200).mean()
    a = (C / C.shift(5) - 1 <= BOOK_STRATEGIES["A"]["drop"]) & (C > sma200)
    vol20 = V.rolling(20).mean()
    strong = (C - L) / (H - L).replace(0, np.nan) > 0.5
    gate = (spy_close > spy_close.rolling(200).mean()).reindex(C.index).ffill().fillna(False)
    b = (O / C.shift(1) - 1 >= BOOK_STRATEGIES["B"]["gap"]) & (V >= BOOK_STRATEGIES["B"]["vol_x"] * vol20) & strong
    b = b & gate.values[:, None]
    # ranking keys when there are more signals than free slots
    a_rank = -(C / C.shift(5) - 1)          # bigger drop first
    b_rank = V / vol20                      # bigger volume surge first
    return {"A": (a.fillna(False), a_rank), "B": (b.fillna(False), b_rank)}


def simulate_book(key, sig, rank, O, C, start_i):
    """Day by day from start_i: exits at the close of each position's last
    session, then new entries at today's open for yesterday's signals (only
    into free slots, best-ranked first), then mark to market at the close.
    Position size = current equity / slots (compounding), whole shares."""
    cfg = BOOK_STRATEGIES[key]
    hold, slots = cfg["hold"], cfg["slots"]
    idx, cols = C.index, list(C.columns)
    Ov, Cv = O.values, C.values
    S = sig.values
    R = rank.values
    cash, positions, trades, log = 100000.0, [], [], []
    last_exit = {}   # j -> session index of the last exit: a signal on/before that day is skipped (same as the research engine)

    def fee(notional):
        return max(5.0, notional * BOOK_COST_PCT / 100)

    for k in range(start_i, len(idx)):
        # exits (close of the hold-th session, counted from the entry session)
        keep = []
        for p in positions:
            if k - p["entry_i"] + 1 >= hold and np.isfinite(Cv[k, p["j"]]):
                px = Cv[k, p["j"]] * (1 - BOOK_SLIPPAGE_PCT / 100)
                gross = p["shares"] * px
                f = fee(gross)
                cash += gross - f
                cost = p["shares"] * p["entry_px"] + p["fee"]
                trades.append({"ticker": cols[p["j"]], "entry_date": idx[p["entry_i"]].strftime("%Y-%m-%d"),
                               "exit_date": idx[k].strftime("%Y-%m-%d"), "entry": round(p["entry_px"], 2),
                               "exit": round(px, 2), "shares": p["shares"], "fees": round(p["fee"] + f, 2),
                               "pnl_usd": round(gross - f - cost, 2), "pnl_pct": round((gross - f - cost) / cost * 100, 2)})
                last_exit[p["j"]] = k
            else:
                keep.append(p)
        positions = keep
        # entries at today's open for yesterday's signals
        if k - 1 >= 0:
            held = {p["j"] for p in positions}
            cand = [j for j in np.flatnonzero(S[k - 1]) if j not in held and np.isfinite(Ov[k, j]) and Ov[k, j] > 0
                    and k - 1 > last_exit.get(j, -1)]
            cand.sort(key=lambda j: -(R[k - 1, j] if np.isfinite(R[k - 1, j]) else -9e9))
            equity = cash + sum(p["shares"] * (Cv[k - 1, p["j"]] if np.isfinite(Cv[k - 1, p["j"]]) else p["entry_px"]) for p in positions)
            for j in cand:
                if len(positions) >= slots:
                    break
                px = Ov[k, j] * (1 + BOOK_SLIPPAGE_PCT / 100)
                budget = min(equity / slots, cash)
                shares = int(budget // px)
                while shares > 0 and shares * px + fee(shares * px) > cash:
                    shares -= 1
                if shares <= 0:
                    continue
                f = fee(shares * px)
                cash -= shares * px + f
                positions.append({"j": j, "entry_i": k, "entry_px": px, "shares": shares, "fee": f})
        value = cash + sum(p["shares"] * (Cv[k, p["j"]] if np.isfinite(Cv[k, p["j"]]) else p["entry_px"]) for p in positions)
        log.append({"date": idx[k].strftime("%Y-%m-%d"), "value": round(value, 2), "positions": len(positions)})

    last = len(idx) - 1
    holdings = [{"ticker": cols[p["j"]], "entry_date": idx[p["entry_i"]].strftime("%Y-%m-%d"),
                 "entry": round(p["entry_px"], 2), "shares": p["shares"],
                 "last": round(float(Cv[last, p["j"]]), 2) if np.isfinite(Cv[last, p["j"]]) else None,
                 "sessions_held": last - p["entry_i"] + 1, "sessions_left": max(hold - (last - p["entry_i"] + 1), 0)}
                for p in positions]
    pending = []
    held = {h["ticker"] for h in holdings}
    for j in np.flatnonzero(S[last]):
        t = cols[j]
        pending.append({"ticker": t, "signal_date": idx[last].strftime("%Y-%m-%d"), "close": round(float(Cv[last, j]), 2),
                        "rank": round(float(R[last, j]), 3) if np.isfinite(R[last, j]) else None, "already_held": t in held})
    pending.sort(key=lambda x: -(x["rank"] or 0))
    closed = trades
    wins = [t for t in closed if t["pnl_usd"] > 0]
    losses = [t for t in closed if t["pnl_usd"] <= 0]
    value = log[-1]["value"] if log else 100000.0
    return {
        "key": key, "label": cfg["label"], "hold_sessions": hold, "slots": slots,
        "start_date": log[0]["date"] if log else None, "value": value,
        "total_return_pct": round((value / 100000 - 1) * 100, 2),
        "cash": round(cash, 2), "holdings": holdings, "pending": pending,
        "trades": trades[-300:], "daily_log": log[-400:],
        "stats": {"closed": len(closed), "win_rate_pct": round(len(wins) / len(closed) * 100, 1) if closed else None,
                  "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 2) if wins else None,
                  "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 2) if losses else None,
                  "avg_trade_pct": round(sum(t["pnl_pct"] for t in closed) / len(closed), 2) if closed else None,
                  "fees_usd": round(sum(t["fees"] for t in closed) + sum(p["fee"] for p in positions), 2)},
    }


def run_strategy_book(store, force=False):
    """Once per completed US session (idempotent): download ~2 years of
    daily bars for the S&P 500, drop today's bar if the session is still
    running, rebuild both strategy portfolios from scratch."""
    sp = get_sp500_tickers()
    if not sp:
        return
    try:
        spy = yf.Ticker("SPY").history(period=BOOK_PERIOD)["Close"].dropna()
        spy.index = spy.index.tz_localize(None) if getattr(spy.index, "tz", None) is not None else spy.index
    except Exception as e:
        print(f"Strategy-book SPY fetch failed: {e}")
        return
    ny = _ny_now()
    last_bar = spy.index[-1].date()
    session_complete = not (last_bar == ny.date() and (ny.hour, ny.minute) < US_CLOSE_NY)
    session = last_bar if session_complete else spy.index[-2].date()
    book = store.get("strategy_book") or {}
    if not force and book.get("session") == session.isoformat() and book.get("tickers"):
        return  # (a book built before v5.13.1 has no ticker list - rebuild it once)
    O, H, L, C, V = _download_book_panels(sorted(set(sp)))
    if C.empty:
        return
    for df in (O, H, L, C, V):
        df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
    keep = C.index.date <= session
    O, H, L, C, V = O[keep], H[keep], L[keep], C[keep], V[keep]
    spy = spy[spy.index.date <= session]
    sigs = compute_book_signals(O, H, L, C, V, spy)
    start_i = 200
    out = {"session": session.isoformat(), "generated_at": datetime.now(timezone.utc).isoformat(),
           # first session the book ran LIVE (everything before it is backfill) - kept across rebuilds
           "live_start": (store.get("strategy_book") or {}).get("live_start") or "2026-10-01",
           "universe": int(C.shape[1]), "bars": int(C.shape[0]), "tickers": sorted(C.columns),
           "research": "theory lab stages 1-3 (26y, 500 stocks): A +1.0%/trade vs market t=4.9, 9/9 variants; "
                       "B +2.0%/trade t=3.6, 10/11 variants (median ~0 - a few big winners carry it)"}
    for key, (sig, rank) in sigs.items():
        res = simulate_book(key, sig, rank, O, C, start_i)
        if res.get("start_date"):
            sw = spy[spy.index.date >= date.fromisoformat(res["start_date"])]
            res["spy_same_window_pct"] = round((float(sw.iloc[-1]) / float(sw.iloc[0]) - 1) * 100, 2) if len(sw) > 1 else None
        out[key] = res
    store["strategy_book"] = out


# ===========================================================================
# v5.15.0 - 🚨 unusual moves + 📰 "why?" + 📚 event log.
# Every run: any stock on one of the user's lists that moves >= 7% today
# triggers ONE Telegram alert per ticker per day, with the latest free
# Yahoo headlines and a keyword-based guess at the kind of news. Every
# event is logged, and its outcome 5/10/20 sessions later is filled in
# from the daily snapshots, so over time the log shows how each kind of
# news-driven move actually played out. Information only - it does not
# change any recommendation (no historical news data exists to test a rule).
# ===========================================================================
MOVE_ALERT_PCT = 7.0
MOVE_EVENTS_MAX = 600
MOVE_OUTCOME_SESSIONS = (5, 10, 20)
NEWS_MAX_AGE_DAYS = 3
NEWS_CATEGORIES = [
    ("earnings", "דוח כספי / תחזית", ["earnings", "results", "quarter", "guidance", "outlook", "forecast", "eps", "revenue", "beats", "misses", "profit warning"]),
    ("rating", "שינוי דירוג אנליסטים", ["downgrade", "upgrade", "price target", "rating", "initiates", "analyst", "overweight", "underweight", "outperform"]),
    ("competition", "תחרות / היצע", ["competition", "competitor", "rival", "capacity", "market share", "price war", "pricing pressure", "expansion", "expand", "double"]),
    ("legal", "משפט / רגולציה", ["lawsuit", "sues", "probe", "investigation", "sec ", "fda", "regulator", "antitrust", "recall", "ban", "tariff", "fine"]),
    ("deal", "מיזוג / רכישה", ["acquire", "acquisition", "merger", "buyout", "takeover", "deal", "spin-off", "stake", "bid"]),
    ("offering", "הנפקה / דילול", ["offering", "dilution", "convertible", "share sale", "secondary"]),
    ("macro", "שוק / מאקרו", ["fed ", "federal reserve", "inflation", "rate cut", "rate hike", "jobs report", "wall street", "stock market"]),
]


def _parse_news_item(item):
    """yfinance has two shapes: old flat dicts, and (1.x) {'content': {...}}."""
    c = item.get("content") if isinstance(item.get("content"), dict) else item
    title = c.get("title")
    if not title:
        return None
    provider = c.get("provider") or {}
    publisher = provider.get("displayName") if isinstance(provider, dict) else None
    publisher = publisher or c.get("publisher")
    url = ((c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else None) \
        or ((c.get("clickThroughUrl") or {}).get("url") if isinstance(c.get("clickThroughUrl"), dict) else None) or c.get("link")
    when = c.get("pubDate") or c.get("displayTime")
    if not when and c.get("providerPublishTime"):
        try:
            when = datetime.fromtimestamp(int(c["providerPublishTime"]), tz=timezone.utc).isoformat()
        except Exception:
            when = None
    return {"title": str(title)[:200], "publisher": publisher, "url": url, "published": when}


def fetch_headlines(ticker, limit=5):
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"News fetch failed for {ticker}: {e}")
        return []
    out, cutoff = [], datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    for it in raw:
        n = _parse_news_item(it) if isinstance(it, dict) else None
        if not n:
            continue
        try:
            if n["published"] and datetime.fromisoformat(str(n["published"]).replace("Z", "+00:00")) < cutoff:
                continue
        except Exception:
            pass
        out.append(n)
        if len(out) >= limit:
            break
    return out


def classify_headlines(headlines):
    """Keyword vote over the titles. Returns (key, Hebrew label)."""
    if not headlines:
        return "none", "ללא חדשה ברורה"
    text = " ".join(h["title"].lower() for h in headlines)
    best, best_n = None, 0
    for key, label, words in NEWS_CATEGORIES:
        n = sum(text.count(w) for w in words)
        if n > best_n:
            best, best_n = (key, label), n
    return best if best else ("other", "חדשה אחרת")


def _followed_lists(store):
    """{ticker: [list labels]} for everything the user follows in the app."""
    lists = {}

    def add(t, label):
        if t:
            lists.setdefault(t, [])
            if label not in lists[t]:
                lists[t].append(label)
    book = store.get("strategy_book") or {}
    for k, lab in (("A", "תיק A"), ("B", "תיק B")):
        for h in ((book.get(k) or {}).get("holdings") or []):
            add(h["ticker"], lab)
    for h in ((store.get("live_portfolio") or {}).get("holdings") or []):
        add(h["ticker"], "התיק החי")
    for h in ((store.get("monthly_portfolio") or {}).get("holdings") or []):
        add(h["ticker"], "תיק חודשי")
    for p in ((store.get("long_term_picks") or {}).get("picks") or []):
        add(p["ticker"], "טווח ארוך")
    today = date.today().isoformat()
    for e in store.get("history", []):
        if e.get("date") == today and e.get("top10"):
            add(e["ticker"], "Top10")
    for h in load_json(MY_PORTFOLIO_FILE, []) or []:
        add(h.get("ticker"), "התיק שלי")
    return lists


def _price_session_date():
    """Which US session today's live % change belongs to: the current one
    from the 09:30 New York open, otherwise the previous weekday's (so a
    weekend or pre-market run never re-alerts Friday's move as a new day)."""
    ny = _ny_now()
    d = ny.date()
    if d.weekday() >= 5 or (ny.hour, ny.minute) < (9, 30):
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
    return d.isoformat()


def scan_unusual_moves(store, send=True):
    prices = (load_json(CURRENT_PRICES_FILE, {}) or {}).get("prices") or {}
    today = _price_session_date()
    log = store.setdefault("move_events", [])
    already = {(e["date"], e["ticker"]) for e in log}
    lists = _followed_lists(store)
    sector = {e["ticker"]: e.get("sector") for e in store.get("history", []) if e.get("sector")}
    new = []
    for t, labels in lists.items():
        lp = prices.get(t) or {}
        pct = lp.get("pct_change")
        if pct is None or not np.isfinite(pct) or abs(pct) < MOVE_ALERT_PCT or (today, t) in already:
            continue
        heads = fetch_headlines(t)
        cat, cat_label = classify_headlines(heads)
        ev = {"date": today, "ticker": t, "pct": round(float(pct), 2), "price": _round_price(lp.get("price")),
              "lists": labels, "sector": sector.get(t), "category": cat, "category_label": cat_label,
              "headlines": heads[:3], "detected_at": datetime.now(timezone.utc).isoformat(), "outcome": {}}
        log.append(ev)
        new.append(ev)
    store["move_events"] = log[-MOVE_EVENTS_MAX:]
    if send and new:
        for ev in new:
            arrow = "🔺" if ev["pct"] > 0 else "🔻"
            lines = [f"{arrow} {ev['ticker']} {ev['pct']:+.1f}% היום · {', '.join(ev['lists'])}",
                     f"📰 סיבה משוערת: {ev['category_label']}"]
            for h in ev["headlines"]:
                lines.append(f"• {h['title']}" + (f" ({h['publisher']})" if h.get("publisher") else ""))
            if not ev["headlines"]:
                lines.append("לא נמצאו כותרות מהימים האחרונים ב-Yahoo")
            lines.append("מידע בלבד - לא משנה אף המלצה באפליקציה.")
            try:
                send_telegram_message("🚨 תנועה חריגה\n" + "\n".join(lines))
            except Exception as e:
                print(f"Move alert telegram failed: {e}")
    return new


def update_move_outcomes(store):
    """Fill in the return 5/10/20 sessions after each event, measured from
    the event day's close (the first clean daily snapshot after the event)."""
    by = {}
    for e in store.get("history", []):
        if e.get("price") is not None and not e.get("stale_snapshot"):
            by.setdefault(e["ticker"], {})[e["date"]] = float(e["price"])
    for ev in store.get("move_events", []):
        series = by.get(ev["ticker"])
        if not series:
            continue
        days = sorted(d for d in series if d > ev["date"])
        if not days:
            continue
        base = series[days[0]]
        ev["base_price"] = round(base, 2)
        for n in MOVE_OUTCOME_SESSIONS:
            if len(days) > n and str(n) not in ev["outcome"]:
                ev["outcome"][str(n)] = round((series[days[n]] / base - 1) * 100, 2)
    stats = {}
    for ev in store.get("move_events", []):
        key = (ev["category"], "up" if ev["pct"] > 0 else "down")
        st = stats.setdefault(key, {"category": ev["category"], "label": ev["category_label"],
                                    "direction": key[1], "n": 0, **{f"n{k}": 0 for k in MOVE_OUTCOME_SESSIONS},
                                    **{f"sum{k}": 0.0 for k in MOVE_OUTCOME_SESSIONS}})
        st["n"] += 1
        for k in MOVE_OUTCOME_SESSIONS:
            v = ev["outcome"].get(str(k))
            if v is not None:
                st[f"n{k}"] += 1
                st[f"sum{k}"] += v
    rows = []
    for st in stats.values():
        row = {"category": st["category"], "label": st["label"], "direction": st["direction"], "events": st["n"]}
        for k in MOVE_OUTCOME_SESSIONS:
            row[f"avg_{k}d_pct"] = round(st[f"sum{k}"] / st[f"n{k}"], 2) if st[f"n{k}"] else None
            row[f"n_{k}d"] = st[f"n{k}"]
        rows.append(row)
    rows.sort(key=lambda r: -r["events"])
    store["move_event_stats"] = {"updated_at": datetime.now(timezone.utc).isoformat(), "rows": rows,
                                 "total_events": len(store.get("move_events", []))}


# ===========================================================================
# v5.16.0 - promotion / demotion rule. A list labeled 🧪 earns 🟢 "promising"
# after 30 live trading days ahead of the S&P 500, and ✅ after 60 days if it
# beats the S&P 500 by 3%+ AND in more than half of the months. The two
# validated strategies keep ✅ only while their LIVE record (backfill
# excluded) is not trailing the S&P 500 by more than 3% after 30+ days.
# ===========================================================================
PROMISING_SESSIONS = 30
PROVEN_SESSIONS = 60
PROVEN_EXCESS_PCT = 3.0
DEMOTE_EXCESS_PCT = -3.0


def _series_from_log(log, value_key="value_end", start_key="value_start", since=None):
    """[(date, value)] with a base point, from a sim daily_log."""
    rows = [r for r in (log or []) if r.get(value_key) is not None and (since is None or r["date"] >= since)]
    if not rows:
        return []
    base = rows[0].get(start_key)
    out = [(rows[0]["date"], float(base))] if base else []
    out += [(r["date"], float(r[value_key])) for r in rows]
    return out


def _value_on_or_before(series, d):
    v = None
    for dd, val in series:
        if dd <= d:
            v = val
        else:
            break
    return v


def evaluate_list(series, spx, validated=False):
    """series/spx: sorted [(date, value)]. Returns the status record."""
    if len(series) < 2:
        return {"status": "validated" if validated else "experiment", "sessions": 0}
    start, end = series[0][0], series[-1][0]
    ret = (series[-1][1] / series[0][1] - 1) * 100
    s0, s1 = _value_on_or_before(spx, start), _value_on_or_before(spx, end)
    spx_ret = (s1 / s0 - 1) * 100 if s0 and s1 else None
    excess = round(ret - spx_ret, 2) if spx_ret is not None else None
    sessions = len(series) - 1
    months = {}
    for d, v in series:
        months.setdefault(d[:7], []).append((d, v))
    beat = total = 0
    prev_end = None
    for m in sorted(months):
        pts = months[m]
        a_d, a_v = (prev_end if prev_end else pts[0])
        b_d, b_v = pts[-1]
        prev_end = pts[-1]
        if b_d == a_d:
            continue
        sa_, sb_ = _value_on_or_before(spx, a_d), _value_on_or_before(spx, b_d)
        if not (sa_ and sb_):
            continue
        total += 1
        beat += (b_v / a_v - 1) > (sb_ / sa_ - 1)
    beat_pct = round(beat / total * 100, 1) if total else None
    if validated:
        status = "warning" if sessions >= PROMISING_SESSIONS and excess is not None and excess < DEMOTE_EXCESS_PCT else "validated"
    elif sessions >= PROVEN_SESSIONS and excess is not None and excess >= PROVEN_EXCESS_PCT and (beat_pct or 0) > 50:
        status = "proven"
    elif sessions >= PROMISING_SESSIONS and excess is not None and excess > 0:
        status = "promising"
    else:
        status = "experiment"
    return {"status": status, "sessions": sessions, "start": start, "end": end, "return_pct": round(ret, 2),
            "spx_return_pct": round(spx_ret, 2) if spx_ret is not None else None, "excess_pct": excess,
            "months": total, "months_beating_pct": beat_pct}


def build_list_status(store):
    spx = _series_from_log((store.get("index_sim_sp500") or {}).get("daily_log"))
    out = {}
    lists = [
        ("top10", "⚡ Top10", "portfolio_sim", False),
        ("long_term", "🏛 טווח ארוך (כל הרשימה)", "long_term_sim", False),
        ("long_enter", "🏛 תזמון חיובי", "portfolio_sim_long_enter", False),
        ("long_dip", "🏛 בתיקון", "portfolio_sim_long_dip", False),
        ("long_value", "🏛 💎 זולה מול עצמה", "portfolio_sim_long_value", False),
    ]
    for key, label, sim_key, validated in lists:
        log = (store.get(sim_key) or {}).get("daily_log") or []
        ser = _series_from_log(log) if log and "value_start" in log[0] else [(r["date"], float(r["value_end"])) for r in log if r.get("value_end") is not None]
        out[key] = {"label": label, **evaluate_list(ser, spx, validated)}
    book = store.get("strategy_book") or {}
    live_start = book.get("live_start")
    for k in ("A", "B"):
        b = book.get(k) or {}
        ser = [(r["date"], float(r["value"])) for r in (b.get("daily_log") or []) if live_start and r["date"] >= live_start]
        out[f"book_{k}"] = {"label": b.get("label", k), "live_since": live_start, **evaluate_list(ser, spx, validated=True)}
    out["_rules"] = {"promising_sessions": PROMISING_SESSIONS, "proven_sessions": PROVEN_SESSIONS,
                     "proven_excess_pct": PROVEN_EXCESS_PCT, "demote_excess_pct": DEMOTE_EXCESS_PCT}
    store["list_status"] = out


def main():
    state = load_json(STATE_FILE, {})
    prediction_store = load_prediction_store()
    ensure_formula_blend_initialized(prediction_store)

    if not is_market_trading_day():
        # Still refresh watchlist/live prices for display on off days, but
        # skip grading/new predictions - nothing else changed since is a
        # simple early return, same as before.
        run_watchlist_alerts(state, prediction_store)
        try:
            update_my_portfolio(prediction_store)
        except Exception as e:
            print(f"My-portfolio update failed, continuing without it: {type(e).__name__}: {e}")
        print("Market hasn't traded today yet (weekend/holiday/pre-open) - "
              "skipping mover alerts, grading, and new predictions.")
        # explicit, timestamped signal the frontend's "🔄 עדכן ניתוח" button
        # can poll for - without this, a click on a non-trading day looked
        # identical to a stuck/failed run (nothing else in the store
        # changes when this early-return path is taken), leaving the "in
        # progress" message stuck until the client-side timeout instead of
        # explaining why nothing happened.
        prediction_store["last_run_status"] = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "traded_today": False,
            "backend_version": BACKEND_VERSION,
        }
        save_json(PREDICTIONS_FILE, prediction_store)
        save_json(STATE_FILE, state)
        return

    run_market_wide_alerts(state, prediction_store)  # cross-references yesterday's predictions

    today_str = date.today().isoformat()

    try:
        try:
            mark_stale_snapshot_days(prediction_store)
        except Exception as e:
            print(f"Stale-day marking failed: {e}")
        grade_pending_predictions(prediction_store)
        update_portfolio_simulation(prediction_store)
        update_portfolio_simulation(prediction_store, "top10_experimental", "portfolio_sim_experimental")
        update_portfolio_simulation(prediction_store, "top10_leading", "portfolio_sim_leading")
        update_portfolio_simulation(prediction_store, "top10_original", "portfolio_sim_original")
        update_portfolio_simulation(prediction_store, "top10_fast_rs", "portfolio_sim_fast_rs")
        update_portfolio_simulation(prediction_store, "top10_dual_momentum_lowvol", "portfolio_sim_dual_momentum_lowvol")
        update_portfolio_simulation(prediction_store, "top10_analyst_momentum", "portfolio_sim_analyst_momentum")
        update_portfolio_simulation(prediction_store, "rec_short", "portfolio_sim_recommendations")
        update_portfolio_simulation(prediction_store, "rec_long_enter", "portfolio_sim_long_enter")
        update_portfolio_simulation(prediction_store, "rec_long_dip", "portfolio_sim_long_dip")
        update_portfolio_simulation(prediction_store, "rec_long_value", "portfolio_sim_long_value")
        calibrate_formula_blend(prediction_store)
        build_formula_comparison(prediction_store)

        try:
            update_index_benchmark_sim(prediction_store, "index_sim_sp500", "^GSPC")
            update_index_benchmark_sim(prediction_store, "index_sim_ta125", "^TA125.TA")
            build_benchmark_comparison(prediction_store)
        except Exception as e:
            print(f"Benchmark comparison update failed, continuing without it: {type(e).__name__}: {e}")

        try:
            update_three_layer_views(prediction_store)
        except Exception as e:
            print(f"Three-layer views update failed, continuing without it: {type(e).__name__}: {e}")

        last_factor_run = (prediction_store.get("factor_analysis") or {}).get("updated_at", "")[:10]
        if last_factor_run != today_str:
            analysis = analyze_factor_performance(prediction_store)
            send_factor_analysis_report(analysis)

        run_predictions(prediction_store)
    except Exception as e:
        # Never let a prediction-engine bug wipe out the rest of the run -
        # watchlist alerts and price data must still get saved below.
        print(f"Prediction engine failed, continuing without it: {e}")

    # Watchlist/live-price refresh runs AFTER run_predictions now (backlog
    # item 1 fix, 2026-09-16): current_prices.json and the live "% today"
    # figures get written against TODAY's fresh Top10, not yesterday's
    # stale picks - this was causing a one-cycle lag / bogus 0.00% whenever
    # Top10 rotated. update_my_portfolio/manage_monthly_portfolio moved
    # down with it to keep their relative order unchanged.
    run_watchlist_alerts(state, prediction_store)

    try:
        scan_unusual_moves(prediction_store)
        update_move_outcomes(prediction_store)
    except Exception as e:
        print(f"Unusual-move scan failed: {type(e).__name__}: {e}")

    try:
        update_my_portfolio(prediction_store)
    except Exception as e:
        print(f"My-portfolio update failed, continuing without it: {type(e).__name__}: {e}")

    try:
        today_entries_for_mp = [e for e in prediction_store["history"] if e["date"] == today_str]
        manage_monthly_portfolio(prediction_store, today_entries_for_mp)
    except Exception as e:
        print(f"Monthly portfolio management failed: {type(e).__name__}: {e}")

    try:
        manage_long_term_picks(prediction_store, today_entries_for_mp)
        build_strategy_comparison(prediction_store)
    except Exception as e:
        print(f"Long-term picks failed: {type(e).__name__}: {e}")

    try:
        run_strategy_book(prediction_store)
    except Exception as e:
        print(f"Strategy book failed: {type(e).__name__}: {e}")

    try:
        backfill_rec_flags(prediction_store)
        build_recommendations(prediction_store)
    except Exception as e:
        print(f"Recommendations failed: {type(e).__name__}: {e}")

    try:
        build_rec_trade_record(prediction_store)
        manage_live_portfolio(prediction_store, today_entries_for_mp)
        build_portfolios_view(prediction_store)
        build_strategy_comparison(prediction_store)
        build_list_status(prediction_store)
    except Exception as e:
        print(f"Portfolios failed: {type(e).__name__}: {e}")

    prediction_store["last_run_status"] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "traded_today": True,
        "backend_version": BACKEND_VERSION,
    }
    save_json(PREDICTIONS_FILE, prediction_store)
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
