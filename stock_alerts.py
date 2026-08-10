"""
Stock price alert checker.

Two independent features:
1. Target-price alerts: reads watchlist.json, checks each stock against
   its target price/condition, sends a Telegram alert when crossed.
2. Market-wide big-move alerts: scans the WHOLE US market (via Yahoo
   Finance's day_gainers/day_losers screeners) and a curated list of
   major Israeli (TASE) stocks (ta_tickers.json), and reports any stock
   that moved more than MOVE_THRESHOLD_PCT in a day, grouped into a
   separate summary message per market. Sent at most once per day per
   stock/market.

State (already-sent alerts) is kept in state.json so re-runs don't spam.
"""

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import requests
import yfinance as yf

BASE_DIR = Path(__file__).parent
WATCHLIST_FILE = BASE_DIR / "watchlist.json"
TA_TICKERS_FILE = BASE_DIR / "ta_tickers.json"
STATE_FILE = BASE_DIR / "state.json"
CURRENT_PRICES_FILE = BASE_DIR / "current_prices.json"

MOVE_THRESHOLD_PCT = 10.0
US_SCREENER_COUNT = 250  # how many top gainers/losers to pull from Yahoo

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


def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing Telegram credentials, skipping send. Message was:")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text})
    if not resp.ok:
        print(f"Failed to send Telegram message: {resp.status_code} {resp.text}")


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


def check_condition(price, target, condition):
    if condition == "above":
        return price >= target
    if condition == "below":
        return price <= target
    raise ValueError(f"Unknown condition: {condition}")


def run_watchlist_alerts(state):
    watchlist = load_json(WATCHLIST_FILE, [])
    current_prices = {}
    for item in watchlist:
        ticker = item["ticker"]
        name = item.get("name", ticker)
        target = item["target_price"]
        condition = item["condition"]
        key = f"{ticker}_{condition}_{target}"

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

        triggered_before = state.get(key, False)
        condition_now = check_condition(price, target, condition)
        print(f"{name} ({ticker}): price={price:.2f}, target={target} ({condition}), "
              f"met={condition_now}, already_alerted={triggered_before}")

        if condition_now and not triggered_before:
            direction = "עלה מעל" if condition == "above" else "ירד מתחת ל"
            msg = (f"🔔 התראת מניה\n{name} ({ticker})\n"
                   f"{direction} {target}\nמחיר נוכחי: {price:.2f}")
            send_telegram_message(msg)
            state[key] = True
        elif not condition_now and triggered_before:
            state[key] = False

    save_json(CURRENT_PRICES_FILE, {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prices": current_prices,
    })


# ---------- market-wide big-move alerts ----------

def get_us_movers(threshold, today, state):
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
                    movers.append(f"{name} ({symbol}): {pct:+.1f}% (מחיר: {price})")
                    state[move_key] = True
                seen_symbols.add(symbol)
    return movers


def get_il_movers(threshold, today, state):
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
                movers.append(f"{ticker}: {pct:+.1f}% (מ-{prev_close:.2f} ל-{price:.2f})")
                state[move_key] = True
    return movers


def run_market_wide_alerts(state):
    today = date.today().isoformat()

    us_movers = get_us_movers(MOVE_THRESHOLD_PCT, today, state)
    if us_movers:
        msg = (f"📈📉 תנודה חדה - שוק ארה\"ב (מעל {MOVE_THRESHOLD_PCT:.0f}%)\n\n"
               + "\n".join(us_movers))
        send_telegram_message(msg)

    il_movers = get_il_movers(MOVE_THRESHOLD_PCT, today, state)
    if il_movers:
        msg = (f"📈📉 תנודה חדה - בורסת תל אביב (מעל {MOVE_THRESHOLD_PCT:.0f}%)\n\n"
               + "\n".join(il_movers))
        send_telegram_message(msg)


def main():
    state = load_json(STATE_FILE, {})
    run_watchlist_alerts(state)
    run_market_wide_alerts(state)
    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
