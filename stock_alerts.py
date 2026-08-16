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
   stock/market. Each mover is cross-referenced against the prediction
   engine's most recent prior call for that ticker (direction + score),
   so you can see whether the big move matches what was predicted.

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
PREDICTIONS_FILE = BASE_DIR / "predictions.json"

MOVE_THRESHOLD_PCT = 10.0
US_SCREENER_COUNT = 250  # how many top gainers/losers to pull from Yahoo
PREDICTION_SCORE_THRESHOLD = 1.5  # medium-high confidence: more alerts, less extreme
PREDICTION_STRONG_THRESHOLD = 3.0  # very strong confidence: highlighted in bold

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
        msg = (f"📈📉 תנודה חדה - שוק ארה\"ב (מעל {MOVE_THRESHOLD_PCT:.0f}%)\n\n"
               + "\n\n".join(us_movers))
        send_telegram_message(msg)

    il_movers = get_il_movers(MOVE_THRESHOLD_PCT, today, state, prediction_store)
    if il_movers:
        msg = (f"📈📉 תנודה חדה - בורסת תל אביב (מעל {MOVE_THRESHOLD_PCT:.0f}%)\n\n"
               + "\n\n".join(il_movers))
        send_telegram_message(msg)


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


# ---------- next-day prediction engine (with self-grading / learning) ----------
# Uses only signals knowable BEFORE a move happens (not post-event facts like
# "beat earnings" or "M&A rumor" - those are only known in hindsight).

def get_prediction_factors(ticker):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    fast = stock.fast_info
    price = fast.get("last_price")
    year_high = fast.get("year_high")
    year_low = fast.get("year_low")

    run_up_30d = None
    try:
        hist = stock.history(period="35d")
        closes = hist["Close"].dropna()
        if len(closes) >= 22:
            run_up_30d = (closes.iloc[-1] - closes.iloc[-22]) / closes.iloc[-22] * 100
    except Exception:
        pass

    target_mean = info.get("targetMeanPrice")
    upside_pct = None
    if target_mean and price:
        upside_pct = (target_mean - price) / price * 100

    short_pct = info.get("shortPercentOfFloat")
    if short_pct is not None:
        short_pct = short_pct * 100

    range_pos = None
    if price and year_high and year_low and year_high > year_low:
        range_pos = (price - year_low) / (year_high - year_low) * 100

    return {
        "price": price,
        "run_up_30d": run_up_30d,
        "upside_pct": upside_pct,
        "short_pct": short_pct,
        "range_pos": range_pos,
    }


def compute_prediction_score(factors):
    score = 0.0
    if factors["range_pos"] is not None:
        if factors["range_pos"] >= 80:
            score -= 2  # trading near 52-week high -> stretched, more downside risk
        elif factors["range_pos"] <= 20:
            score += 2  # trading near 52-week low -> cheap, more upside potential
    if factors["upside_pct"] is not None:
        if factors["upside_pct"] >= 15:
            score += 2  # analysts see a lot of room above current price -> bullish
        elif factors["upside_pct"] <= -10:
            score -= 2  # already trading above target -> priced for perfection
    if factors["run_up_30d"] is not None:
        score -= factors["run_up_30d"] * 0.06  # penalize stocks that already ran hard
    if factors["short_pct"] is not None:
        score += factors["short_pct"] * 0.05  # short-squeeze potential
    return round(score, 2)


def build_prediction_universe():
    tickers = set()
    for item in load_json(WATCHLIST_FILE, []):
        tickers.add(item["ticker"])
    for t in load_json(TA_TICKERS_FILE, []):
        tickers.add(t)
    for screen_name in ["day_gainers", "day_losers", "most_actives"]:
        try:
            result = yf.screen(screen_name, count=100)
            for q in result.get("quotes", []):
                if q.get("symbol"):
                    tickers.add(q["symbol"])
        except Exception as e:
            print(f"Error screening {screen_name} for prediction universe: {e}")
    return tickers


def load_prediction_store():
    return load_json(PREDICTIONS_FILE, {"history": [], "accuracy": {}})


def recompute_accuracy(store):
    graded = [e for e in store["history"] if e.get("graded")]
    strong_graded = [e for e in graded if e.get("strong")]

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
        "us": calc([e for e in graded if not is_israeli(e["ticker"])]),
        "il": calc([e for e in graded if is_israeli(e["ticker"])]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def grade_pending_predictions(store):
    """Check yesterday-or-earlier predictions against the actual price now,
    mark them correct/incorrect, so we can measure and improve the formula."""
    today = date.today().isoformat()
    changed = False
    for entry in store["history"]:
        if entry.get("graded") or entry.get("date") == today:
            continue
        ticker = entry["ticker"]
        try:
            current_price = yf.Ticker(ticker).fast_info.get("last_price")
        except Exception as e:
            print(f"Error grading {ticker}: {e}")
            continue
        if current_price is None or not entry.get("price"):
            continue
        actual_pct = (current_price - entry["price"]) / entry["price"] * 100
        actual_direction = "up" if actual_pct >= 0 else "down"
        entry["actual_price"] = round(current_price, 4)
        entry["actual_pct_change"] = round(actual_pct, 2)
        entry["actual_direction"] = actual_direction
        entry["correct"] = actual_direction == entry["predicted"]
        entry["graded"] = True
        changed = True
    if changed:
        recompute_accuracy(store)
    return changed


def run_predictions(store):
    today = date.today().isoformat()
    already_predicted_today = any(e.get("date") == today for e in store["history"])
    if already_predicted_today:
        print("Predictions already run today, skipping.")
        return

    universe = build_prediction_universe()
    us_strong = []
    il_strong = []

    for ticker in universe:
        try:
            factors = get_prediction_factors(ticker)
        except Exception as e:
            print(f"Prediction error for {ticker}: {e}")
            continue
        if factors["price"] is None:
            continue
        score = compute_prediction_score(factors)
        predicted = "up" if score >= 0 else "down"
        is_strong = abs(score) >= PREDICTION_STRONG_THRESHOLD

        store["history"].append({
            "date": today, "ticker": ticker, "score": score, "predicted": predicted,
            "strong": is_strong, "graded": False, **factors,
        })

        if abs(score) >= PREDICTION_SCORE_THRESHOLD:
            direction = "עלייה" if predicted == "up" else "ירידה"
            line = f"{ticker}: {direction} צפויה (ציון {score})"
            if is_strong:
                line = f"*{line}* 🔥"  # bold + flame for very-strong signals
            if is_israeli(ticker):
                il_strong.append(line)
            else:
                us_strong.append(line)

    if us_strong:
        send_telegram_message(
            "🔮 חיזוי ליום המסחר הבא - שוק ארה\"ב\n\n" + "\n".join(us_strong),
            parse_mode="Markdown",
        )
    if il_strong:
        send_telegram_message(
            "🔮 חיזוי ליום המסחר הבא - בורסת ת\"א\n\n" + "\n".join(il_strong),
            parse_mode="Markdown",
        )


def main():
    state = load_json(STATE_FILE, {})
    prediction_store = load_prediction_store()

    run_watchlist_alerts(state)
    run_market_wide_alerts(state, prediction_store)  # cross-references yesterday's predictions

    grade_pending_predictions(prediction_store)
    run_predictions(prediction_store)
    save_json(PREDICTIONS_FILE, prediction_store)

    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
