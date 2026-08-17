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
from datetime import date, datetime, timezone
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

SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

MOVE_THRESHOLD_PCT = 10.0
US_SCREENER_COUNT = 250  # how many top gainers/losers to pull from Yahoo
PREDICTION_SCORE_THRESHOLD = 1.5  # medium-high confidence: more alerts, less extreme
PREDICTION_STRONG_THRESHOLD = 3.0  # very strong confidence: highlighted in bold
PREFILTER_THRESHOLD = 1.0   # only fetch fundamentals (slow) for tickers past this technical-only score
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


def run_watchlist_alerts(state):
    watchlist = load_json(WATCHLIST_FILE, [])
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

def compute_rsi(closes, period=14):
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    valid = rsi.dropna()
    return float(valid.iloc[-1]) if len(valid) else None


def compute_macd_bullish(closes):
    if len(closes) < 26:
        return None
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return bool(macd_line.iloc[-1] > signal_line.iloc[-1])


def compute_technical_factors(closes, volumes):
    """Everything derivable from price/volume history alone - no network
    calls per ticker, so this is cheap enough to run on the whole universe."""
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

    return {
        "price": price,
        "range_pos": range_pos,
        "run_up_30d": run_up_30d,
        "rsi": rsi,
        "macd_bullish": macd_bullish,
        "ma_trend": ma_trend,
        "vol_ratio": vol_ratio,
    }


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

    return {"upside_pct": upside_pct, "short_pct": short_pct}


def get_market_regime():
    """Is the overall market (S&P 500) trending up or down right now?
    Used as a small nudge on every score, not an override - a stock can
    still score bullish in a down market and vice versa."""
    try:
        spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True)["Close"].dropna()
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
    score = 0.0

    if factors.get("range_pos") is not None:
        if factors["range_pos"] >= 80:
            score -= 2  # trading near 52-week high -> stretched, more downside risk
        elif factors["range_pos"] <= 20:
            score += 2  # trading near 52-week low -> cheap, more upside potential

    if factors.get("upside_pct") is not None:
        if factors["upside_pct"] >= 15:
            score += 2  # analysts see a lot of room above current price -> bullish
        elif factors["upside_pct"] <= -10:
            score -= 2  # already trading above target -> priced for perfection

    if factors.get("run_up_30d") is not None:
        score -= factors["run_up_30d"] * 0.06  # penalize stocks that already ran hard

    if factors.get("short_pct") is not None:
        score += factors["short_pct"] * 0.05  # short-squeeze potential

    if factors.get("rsi") is not None:
        if factors["rsi"] <= 30:
            score += 1.5  # oversold
        elif factors["rsi"] >= 70:
            score -= 1.5  # overbought

    if factors.get("macd_bullish") is not None:
        score += 1.2 if factors["macd_bullish"] else -1.2

    if factors.get("ma_trend") == "golden":
        score += 1.5
    elif factors.get("ma_trend") == "death":
        score -= 1.5

    if factors.get("vol_ratio") is not None and factors.get("run_up_30d") is not None:
        if factors["vol_ratio"] >= 1.5:
            # a volume surge CONFIRMS whatever direction the stock is already moving in
            score += 1.0 if factors["run_up_30d"] > 0 else -1.0

    if market_regime.get("bullish") is not None:
        score += 0.3 if market_regime["bullish"] else -0.3

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
    print(f"Prediction universe: {len(universe)} tickers")
    market_regime = get_market_regime()
    print(f"Market regime: {market_regime}")

    technical = {}
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
                closes = data[symbol]["Close"] if len(batch) > 1 else data["Close"]
                volumes = data[symbol]["Volume"] if len(batch) > 1 else data["Volume"]
            except Exception:
                continue
            try:
                tf = compute_technical_factors(closes, volumes)
                if tf:
                    technical[symbol] = tf
            except Exception as e:
                print(f"Technical analysis error for {symbol}: {e}")
        time.sleep(1)

    print(f"Technical factors computed for {len(technical)} tickers")

    # stage 1: cheap technical-only score to decide who's worth the slow .info() call
    prelim_scores = {}
    for symbol, tf in technical.items():
        prelim_scores[symbol] = compute_prediction_score(tf, market_regime)

    candidates = [s for s, sc in prelim_scores.items() if abs(sc) >= PREFILTER_THRESHOLD]
    print(f"{len(candidates)} tickers passed the pre-filter, fetching fundamentals for those...")

    us_strong = []
    il_strong = []

    for symbol in technical:
        factors = dict(technical[symbol])
        if symbol in candidates:
            try:
                fund = get_fundamental_factors(symbol)
                factors.update(fund)
            except Exception as e:
                print(f"Fundamentals error for {symbol}: {e}")

        score = compute_prediction_score(factors, market_regime)
        predicted = "up" if score >= 0 else "down"
        is_strong = abs(score) >= PREDICTION_STRONG_THRESHOLD

        entry = {
            "date": today, "ticker": symbol, "score": score, "predicted": predicted,
            "strong": is_strong, "graded": False,
        }
        for k, v in factors.items():
            if isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            entry[k] = v
        store["history"].append(entry)

        if abs(score) >= PREDICTION_SCORE_THRESHOLD:
            direction = "עלייה" if predicted == "up" else "ירידה"
            line = f"{symbol}: {direction} צפויה (ציון {score})"
            if is_strong:
                line = f"*{line}* 🔥"  # bold + flame for very-strong signals
            if is_israeli(symbol):
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

    store["market_regime"] = market_regime


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
