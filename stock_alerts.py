"""
Stock price alert checker.
Reads watchlist.json, fetches current prices via Yahoo Finance,
and sends a Telegram message when a target price condition is met.
Keeps track of already-triggered alerts in state.json so it doesn't
spam the same alert on every run.
"""

import json
import os
import sys
from pathlib import Path

import requests
import yfinance as yf

BASE_DIR = Path(__file__).parent
WATCHLIST_FILE = BASE_DIR / "watchlist.json"
STATE_FILE = BASE_DIR / "state.json"

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


def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    # fast_info is quicker and more reliable than history() for a single price
    price = stock.fast_info.get("last_price")
    if price is None:
        hist = stock.history(period="1d")
        if hist.empty:
            return None
        price = hist["Close"].iloc[-1]
    return float(price)


def check_condition(price, target, condition):
    if condition == "above":
        return price >= target
    if condition == "below":
        return price <= target
    raise ValueError(f"Unknown condition: {condition}")


def main():
    watchlist = load_json(WATCHLIST_FILE, [])
    state = load_json(STATE_FILE, {})

    if not watchlist:
        print("Watchlist is empty, nothing to check.")
        return

    for item in watchlist:
        ticker = item["ticker"]
        name = item.get("name", ticker)
        target = item["target_price"]
        condition = item["condition"]
        key = f"{ticker}_{condition}_{target}"

        try:
            price = get_current_price(ticker)
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            continue

        if price is None:
            print(f"No price data for {ticker}")
            continue

        triggered_before = state.get(key, False)
        condition_now = check_condition(price, target, condition)

        print(f"{name} ({ticker}): price={price:.2f}, target={target} ({condition}), "
              f"met={condition_now}, already_alerted={triggered_before}")

        if condition_now and not triggered_before:
            direction = "עלה מעל" if condition == "above" else "ירד מתחת ל"
            msg = (f"🔔 התראת מניה\n"
                   f"{name} ({ticker})\n"
                   f"{direction} {target}\n"
                   f"מחיר נוכחי: {price:.2f}")
            send_telegram_message(msg)
            state[key] = True
        elif not condition_now and triggered_before:
            # price moved back past the target, reset so a future crossing alerts again
            state[key] = False

    save_json(STATE_FILE, state)


if __name__ == "__main__":
    main()
