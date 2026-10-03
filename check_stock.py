"""Entry point for the on-demand 'בדוק מניה' feature.

Run standalone (not part of the main 15-minute check_prices.yml cycle):
    python check_stock.py TICKER

Triggered by .github/workflows/check_stock.yml via workflow_dispatch with a
`ticker` input. Writes a single result file (stock_check_result.json) that
the frontend polls for after it kicks off the run - see the "בדוק מניה"
section in index.html. Overwrites the previous result each time; this is a
one-shot lookup tool, not something that needs history.
"""
import json
import os
import re
import sys

from stock_alerts import analyze_single_ticker

RESULT_FILE = "stock_check_result.json"
TICKER_RE = re.compile(r"[A-Za-z0-9.^=-]{1,15}")


def main():
    ticker = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CHECK_TICKER", "")).strip()
    if not ticker:
        result = {"error": "לא התקבל טיקר לבדיקה"}
    elif not TICKER_RE.fullmatch(ticker):
        # security (v5.17.1): only plain ticker characters ever reach the analysis / the result file
        result = {"error": "סימול לא תקין - מותרות רק אותיות באנגלית, ספרות, נקודה, מקף ו-^"}
    else:
        try:
            result = analyze_single_ticker(ticker)
        except Exception as e:
            result = {"ticker": ticker.upper(), "error": f"שגיאה בניתוח: {type(e).__name__}"}

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
