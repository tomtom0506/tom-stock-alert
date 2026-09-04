"""Entry point for the "תחזית למחר" feature (item 19) - a separate, on-demand
or once-daily forecast for the NEXT trading session, independent of the
official once-a-day Top10 pipeline in stock_alerts.py's run_predictions().

Run standalone:
    python tomorrow_forecast.py

Triggered by .github/workflows/tomorrow_forecast.yml, both on a daily
schedule and via workflow_dispatch for on-demand runs. Writes its result
into predictions.json under the "tomorrow_forecast" key (see
run_tomorrow_forecast) - never touches the tracked/graded "history" list,
so this can run as often as needed without affecting accuracy tracking.
"""
import json

from stock_alerts import (
    PREDICTIONS_FILE, load_json, save_json, run_tomorrow_forecast,
)


def main():
    store = load_json(PREDICTIONS_FILE, {})
    store.setdefault("history", [])
    result = run_tomorrow_forecast(store)
    save_json(PREDICTIONS_FILE, store)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
