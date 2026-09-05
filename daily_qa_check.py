"""
Daily QA check for tom-stock-alert.

Read-only health check over the live data files (predictions.json,
current_prices.json), run once per day by its own separate GitHub Actions
workflow (daily_qa_check.yml). Does NOT touch prediction/scoring logic in
any way - it only verifies the data the main engine already produced is
sane, and sends one Telegram summary message.

Deliberately reuses is_market_trading_day() and the Telegram helpers from
stock_alerts.py rather than re-implementing the trading-calendar logic -
the main daily engine's own market-day check is the single source of
truth for "was today supposed to be a trading day" (US calendar, gates
Top10/grading/monthly-portfolio for ALL tickers including .TA - see that
function's docstring for why), so this script must never second-guess it
with a separate calendar of its own.

No new persisted state file is introduced. Every check either reads
today's data directly, or - for "did this change since yesterday" checks
(stagnation, accuracy trend) - reads the relevant file's own git history
via `git show`/`git log`, since the repo is committed to daily by the
existing workflows anyway. This keeps the script fully read-only with
respect to the app's own data.
"""
import json
import py_compile
import subprocess
from datetime import date

from stock_alerts import (
    BASE_DIR, CURRENT_PRICES_FILE, PREDICTIONS_FILE,
    load_json, send_telegram_message, is_market_trading_day,
)

STAGNATION_PRICE_MATCH_THRESHOLD = 0.70  # 70%+ tickers unchanged from prior snapshot -> suspicious
ACCURACY_LOW_THRESHOLD = 40.0
ACCURACY_LOW_STREAK_DAYS = 3
ACCURACY_ROLLING_WINDOW = 5


def git_show(path_in_repo, rev):
    """Content of a repo-relative file at a given git revision, or None."""
    try:
        out = subprocess.run(
            ["git", "show", f"{rev}:{path_in_repo}"],
            cwd=BASE_DIR, capture_output=True, text=True, check=True, timeout=15,
        )
        return out.stdout
    except Exception as e:
        print(f"git show {rev}:{path_in_repo} failed: {e}")
        return None


def find_previous_snapshot(filename, before_date_str):
    """Walk this file's commit history and return the parsed JSON content
    from the most recent commit strictly before before_date_str, so a
    commit already made today (if any landed before this QA run) is
    skipped and we get a true prior-day snapshot to compare against."""
    try:
        log = subprocess.run(
            ["git", "log", "--format=%H|%cI", "--", filename],
            cwd=BASE_DIR, capture_output=True, text=True, check=True, timeout=15,
        )
    except Exception as e:
        print(f"git log for {filename} failed: {e}")
        return None
    for line in log.stdout.splitlines():
        if "|" not in line:
            continue
        sha, commit_iso = line.split("|", 1)
        if commit_iso[:10] < before_date_str:
            content = git_show(filename, sha)
            if content is None:
                continue
            try:
                return json.loads(content)
            except Exception:
                continue
    return None


def check_engine_ran_today(store, today_str, issues, info):
    history_dates = {e["date"] for e in store.get("history", [])}
    if today_str not in history_dates:
        issues.append(f"המנוע לא רשם רשומות עבור היום ({today_str}) - ייתכן שנתקע.")
    else:
        info.append(f"המנוע רץ היום ({today_str}).")


def check_duplicates(store, today_str, issues):
    todays = [e for e in store.get("history", []) if e["date"] == today_str]
    seen, dupes = set(), set()
    for e in todays:
        key = (e["ticker"], e["date"])
        if key in seen:
            dupes.add(e["ticker"])
        seen.add(key)
    if dupes:
        issues.append(f"רשומות כפולות (טיקר+תאריך) היום: {', '.join(sorted(dupes))}")


def check_bad_prices(prices_data, issues):
    bad = [t for t, p in (prices_data or {}).items() if p.get("price") is None or p.get("price") <= 0]
    if bad:
        issues.append(f"מחירים פסולים (0/שלילי/None): {', '.join(sorted(bad))}")


def check_support_resistance(store, today_str, issues):
    todays = [e for e in store.get("history", []) if e["date"] == today_str]
    bad = {
        e["ticker"] for e in todays
        if e.get("support") is not None and e.get("resistance") is not None
        and e["support"] >= e["resistance"]
    }
    if bad:
        issues.append(f"הפרת תמיכה≥התנגדות: {', '.join(sorted(bad))}")


def check_required_fields(store, today_str, issues):
    todays = [e for e in store.get("history", []) if e["date"] == today_str]
    missing_overall = {e["ticker"] for e in todays if "overall_score" not in e}
    missing_engine = {e["ticker"] for e in todays if "engine_version" not in e}
    sell_queue = store.get("sell_queue") or {}
    if missing_overall:
        issues.append(f"overall_score חסר עבור: {', '.join(sorted(missing_overall))}")
    if missing_engine:
        issues.append(f"engine_version חסר עבור: {', '.join(sorted(missing_engine))}")
    if todays and sell_queue.get("date") != today_str:
        issues.append(f"sell_queue לא עודכן היום (תאריך אחרון: {sell_queue.get('date')})")


def check_monthly_portfolio_prices(store, prices_data, issues):
    holdings = (store.get("monthly_portfolio") or {}).get("holdings", [])
    missing = {h["ticker"] for h in holdings if h["ticker"] not in (prices_data or {})}
    if missing:
        issues.append(f"טיקרים בתיק החודשי חסרים מרשימת המחירים החיים: {', '.join(sorted(missing))}")


def check_syntax(issues):
    for fname in ("stock_alerts.py", "check_stock.py", "tomorrow_forecast.py"):
        path = BASE_DIR / fname
        if not path.exists():
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as e:
            issues.append(f"שגיאת syntax ב-{fname}: {e}")


def check_stagnation(prices_data, today_str, issues, info):
    prev = find_previous_snapshot("current_prices.json", today_str)
    if not prev:
        info.append("אין נתוני מחירים קודמים להשוואת סטגנציה (הרצה ראשונה?).")
        return
    prev_prices = prev.get("prices", {})
    common = [t for t in prices_data if t in prev_prices]
    if not common:
        return
    unchanged = sum(1 for t in common if prices_data[t].get("price") == prev_prices[t].get("price"))
    ratio = unchanged / len(common)
    if ratio >= STAGNATION_PRICE_MATCH_THRESHOLD:
        issues.append(
            f"חשד לפיד מחירים תקוע: {unchanged}/{len(common)} טיקרים ({ratio:.0%}) "
            f"עם מחיר זהה בדיוק לאתמול."
        )


def check_accuracy_streak(store, issues):
    history = store.get("history", [])
    graded = [e for e in history if e.get("graded")]
    dates = sorted({e["date"] for e in graded}, reverse=True)
    dates = dates[:ACCURACY_ROLLING_WINDOW + ACCURACY_LOW_STREAK_DAYS - 1]
    if len(dates) < ACCURACY_ROLLING_WINDOW + ACCURACY_LOW_STREAK_DAYS - 1:
        return  # not enough graded trading-day history yet to judge a streak

    def rolling_avg_ending(end_idx, subset_filter):
        window_dates = set(dates[end_idx:end_idx + ACCURACY_ROLLING_WINDOW])
        subset = [e for e in graded if e["date"] in window_dates and subset_filter(e)]
        if not subset:
            return None
        return sum(1 for e in subset if e["correct"]) / len(subset) * 100

    categories = {
        "כללי": lambda e: True,
        "Top10": lambda e: e.get("top10"),
        "Top10 (עלייה)": lambda e: e.get("top10") and e.get("predicted") == "up",
    }
    for label, f in categories.items():
        streak_all_low = True
        for i in range(ACCURACY_LOW_STREAK_DAYS):
            avg = rolling_avg_ending(i, f)
            if avg is None or avg >= ACCURACY_LOW_THRESHOLD:
                streak_all_low = False
                break
        if streak_all_low:
            issues.append(
                f"🔴 נורה אדומה: דיוק '{label}' מתחת ל-{ACCURACY_LOW_THRESHOLD}% "
                f"(ממוצע נגלגל {ACCURACY_ROLLING_WINDOW} ימי מסחר) "
                f"במשך {ACCURACY_LOW_STREAK_DAYS} ימי מסחר רצופים."
            )


def main():
    today_str = date.today().isoformat()

    if not is_market_trading_day():
        send_telegram_message("ℹ️ בדיקת QA יומית: אין מסחר היום (סופ״ש/חג ארה״ב) - לא בוצעה בדיקה.")
        print("Not a trading day, QA check skipped.")
        return

    store = load_json(PREDICTIONS_FILE, {})
    prices_data = load_json(CURRENT_PRICES_FILE, {}).get("prices", {})

    issues, info = [], []

    check_engine_ran_today(store, today_str, issues, info)
    check_duplicates(store, today_str, issues)
    check_bad_prices(prices_data, issues)
    check_support_resistance(store, today_str, issues)
    check_required_fields(store, today_str, issues)
    check_monthly_portfolio_prices(store, prices_data, issues)
    check_syntax(issues)
    check_stagnation(prices_data, today_str, issues, info)
    check_accuracy_streak(store, issues)

    if issues:
        header = f"⚠️ בדיקת QA יומית ({today_str}) - נמצאו בעיות:"
        body = "\n".join(f"• {i}" for i in issues)
    else:
        header = f"✅ בדיקת QA יומית ({today_str}) - הכל תקין."
        body = "\n".join(f"• {i}" for i in info)

    msg = header + (("\n" + body) if body else "")
    send_telegram_message(msg)
    print(msg)


if __name__ == "__main__":
    main()
