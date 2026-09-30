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

Most checks read today's data directly, or - for "did this change since
yesterday" checks (stagnation, accuracy trend) - read the relevant file's
own git history via `git show`/`git log`, since the repo is committed to
daily by the existing workflows anyway.

Since v5.5.0 this script ALSO writes three small accumulating files (the
workflow now commits them back - see daily_qa_check.yml): qa_outliers_log
(every data_suspect ticker ever flagged, for audit), qa_daily_log (one
enriched record per trading day - accuracy, market regime, actual index
return, suspect count), and qa_conclusions.md (a human-readable,
always-current breakdown of accuracy conditioned on market state,
regenerated fully from qa_daily_log every run, with a minimum-sample
guard so a small/skewed window never gets reported as a firm conclusion -
this is what a check-in conversation reads, not the daily Telegram
message). Everything about the app's OWN data (predictions.json,
current_prices.json, the code files) is still read-only.
"""
import json
import py_compile
import subprocess
from datetime import date, datetime, timedelta, timezone

from stock_alerts import (
    BASE_DIR, CURRENT_PRICES_FILE, PREDICTIONS_FILE,
    load_json, save_json, send_telegram_message, is_market_trading_day,
)

STAGNATION_PRICE_MATCH_THRESHOLD = 0.70  # 70%+ tickers unchanged from prior snapshot -> suspicious
ACCURACY_LOW_THRESHOLD = 40.0
ACCURACY_LOW_STREAK_DAYS = 3
ACCURACY_ROLLING_WINDOW = 5
MIN_SAMPLES_FOR_CONCLUSION = 5  # don't report a conditional-accuracy bucket until it has at least this many days

QA_OUTLIERS_LOG = BASE_DIR / "qa_outliers_log.json"
QA_DAILY_LOG = BASE_DIR / "qa_daily_log.json"
QA_CONCLUSIONS_FILE = BASE_DIR / "qa_conclusions.md"


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
    # v5.8.0: only the Top10 is guaranteed a breakdown (watchlist/crypto get
    # one too, but the rest of the scanned universe never does, by design -
    # see run_predictions). The old version checked EVERY scanned ticker, so
    # it fired daily with hundreds of names and its warning became noise.
    missing_overall = {e["ticker"] for e in todays if e.get("top10") and e.get("overall_score") is None}
    missing_engine = {e["ticker"] for e in todays if "engine_version" not in e}
    sell_queue = store.get("sell_queue") or {}
    if missing_overall:
        issues.append(f"overall_score חסר עבור: {', '.join(sorted(missing_overall))}")
    if missing_engine:
        issues.append(f"engine_version חסר עבור: {', '.join(sorted(missing_engine))}")
    if todays and sell_queue.get("date") != today_str:
        issues.append(f"sell_queue לא עודכן היום (תאריך אחרון: {sell_queue.get('date')})")
    hold_queue = store.get("hold_queue")
    if todays and hold_queue is not None and hold_queue.get("date") != today_str:
        issues.append(f"hold_queue לא עודכן היום (תאריך אחרון: {hold_queue.get('date')})")


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


STALE_FEED_MAX_HOURS = 30   # on a trading day, a price file older than this means the feed stopped updating


def check_stagnation(prices_data, today_str, issues, info, updated_at=None):
    """Did prices change between the last two sessions?

    v5.10.1 fix (false alarm 30.9.2026): this used to compare the current
    file against the last commit before TODAY. But QA runs at 04:00 UTC,
    before any market opens, so the current file IS the last commit from
    the previous evening - the check was comparing that snapshot to
    (effectively) itself and flagged 100% "unchanged". Now the reference
    point is the file's own timestamp: compare against the last commit from
    before the day the current prices were written, i.e. the previous
    session. A genuinely stuck feed is caught separately by age
    (STALE_FEED_MAX_HOURS)."""
    ref_date = (updated_at or "")[:10] or today_str
    if updated_at:
        try:
            upd = datetime.fromisoformat(updated_at)
            age_h = (datetime.now(timezone.utc) - upd).total_seconds() / 3600
            # a whole weekday passed with no update = stuck (weekends are not
            # counted, so Monday morning after a Friday close is fine; a
            # market holiday can still trip this - rare, and worth a look anyway)
            d, today_d, missed = upd.date() + timedelta(days=1), date.fromisoformat(today_str), 0
            while d < today_d:
                missed += d.weekday() < 5
                d += timedelta(days=1)
            if age_h > STALE_FEED_MAX_HOURS and missed:
                issues.append(f"פיד המחירים לא התעדכן {age_h:.0f} שעות (עדכון אחרון {updated_at[:16].replace('T', ' ')} UTC).")
        except (ValueError, TypeError):
            pass
    prev = find_previous_snapshot("current_prices.json", ref_date)
    if not prev:
        info.append("אין נתוני מחירים קודמים להשוואת סטגנציה (הרצה ראשונה?).")
        return
    prev_prices = prev.get("prices", {})
    prev_date = (prev.get("updated_at") or "")[:10] or "?"
    common = [t for t in prices_data if t in prev_prices]
    if not common:
        return
    unchanged = sum(1 for t in common if prices_data[t].get("price") == prev_prices[t].get("price"))
    ratio = unchanged / len(common)
    if ratio >= STAGNATION_PRICE_MATCH_THRESHOLD:
        issues.append(
            f"חשד לפיד מחירים תקוע: {unchanged}/{len(common)} טיקרים ({ratio:.0%}) "
            f"עם מחיר זהה בדיוק לסשן הקודם ({prev_date})."
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


def check_data_suspect_flags(store, today_str, outliers_section):
    """Surfaces today's data_suspect tickers (see compute_technical_factors
    in stock_alerts.py, added v5.5.0 after the NFE incident) prominently in
    the Telegram message, and appends each to the cumulative outliers log
    so a repeating pattern (same ticker, same kind of event) is visible
    over time instead of each flag disappearing after one day's message."""
    todays = [e for e in store.get("history", []) if e["date"] == today_str]
    flagged = [e for e in todays if e.get("data_suspect")]
    if not flagged:
        return

    for e in flagged:
        outliers_section.append(
            f"{e['ticker']}: {e.get('data_suspect_reason') or 'תנועת מחיר קיצונית לא מוסברת'} "
            f"- הוצא אוטומטית מ-Top10/תחזיות חזקות."
        )

    log = load_json(QA_OUTLIERS_LOG, [])
    existing_keys = {(rec.get("date"), rec.get("ticker")) for rec in log}
    for e in flagged:
        key = (today_str, e["ticker"])
        if key in existing_keys:
            continue
        log.append({
            "date": today_str,
            "ticker": e["ticker"],
            "reason": e.get("data_suspect_reason"),
        })
    save_json(QA_OUTLIERS_LOG, log)


# ---------------------------------------------------------------------------
# v5.8.0 - LOGIC checks. Everything above verifies the data is well-formed;
# none of it could notice a feature whose condition can never fire (the
# permanently-empty sell queue) or a score whose meaning is inverted (the
# direction-blind abs() score). Both bugs were found by Tomer, not by QA,
# on 29.9.2026 - these checks exist so the next one of that kind is caught
# here first. Each is deliberately cheap and reads only files already in
# the repo.
# ---------------------------------------------------------------------------
DEAD_FEATURE_DAYS = 10          # trading days in a row with an empty output before it's flagged
ANALYST_GAP_POINTS = 50         # |our overall - analyst score| above this -> listed for manual review
REDUNDANT_CORR = 0.85           # two score components this correlated are effectively one signal (the pre-5.8.0 original/leading pair ran at ~0.90)
MIN_CORR_SAMPLES = 10
PRICE_MAX_DECIMALS = 2
STALE_SIM_DAYS = 5              # a parallel portfolio sim not advanced for this many days is stuck
# v5.12.0: sims driven by recommendation flags only advance on days that
# actually had a recommendation of that kind - quiet stretches are normal
SPARSE_SIMS = {"portfolio_sim_recommendations", "portfolio_sim_long_enter", "portfolio_sim_long_dip"}

# metrics recorded daily (see record_daily_log) whose outputs should NOT be
# empty for DEAD_FEATURE_DAYS straight trading days in normal operation
DEAD_FEATURE_METRICS = {
    "sell_queue_count": "תור המכירה",
    "hold_queue_count": "תור ההחזקה",
    "verdict_buy_count": "תג ✅ מומלץ",
    # verdict_avoid_count is recorded but NOT checked here: verdicts are only
    # computed for Top10 + watchlist + crypto names, which are mostly bullish
    # picks, so days with no ❌ at all are normal - it would false-alarm.
}


def check_dead_features(daily_log, issues):
    recent = daily_log[-DEAD_FEATURE_DAYS:]
    if len(recent) < DEAD_FEATURE_DAYS:
        return  # not enough days recorded yet
    for key, label in DEAD_FEATURE_METRICS.items():
        vals = [r.get(key) for r in recent]
        if all(v is not None for v in vals) and all(v == 0 for v in vals):
            issues.append(
                f"פיצ'ר מת? '{label}' ריק {DEAD_FEATURE_DAYS} ימי מסחר ברציפות - "
                f"לבדוק שהתנאי שלו בכלל יכול להתקיים."
            )


def check_direction_consistency(store, today_str, issues):
    """The timing score is directional by design (50 = neutral). An up
    prediction scoring below 50, or a 'buy' verdict on a down prediction,
    means the score and the prediction disagree - exactly the class of bug
    the old abs() formula had."""
    todays = [e for e in store.get("history", []) if e["date"] == today_str and e.get("timing_score") is not None]
    bad = []
    for e in todays:
        t = e["timing_score"]
        if (e.get("predicted") == "up" and t < 50) or (e.get("predicted") == "down" and t > 50):
            bad.append(f"{e['ticker']} ({e.get('predicted')}, {t})")
        elif e.get("verdict") == "buy" and e.get("predicted") == "down":
            bad.append(f"{e['ticker']} (buy על תחזית ירידה)")
    if bad:
        issues.append(f"סתירת כיוון בין הציון לתחזית: {', '.join(bad[:8])}")


def check_analyst_gap(store, today_str, info):
    todays = [e for e in store.get("history", []) if e["date"] == today_str]
    gaps = []
    for e in todays:
        a = (e.get("score_components") or {}).get("analyst")
        o = e.get("overall_score")
        if a is not None and o is not None and abs(o - a) > ANALYST_GAP_POINTS:
            gaps.append((abs(o - a), f"{e['ticker']} (שלנו {o} מול אנליסטים {a})"))
    if gaps:
        gaps.sort(reverse=True)
        info.append("פער חריג מול אנליסטים (לבדיקה ידנית, לא תקלה): " + ", ".join(g[1] for g in gaps[:5]))


def _corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def check_component_redundancy(store, today_str, issues):
    todays = [e for e in store.get("history", []) if e["date"] == today_str and e.get("score_components")]
    keys = sorted({k for e in todays for k in e["score_components"]})
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs = [(e["score_components"][a], e["score_components"][b]) for e in todays
                     if e["score_components"].get(a) is not None and e["score_components"].get(b) is not None]
            if len(pairs) < MIN_CORR_SAMPLES:
                continue
            c = _corr([p[0] for p in pairs], [p[1] for p in pairs])
            if c is not None and c > REDUNDANT_CORR:
                issues.append(f"רכיבי ציון כפולים: '{a}' ו-'{b}' במתאם {c:.2f} (n={len(pairs)}) - בפועל אותו סיגנל")


def _too_many_decimals(v):
    if not isinstance(v, float):
        return False
    return abs(round(v, PRICE_MAX_DECIMALS) - v) > 1e-9


def check_price_format(store, issues):
    """Prices the UI shows straight from these files must already be rounded
    (the '518.0999755859375' case)."""
    bad = []
    check = load_json(BASE_DIR / "stock_check_result.json", {})
    for k in ("price", "support", "resistance"):
        if _too_many_decimals(check.get(k)):
            bad.append(f"stock_check_result.{k}={check.get(k)}")
    for qkey in ("sell_queue", "hold_queue"):
        for item in ((store.get(qkey) or {}).get("items") or []):
            for k in ("price", "support", "resistance", "pick_price"):
                if _too_many_decimals(item.get(k)):
                    bad.append(f"{qkey}:{item.get('ticker')}.{k}")
    if bad:
        issues.append(f"מחירים לא מעוגלים בקבצים שמוצגים למשתמש: {', '.join(bad[:6])}")


def check_calibration_text(store, issues):
    """Auto-generated notes must agree with the numbers they describe."""
    blend = store.get("formula_blend") or {}
    hist = blend.get("history") or []
    if hist:
        last = hist[-1]
        old_a, new_a, note = last.get("old_alpha"), last.get("new_alpha"), last.get("note") or ""
        if old_a is not None and new_a is not None and new_a != old_a:
            expected = "לטובת יחס סיכוי/סיכון" if new_a > old_a else "לטובת עוצמת חיזוי טהורה"
            if expected not in note:
                issues.append(f"הערת הכיול סותרת את השינוי ב-alpha ({old_a}→{new_a}): '{note[:80]}'")
    alpha = blend.get("alpha")
    comp = store.get("formula_comparison") or {}
    for f in comp.get("formulas") or []:
        if f.get("key") == "current" and alpha is not None and f"alpha={alpha:.2f}" not in (f.get("label") or ""):
            issues.append(f"תווית 'הנוסחה האמיתית' לא תואמת את alpha הנוכחי ({alpha:.2f}): {f.get('label')}")


def check_stale_sims(store, today_str, issues):
    today = date.fromisoformat(today_str)
    stale = []
    for key, val in store.items():
        if not key.startswith("portfolio_sim") or not isinstance(val, dict) or key in SPARSE_SIMS:
            continue
        last = val.get("last_processed_date")
        if not last:
            continue
        try:
            age = (today - date.fromisoformat(last)).days
        except ValueError:
            continue
        if age > STALE_SIM_DAYS:
            stale.append(f"{key} ({last})")
    if stale:
        issues.append(f"סימולציות תיק שלא התקדמו מעל {STALE_SIM_DAYS} ימים: {', '.join(stale)}")


def record_daily_log(store, today_str, prices_payload):
    """Appends one enriched record for today to qa_daily_log.json - the
    accumulating evidence base generate_conclusions() reads from. Skips if
    today's record already exists (idempotent - a workflow_dispatch re-run
    on the same day shouldn't duplicate the day's entry)."""
    log = load_json(QA_DAILY_LOG, [])
    if any(rec.get("date") == today_str for rec in log):
        return log  # already recorded today, nothing to do

    acc = store.get("accuracy") or {}
    regime = store.get("market_regime") or {}
    indices = (prices_payload or {}).get("indices") or {}
    todays = [e for e in store.get("history", []) if e["date"] == today_str]

    record = {
        "date": today_str,
        "market_regime_bullish": regime.get("bullish"),
        "sp500_pct_change": (indices.get("sp500") or {}).get("pct_change"),
        "ta125_pct_change": (indices.get("ta125") or {}).get("pct_change"),
        "strong_daily_accuracy": acc.get("top10_accuracy_daily"),
        "data_suspect_count": sum(1 for e in todays if e.get("data_suspect")),
        # v5.8.0 - dead-feature counters (see check_dead_features)
        "sell_queue_count": len(((store.get("sell_queue") or {}).get("items")) or []),
        "hold_queue_count": len(((store.get("hold_queue") or {}).get("items")) or []) if store.get("hold_queue") else None,
        "verdict_buy_count": sum(1 for e in todays if e.get("verdict") == "buy") if any("verdict" in e for e in todays) else None,
        "verdict_avoid_count": sum(1 for e in todays if e.get("verdict") == "avoid") if any("verdict" in e for e in todays) else None,
    }
    log.append(record)
    save_json(QA_DAILY_LOG, log)
    return log


def generate_conclusions(daily_log):
    """Regenerates qa_conclusions.md FROM SCRATCH every run, from the full
    accumulated qa_daily_log - always reflects all evidence gathered so
    far, not just today's. Each bucket is reported only once it has at
    least MIN_SAMPLES_FOR_CONCLUSION days, specifically so a conclusion
    like the "4%/25" one Tomer flagged (a small, skewed recent window)
    never gets presented here as if it were a firm, statistically
    meaningful finding."""

    def bucket_avg(records, key_filter):
        vals = [r["strong_daily_accuracy"] for r in records if key_filter(r) and r.get("strong_daily_accuracy") is not None]
        return (round(sum(vals) / len(vals), 1) if vals else None), len(vals)

    lines = [
        "# יומן מסקנות - tom-stock-alert",
        "",
        f"מעודכן אוטומטית מכל בדיקת QA (מחזיק {len(daily_log)} ימי מסחר שנרשמו עד כה). "
        f"בקטגוריה מוצג מסקנה רק מ-{MIN_SAMPLES_FOR_CONCLUSION} ימים ומעלה - פחות מזה מוצג כ'אין מספיק נתונים'.",
        "",
        "## דיוק מותנה מצב-שוק (תג שורי/דובי)",
    ]
    for label, filt in (
        ("ימים שהתג היה שורי", lambda r: r.get("market_regime_bullish") is True),
        ("ימים שהתג היה דובי", lambda r: r.get("market_regime_bullish") is False),
    ):
        avg, n = bucket_avg(daily_log, filt)
        if n >= MIN_SAMPLES_FOR_CONCLUSION:
            lines.append(f"- **{label}**: דיוק ממוצע {avg}% (מבוסס {n} ימים)")
        else:
            lines.append(f"- {label}: אין מספיק נתונים עדיין ({n}/{MIN_SAMPLES_FOR_CONCLUSION} ימים)")

    lines += ["", "## דיוק מותנה תשואת מדד בפועל (S&P 500 אותו יום)"]
    for label, filt in (
        ("ימים שהמדד עלה בפועל", lambda r: (r.get("sp500_pct_change") or 0) > 0),
        ("ימים שהמדד ירד בפועל", lambda r: (r.get("sp500_pct_change") or 0) < 0),
    ):
        avg, n = bucket_avg(daily_log, filt)
        if n >= MIN_SAMPLES_FOR_CONCLUSION:
            lines.append(f"- **{label}**: דיוק ממוצע {avg}% (מבוסס {n} ימים)")
        else:
            lines.append(f"- {label}: אין מספיק נתונים עדיין ({n}/{MIN_SAMPLES_FOR_CONCLUSION} ימים)")

    total_suspect = sum(r.get("data_suspect_count") or 0 for r in daily_log)
    lines += ["", "## חריגות נתונים", f"סה\"כ {total_suspect} סימוני 'נתון חשוד' מאז שהמנגנון הופעל (ראה qa_outliers_log.json לפירוט מלא)."]

    QA_CONCLUSIONS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    today_str = date.today().isoformat()

    if not is_market_trading_day():
        send_telegram_message("ℹ️ בדיקת QA יומית: אין מסחר היום (סופ״ש/חג ארה״ב) - לא בוצעה בדיקה.")
        print("Not a trading day, QA check skipped.")
        return

    store = load_json(PREDICTIONS_FILE, {})
    prices_payload = load_json(CURRENT_PRICES_FILE, {})
    prices_data = prices_payload.get("prices", {})

    issues, info, outliers = [], [], []

    check_engine_ran_today(store, today_str, issues, info)
    check_duplicates(store, today_str, issues)
    check_bad_prices(prices_data, issues)
    check_support_resistance(store, today_str, issues)
    check_required_fields(store, today_str, issues)
    check_monthly_portfolio_prices(store, prices_data, issues)
    check_syntax(issues)
    check_stagnation(prices_data, today_str, issues, info, prices_payload.get("updated_at"))
    check_accuracy_streak(store, issues)
    check_data_suspect_flags(store, today_str, outliers)

    check_direction_consistency(store, today_str, issues)
    check_analyst_gap(store, today_str, info)
    check_component_redundancy(store, today_str, issues)
    check_price_format(store, issues)
    check_calibration_text(store, issues)
    check_stale_sims(store, today_str, issues)

    daily_log = record_daily_log(store, today_str, prices_payload)
    check_dead_features(daily_log, issues)
    generate_conclusions(daily_log)

    if issues:
        header = f"⚠️ בדיקת QA יומית ({today_str}) - נמצאו בעיות:"
        body = "\n".join(f"• {i}" for i in issues)
        review = [i for i in info if i.startswith("פער חריג")]
        if review:
            body += "\n\n🔍 לבדיקה ידנית:\n" + "\n".join(f"• {i}" for i in review)
    else:
        header = f"✅ בדיקת QA יומית ({today_str}) - הכל תקין."
        body = "\n".join(f"• {i}" for i in info)

    if outliers:
        body += "\n\n🔎 חריגות נתונים שסוננו אוטומטית:\n" + "\n".join(f"• {o}" for o in outliers)

    msg = header + (("\n" + body) if body else "")
    send_telegram_message(msg)
    print(msg)


if __name__ == "__main__":
    main()
