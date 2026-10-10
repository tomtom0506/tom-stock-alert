"""
🔁 בדיקת התאמה A/B - הקוד החי מול מנוע המחקר (אופליין בלבד).
מופעל רק דרך .github/workflows/ab_parity_check.yml (workflow_dispatch).
לא כותב לשום קובץ של האפליקציה החיה - התוצאות נשמרות כ-artifact.

Question it answers (audit 10.10.2026, item 6): does the LIVE strategy book
(stock_alerts.compute_book_signals + simulate_book) produce the same
signals and trades as the rules that were actually validated in the theory
lab (research/theory_lab_stage3.py), on the same days? Target: 95%+ match.
Waiting months for live performance only tells us something if the live
code is the thing that was tested.

Both sides run on the SAME downloaded panel (the live downloader, S&P 500,
2 years, auto-adjusted), so any difference is a difference in the RULES,
not in the data:

  * Signal level, per strategy: for every session in the window, the set of
    tickers with a live signal vs the set with a research signal. Reported
    as overall match = |both| / |either| (Jaccard), plus counts of live-only
    and research-only signals, with examples.
  * Trade level: the research harness takes EVERY signal (no slot limit),
    the live book has 10 (A) / 20 (B) slots. So the check is (1) every live
    trade should exist in the research trade list with the same ticker,
    entry and exit date, and (2) how many research trades the live book
    skipped because its slots were full - that's a real difference in what
    is being measured, reported separately, not counted as a mismatch.

B is compared against BOTH research versions: the base rule that produced
t=3.6 (no market gate) and the "B+exposure" variant (no new trades while
SPY is below its 200-day average), because the live code uses the gate.
"""
import json
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import stock_alerts as live  # noqa: E402  (read-only use of the live functions)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_ab_parity"
TARGET_MATCH_PCT = 95.0
WINDOW_SESSIONS = 250          # signal comparison window (about a year), plus the live window separately
EXAMPLES = 15


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# research rules - copied verbatim from research/theory_lab_stage3.py main()
# (sigA / sigB / spy_gate), so this file must be updated if those change.
# ---------------------------------------------------------------------------
def research_signals(O, H, L, C, V, spy):
    uptrend = C > C.rolling(200).mean()
    vol20 = V.rolling(20).mean()
    drop5 = C / C.shift(5) - 1
    gap = O / C.shift(1) - 1
    strong = (C - L) / (H - L).replace(0, np.nan) > 0.5
    spy_gate = (spy > spy.rolling(200).mean()) | spy.rolling(200).mean().isna()
    sigA = (drop5 <= -0.12) & uptrend
    sigB = (gap >= 0.05) & (V >= 3 * vol20) & strong
    gate = spy_gate.reindex(C.index).fillna(True)          # Lab2.trades: gate.reindex(idx).fillna(True)
    sigB_gated = sigB & gate.values[:, None]
    return {"A": sigA.fillna(False), "B": sigB.fillna(False), "B_gated": sigB_gated.fillna(False)}


def research_trades(sig, O, C, hold, start_i):
    """Lab2.trades entry/exit rules: enter at the open after the signal,
    exit at the close of the hold-th session, no overlapping trades per
    ticker. Costs are irrelevant here - only (ticker, entry, exit) matter."""
    Ov, Cv, S = O.values, C.values, sig.values
    idx, cols, n = C.index, list(C.columns), len(C.index)
    out = set()
    for j, t in enumerate(cols):
        nxt = -1
        for i in np.flatnonzero(S[:, j]):
            if i < start_i - 1 or i <= nxt or i + 2 >= n:
                continue
            e = i + 1
            x = min(e + hold - 1, n - 1)
            if not (np.isfinite(Ov[e, j]) and Ov[e, j] > 0 and np.isfinite(Cv[x, j])):
                continue
            if e + hold - 1 <= n - 1:          # only completed research trades are comparable
                out.add((t, idx[e].strftime("%Y-%m-%d"), idx[x].strftime("%Y-%m-%d")))
            nxt = x
    return out


def compare_signals(live_sig, res_sig, dates):
    both = live_only = res_only = 0
    ex_live, ex_res = [], []
    for d in dates:
        a = set(live_sig.columns[live_sig.loc[d].values])
        b = set(res_sig.columns[res_sig.loc[d].values])
        both += len(a & b)
        for t in sorted(a - b):
            live_only += 1
            if len(ex_live) < EXAMPLES:
                ex_live.append(f"{d.strftime('%Y-%m-%d')} {t}")
        for t in sorted(b - a):
            res_only += 1
            if len(ex_res) < EXAMPLES:
                ex_res.append(f"{d.strftime('%Y-%m-%d')} {t}")
    either = both + live_only + res_only
    return {"sessions": len(dates), "both": both, "live_only": live_only, "research_only": res_only,
            "match_pct": round(both / either * 100, 1) if either else None,
            "examples_live_only": ex_live, "examples_research_only": ex_res}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = live.get_sp500_tickers()
    if not sp:
        log("no S&P 500 list - abort")
        sys.exit(1)
    log(f"universe: {len(sp)} tickers, live downloader, period {live.BOOK_PERIOD}")
    spy = yf.Ticker("SPY").history(period=live.BOOK_PERIOD)["Close"].dropna()
    spy.index = spy.index.tz_localize(None) if getattr(spy.index, "tz", None) is not None else spy.index
    O, H, L, C, V = live._download_book_panels(sorted(set(sp)))
    for df in (O, H, L, C, V):
        df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
    # last COMPLETE session only, same rule as run_strategy_book
    ny = live._ny_now()
    if len(C.index) and C.index[-1].date() == ny.date() and (ny.hour, ny.minute) < live.US_CLOSE_NY:
        O, H, L, C, V = O.iloc[:-1], H.iloc[:-1], L.iloc[:-1], C.iloc[:-1], V.iloc[:-1]
    spy = spy[spy.index <= C.index[-1]]
    log(f"panel: {C.shape[1]} tickers x {C.shape[0]} sessions, last {C.index[-1].date()}")

    live_sigs = live.compute_book_signals(O, H, L, C, V, spy)
    res = research_signals(O, H, L, C, V, spy)
    start_i = 200
    live_start = date(2026, 10, 1)
    window = C.index[max(start_i, len(C.index) - WINDOW_SESSIONS):]
    live_window = [d for d in C.index if d.date() >= live_start]

    out = {"meta": {"generated_at": datetime.now(timezone.utc).isoformat(), "tickers": int(C.shape[1]),
                    "sessions": int(C.shape[0]), "first": str(C.index[0].date()), "last": str(C.index[-1].date()),
                    "window_first": str(window[0].date()), "live_start": live_start.isoformat(),
                    "target_match_pct": TARGET_MATCH_PCT, "backend_version": live.BACKEND_VERSION},
           "signals": {}, "trades": {}}
    pairs = [("A", "A"), ("B", "B"), ("B", "B_gated")]
    for live_key, res_key in pairs:
        name = f"live {live_key} vs research {res_key}"
        out["signals"][name] = {
            "last_year": compare_signals(live_sigs[live_key][0], res[res_key], window),
            "live_window": compare_signals(live_sigs[live_key][0], res[res_key], live_window) if live_window else None,
        }
        log(f"{name}: signal match {out['signals'][name]['last_year']['match_pct']}%")

    for live_key, res_key in pairs:
        cfg = live.BOOK_STRATEGIES[live_key]
        sim = live.simulate_book(live_key, live_sigs[live_key][0], live_sigs[live_key][1], O, C, start_i)
        live_tr = {(t["ticker"], t["entry_date"], t["exit_date"]) for t in sim["trades"]}
        first_live = min((t[1] for t in live_tr), default=None)
        res_tr = {t for t in research_trades(res[res_key], O, C, cfg["hold"], start_i) if first_live and t[1] >= first_live}
        last_exit = max((t[2] for t in live_tr), default=None)
        res_tr = {t for t in res_tr if last_exit and t[2] <= last_exit}
        in_both = live_tr & res_tr
        name = f"live {live_key} vs research {res_key}"
        out["trades"][name] = {
            "live_trades": len(live_tr), "research_trades": len(res_tr), "live_also_in_research": len(in_both),
            "live_trade_match_pct": round(len(in_both) / len(live_tr) * 100, 1) if live_tr else None,
            "research_trades_not_taken_live": len(res_tr - live_tr),
            "examples_live_not_in_research": [" ".join(t) for t in sorted(live_tr - res_tr)[:EXAMPLES]],
            "note": f"live book has {cfg['slots']} slots; research takes every signal - "
                    f"research trades not taken live are mostly 'slots were full', not a rule mismatch",
        }
        log(f"{name}: live trades in research {out['trades'][name]['live_trade_match_pct']}%")

    (OUTPUT_DIR / "ab_parity_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))

    def verdict(p):
        return "—" if p is None else ("✅ עובר" if p >= TARGET_MATCH_PCT else "❌ לא עובר")
    lines = ["# 🔁 בדיקת התאמה A/B - קוד חי מול מנוע המחקר", "",
             f"נתונים: {out['meta']['tickers']} מניות · {out['meta']['first']} עד {out['meta']['last']} · backend {live.BACKEND_VERSION}",
             f"יעד: {TARGET_MATCH_PCT:.0f}%+ התאמה. שני הצדדים רצו על אותם נתונים בדיוק, כך שכל פער הוא פער בכללים.", "",
             "## התאמת אותות (שנה אחרונה)"]
    for name, v in out["signals"].items():
        s = v["last_year"]
        lines.append(f"- **{name}:** {s['match_pct']}% {verdict(s['match_pct'])} · משותפים {s['both']} · "
                     f"רק בחי {s['live_only']} · רק במחקר {s['research_only']}")
        if v["live_window"]:
            w = v["live_window"]
            lines.append(f"  - מאז שהאסטרטגיות חיות ({live_start.strftime('%d.%m')}): {w['match_pct']}% "
                         f"({w['both']} משותפים, {w['live_only']} רק בחי, {w['research_only']} רק במחקר)")
        if s["examples_live_only"]:
            lines.append(f"  - דוגמאות רק בחי: {', '.join(s['examples_live_only'][:6])}")
        if s["examples_research_only"]:
            lines.append(f"  - דוגמאות רק במחקר: {', '.join(s['examples_research_only'][:6])}")
    lines += ["", "## התאמת עסקאות"]
    for name, t in out["trades"].items():
        lines.append(f"- **{name}:** {t['live_trade_match_pct']}% מהעסקאות החיות קיימות גם במחקר "
                     f"{verdict(t['live_trade_match_pct'])} ({t['live_also_in_research']}/{t['live_trades']}) · "
                     f"{t['research_trades_not_taken_live']} עסקאות מחקר לא נלקחו בחי (בעיקר כי המשבצות היו מלאות)")
    lines += ["", "## איך לקרוא",
              "- התאמה נמוכה מול 'research B' וגבוהה מול 'research B_gated' = ההבדל הוא שער ה-S&P בלבד. "
              "הגרסה שנבדקה במחקר עם t=3.6 היא בלי שער; הגרסה עם השער נבדקה רק בבדיקת העמלות. זו החלטה, לא באג.",
              "- הבדל בין 'כל אות' (מחקר) ל-'10/20 משבצות' (חי) משנה את מה שנמדד: הביצועים החיים יהיו של תת-קבוצה של העסקאות שנבדקו."]
    (OUTPUT_DIR / "ab_parity_report.md").write_text("\n".join(lines))
    log("Done")


if __name__ == "__main__":
    main()
