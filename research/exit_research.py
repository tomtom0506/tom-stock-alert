"""
🎯 מחקר יציאה: A/B - לפי זמן, או לפי מחיר או זמן (הראשון מביניהם). אופליין בלבד.
מופעל רק דרך .github/workflows/exit_research.yml (workflow_dispatch).
לא כותב לשום קובץ של האפליקציה החיה - התוצאות נשמרות כ-artifact.

Tomer, 14.10.2026: "chart reading is a proven method - also give a price to
sell at, besides the time rule; whichever comes first." The validated A/B
rules exit on TIME only. This tests, on the SAME entries (theory_lab_stage3
harness, S&P 500, 26 years), whether adding chart levels helps:

  TIME          the validated rule (A: 10 sessions, B: 60)
  TARGET        + sell at the resistance line (the app's own
                compute_technical_factors on the signal day; above the
                entry, else the 52-week high), whichever comes first
  STOP          + sell if the support line breaks
  FIRST         + both target and stop (Tomer's idea)

Fills: a target is a limit (filled at the target, or at the open if it
gaps above); a stop fills at the stop (or the open if it gaps below); on a
day that touches both, the stop is assumed first (the conservative side).
0.15% per side as everywhere in the lab.

Same entries for every exit rule, so the comparison is PAIRED: for each
trade, (exit rule) - (time rule). Trades overlap in time, so the t-stat is
on YEARLY averages of that difference, not on trades.

Decision (fixed before running): an exit rule is adopted only if its
paired improvement over TIME is positive with t >= 2 across years, positive
in the holdout (2017+), and the rule itself still passes the lab's bar
(avg excess > 0, t >= 2).
"""
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import theory_lab as S1           # noqa: E402
import theory_lab_stage2 as S2    # noqa: E402
import theory_lab_stage3 as S3    # noqa: E402
import stock_alerts as live       # noqa: E402  (support/resistance exactly as the app computes them)

OUTPUT_DIR = HERE / "output_exit_research"
COST = S1.COST_PER_SIDE
WINDOW = 252


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def entries(lab, signal, hold):
    """The validated rule's entries: (i_signal, j, e, x_time), no overlap per ticker."""
    C, O = lab.C.values, lab.O.values
    sig = signal.reindex(index=lab.idx, columns=lab.u).fillna(False).values
    n = len(lab.idx)
    out = []
    for j in range(len(lab.u)):
        nxt = -1
        for i in np.flatnonzero(sig[:, j]):
            if i <= nxt or i + 2 >= n or i < WINDOW:
                continue
            e = i + 1
            x = min(e + hold - 1, n - 1)
            if not (np.isfinite(O[e, j]) and O[e, j] > 0 and np.isfinite(C[x, j])):
                continue
            out.append((i, j, e, x))
            nxt = x
    return out


def levels(lab, i, j, entry):
    t = lab.u[j]
    sl = slice(i - WINDOW + 1, i + 1)
    try:
        f = live.compute_technical_factors(lab.C[t].iloc[sl], lab.V[t].iloc[sl], lab.H[t].iloc[sl], lab.L[t].iloc[sl])
    except Exception:
        return None, None, None
    if not f:
        return None, None, None
    tgt = None
    for k in ("resistance", "year_high"):
        v = f.get(k)
        if v and v > entry * 1.005:
            tgt = float(v)
            break
    far = f.get("year_high")
    far = float(far) if isinstance(far, (int, float)) and not isinstance(far, bool) and entry * 1.005 < far <= entry * 1.4 else tgt
    stop = None
    for k in ("support",):
        v = f.get(k)
        if v and v < entry * 0.995:
            stop = float(v)
            break
    return tgt, stop, far


def run_exits(lab, ents, use_target, use_stop, lv, far=False):
    C, O, H, L = lab.C.values, lab.O.values, lab.H.values, lab.L.values
    rows = []
    for (i, j, e, x), (tgt, stop, far_t) in zip(ents, lv):
        entry = O[e, j]
        px, xi, why = C[x, j], x, "time"
        tg = (far_t if far else tgt) if use_target else None
        st = stop if use_stop else None
        if tg or st:
            for k in range(e, x + 1):
                o, h, l = O[k, j], H[k, j], L[k, j]
                if not (np.isfinite(h) and np.isfinite(l)):
                    continue
                if st and l <= st:
                    px, xi, why = (min(o, st) if k > e and np.isfinite(o) else st), k, "stop"
                    break
                if tg and h >= tg:
                    px, xi, why = (max(o, tg) if k > e and np.isfinite(o) else tg), k, "target"
                    break
        net = (px / entry) * (1 - COST) ** 2 - 1
        rows.append((lab.idx[i], lab.u[j], net, net - lab.bench(i, xi), xi - e + 1, why))
    return pd.DataFrame(rows, columns=["date", "ticker", "ret", "excess", "days", "why"]).set_index("date")


def paired(base, alt):
    d = (alt["excess"].values - base["excess"].values)
    s = pd.Series(d, index=pd.to_datetime(base.index))
    by_year = s.groupby(s.index.year).mean()
    hold = by_year[by_year.index >= 2017]
    return {"avg_improvement_pct": round(float(d.mean()) * 100, 3),
            "median_improvement_pct": round(float(np.median(d)) * 100, 3),
            "better_trades_pct": round(float((d > 0).mean()) * 100, 1),
            "t_years": _t(list(by_year.values)),
            "years": int(len(by_year)),
            "holdout_avg_improvement_pct": round(float(hold.mean()) * 100, 3) if len(hold) else None,
            "by_year_pct": {int(y): round(float(v) * 100, 2) for y, v in by_year.items()}}


def _t(x):
    x = [v for v in x if np.isfinite(v)]
    if len(x) < 3:
        return None
    sd = float(np.std(x, ddof=1))
    return round(float(np.mean(x)) / (sd / np.sqrt(len(x))), 2) if sd > 0 else None


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe, src = S2.get_universe()
    frames = S2.download_chunked(universe + ["SPY"])
    lab = S2.Lab2(frames, [t for t in universe if t in frames])
    C, V, O, H, L = lab.C, lab.V, lab.O, lab.H, lab.L
    uptrend = C > lab.sma200
    vol20 = V.rolling(20).mean()
    drop5 = C / C.shift(5) - 1
    gap = O / C.shift(1) - 1
    strong = (C - L) / (H - L).replace(0, np.nan) > 0.5
    strategies = {"A": ((drop5 <= -0.12) & uptrend, 10), "B": ((gap >= 0.05) & (V >= 3 * vol20) & strong, 60)}
    log(f"{len(lab.u)} tickers, {len(lab.idx)} sessions")

    out = {}
    for name, (sig, hold) in strategies.items():
        ents = entries(lab, sig, hold)
        log(f"{name}: {len(ents)} entries, computing chart levels...")
        lv = [levels(lab, i, j, O.values[e, j]) for (i, j, e, x) in ents]
        cov = {"with_target": sum(1 for a, b, c in lv if a), "with_stop": sum(1 for a, b, c in lv if b),
               "with_far": sum(1 for a, b, c in lv if c)}
        res = {}
        if not ents:
            out[name] = {"coverage": cov, "results": {}}
            log(f"{name}: no entries")
            continue
        base = run_exits(lab, ents, False, False, lv)
        for rule, ut, us in (("TIME", False, False), ("TARGET", True, False), ("STOP", False, True), ("FIRST", True, True), ("FAR", True, True)):
            df = base if rule == "TIME" else run_exits(lab, ents, ut, us, lv, far=(rule == "FAR"))
            d = S3.describe(lab, df[["ret", "excess", "days"]], name, {"exit": rule}, with_placebo=(rule == "FIRST"))
            d["exit_reasons_pct"] = {k: round(v * 100, 1) for k, v in df["why"].value_counts(normalize=True).items()}
            d["avg_days"] = round(float(df["days"].mean()), 1)
            if rule != "TIME":
                d["vs_time"] = paired(base, df)
                v = d["vs_time"]
                d["verdict"] = "ADOPT" if ((v["avg_improvement_pct"] or 0) > 0 and (v["t_years"] or 0) >= 2
                                            and (v["holdout_avg_improvement_pct"] or 0) > 0
                                            and (d.get("avg_excess_pct") or 0) > 0 and (d.get("t_all") or 0) >= 2) else "KEEP TIME"
            res[rule] = d
            log(f"{name} {rule}: n={d.get('n')} excess={d.get('avg_excess_pct')}% t={d.get('t_all')} "
                f"vs time={d.get('vs_time', {}).get('avg_improvement_pct')} -> {d.get('verdict', '-')}")
        out[name] = {"coverage": cov, "results": res}

    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "universe_source": src, "tickers": len(lab.u),
            "first": str(lab.idx[0].date()), "last": str(lab.idx[-1].date()), "runtime_sec": round(time.time() - t0, 1),
            "backend_version": live.BACKEND_VERSION}
    (OUTPUT_DIR / "exit_summary.json").write_text(json.dumps({"meta": meta, **out}, ensure_ascii=False, indent=1, default=str))

    Lr = ["# 🎯 מחקר יציאה - זמן מול מחיר-או-זמן", "",
          f"{meta['tickers']} מניות S&P 500 · {meta['first']} עד {meta['last']} · אותן כניסות בדיוק לכל כלל יציאה · {meta['runtime_sec']} ש'", "",
          "**כלל ההחלטה (נקבע מראש):** כלל יציאה מאומץ רק אם השיפור שלו מול 'זמן בלבד' חיובי עם t ≥ 2 על פני השנים, "
          "חיובי גם מ-2017, והאסטרטגיה איתו עדיין עוברת את הרף (עודף > 0, t ≥ 2).", ""]
    lab_names = {"TIME": "זמן בלבד (הכלל הנוכחי)", "TARGET": "יעד (התנגדות) או זמן", "STOP": "סטופ (תמיכה) או זמן", "FIRST": "יעד, סטופ או זמן - הראשון", "FAR": "סטופ קרוב + יעד רחוק (שיא שנתי) - שיטת החבר"}
    for name in ("A", "B"):
        o = out[name]
        if not o["results"]:
            Lr += [f"## אסטרטגיה {name}", "אין עסקאות.", ""]
            continue
        Lr += [f"## אסטרטגיה {name}", f"רמות: יעד נמצא ל-{o['coverage']['with_target']}, סטופ ל-{o['coverage']['with_stop']} מתוך {o['results']['TIME']['n']} עסקאות", "",
               "| יציאה | עודף ממוצע | חציון | הצלחה | t | ימים | שיפור מול זמן | t שיפור | מ-2017 | סיבות יציאה | |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for rule in ("TIME", "TARGET", "STOP", "FIRST", "FAR"):
            d = o["results"][rule]
            v = d.get("vs_time") or {}
            reasons = " ".join(f"{k}:{p}%" for k, p in d["exit_reasons_pct"].items())
            Lr.append(f"| {lab_names[rule]} | {d['avg_excess_pct']:+.2f}% | {d['median_excess_pct']:+.2f}% | {d['win_rate_pct']}% | {d['t_all']} | {d['avg_days']} | "
                      f"{('%+.2f%%' % v['avg_improvement_pct']) if v else '—'} | {v.get('t_years', '—') if v else '—'} | "
                      f"{('%+.2f%%' % v['holdout_avg_improvement_pct']) if v and v.get('holdout_avg_improvement_pct') is not None else '—'} | {reasons} | "
                      f"{'✅ לאמץ' if d.get('verdict') == 'ADOPT' else ('—' if rule == 'TIME' else '❌ להשאיר זמן')} |")
        Lr.append("")
    Lr += ["## מגבלות",
           "- קווי התמיכה וההתנגדות מחושבים בדיוק כמו באפליקציה, ביום האות, על שנה אחורה.",
           "- מילוי יעד/סטופ מניח שהפקודה בוצעה בדיוק ברמה (או בפתיחה אם המחיר קפץ מעבר לה). בפועל ייתכן החלקה קטנה.",
           "- הטיית שורדים: רשימת ה-S&P של היום, כמו בכל שלבי המעבדה - משפיעה באותה מידה על כל כללי היציאה."]
    (OUTPUT_DIR / "exit_report.md").write_text("\n".join(Lr))
    log("Done")


if __name__ == "__main__":
    main()
