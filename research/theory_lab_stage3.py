"""
🧪 מעבדת תיאוריות - שלב 3: בדיקת חוסן - אופליין בלבד.
מופעל רק דרך .github/workflows/theory_lab_stage3.yml (workflow_dispatch).

Stage 2 (2.10.2026) passed two theories with ONE parameter set each:
  A. 12%+ drop in 5 days while above the 200-day average, hold 10 days
  B. 5%+ opening gap on 3x volume, close in the upper half, hold 60 days
Stage 3 asks whether that is a real effect or a lucky parameter choice:
  1. Parameter neighborhood: A = drop 10/12/15% x hold 5/10/15,
     B = gap 4/5/6% x hold 20/40/60, plus volume x2 / x4 at the base gap.
     A real effect should hold across most of its neighborhood.
  2. Cost stress: 2x and 3x the cost per side (opening gaps are expensive
     to trade - real slippage on those days is larger).
  3. Concentration: median and trimmed (2.5% each side) excess, so a few
     huge winners can't carry the result.
  4. Transparency: run timing per step and rows actually loaded per ticker.
Universe, data and benchmark exactly as stage 2.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory_lab as S1  # noqa: E402
import theory_lab_stage2 as S2  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output_theory_lab_stage3"
COST = S1.COST_PER_SIDE
ROBUST_SHARE = 0.75   # >= 75% of the neighborhood must be positive with t >= 2


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def describe(lab, df, name, params, with_placebo=False):
    if len(df) == 0:
        return {"name": name, **params, "n": 0}
    ex = df["excess"].values
    lo, hi = np.percentile(ex, [2.5, 97.5])
    trimmed = ex[(ex >= lo) & (ex <= hi)]
    periods = S2.period_table(df["excess"])
    t_all = S1.tstat(ex)
    t_17 = S1.tstat(df[df.index >= S2.HOLDOUT_START]["excess"].values)
    return {"name": name, **params, "n": int(len(df)),
            "win_rate_pct": round(float((df["ret"] > 0).mean() * 100), 1),
            "avg_excess_pct": round(float(ex.mean() * 100), 3),
            "median_excess_pct": round(float(np.median(ex) * 100), 3),
            "trimmed_excess_pct": round(float(trimmed.mean() * 100), 3),
            "t_all": round(t_all, 2) if t_all is not None else None,
            "t_2017": round(t_17, 2) if t_17 is not None else None,
            "positive_periods": sum(1 for p in periods if (p["avg_excess_pct"] or 0) > 0 and p["n"] > 0),
            "periods": periods,
            "placebo_pct": lab.placebo(df) if with_placebo else None}


def robust(rows):
    good = [r for r in rows if r.get("n") and (r.get("avg_excess_pct") or 0) > 0 and (r.get("t_all") or 0) >= 2]
    return {"variants": len(rows), "good": len(good), "share": round(len(good) / len(rows), 2) if rows else 0,
            "robust": bool(rows) and len(good) / len(rows) >= ROBUST_SHARE}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timing = {}
    t0 = time.time()
    universe, src = S2.get_universe()
    frames = S2.download_chunked(universe + ["SPY"])
    timing["download_sec"] = round(time.time() - t0, 1)
    rows_per = {t: int(len(f)) for t, f in frames.items()}
    vals = np.array(list(rows_per.values()))
    data_report = {"universe_source": src, "requested": len(universe) + 1, "loaded": len(frames),
                   "total_rows": int(vals.sum()), "rows_min": int(vals.min()), "rows_median": int(np.median(vals)),
                   "rows_max": int(vals.max()), "tickers_with_full_26y": int((vals >= 6400).sum()),
                   "missing": sorted(set(universe + ["SPY"]) - set(frames))}
    log(f"Data: {data_report}")

    t1 = time.time()
    lab = S2.Lab2(frames, [t for t in universe if t in frames])
    C, V, O, H, L = lab.C, lab.V, lab.O, lab.H, lab.L
    spy = frames["SPY"]["Close"]
    spy_gate = (spy > spy.rolling(200).mean()) | spy.rolling(200).mean().isna()
    uptrend = C > lab.sma200
    vol20 = V.rolling(20).mean()
    drop5 = C / C.shift(5) - 1
    gap = O / C.shift(1) - 1
    strong = (C - L) / (H - L).replace(0, np.nan) > 0.5
    timing["setup_sec"] = round(time.time() - t1, 1)

    t2 = time.time()
    A, B, stress = [], [], []
    for x in (0.10, 0.12, 0.15):
        sig = (drop5 <= -x) & uptrend
        for hold in (5, 10, 15):
            base = (x == 0.12 and hold == 10)
            A.append(describe(lab, lab.trades(sig, hold), "A", {"drop_pct": int(x * 100), "hold": hold}, with_placebo=base))
            log(f"  A drop {x:.0%} hold {hold}: {A[-1].get('avg_excess_pct')} t={A[-1].get('t_all')}")
    for g in (0.04, 0.05, 0.06):
        sig = (gap >= g) & (V >= 3 * vol20) & strong
        for hold in (20, 40, 60):
            base = (g == 0.05 and hold == 60)
            B.append(describe(lab, lab.trades(sig, hold), "B", {"gap_pct": int(g * 100), "volume_x": 3, "hold": hold}, with_placebo=base))
            log(f"  B gap {g:.0%} hold {hold}: {B[-1].get('avg_excess_pct')} t={B[-1].get('t_all')}")
    for vm in (2, 4):
        sig = (gap >= 0.05) & (V >= vm * vol20) & strong
        B.append(describe(lab, lab.trades(sig, 60), "B", {"gap_pct": 5, "volume_x": vm, "hold": 60}))
    sigA = (drop5 <= -0.12) & uptrend
    sigB = (gap >= 0.05) & (V >= 3 * vol20) & strong
    for mult in (2, 3):
        c = COST * mult
        stress.append(describe(lab, lab.trades(sigA, 10, cost=c), "A", {"cost_per_side_pct": round(c * 100, 2)}))
        stress.append(describe(lab, lab.trades(sigB, 60, cost=c), "B", {"cost_per_side_pct": round(c * 100, 2)}))
        stress.append(describe(lab, lab.trades(sigB, 60, gate=spy_gate, cost=c), "B+exposure", {"cost_per_side_pct": round(c * 100, 2)}))
    timing["tests_sec"] = round(time.time() - t2, 1)
    timing["total_sec"] = round(time.time() - t0, 1)

    out = {"meta": {"generated_at": datetime.now(timezone.utc).isoformat(), "first_date": str(lab.idx[0].date()),
                    "last_date": str(lab.idx[-1].date()), "usable_tickers": len(lab.u), "cost_per_side_pct": COST * 100,
                    "robust_rule": f">= {int(ROBUST_SHARE*100)}% of the neighborhood with excess > 0 and t >= 2",
                    "timing": timing, "data": data_report},
           "A_drop_neighborhood": A, "A_robustness": robust(A),
           "B_gap_neighborhood": B, "B_robustness": robust(B),
           "cost_stress": stress}
    (OUTPUT_DIR / "stage3_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    lines = ["# 🧪 שלב 3 - בדיקת חוסן", "", f"זמנים: הורדה {timing['download_sec']} ש' · חישוב {timing['tests_sec']} ש' · סה\"כ {timing['total_sec']} ש'",
             f"נתונים: {data_report['loaded']}/{data_report['requested']} מניות · {data_report['total_rows']:,} שורות · חציון {data_report['rows_median']} שורות למניה", "",
             f"A (ירידה חדה): {out['A_robustness']['good']}/{out['A_robustness']['variants']} וריאציות טובות -> {'חסין' if out['A_robustness']['robust'] else 'לא חסין'}",
             f"B (קפיצת חדשות): {out['B_robustness']['good']}/{out['B_robustness']['variants']} וריאציות טובות -> {'חסין' if out['B_robustness']['robust'] else 'לא חסין'}"]
    (OUTPUT_DIR / "stage3_report.md").write_text("\n".join(lines))
    log("Done")


if __name__ == "__main__":
    main()
