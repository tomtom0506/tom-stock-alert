"""
🇮🇱 האם אסטרטגיות A ו-B עובדות גם בבורסת תל אביב? (אופליין בלבד)
מופעל רק דרך .github/workflows/ta_ab_check.yml (workflow_dispatch).
לא כותב לשום קובץ של האפליקציה החיה - התוצאות נשמרות כ-artifact.

Agreed 11.10.2026: Israeli stocks enter the "3 to buy now" short-term list
only if the two research-validated rules also pass on TASE. Same harness,
same rules, same bar as theory_lab_stage3 (S&P 500, where both passed):

  * A: 12%+ drop in 5 days while above the 200-day average, hold 10
  * B: 5%+ opening gap on 3x the 20-day volume, close in the upper half, hold 60
  * entry at the next open, exit at the close, 0.15% per side, measured
    against the equal-weight universe ("excess"), discovery/holdout split at
    2017, placebo vs 200 random-entry versions, parameter neighborhood.

PASS (per strategy, fixed before running): base rule with avg excess > 0,
t >= 2 overall, positive in the holdout (2017+), placebo >= 95%, and >= 75%
of the parameter neighborhood positive with t >= 2 - plus still positive at
2x costs (TASE spreads are wider than on the S&P 500).

Universe: the app's own TASE list (ta_tickers.json, ~57 large names) - not
the full TA-125 (there is no free constituents file); survivorship bias
applies, as in every lab stage. Fewer names than the S&P 500 test means
fewer trades, so a FAIL can also mean "not enough evidence".
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
import theory_lab as S1           # noqa: E402
import theory_lab_stage2 as S2    # noqa: E402
import theory_lab_stage3 as S3    # noqa: E402

OUTPUT_DIR = HERE / "output_ta_ab_check"
TA_FILE = HERE.parent / "ta_tickers.json"
INDEX = "^TA125.TA"
COST = S1.COST_PER_SIDE


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


JUMP_LIMIT = 0.5      # a >50% close-to-close (or open-vs-previous-close) move = unadjusted split / bad data


def clean_frames(frames):
    """Run 1 (11.10.2026) was unusable: Yahoo's TASE history has unadjusted
    splits (ROBO.TA shows +9603% in one day - already in qa_outliers_log), and
    one such day blows up the equal-weight benchmark, so every trade's
    "excess" became nonsense (-4305% average). Each ticker keeps only the
    history AFTER its last impossible jump; if less than ~2 years remain it's
    dropped. Every cut is listed in the report."""
    out, cut = {}, {}
    for t, df in frames.items():
        c = df["Close"]
        r = c.pct_change().abs()
        g = (df["Open"] / c.shift(1) - 1).abs()
        bad = df.index[(r > JUMP_LIMIT) | (g > JUMP_LIMIT)]
        if len(bad):
            df = df[df.index > bad[-1]]
            cut[t] = f"{len(bad)} jumps, last {bad[-1].date()}, kept {len(df)} rows"
        if len(df) >= 500:
            out[t] = df
        elif t in cut:
            cut[t] += " -> dropped"
    return out, cut


def verdict(base, neighborhood, stress2):
    if not base.get("n"):
        return "FAIL", "אין עסקאות"
    checks = {
        "תשואה עודפת חיובית": (base.get("avg_excess_pct") or 0) > 0,
        "t כולל ≥ 2": (base.get("t_all") or 0) >= 2,
        "חיובי בבקרה (2017+)": (base.get("t_2017") or 0) > 0,
        "פלצבו ≥ 95%": (base.get("placebo_pct") or 0) >= 95,
        "חוסן פרמטרים ≥ 75%": neighborhood["robust"],
        "חיובי גם בעמלה כפולה": (stress2.get("avg_excess_pct") or 0) > 0,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return ("PASS" if not failed else "FAIL"), ("עבר את כל התנאים" if not failed else "נכשל ב: " + ", ".join(failed))


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = sorted(set(json.loads(TA_FILE.read_text(encoding="utf-8"))))
    log(f"TASE universe: {len(universe)} tickers + {INDEX}")
    frames = S2.download_chunked(universe + [INDEX])
    frames, cut = clean_frames(frames)
    if cut:
        log(f"data cleaning: {cut}")
    loaded = [t for t in universe if t in frames]
    log(f"loaded {len(loaded)}/{len(universe)}; index {'ok' if INDEX in frames else 'MISSING'}")
    if len(loaded) < 20:
        log("too few tickers - abort")
        sys.exit(1)

    lab = S2.Lab2(frames, loaded)
    C, V, O, H, L = lab.C, lab.V, lab.O, lab.H, lab.L
    uptrend = C > lab.sma200
    vol20 = V.rolling(20).mean()
    drop5 = C / C.shift(5) - 1
    gap = O / C.shift(1) - 1
    strong = (C - L) / (H - L).replace(0, np.nan) > 0.5

    A, B = [], []
    for x in (0.10, 0.12, 0.15):
        sig = (drop5 <= -x) & uptrend
        for hold in (5, 10, 15):
            A.append(S3.describe(lab, lab.trades(sig, hold), "A", {"drop_pct": int(x * 100), "hold": hold},
                                 with_placebo=(x == 0.12 and hold == 10)))
    for g in (0.04, 0.05, 0.06):
        sig = (gap >= g) & (V >= 3 * vol20) & strong
        for hold in (20, 40, 60):
            B.append(S3.describe(lab, lab.trades(sig, hold), "B", {"gap_pct": int(g * 100), "hold": hold},
                                 with_placebo=(g == 0.05 and hold == 60)))
    sigA = (drop5 <= -0.12) & uptrend
    sigB = (gap >= 0.05) & (V >= 3 * vol20) & strong
    stressA = S3.describe(lab, lab.trades(sigA, 10, cost=COST * 2), "A", {"cost_x": 2})
    stressB = S3.describe(lab, lab.trades(sigB, 60, cost=COST * 2), "B", {"cost_x": 2})
    baseA = next(r for r in A if r.get("drop_pct") == 12 and r.get("hold") == 10)
    baseB = next(r for r in B if r.get("gap_pct") == 5 and r.get("hold") == 60)
    rA, rB = S3.robust(A), S3.robust(B)
    vA, why_A = verdict(baseA, rA, stressA)
    vB, why_B = verdict(baseB, rB, stressB)

    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "tickers": len(loaded), "cleaned": cut,
            "first": str(lab.idx[0].date()), "last": str(lab.idx[-1].date()), "cost_per_side_pct": COST * 100,
            "runtime_sec": round(time.time() - t0, 1), "missing": sorted(set(universe) - set(loaded))}
    out = {"meta": meta, "A": {"verdict": vA, "why": why_A, "base": baseA, "robustness": rA, "cost_x2": stressA, "neighborhood": A},
           "B": {"verdict": vB, "why": why_B, "base": baseB, "robustness": rB, "cost_x2": stressB, "neighborhood": B}}
    (OUTPUT_DIR / "ta_ab_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))

    def line(r):
        if not r.get("n"):
            return "אין עסקאות"
        return (f"{r['n']} עסקאות · עודף ממוצע {r['avg_excess_pct']:+.2f}% (חציון {r['median_excess_pct']:+.2f}%) · "
                f"t={r['t_all']} · מ-2017 t={r['t_2017']} · הצלחה {r['win_rate_pct']}%"
                + (f" · פלצבו {r['placebo_pct']}%" if r.get("placebo_pct") is not None else ""))
    Lr = ["# 🇮🇱 אסטרטגיות A/B על בורסת תל אביב", "",
          f"{meta['tickers']} מניות מרשימת ת\"א של האפליקציה · {meta['first']} עד {meta['last']} · עמלה {COST * 100:.2f}% לצד · {meta['runtime_sec']} ש'", "",
          f"## A - ירידה חדה: {'✅ עובר' if vA == 'PASS' else '❌ לא עובר'}", f"- {why_A}", f"- בסיס: {line(baseA)}",
          f"- חוסן: {rA['good']}/{rA['variants']} וריאציות טובות · עמלה כפולה: {line(stressA)}", "",
          f"## B - קפיצת חדשות: {'✅ עובר' if vB == 'PASS' else '❌ לא עובר'}", f"- {why_B}", f"- בסיס: {line(baseB)}",
          f"- חוסן: {rB['good']}/{rB['variants']} וריאציות טובות · עמלה כפולה: {line(stressB)}", "",
          "## מגבלות",
          "- רשימת המניות של האפליקציה (~57), לא כל ת\"א 125 - ופחות מניות = פחות עסקאות, אז 'לא עובר' יכול לנבוע גם מחוסר ראיות.",
          "- הטיית שורדים: הרשימה של היום, בלי מניות שנמחקו.",
          "- מרווחי קנייה/מכירה בת\"א רחבים יותר - לכן נבדקה גם עמלה כפולה."]
    if cut:
        Lr.append("- ניקוי נתונים (קפיצה של מעל 50% ביום = פיצול לא מתוקנן): " + "; ".join(f"{t}: {v}" for t, v in cut.items()))
    if meta["missing"]:
        Lr.append(f"- לא נמצאו נתונים ל: {', '.join(meta['missing'])}")
    (OUTPUT_DIR / "ta_ab_report.md").write_text("\n".join(Lr))
    log(f"A: {vA} ({why_A}) | B: {vB} ({why_B})")


if __name__ == "__main__":
    main()
