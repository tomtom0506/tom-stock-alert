"""
🔍 מחקר: אסטרטגיה A לפי סיבת הירידה (אופליין). מופעל מ-research.yml (script = a_reason_research).

Tomer, 17.10.2026: buy only a strong stock that fell for no company-specific
reason (no technology or management problem) - if the market comes back, it
comes back. Academic work on short-term reversal points the same way: drops
WITHOUT company news tend to revert, drops on company news tend to continue.

Same harness and entries as the validated A (theory_lab_stage3: 12%+ drop in 5
days above the 200-day average, hold 10, 0.15% per side, excess vs the
equal-weight universe). Each A trade is split by WHY it fell, from prices only:

  idio = stock 5-day return - its sector ETF 5-day return
         (GICS sector from the current S&P 500 list -> SPDR sector ETF;
          before an ETF existed, SPY)
  MARKET/SECTOR drop   idio >= -T   (most of the drop is the market/sector)
  COMPANY drop         idio <  -T   (the stock fell alone)
  T = 4 / 6 / 8 points (robustness - a real effect shouldn't depend on one T)

Decision rule (fixed before running): the screen switches to "market/sector
drops only" if, at T = 6, that group's average excess is ABOVE full A, with
t >= 2, positive in the holdout (2017+), AND the same holds at T = 4 and 8
(direction at least). Otherwise A stays as it is.
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
import theory_lab_stage2 as S2    # noqa: E402
import theory_lab_stage3 as S3    # noqa: E402

OUT = HERE / "output_a_reason"
SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
SECTOR_ETF = {"Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
              "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
              "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
              "Communication Services": "XLC"}
THRESHOLDS = (4, 6, 8)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    universe, src = S2.get_universe()
    meta = pd.read_csv(SP500_CSV)
    meta["Symbol"] = meta["Symbol"].str.replace(".", "-", regex=False)
    sector_of = dict(zip(meta["Symbol"], meta["GICS Sector"]))
    frames = S2.download_chunked(universe + ["SPY"] + sorted(set(SECTOR_ETF.values())))
    lab = S2.Lab2(frames, [t for t in universe if t in frames])
    C, V, O, H, L = lab.C, lab.V, lab.O, lab.H, lab.L
    log(f"{len(lab.u)} tickers, {len(lab.idx)} sessions")

    def close_of(t):
        f = frames.get(t)
        return f["Close"].reindex(lab.idx) if f is not None else pd.Series(np.nan, index=lab.idx)
    spy5 = close_of("SPY") / close_of("SPY").shift(5) - 1
    etf5 = {e: close_of(e) / close_of(e).shift(5) - 1 for e in set(SECTOR_ETF.values())}

    uptrend = C > lab.sma200
    drop5 = C / C.shift(5) - 1
    sigA = (drop5 <= -0.12) & uptrend
    tr = lab.trades(sigA, 10)                     # the validated A, unchanged
    log(f"A trades: {len(tr)}")

    rows = []
    for d, r in tr.iterrows():
        t = r["ticker"]
        etf = SECTOR_ETF.get(sector_of.get(t))
        s5 = float(drop5.at[d, t])
        b5 = etf5[etf].get(d) if etf else np.nan
        bench_kind = "sector"
        if b5 is None or not np.isfinite(b5):
            b5, bench_kind = spy5.get(d), "spy"
        rows.append({"idio": (s5 - float(b5)) * 100 if b5 is not None and np.isfinite(b5) else np.nan,
                     "stock5": s5 * 100, "bench5": float(b5) * 100 if b5 is not None and np.isfinite(b5) else np.nan,
                     "bench": bench_kind})
    info = pd.DataFrame(rows, index=tr.index)
    tr = pd.concat([tr, info], axis=1)
    tr = tr[np.isfinite(tr["idio"])]

    full = S3.describe(lab, tr[["ret", "excess", "days"]], "A", {"group": "all"}, with_placebo=False)
    res = {"all": full}
    for T in THRESHOLDS:
        mkt = tr[tr["idio"] >= -T]
        com = tr[tr["idio"] < -T]
        res[f"market_T{T}"] = S3.describe(lab, mkt[["ret", "excess", "days"]], "A", {"group": "market/sector", "T": T}, with_placebo=(T == 6))
        res[f"company_T{T}"] = S3.describe(lab, com[["ret", "excess", "days"]], "A", {"group": "company", "T": T}, with_placebo=(T == 6))
        log(f"T={T}: market/sector n={res[f'market_T{T}'].get('n')} x={res[f'market_T{T}'].get('avg_excess_pct')} | "
            f"company n={res[f'company_T{T}'].get('n')} x={res[f'company_T{T}'].get('avg_excess_pct')}")

    def better(r):
        return (r.get("n") and (r.get("avg_excess_pct") or -9) > (full.get("avg_excess_pct") or 0))
    m6 = res["market_T6"]
    adopt = bool(better(m6) and (m6.get("t_all") or 0) >= 2 and (m6.get("t_2017") or 0) > 0
                 and better(res["market_T4"]) and better(res["market_T8"]))
    meta_out = {"generated_at": datetime.now(timezone.utc).isoformat(), "universe_source": src, "tickers": len(lab.u),
                "first": str(lab.idx[0].date()), "last": str(lab.idx[-1].date()), "trades": int(len(tr)),
                "adopt_market_only": adopt, "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "summary.json").write_text(json.dumps({"meta": meta_out, "results": res}, ensure_ascii=False, indent=1, default=str))

    def line(r):
        if not r.get("n"):
            return "אין עסקאות"
        return (f"{r['n']} עסקאות · עודף ממוצע {r['avg_excess_pct']:+.2f}% (חציון {r['median_excess_pct']:+.2f}%) · "
                f"t={r['t_all']} · מ-2017 t={r['t_2017']} · הצלחה {r['win_rate_pct']}%"
                + (f" · פלצבו {r['placebo_pct']}%" if r.get("placebo_pct") is not None else ""))
    L = ["# 🔍 אסטרטגיה A לפי סיבת הירידה", "",
         f"{meta_out['tickers']} מניות S&P 500 · {meta_out['first']} עד {meta_out['last']} · אותן עסקאות A בדיוק · {meta_out['runtime_sec']} ש'", "",
         "**ההחלטה: " + ("✅ לעבור ל'ירידת שוק/ענף בלבד'" if adopt else "❌ להשאיר את A כמו שהיא") + "**", "",
         f"- **כל A:** {line(full)}"]
    for T in THRESHOLDS:
        L += [f"- **סף {T} נק' - ירידת שוק/ענף:** {line(res[f'market_T{T}'])}",
              f"- **סף {T} נק' - ירידה של החברה:** {line(res[f'company_T{T}'])}"]
    L += ["", "## איך לקרוא",
          "- 'ירידת שוק/ענף' = המניה ירדה לכל היותר בסף הנקודות יותר מקרן הענף שלה באותם 5 ימים.",
          "- הסיווג לפי מחירים בלבד. ירידה של החברה עצמה כמעט תמיד נובעת מחדשות עליה (דוח, הנהלה, תחרות), אבל לא כל ירידה כזו היא בעיה אמיתית.",
          "- הענף לפי רשימת ה-S&P של היום (חברה שעברה ענף מסווגת לפי הענף הנוכחי). לפני שקרן ענף קיימת - מול ה-S&P."]
    (OUT / "report.md").write_text("\n".join(L))
    log("Done")


if __name__ == "__main__":
    main()
