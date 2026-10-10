"""
📐 בדק היסטורי - באיזה אופק ציון ה-Top10 מנבא? (אופליין בלבד)
מופעל רק דרך .github/workflows/top10_horizon_backtest.yml (workflow_dispatch).
לא כותב לשום קובץ של האפליקציה החיה - התוצאות נשמרות כ-artifact.

Audit 10.10.2026, item 2: in 36 live days the score had no 1-day power
(IC ~0) but a positive IC at 5 and 10 sessions - in a single bull regime,
with only 2-3 independent periods. This replays the LIVE scoring code on
~10 years of S&P 500 history to see whether that holds.

What is replayed exactly: stock_alerts.compute_technical_factors on the
same trailing 1-year window the live engine downloads, RS Rating as the
same cross-sectional percentile of run_up_180d, the same market regime
(SPY SMA50 vs SMA200 + 10-day change), compute_prediction_score x the
Minervini trend-template multiplier, and compute_risk_reward_score (the
live Top10 is ~99% risk/reward-weighted since the 6.10 calibration).

Known limits (also written into the report):
  * Fundamentals/analyst inputs (upside, recommendation, short interest)
    have no history in Yahoo - this is the TECHNICAL part of the score,
    which is most of it. Sector diversification of the Top10 is skipped
    for the same reason (no historical sectors).
  * Survivorship bias: today's S&P 500 list, so delisted losers are
    missing - results lean optimistic. That's why the bar below is strict.
  * Sampled every 5th session (runtime); independent-window t-stats are
    computed on every h-th sample so overlapping windows are never
    counted twice.

The formula is fixed (nothing is fitted here), so "walk-forward" means: the
same fixed rule judged separately per year, in a DISCOVERY period
(<= 2021) and a HOLDOUT period (2022 onward), and per market regime.

Decision rule fixed BEFORE running (week 2 of the audit plan):
  PASS for horizon h = mean IC > 0.03 AND t > 2 on independent windows AND
  Top10 excess over the universe above one full round trip of costs
  (0.30%) per holding period - in the full sample AND positive in holdout.
"""
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import stock_alerts as live  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output_top10_horizon"
YEARS = "10y"
WINDOW = 252                 # live PRICE_HISTORY_PERIOD = "1y"
STEP = 5                     # evaluate every 5th session
HORIZONS = (1, 5, 10, 20)
TOP_N = 10
ROUND_TRIP_COST_PCT = 0.30   # 0.15% per side, same as the theory lab
HOLDOUT_START = "2022-01-01"
PASS_IC, PASS_T = 0.03, 2.0
CHUNK = 100


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def download(tickers):
    frames = {}
    for k in range(0, len(tickers), CHUNK):
        part = tickers[k:k + CHUNK]
        log(f"  downloading {k + 1}-{k + len(part)} / {len(tickers)}")
        try:
            data = yf.download(" ".join(part), period=YEARS, group_by="ticker", threads=True,
                               progress=False, auto_adjust=True)
        except Exception as e:
            log(f"  chunk failed: {e}")
            continue
        for t in part:
            try:
                # newer yfinance returns (ticker, field) columns even for ONE
                # ticker with group_by="ticker" - handle both shapes
                if isinstance(data.columns, pd.MultiIndex):
                    lvl0 = data.columns.get_level_values(0)
                    df = data[t] if t in lvl0 else data.xs(t, axis=1, level=1)
                else:
                    df = data
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
                if len(df) > WINDOW + 30:
                    frames[t] = df
            except Exception:
                continue
        time.sleep(2)
    return frames


def ticker_factors(args):
    """Technical factors for one ticker on every sample date (worker)."""
    t, df, sample_dates = args
    out = {}
    pos = {d: i for i, d in enumerate(df.index)}
    C, V, H, L = df["Close"], df["Volume"], df["High"], df["Low"]
    for d in sample_dates:
        i = pos.get(d)
        if i is None or i < WINDOW - 1:
            continue
        sl = slice(i - WINDOW + 1, i + 1)
        try:
            tf = live.compute_technical_factors(C.iloc[sl], V.iloc[sl], H.iloc[sl], L.iloc[sl])
        except Exception:
            tf = None
        if not tf:
            continue
        keep = {k: v for k, v in tf.items() if not k.startswith("_") and not isinstance(v, (list, dict))}
        keep["tt_mult"] = (tf.get("trend_template") or {}).get("multiplier", 1.0)
        out[d] = keep
    return t, out


def spearman(a, b):
    if len(a) < 30:
        return None
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(ra.corr(rb))


def tstat(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if len(x) < 3:
        return None
    s = np.std(x, ddof=1)
    return round(float(np.mean(x) / (s / math.sqrt(len(x)))), 2) if s > 0 else None


def summarize(rows, h, label):
    """rows: per-sample dicts for horizon h, oldest first."""
    rows = [r for r in rows if r["ic"] is not None]
    if not rows:
        return {"label": label, "n": 0}
    step = max(1, h // STEP) if h > STEP else 1
    indep = rows[::step]
    ics = [r["ic"] for r in rows]
    exc_s = [r["top_score_excess"] for r in rows if r["top_score_excess"] is not None]
    exc_rr = [r["top_rr_excess"] for r in rows if r["top_rr_excess"] is not None]
    return {
        "label": label, "samples": len(rows), "independent": len(indep),
        "mean_ic": round(float(np.mean(ics)), 4), "positive_share_pct": round(float(np.mean([v > 0 for v in ics]) * 100), 1),
        "t_ic_independent": tstat([r["ic"] for r in indep]),
        "decile_spread_pct": round(float(np.mean([r["spread"] for r in rows])), 3),
        "top10_score_excess_pct": round(float(np.mean(exc_s)), 3) if exc_s else None,
        "t_top10_score_independent": tstat([r["top_score_excess"] for r in indep]),
        "top10_rr_excess_pct": round(float(np.mean(exc_rr)), 3) if exc_rr else None,
        "t_top10_rr_independent": tstat([r["top_rr_excess"] for r in indep]),
    }


def passes(s):
    if not s or not s.get("samples"):
        return False
    exc = s.get("top10_rr_excess_pct")
    return (s["mean_ic"] > PASS_IC and (s["t_ic_independent"] or 0) > PASS_T
            and exc is not None and exc > ROUND_TRIP_COST_PCT)


def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = live.get_sp500_tickers()
    if not universe:
        log("no S&P 500 list - abort")
        sys.exit(1)
    log(f"universe {len(universe)}, downloading {YEARS}...")
    frames = download(sorted(set(universe)))
    spy_df = download(["SPY"]).get("SPY")
    log(f"downloaded {len(frames)} tickers, SPY {'ok' if spy_df is not None else 'MISSING'}")
    if spy_df is None or len(frames) < 100:
        log("download failed - abort")
        sys.exit(1)
    spy = spy_df["Close"]
    dl_sec = round(time.time() - t0, 1)

    all_dates = spy.index
    sample_dates = list(all_dates[WINDOW + 10: len(all_dates) - 1: STEP])
    log(f"{len(frames)} tickers, {len(sample_dates)} sample dates {sample_dates[0].date()} -> {sample_dates[-1].date()}")

    t1 = time.time()
    factors = {}
    workers = max(1, (os.cpu_count() or 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (t, out) in enumerate(ex.map(ticker_factors, [(t, df, sample_dates) for t, df in frames.items()], chunksize=4)):
            factors[t] = out
            if (k + 1) % 50 == 0:
                log(f"  factors {k + 1}/{len(frames)}")
    fac_sec = round(time.time() - t1, 1)

    close = pd.DataFrame({t: df["Close"] for t, df in frames.items()}).reindex(all_dates)
    spy_sma50, spy_sma200 = spy.rolling(50).mean(), spy.rolling(200).mean()
    fwd = {h: (close.shift(-h) / close - 1) * 100 for h in HORIZONS}

    per_h = {h: [] for h in HORIZONS}
    for d in sample_dates:
        regime = {"bullish": bool(spy_sma50.loc[d] > spy_sma200.loc[d]) if np.isfinite(spy_sma200.loc[d]) else None,
                  "recent_10d_pct": round(float(spy.loc[d] / spy.shift(10).loc[d] - 1) * 100, 2)}
        day = {t: dict(f[d]) for t, f in factors.items() if d in f}
        if len(day) < 100:
            continue
        perf = sorted((t, v["run_up_180d"]) for t, v in day.items() if v.get("run_up_180d") is not None)
        ranked = sorted(perf, key=lambda p: p[1])
        for r, (t, _) in enumerate(ranked):
            day[t]["rs_rating"] = round(r / max(len(ranked) - 1, 1) * 100, 1)
        entries = []
        for t, v in day.items():
            sc = round(live.compute_prediction_score(v, regime) * v.get("tt_mult", 1.0), 2)
            e = dict(v, ticker=t, score=sc, predicted="up" if sc >= 0 else "down")
            e["rr"] = live.compute_risk_reward_score(e)
            entries.append(e)
        pool = [e for e in entries if e["predicted"] == "up" and abs(e["score"]) >= live.PREDICTION_SCORE_THRESHOLD
                and not e.get("data_suspect")]
        top_s = {e["ticker"] for e in sorted(pool, key=lambda e: e["score"], reverse=True)[:TOP_N]}
        top_r = {e["ticker"] for e in sorted(pool, key=lambda e: e["rr"], reverse=True)[:TOP_N]}
        for h in HORIZONS:
            vals = [(e["score"], fwd[h].at[d, e["ticker"]], e["ticker"]) for e in entries]
            vals = [x for x in vals if x[1] is not None and np.isfinite(x[1]) and abs(x[1]) < 300]
            if len(vals) < 100:
                continue
            rets = [x[1] for x in vals]
            univ = float(np.mean(rets))
            srt = sorted(vals, key=lambda x: x[0], reverse=True)
            k = len(srt) // 10
            ts = [x[1] for x in vals if x[2] in top_s]
            tr = [x[1] for x in vals if x[2] in top_r]
            per_h[h].append({
                "date": d.strftime("%Y-%m-%d"), "bull": regime["bullish"], "n": len(vals),
                "ic": spearman([x[0] for x in vals], rets),
                "spread": float(np.mean([x[1] for x in srt[:k]]) - np.mean([x[1] for x in srt[-k:]])),
                "top_score_excess": float(np.mean(ts) - univ) if ts else None,
                "top_rr_excess": float(np.mean(tr) - univ) if tr else None,
            })

    results = {}
    for h in HORIZONS:
        rows = per_h[h]
        disc = [r for r in rows if r["date"] < HOLDOUT_START]
        hold = [r for r in rows if r["date"] >= HOLDOUT_START]
        years = sorted({r["date"][:4] for r in rows})
        results[h] = {
            "all": summarize(rows, h, "הכל"),
            "discovery": summarize(disc, h, f"עד {HOLDOUT_START[:4]}"),
            "holdout": summarize(hold, h, f"מ-{HOLDOUT_START[:4]}"),
            "bull": summarize([r for r in rows if r["bull"]], h, "שוק שורי"),
            "bear": summarize([r for r in rows if r["bull"] is False], h, "שוק דובי"),
            "by_year": {y: summarize([r for r in rows if r["date"].startswith(y)], h, y) for y in years},
        }
        a, ho = results[h]["all"], results[h]["holdout"]
        results[h]["verdict"] = "PASS" if passes(a) and (ho.get("mean_ic") or 0) > 0 and (ho.get("top10_rr_excess_pct") or 0) > 0 else "FAIL"
        log(f"h={h}: IC {a.get('mean_ic')} t={a.get('t_ic_independent')} rr-excess {a.get('top10_rr_excess_pct')} -> {results[h]['verdict']}")

    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "tickers": len(frames),
            "samples": len(sample_dates), "first": str(sample_dates[0].date()), "last": str(sample_dates[-1].date()),
            "backend_version": live.BACKEND_VERSION, "step_sessions": STEP, "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
            "timing_sec": {"download": dl_sec, "factors": fac_sec, "total": round(time.time() - t0, 1)},
            "workers": workers,
            "limits": ["technical part of the score only (no historical fundamentals/analysts)",
                       "survivorship bias: today's S&P 500 list", "no sector diversification in the Top10",
                       "sampled every 5th session"]}
    (OUTPUT_DIR / "top10_horizon_summary.json").write_text(
        json.dumps({"meta": meta, "results": {str(h): v for h, v in results.items()}}, ensure_ascii=False, indent=1))

    def line(s):
        if not s.get("samples"):
            return "אין מספיק נתונים"
        s = {k: ("—" if v is None else v) for k, v in s.items()}
        if "—" in (s["top10_rr_excess_pct"], s["top10_score_excess_pct"]):
            return f"IC {s['mean_ic']:+.3f} (t={s['t_ic_independent']}, {s['independent']} חלונות בלתי תלויים) · אין מספיק ימים עם Top10"
        return (f"IC {s['mean_ic']:+.3f} (t={s['t_ic_independent']}, {s['independent']} חלונות בלתי תלויים, "
                f"{s['positive_share_pct']}% חיוביים) · Top10 (סיכוי/סיכון) מעל היקום {s['top10_rr_excess_pct']:+.2f}% "
                f"(t={s['t_top10_rr_independent']}) · Top10 (ציון) {s['top10_score_excess_pct']:+.2f}% · עשירונים {s['decile_spread_pct']:+.2f}%")
    L = ["# 📐 בדק היסטורי - באיזה אופק ציון ה-Top10 מנבא?", "",
         f"{meta['tickers']} מניות S&P 500 · {meta['samples']} תאריכי בדיקה ({meta['first']} עד {meta['last']}, כל 5 ימי מסחר) · "
         f"backend {live.BACKEND_VERSION} · זמן ריצה {meta['timing_sec']['total']} ש'", "",
         f"**כלל ההחלטה (נקבע מראש):** IC מעל {PASS_IC} עם t מעל {PASS_T} בחלונות בלתי תלויים, ותשואה עודפת של ה-Top10 "
         f"מעל {ROUND_TRIP_COST_PCT}% (עמלה מלאה הלוך-חזור) לתקופה - בכל התקופה, וחיובי גם בתקופת הבקרה (מ-{HOLDOUT_START[:4]}).", ""]
    for h in HORIZONS:
        r = results[h]
        L += [f"## {h} {'יום' if h == 1 else 'ימי מסחר'} - {'✅ עובר' if r['verdict'] == 'PASS' else '❌ לא עובר'}",
              f"- **הכל:** {line(r['all'])}", f"- **{r['discovery']['label']}:** {line(r['discovery'])}",
              f"- **{r['holdout']['label']} (בקרה):** {line(r['holdout'])}",
              f"- **שוק שורי:** {line(r['bull'])}", f"- **שוק דובי:** {line(r['bear'])}",
              "- **לפי שנה (IC):** " + " · ".join(f"{y}: {v['mean_ic']:+.3f}" for y, v in r["by_year"].items() if v.get("samples")), ""]
    L += ["## מגבלות", "- נבדק רק החלק הטכני של הציון: לנתוני אנליסטים ופונדמנטלים אין היסטוריה ב-Yahoo.",
          "- הטיית שורדים: רשימת ה-S&P 500 של היום, בלי מניות שנמחקו - התוצאות נוטות להיות אופטימיות מדי.",
          "- ה-Top10 בבדיקה בלי פיזור סקטוריאלי, כי אין סקטורים היסטוריים.",
          "- זו בדיקה של אות, לא סימולציית תיק: התשואה העודפת היא ממוצע פשוט לתקופה, והעמלה מנוכה כאילו כל התיק מתחלף בכל תקופה."]
    (OUTPUT_DIR / "top10_horizon_report.md").write_text("\n".join(L))
    log("Done")


if __name__ == "__main__":
    main()
