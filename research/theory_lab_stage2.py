"""
🧪 מעבדת תיאוריות - שלב 2 (מחקר ממוקד) - אופליין בלבד.
מופעל רק דרך .github/workflows/theory_lab_stage2.yml (workflow_dispatch).

Stage 1 (theory_lab.py, 2.10.2026) left three candidates that looked good
in only ONE of its two periods, plus the 200-day exposure rule that passed.
Stage 2 asks, for those three only, the question stage 1 couldn't settle:
is the edge CONSISTENT, or did it come from one lucky stretch?

Changes vs stage 1 (all decided before running):
  1. Universe ~500: today's S&P 500 constituents (free list on GitHub),
     falling back to stage 1's 88 names if the list can't be fetched.
     With 88 names a "top decile" was 8 stocks - too noisy to trust.
  2. Five separate 5-year periods instead of two halves.
  3. Each candidate also tested WITH the 200-day exposure rule (no new
     trades / cash while the S&P 500 is below its 200-day average).
  4. Benchmark fix: a trade enters at the open of day t+1, so its
     equal-weight benchmark now also starts at that open (stage 1 started
     it at the close of day t, which charged every trade one overnight move
     it could never have captured - a small bias against every theory).

PASS (stage 2): excess > 0 in at least 4 of 5 periods, AND overall t >= 3,
AND 2017+ t >= 2, AND (trade theories) placebo >= 95%.
Survivorship is WORSE with today's S&P 500 list (it is literally the list
of winners) - that is why every result is judged against the equal-weight
average of the SAME list, never against absolute returns.
"""
import io
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory_lab as S1  # noqa: E402  (stage-1 helpers: tstat, series_stats, COST_PER_SIDE ...)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_theory_lab_stage2"
LOOKBACK_YEARS = 26
PERIODS = [("2001-01-01", "2006-01-01"), ("2006-01-01", "2011-01-01"), ("2011-01-01", "2016-01-01"),
           ("2016-01-01", "2021-01-01"), ("2021-01-01", "2030-01-01")]
HOLDOUT_START = "2017-01-01"
COST = S1.COST_PER_SIDE
PLACEBO_RUNS = 200
CONSTITUENTS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
CHUNK = 100


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def get_universe():
    try:
        r = requests.get(CONSTITUENTS_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        tickers = [str(t).strip().replace(".", "-") for t in df["Symbol"].tolist()]
        if len(tickers) > 400:
            return sorted(set(tickers)), "S&P 500 (current list)"
    except Exception as e:
        log(f"constituents fetch failed ({e}) - falling back to stage-1 universe")
    return list(dict.fromkeys(S1.BASE_UNIVERSE)), "stage-1 universe (fallback)"


def download_chunked(tickers):
    frames = {}
    for k in range(0, len(tickers), CHUNK):
        part = tickers[k:k + CHUNK]
        log(f"  downloading {k + 1}-{k + len(part)} / {len(tickers)}")
        frames.update(S1.download(part))
    return frames


# ---------------------------------------------------------------------------
# harness (stage-1 engine with the corrected open-to-close benchmark)
# ---------------------------------------------------------------------------
class Lab2(S1.Lab):
    def __init__(self, frames, universe):
        super().__init__(frames, universe)
        # equal-weight intraday return of the entry day (open -> close)
        self.ew_intra = (self.C / self.O - 1).mean(axis=1).fillna(0).values

    def bench(self, i_signal, x):
        e = i_signal + 1
        ewc = self.ew_cum.values
        return (1 + self.ew_intra[e]) * (ewc[x] / ewc[e]) - 1

    def trades(self, signal, hold, gate=None, cost=None):
        """Same rules as stage 1 (enter open t+1, exit close of t+hold, no
        overlapping trades per ticker). gate: bool Series by date - no NEW
        trades on days it is False (the exposure rule)."""
        C, O, idx = self.C.values, self.O.values, self.idx
        sig = signal.reindex(index=idx, columns=self.u).fillna(False)
        if gate is not None:
            sig = sig & gate.reindex(idx).fillna(True).values[:, None]
        sig = sig.values
        n = len(idx)
        out = []
        for j, t in enumerate(self.u):
            nxt = -1
            for i in np.flatnonzero(sig[:, j]):
                if i <= nxt or i + 2 >= n:
                    continue
                e = i + 1
                x = min(e + hold - 1, n - 1)
                if not (np.isfinite(O[e, j]) and O[e, j] > 0 and np.isfinite(C[x, j])):
                    continue
                net = (C[x, j] / O[e, j]) * (1 - (COST if cost is None else cost)) ** 2 - 1
                out.append((idx[i], t, net, net - self.bench(i, x), x - e + 1))
                nxt = x
        return pd.DataFrame(out, columns=["date", "ticker", "ret", "excess", "days"]).set_index("date")

    def placebo(self, df):
        if len(df) < S1.MIN_TRADES:
            return None
        rng = np.random.default_rng(11)
        C, O = self.C.values, self.O.values
        n_days, n_t = C.shape
        holds = df["days"].values.astype(int)
        real = df["excess"].mean()
        means = []
        for _ in range(PLACEBO_RUNS):
            h = np.repeat(holds, 3)
            i = rng.integers(200, n_days - holds.max() - 2, size=len(h))
            j = rng.integers(0, n_t, size=len(h))
            e, x = i + 1, i + h
            o, c = O[e, j], C[x, j]
            ok = np.isfinite(o) & np.isfinite(c) & (o > 0)
            if ok.sum() == 0:
                continue
            net = (c[ok] / o[ok]) * (1 - COST) ** 2 - 1
            b = (1 + self.ew_intra[e[ok]]) * (self.ew_cum.values[x[ok]] / self.ew_cum.values[e[ok]]) - 1
            means.append((net - b)[:len(holds)].mean())
        return round(float(np.mean(np.array(means) < real) * 100), 1) if means else None


def period_table(excess_series):
    rows = []
    for a, b in PERIODS:
        part = excess_series[(excess_series.index >= a) & (excess_series.index < b)]
        t = S1.tstat(part.values)
        rows.append({"period": f"{a[:4]}-{str(int(b[:4]) - 1) if b < '2030' else 'היום'}", "n": int(len(part)),
                     "avg_excess_pct": round(float(part.mean() * 100), 3) if len(part) else None,
                     "t": round(t, 2) if t is not None else None})
    return rows


def judge(periods, t_all, t_hold, placebo=None, need_placebo=True):
    pos = sum(1 for p in periods if (p["avg_excess_pct"] or 0) > 0 and p["n"] > 0)
    ok = pos >= 4 and (t_all or 0) >= 3 and (t_hold or 0) >= 2 and ((not need_placebo) or (placebo or 0) >= 95)
    if ok:
        return "✅ עובר - עקבי", pos
    if pos >= 4:
        return "🟡 עקבי בכיוון, לא מספיק מובהק", pos
    if (t_hold or 0) >= 2:
        return "❌ עובד רק לאחרונה", pos
    return "❌ נכשל", pos


def trade_result(lab, name, df, note):
    if len(df) == 0:
        return {"theory": name, "type": "trades", "n": 0, "verdict": "אין מספיק נתונים", "note": note}
    t_all = S1.tstat(df["excess"].values)
    hold = df[df.index >= HOLDOUT_START]
    t_hold = S1.tstat(hold["excess"].values)
    periods = period_table(df["excess"])
    pl = lab.placebo(df)
    wins = df[df["ret"] > 0]
    v, pos = judge(periods, t_all, t_hold, pl)
    return {"theory": name, "type": "trades", "note": note, "n": int(len(df)),
            "win_rate_pct": round(len(wins) / len(df) * 100, 1),
            "avg_trade_pct": round(float(df["ret"].mean() * 100), 3),
            "avg_excess_pct": round(float(df["excess"].mean() * 100), 3),
            "t_all": round(t_all, 2) if t_all is not None else None,
            "t_2017": round(t_hold, 2) if t_hold is not None else None,
            "placebo_pct": pl, "periods": periods, "positive_periods": pos, "verdict": v}


def monthly_momentum(name, C, score, spy_gate=None, top_frac=0.1, min_names=10, note=""):
    me = C.resample("ME").last()
    sc = score.reindex(me.index, method="ffill")
    fwd = me.pct_change().shift(-1)
    ew = fwd.mean(axis=1)
    gate = spy_gate.reindex(me.index, method="ffill") if spy_gate is not None else None
    rows, prev = [], set()
    for d in me.index[:-1]:
        if gate is not None and not bool(gate.loc[d]):
            cost = 2 * COST * (1.0 if prev else 0.0) / 2  # sell everything once
            rows.append((d, -cost, ew.loc[d]))
            prev = set()
            continue
        s = sc.loc[d].dropna()
        s = s[me.loc[d, s.index].notna()]
        if len(s) < min_names * 3:
            continue
        k = max(min_names, int(len(s) * top_frac))
        pick = set(s.sort_values(ascending=False).index[:k])
        turnover = len(pick ^ prev) / (2 * k) if prev else 1.0
        rows.append((d, fwd.loc[d, list(pick)].mean() - 2 * COST * turnover, ew.loc[d]))
        prev = pick
    df = pd.DataFrame(rows, columns=["date", "r", "ew"]).set_index("date").dropna()
    ex = df["r"] - df["ew"]
    t_all, t_hold = S1.tstat(ex.values), S1.tstat(ex[ex.index >= HOLDOUT_START].values)
    periods = period_table(ex)
    v, pos = judge(periods, t_all, t_hold, need_placebo=False)
    st, bst = S1.series_stats(df["r"], 12) or {}, S1.series_stats(df["ew"], 12) or {}
    per_cagr = []
    for a, b in PERIODS:
        p = df[(df.index >= a) & (df.index < b)]
        s1, s2 = S1.series_stats(p["r"], 12) or {}, S1.series_stats(p["ew"], 12) or {}
        per_cagr.append({"cagr_pct": s1.get("cagr_pct"), "bench_cagr_pct": s2.get("cagr_pct"), "max_dd_pct": s1.get("max_dd_pct")})
    for p, c in zip(periods, per_cagr):
        p.update(c)
    return {"theory": name, "type": "monthly", "note": note, "n": int(len(df)),
            "cagr_pct": st.get("cagr_pct"), "bench_cagr_pct": bst.get("cagr_pct"),
            "max_dd_pct": st.get("max_dd_pct"), "bench_max_dd_pct": bst.get("max_dd_pct"),
            "sharpe": st.get("sharpe"), "bench_sharpe": bst.get("sharpe"),
            "months_beating_pct": round(float((ex > 0).mean() * 100), 1),
            "t_all": round(t_all, 2) if t_all is not None else None,
            "t_2017": round(t_hold, 2) if t_hold is not None else None,
            "periods": periods, "positive_periods": pos, "verdict": v}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe, universe_src = get_universe()
    log(f"Universe: {len(universe)} tickers ({universe_src})")
    frames = download_chunked(universe + ["SPY"])
    log(f"Loaded {len(frames)} tickers")
    lab = Lab2(frames, [t for t in universe if t in frames])
    C, V, O, H, L = lab.C, lab.V, lab.O, lab.H, lab.L
    spy = frames["SPY"]["Close"]
    spy_gate = (spy > spy.rolling(200).mean()) | spy.rolling(200).mean().isna()
    uptrend = C > lab.sma200
    vol20 = V.rolling(20).mean()
    results = []

    def add(r):
        log(f"  {r['theory']}: {r['verdict']}")
        results.append(r)

    # candidate 1: 12-1 momentum
    mom = C.shift(21) / C.shift(252) - 1
    add(monthly_momentum("מומנטום 12-1 · העשירון העליון", C, mom))
    add(monthly_momentum("מומנטום 12-1 · העשירון העליון + כלל חשיפה", C, mom, spy_gate=spy_gate,
                         note="במזומן בכל חודש שבו S&P מתחת לממוצע 200"))

    # candidate 2: sharp 5-day drop in an uptrend
    drop = (C / C.shift(5) - 1 <= -0.12) & uptrend
    add(trade_result(lab, "ירידה של 12%+ ב-5 ימים במגמה עולה · החזקה 10", lab.trades(drop, 10), ""))
    add(trade_result(lab, "ירידה של 12%+ ב-5 ימים במגמה עולה · החזקה 10 + כלל חשיפה", lab.trades(drop, 10, spy_gate), ""))

    # candidate 3: gap & go
    gap = O / C.shift(1) - 1
    strong_close = (C - L) / (H - L).replace(0, np.nan) > 0.5
    gg = (gap >= 0.05) & (V >= 3 * vol20) & strong_close
    add(trade_result(lab, "קפיצת פתיחה 5%+ בווליום פי 3 · החזקה 60", lab.trades(gg, 60), ""))
    add(trade_result(lab, "קפיצת פתיחה 5%+ בווליום פי 3 · החזקה 60 + כלל חשיפה", lab.trades(gg, 60, spy_gate), ""))

    # reference: the passing exposure rule on SPY per period
    ref = []
    r = spy.pct_change()
    pos = spy_gate.astype(float).shift(1).fillna(1)
    strat = pos * r - pos.diff().abs().fillna(0) * COST
    for a, b in PERIODS:
        m = (strat.index >= a) & (strat.index < b)
        s1, s2 = S1.series_stats(strat[m], 252) or {}, S1.series_stats(r[m], 252) or {}
        ref.append({"period": f"{a[:4]}-{str(int(b[:4]) - 1) if b < '2030' else 'היום'}",
                    "rule_cagr_pct": s1.get("cagr_pct"), "rule_max_dd_pct": s1.get("max_dd_pct"),
                    "spy_cagr_pct": s2.get("cagr_pct"), "spy_max_dd_pct": s2.get("max_dd_pct")})

    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "universe_source": universe_src,
            "universe_requested": len(universe), "usable_tickers": len(lab.u),
            "first_date": str(lab.idx[0].date()), "last_date": str(lab.idx[-1].date()),
            "periods": PERIODS, "cost_per_side_pct": COST * 100,
            "pass_rule": "excess>0 in >=4/5 periods AND t_all>=3 AND t_2017>=2 AND placebo>=95 (trade theories)",
            "known_limits": ["survivorship bias is strong (today's S&P 500 = the winners) - judge only vs the equal-weight of the same list",
                             "Yahoo data quality", "cash earns 0"]}
    (OUTPUT_DIR / "stage2_summary.json").write_text(
        json.dumps({"meta": meta, "results": results, "exposure_rule_by_period": ref}, ensure_ascii=False, indent=1, default=str))
    lines = ["# 🧪 מעבדת תיאוריות - שלב 2", "",
             f"{meta['usable_tickers']} מניות ({universe_src}) · {meta['first_date']} עד {meta['last_date']} · עלות {COST*100:.2f}% לכל צד", "",
             "| תיאוריה | פסיקה | תקופות חיוביות | t כולל | t 2017+ | פלצבו |", "|---|---|---|---|---|---|"]
    for x in results:
        lines.append(f"| {x['theory']} | {x['verdict']} | {x.get('positive_periods')}/5 | {x.get('t_all')} | {x.get('t_2017')} | {x.get('placebo_pct')} |")
    (OUTPUT_DIR / "stage2_report.md").write_text("\n".join(lines))
    log("Done")


if __name__ == "__main__":
    main()
