"""
🔬 מחקר: קניית ירידות בענקיות המדד (השערה 6).
אופליין בלבד. מופעל רק דרך .github/workflows/research.yml (הפעלה ידנית).
לא נוגע ב-stock_alerts.py, ב-index.html או בשום קובץ של האפליקציה החיה.

אושר ע"י תומר ב-27.11.2026. הכלל, הבקרות ורף ההצלחה נקבעו מראש
(research/2026-11-27-megacap-dip.md) ואסור לשנות אותם אחרי שרואים תוצאות.

Rule:
  megacaps   on each month's last trading day: the 20 S&P 500 members (point-in-time) with the
             highest average daily dollar volume over the past 252 days (size proxy) - used
             for the whole next month
  flag       a megacap closes >= 20% below its 52-week high (split-adjusted close)
  entry      next close, 10% of equity per position, max 10 positions, deepest drop first
  exit       +30% from entry (dividend-adjusted) or 504 trading days (2 years); signal at
             the close, sale at the next close
  idle cash  in SPY (not cash), so the comparison with SPY is fair
Placebo: same megacaps, same exits/slots, entries on random days into a random megacap,
  calibrated to the same number of entries; 20 seeds, median CAGR.
Costs: identical to the other studies (0.1% min $8 + slippage, SPY leg included).

Usage:  python research/megacap_dip_research.py            # test 2001-01-01..2021-11-19
        python research/megacap_dip_research.py --holdout  # 2021-11-20..today - ONE run, after the test results are written
"""

import argparse
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_lowvol_research as B  # noqa: E402

OUTPUT_DIR = HERE / "output_megacap_dip"
TEST_START = pd.Timestamp("2001-01-01")
N_MEGA = 20
DV_DAYS = 252
DD_THRESHOLD = 0.20
GRID = [0.15, 0.25, 0.30]          # reported only, never used for the verdict
SLOTS = 10
TAKE_PROFIT = 0.30
MAX_HOLD = 504
PLACEBO_SEEDS = 20
BAR_EXCESS = 0.02
BAR_EXTRA_DD = 0.05


# ---------------------------------------------------------------- pure helpers (tested)
def megacap_sets(close, volume, member, n=N_MEGA, window=DV_DAYS):
    """{month_end_day: [tickers]} - top-n members by trailing mean dollar volume."""
    dv = (close * volume).rolling(window, min_periods=int(window * 0.8)).mean()
    days = close.index
    s = pd.Series(days, index=days)
    out = {}
    for me in s.groupby([days.year, days.month]).max():
        row = dv.loc[me][member.loc[me].reindex(dv.columns).fillna(False).astype(bool)].dropna()
        if len(row):
            out[me] = list(row.nlargest(n).index)
    return out


def megacap_lookup(sets):
    keys = sorted(sets)

    def on(d):
        j = bisect.bisect_left(keys, d) - 1            # strictly before d
        return sets[keys[j]] if j >= 0 else []
    return on


def drawdown_from_high(close, window=252):
    hi = close.rolling(window, min_periods=int(window * 0.8)).max()
    return close / hi - 1


def deepest_first(dd_row):
    def choose(d, cands, held, k):
        return sorted(cands, key=lambda t: dd_row(d, t))
    return choose


def simulate_dips(days, adj, rets, spy_ret, mega_on, dd, start, end, choose, thr=DD_THRESHOLD,
                  slots=SLOTS, tp=TAKE_PROFIT, max_hold=MAX_HOLD, initial=B.INITIAL):
    """Returns (equity Series, trades list). Idle money sits in SPY."""
    win = days[(days >= start) & (days <= end)]
    spy_val, pos, eq, trades = initial, {}, [], []
    pend_in, pend_out = [], []
    for k, d in enumerate(win):
        if k > 0:
            r = spy_ret.get(d, 0.0)
            spy_val *= 1 + (r if np.isfinite(r) else 0.0)
            for t, p in pos.items():
                r = rets.at[d, t] if t in rets.columns else np.nan
                if np.isfinite(r):
                    p["value"] *= 1 + r
        for t in pend_out:
            if t in pos:
                p = pos.pop(t)
                v = p["value"]
                c = B.trade_cost(v, t) + B.trade_cost(v, B.ETF)
                spy_val += v - c
                trades.append({"ticker": t, "entry": p["entry"], "exit": d.strftime("%Y-%m-%d"),
                               "days": k - p["k"], "return": v / p["cost_basis"] - 1, "reason": p["why"]})
        for t in pend_in:
            if len(pos) >= slots or t in pos or not np.isfinite(adj.at[d, t]):
                continue
            total = spy_val + sum(p["value"] for p in pos.values())
            amt = min(total / slots, spy_val)
            if amt < B.MIN_TRADE:
                continue
            c = B.trade_cost(amt, t) + B.trade_cost(amt, B.ETF)
            spy_val -= amt + c
            pos[t] = {"value": amt, "cost_basis": amt, "entry_px": adj.at[d, t], "k": k,
                      "entry": d.strftime("%Y-%m-%d"), "why": ""}
        eq.append(spy_val + sum(p["value"] for p in pos.values()))
        pend_out = []
        for t, p in pos.items():
            px = adj.at[d, t]
            if np.isfinite(px) and px / p["entry_px"] - 1 >= tp:
                p["why"] = "tp"
                pend_out.append(t)
            elif k - p["k"] >= max_hold:
                p["why"] = "time"
                pend_out.append(t)
        cands = [t for t in mega_on(d) if t not in pos and t in dd.columns
                 and np.isfinite(dd.at[d, t]) and dd.at[d, t] <= -thr]
        pend_in = choose(d, cands, pos, k)
    for t, p in pos.items():
        trades.append({"ticker": t, "entry": p["entry"], "exit": "", "days": len(win) - 1 - p["k"],
                       "return": p["value"] / p["cost_basis"] - 1, "reason": "open"})
    return pd.Series(eq, index=win), trades


def placebo_chooser(mega_on, p_day, seed):
    rng = np.random.default_rng(seed)

    def choose(d, cands, held, k):
        if rng.random() >= p_day:
            return []
        pool = [t for t in mega_on(d) if t not in held]
        return [pool[rng.integers(len(pool))]] if pool else []
    return choose


def verdict(strat, spy, placebo_cagr):
    return (strat["cagr"] >= spy["cagr"] + BAR_EXCESS and strat["cagr"] > placebo_cagr
            and abs(strat["max_dd"]) <= abs(spy["max_dd"]) + BAR_EXTRA_DD)


def trade_stats(trades):
    if not trades:
        return {"n": 0}
    df = pd.DataFrame(trades)
    closed = df[df.reason != "open"]
    return {"n": int(len(df)), "tp": int((df.reason == "tp").sum()), "time": int((df.reason == "time").sum()),
            "open_at_end": int((df.reason == "open").sum()),
            "avg_return": float(closed["return"].mean()) if len(closed) else None,
            "avg_days": float(closed["days"].mean()) if len(closed) else None}


# ---------------------------------------------------------------- data (network)
def download(tickers, start=B.PRICE_START):
    import yfinance as yf
    A, C, V = {}, {}, {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), 100):
        part = tickers[i:i + 100]
        B.log(f"prices {i + 1}-{i + len(part)} / {len(tickers)}")
        try:
            data = yf.download(" ".join(part), start=start, auto_adjust=False, group_by="ticker",
                               threads=True, progress=False)
        except Exception as e:
            B.log(f"  chunk failed: {e}")
            continue
        for t in part:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                if df["Adj Close"].dropna().shape[0] < 260:
                    continue
                idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
                A[t] = pd.Series(df["Adj Close"].values, index=idx)
                C[t] = pd.Series(df["Close"].values, index=idx)
                V[t] = pd.Series(df["Volume"].values, index=idx)
            except Exception:
                continue
        time.sleep(2)
    return pd.DataFrame(A), pd.DataFrame(C), pd.DataFrame(V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="2021-11-20..today - ONE run, after the test results are written")
    args = ap.parse_args()
    tag = "holdout" if args.holdout else "test"
    start = pd.Timestamp(B.HOLDOUT_START) if args.holdout else TEST_START
    end = pd.Timestamp("2100-01-01") if args.holdout else pd.Timestamp(B.HOLDOUT_START) - pd.Timedelta(days=1)
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [(d, s) for d, s in B.load_members() if d >= pd.Timestamp("1998-01-01")]
    ever = sorted(set().union(*[s for _, s in rows]))
    _, spy, irx = B.load_market()
    adj, close, vol = download(ever)
    days = spy.index.intersection(adj.index)
    adj, close, vol, spy = adj.reindex(days), close.reindex(days), vol.reindex(days), spy.reindex(days)
    end = min(end, days[-1])
    member = B.membership_matrix(rows, days, list(adj.columns))
    rets, n_bad = B.clean_returns(adj)
    spy_ret = spy.pct_change(fill_method=None)
    cash_ret = (irx.reindex(days).ffill() / 100 / 252).shift(1).fillna(0.0)
    sets = megacap_sets(close, vol, member)
    mega_on = megacap_lookup(sets)
    dd = drawdown_from_high(close)
    deepest = deepest_first(lambda d, t: dd.at[d, t])

    def run(choose, thr=DD_THRESHOLD):
        return simulate_dips(days, adj, rets, spy_ret, mega_on, dd, start, end, choose, thr=thr)

    B.log("strategy")
    eq, trades = run(deepest)
    res = {"strategy": {**B.metrics(eq, cash_ret), **trade_stats(trades)}}
    spy_eq, _, _ = B.simulate(B.orders_from_targets({d: {B.ETF: 1.0} for d in days}, start, end),
                              pd.DataFrame({B.ETF: spy_ret, B.CASH: cash_ret}), start, end)
    res["spy"] = B.metrics(spy_eq, cash_ret)

    n_days = len(eq)
    p_day = min(1.0, len(trades) / max(n_days, 1))
    plc = []
    for s in range(PLACEBO_SEEDS):
        B.log(f"placebo {s + 1}/{PLACEBO_SEEDS}")
        e, tr = run(placebo_chooser(mega_on, p_day, s))
        plc.append({**B.metrics(e, cash_ret), "n": len(tr)})
    plc_cagr = float(np.median([p["cagr"] for p in plc]))
    res["placebo"] = {"median_cagr": plc_cagr, "median_max_dd": float(np.median([p["max_dd"] for p in plc])),
                      "beat_by_strategy": int(sum(res["strategy"]["cagr"] > p["cagr"] for p in plc)),
                      "median_entries": float(np.median([p["n"] for p in plc]))}
    grid = {}
    for thr in GRID:
        B.log(f"grid {thr}")
        e, tr = run(deepest, thr)
        grid[str(thr)] = {**B.metrics(e, cash_ret), **trade_stats(tr)}
    passed = bool(verdict(res["strategy"], res["spy"], plc_cagr))
    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame({"strategy": eq, "spy": spy_eq}).to_csv(out_dir / "equity_curves.csv")
    mega_hist = {str(k.date()): v for k, v in sets.items() if start <= k <= end}
    (out_dir / "megacaps_by_month.json").write_text(json.dumps(mega_hist, indent=1))
    summary = {"run": tag, "period": [str(start.date()), str(end.date())], "results": res, "grid": grid,
               "pass": passed, "bad_returns_masked": n_bad, "tickers_ever": len(ever),
               "tickers_with_prices": int(adj.shape[1])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "SUMMARY.md").write_text(render_md(summary, mega_hist))
    B.log(f"done: H6 {'PASS' if passed else 'FAIL'}")


def render_md(s, mega_hist):
    R = s["results"]
    st, sp, pl = R["strategy"], R["spy"], R["placebo"]
    L = ["> ⚠️ ענקיות שקרסו ונמחקו (Lehman, WaMu, Enron...) חסרות ב-yfinance. התוצאה כנראה טובה מדי.", "",
         f"# קניית ירידות בענקיות — {'תקופת בקרה' if s['run'] == 'holdout' else 'תקופת בדיקה'} "
         f"{s['period'][0]} עד {s['period'][1]} — {'✅ עומדת ברף' if s['pass'] else '❌ לא עומדת ברף'}", "",
         "| | תשואה שנתית | ירידה מקסימלית | Sharpe |", "|---|---|---|---|",
         f"| **השערה 6 (ירידה 20%)** | {B.pct(st['cagr'])} | {B.pct(st['max_dd'])} | {B.num(st['sharpe'])} |",
         f"| פלצבו (חציון של {PLACEBO_SEEDS}) | {B.pct(pl['median_cagr'])} | {B.pct(pl['median_max_dd'])} | |",
         f"| SPY | {B.pct(sp['cagr'])} | {B.pct(sp['max_dd'])} | {B.num(sp['sharpe'])} |", "",
         f"השערה 6 ניצחה {pl['beat_by_strategy']}/{PLACEBO_SEEDS} ריצות פלצבו.", "",
         f"**עסקאות:** {st.get('n', 0)} (יעד +30%: {st.get('tp', 0)}, עצירת זמן: {st.get('time', 0)}, "
         f"פתוחות בסוף: {st.get('open_at_end', 0)}). תשואה ממוצעת לעסקה סגורה: {B.pct(st.get('avg_return'))}, "
         f"ימי מסחר בממוצע: {B.num(st.get('avg_days'), 0)}.", "",
         "## חוסן (לעיון, לא חלק מהרף)", "", "| סף ירידה | תשואה שנתית | ירידה מקסימלית | עסקאות |", "|---|---|---|---|"]
    for thr, g in s["grid"].items():
        L.append(f"| {float(thr) * 100:.0f}% | {B.pct(g['cagr'])} | {B.pct(g['max_dd'])} | {g.get('n', 0)} |")
    firsts = list(mega_hist.items())[:1] + list(mega_hist.items())[-1:]
    L += ["", "## דוגמה: הענקיות בתחילת התקופה ובסופה", ""]
    for d, v in firsts:
        L.append(f"- {d}: {', '.join(v)}")
    L += ["", f"מניות שהיו במדד מאז 1998: {s['tickers_ever']}, עם מחירים: {s['tickers_with_prices']}. "
          f"תשואות חריגות שסוננו: {s['bad_returns_masked']}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
