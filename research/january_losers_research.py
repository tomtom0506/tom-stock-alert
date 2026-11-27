"""
🔬 מחקר: ההפסדניות של השנה בינואר (השערה 5).
אופליין בלבד. מופעל רק דרך .github/workflows/research.yml (הפעלה ידנית).
לא נוגע ב-stock_alerts.py, ב-index.html או בשום קובץ של האפליקציה החיה.

אושר ע"י תומר ב-21.11.2026. הכלל והרף נקבעו מראש
(research/2026-11-21-january-losers.md) ואסור לשנות אותם אחרי שרואים תוצאות.

Rule, every year E:
  rank day   3rd-to-last trading day of December E
  universe   S&P 500 members on the rank day (fja05680, point-in-time), raw close > $5
  losers     the 20 worst year-to-date returns (last close of E-1 -> rank day, dividend-adjusted)
  entry      close of the 2nd-to-last trading day of December E (sell SPY, buy the 20, equal weight)
  exit       close of the last trading day of January E+1 (sell the 20, buy SPY back)
Costs on a $100,000 position: stocks 0.1% (min $8) + 0.10% slippage per side;
the SPY switch out and back 0.1% (min $8) + 0.05% per side.
Excess = net basket return - SPY return over the same window.
Bar (test Jan 2001-2021 AND holdout Jan 2022-2026): mean excess >= +1%, positive in
>= 60% of years, worst year >= -10%.

Usage:  python research/january_losers_research.py            # test years only
        python research/january_losers_research.py --holdout  # holdout years - ONE run, after the test results are written
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_lowvol_research as B  # noqa: E402  (members, market data, costs)

OUTPUT_DIR = HERE / "output_january_losers"
N_LOSERS = 20
MIN_PRICE = 5.0
CAPITAL = 100_000.0
TEST_ENTRY_YEARS = list(range(2000, 2021))      # Januaries 2001-2021
HOLDOUT_ENTRY_YEARS = list(range(2021, 2026))   # Januaries 2022-2026
BAR_MEAN = 0.01
BAR_HIT = 0.60
BAR_WORST = -0.10


# ---------------------------------------------------------------- pure helpers (tested)
def key_days(days, year):
    """(prev_year_last, rank_day, entry_day, exit_day) for December `year` -> January `year`+1."""
    prev = days[days.year == year - 1]
    dec = days[(days.year == year) & (days.month == 12)]
    jan = days[(days.year == year + 1) & (days.month == 1)]
    if len(prev) == 0 or len(dec) < 3 or len(jan) == 0:
        return None
    return prev[-1], dec[-3], dec[-2], jan[-1]


def pick_losers(adj, close, member_row, base_day, rank_day, entry_day, n=N_LOSERS, min_price=MIN_PRICE):
    mem = member_row[member_row].index.intersection(adj.columns)
    a0, a1, ae = adj.loc[base_day, mem], adj.loc[rank_day, mem], adj.loc[entry_day, mem]
    px = close.loc[rank_day, mem]
    ok = a0.notna() & a1.notna() & ae.notna() & (px > min_price) & (a0 > 0)
    ytd = (a1[ok] / a0[ok] - 1).sort_values()
    return list(ytd.index[:n]), ytd.iloc[:n]


def basket_return(adj, tickers, entry_day, exit_day):
    """Equal-weight buy-and-hold; a stock with no price on the exit day is valued at its
    last price before it (data ended - usually a merger/delisting)."""
    rets = []
    window = adj.loc[entry_day:exit_day, tickers]
    for t in tickers:
        s = window[t].dropna()
        rets.append(float(s.iloc[-1] / s.iloc[0] - 1) if len(s) else 0.0)
    return float(np.mean(rets)) if rets else 0.0, rets


def round_trip_cost(n_stocks, capital=CAPITAL):
    """Fraction of capital lost to costs for one year's switch SPY -> basket -> SPY."""
    pos = capital / n_stocks
    stock_side = max(B.FEE_PCT * pos, B.FEE_MIN) / pos + B.SLIP_STOCK
    spy_side = max(B.FEE_PCT * capital, B.FEE_MIN) / capital + B.SLIP_ETF
    return 2 * stock_side + 2 * spy_side


def summarize(rows):
    ex = np.array([r["excess"] for r in rows])
    n = len(ex)
    sd = ex.std(ddof=1) if n > 1 else float("nan")
    return {"years": n, "mean_excess": float(ex.mean()) if n else None,
            "median_excess": float(np.median(ex)) if n else None,
            "hit_rate": float((ex > 0).mean()) if n else None,
            "worst_excess": float(ex.min()) if n else None, "best_excess": float(ex.max()) if n else None,
            "t_stat": float(ex.mean() / (sd / math.sqrt(n))) if n > 1 and sd > 0 else None}


def verdict(s):
    return (s["years"] > 0 and s["mean_excess"] >= BAR_MEAN and s["hit_rate"] >= BAR_HIT
            and s["worst_excess"] >= BAR_WORST)


# ---------------------------------------------------------------- data (network)
def download_adj_and_close(tickers, start=B.PRICE_START):
    import yfinance as yf
    adj, close = {}, {}
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
                if df["Adj Close"].dropna().shape[0] < 70:
                    continue
                idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
                adj[t] = pd.Series(df["Adj Close"].values, index=idx)
                close[t] = pd.Series(df["Close"].values, index=idx)
            except Exception:
                continue
        time.sleep(2)
    return pd.DataFrame(adj), pd.DataFrame(close)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="holdout Januaries 2022-2026 - ONE run, after the test results are written")
    args = ap.parse_args()
    tag = "holdout" if args.holdout else "test"
    years = HOLDOUT_ENTRY_YEARS if args.holdout else TEST_ENTRY_YEARS
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_m = [(d, s) for d, s in B.load_members() if d >= pd.Timestamp("1998-01-01")]
    ever = sorted(set().union(*[s for _, s in rows_m]))
    _, spy, _ = B.load_market()
    adj, close = download_adj_and_close(ever)
    days = spy.index.intersection(adj.index)
    adj, close, spy = adj.reindex(days), close.reindex(days), spy.reindex(days)
    member = B.membership_matrix(rows_m, days, list(adj.columns))

    cost = round_trip_cost(N_LOSERS)
    out, picks_log = [], []
    for y in years:
        k = key_days(days, y)
        if k is None:
            B.log(f"{y}: missing days - skipped")
            continue
        base, rank, entry, exit_ = k
        tick, ytd = pick_losers(adj, close, member.loc[rank], base, rank, entry)
        if len(tick) < N_LOSERS:
            B.log(f"{y}: only {len(tick)} eligible")
        gross, each = basket_return(adj, tick, entry, exit_)
        spy_ret = float(spy.loc[exit_] / spy.loc[entry] - 1)
        net = gross - cost
        out.append({"january": y + 1, "entry": str(entry.date()), "exit": str(exit_.date()),
                    "n": len(tick), "basket_gross": gross, "basket_net": net, "spy": spy_ret,
                    "excess": net - spy_ret, "avg_ytd_of_losers": float(ytd.mean())})
        picks_log += [{"january": y + 1, "ticker": t, "ytd_at_rank": float(ytd[t]), "return": r}
                      for t, r in zip(tick, each)]
    s = summarize(out)
    passed = bool(verdict(s))
    pd.DataFrame(out).to_csv(out_dir / "years.csv", index=False)
    pd.DataFrame(picks_log).to_csv(out_dir / "picks.csv", index=False)
    summary = {"run": tag, "round_trip_cost": cost, "summary": s, "pass": passed, "years": out,
               "tickers_ever": len(ever), "tickers_with_prices": int(adj.shape[1])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "SUMMARY.md").write_text(render_md(summary))
    B.log(f"done: H5 {'PASS' if passed else 'FAIL'}")


def render_md(m):
    s = m["summary"]
    L = ["> ⚠️ מניות שנמחקו מהמסחר חסרות ב-yfinance, ודווקא בין המפסידות הן נפוצות. התוצאה כנראה טובה מדי.", "",
         f"# ההפסדניות בינואר — {'תקופת בקרה' if m['run'] == 'holdout' else 'תקופת בדיקה'} "
         f"— {'✅ עומדת ברף' if m['pass'] else '❌ לא עומדת ברף'}", "",
         f"עלות מעבר הלוך-חזור (מניות + SPY): {B.pct(m['round_trip_cost'], 2)}", "",
         "| מדד | ערך | רף |", "|---|---|---|",
         f"| שנים | {s['years']} | |",
         f"| עודף ממוצע מול SPY (נטו) | {B.pct(s['mean_excess'], 2)} | ≥ 1.00% |",
         f"| עודף חציוני | {B.pct(s['median_excess'], 2)} | |",
         f"| שנים חיוביות | {B.pct(s['hit_rate'], 0)} | ≥ 60% |",
         f"| השנה הגרועה | {B.pct(s['worst_excess'], 2)} | ≥ -10% |",
         f"| השנה הטובה | {B.pct(s['best_excess'], 2)} | |",
         f"| t-stat | {B.num(s['t_stat'])} | |", "",
         "| ינואר | כניסה | יציאה | מניות | סל ברוטו | סל נטו | SPY | עודף | תשואת המפסידות עד הדירוג |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in m["years"]:
        L.append(f"| {r['january']} | {r['entry']} | {r['exit']} | {r['n']} | {B.pct(r['basket_gross'])} | "
                 f"{B.pct(r['basket_net'])} | {B.pct(r['spy'])} | {B.pct(r['excess'])} | {B.pct(r['avg_ytd_of_losers'])} |")
    L += ["", f"מניות שהיו במדד מאז 1998: {m['tickers_ever']}, עם מחירים: {m['tickers_with_prices']}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
