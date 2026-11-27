"""
🔬 מחקר ליבה: התאוששות רוחב שוק (השערה 1) ותנודתיות נמוכה במקום חצי מזומן (השערה 2).
אופליין בלבד. מופעל רק דרך .github/workflows/research.yml (הפעלה ידנית).
לא נוגע ב-stock_alerts.py, ב-index.html או בשום קובץ של האפליקציה החיה.

אושר ע"י תומר ב-20.11.2026. ההשערות, הכללים ורף ההצלחה נקבעו מראש
(research/2026-11-20-breadth-and-lowvol.md) ואסור לשנות אותם אחרי שרואים תוצאות.

Baseline (the proven exposure rule, as in stock_alerts.compute_exposure_signal):
  S&P 500 (^GSPC) close above its 200-day average -> 100% SPY, else 50% SPY + 50% cash.

H1 breadth thrust: breadth = share of S&P 500 members (point-in-time) whose
  close is above their own 50-day average. Event = breadth goes from below 20%
  to above 50% within 10 trading days. After an event: 100% SPY for 126
  trading days (pre-registered, decided 20.11.2026), then back to baseline.
H2 low-vol: while ^GSPC is below its 200-day average, hold the 30 members with
  the lowest 60-day volatility, equal weight, re-selected monthly; above -> 100% SPY.

Rules shared by every strategy (identical, so the comparison is fair):
  - signal at close of day t, executed at close of the next trading day
  - commission 0.1% per trade, minimum $8; slippage 0.05% (SPY) / 0.10% (stocks)
  - $100,000 start; trades under $200 are skipped (except closing a position)
  - cash earns the 13-week T-bill yield (^IRX) of the previous day
  - test period 20.11.2000-19.11.2021; holdout 20.11.2021-today, run ONCE with --holdout

Usage:  python research/breadth_lowvol_research.py            # test period only
        python research/breadth_lowvol_research.py --holdout  # holdout only - ONE run, at the end
"""

import argparse
import bisect
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output_breadth_lowvol"

# ---------------------------------------------------------------- pre-registered config (do NOT tune)
MEMBERS_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
PRICE_START = "1998-06-01"           # warm-up for the 200-day average and 12-month momentum
TEST_START = "2000-11-20"
HOLDOUT_START = "2021-11-20"
INDEX, ETF, TBILL = "^GSPC", "SPY", "^IRX"
CASH = "CASH"

INITIAL = 100_000.0
FEE_PCT = 0.001
FEE_MIN = 8.0
SLIP_ETF = 0.0005
SLIP_STOCK = 0.0010
MIN_TRADE = 200.0

REGIME_SMA = 200
REDUCED = 0.5

BREADTH_SMA = 50
THRUST_LOW = 0.20
THRUST_HIGH = 0.50
THRUST_WINDOW = 10
THRUST_HOLD = 126                    # trading days of 100% exposure after an event
MIN_BREADTH_NAMES = 200              # fewer members with data than this -> breadth undefined

LOWVOL_N = 30
LOWVOL_DAYS = 60

BAD_RET_UP, BAD_RET_DOWN = 1.0, -0.9  # one-day moves beyond these = data error, treated as 0

DECADES = [("2000-2009", "2000-01-01", "2009-12-31"),
           ("2010-2019", "2010-01-01", "2019-12-31"),
           ("2020-2026", "2020-01-01", "2026-12-31")]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- data (network; not used by the tests)
def load_members():
    """[(date, set(tickers))] from fja05680/sp500 - a community dataset, NOT an official S&P list."""
    df = pd.read_csv(MEMBERS_URL)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return [(d, {t.strip().replace(".", "-") for t in str(s).split(",") if t.strip()})
            for d, s in zip(df["date"], df["tickers"])]


def download_adj(tickers, start=PRICE_START):
    """Dividend- and split-adjusted closes, 100 tickers per request (same as dip_research)."""
    import yfinance as yf
    out = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), 100):
        part = tickers[i:i + 100]
        log(f"prices {i + 1}-{i + len(part)} / {len(tickers)}")
        try:
            data = yf.download(" ".join(part), start=start, auto_adjust=False, group_by="ticker",
                               threads=True, progress=False)
        except Exception as e:
            log(f"  chunk failed: {e}")
            continue
        for t in part:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                s = df["Adj Close"].dropna()
                if len(s) < 70:
                    continue
                if getattr(s.index, "tz", None) is not None:
                    s.index = s.index.tz_localize(None)
                out[t] = s
            except Exception:
                continue
        time.sleep(2)
    return pd.DataFrame(out)


def load_market(start=PRICE_START):
    import yfinance as yf
    data = yf.download(f"{INDEX} {ETF} {TBILL}", start=start, auto_adjust=False, group_by="ticker",
                       progress=False)
    def col(t, c):
        s = data[t][c].dropna()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s
    return col(INDEX, "Close"), col(ETF, "Adj Close"), col(TBILL, "Close")


# ---------------------------------------------------------------- pure helpers (tested)
def membership_matrix(rows, days, tickers):
    """Bool DataFrame days x tickers: was the ticker an S&P 500 member on that day."""
    dates = [d for d, _ in rows]
    cols = {t: i for i, t in enumerate(tickers)}
    m = np.zeros((len(days), len(tickers)), dtype=bool)
    for k, d in enumerate(days):
        j = bisect.bisect_right(dates, d) - 1
        if j < 0:
            continue
        for t in rows[j][1]:
            i = cols.get(t)
            if i is not None:
                m[k, i] = True
    return pd.DataFrame(m, index=days, columns=list(tickers))


def clean_returns(prices):
    """Daily returns; impossible one-day moves (data errors) -> NaN. Returns (rets, n_bad)."""
    r = prices.pct_change(fill_method=None)
    bad = (r > BAD_RET_UP) | (r < BAD_RET_DOWN)
    return r.mask(bad), int(bad.values.sum())


def breadth_series(adj, member, sma_days=BREADTH_SMA, min_names=MIN_BREADTH_NAMES):
    sma = adj.rolling(sma_days, min_periods=sma_days).mean()
    valid = sma.notna() & adj.notna() & member.reindex_like(adj).fillna(False).astype(bool)
    above = (adj > sma) & valid
    n = valid.sum(axis=1)
    b = above.sum(axis=1) / n.replace(0, np.nan)
    b[n < min_names] = np.nan
    return b, n


def thrust_events(breadth, low=THRUST_LOW, high=THRUST_HIGH, window=THRUST_WINDOW):
    """Days where breadth closes above `high` and was below `low` at some close in the
    previous `window` trading days. After an event, a new one needs breadth to go
    below `low` again first (one thrust = one event)."""
    vals = breadth.values
    events, armed_after = [], -1
    for i in range(len(vals)):
        v = vals[i]
        if not np.isfinite(v) or v <= high:
            continue
        lo = vals[max(0, i - window):i]
        lo_idx = [i - len(lo) + k for k, x in enumerate(lo) if np.isfinite(x) and x < low]
        if lo_idx and max(lo_idx) > armed_after:
            events.append(breadth.index[i])
            armed_after = i
    return events


def regime_above(index_close, sma_days=REGIME_SMA):
    sma = index_close.rolling(sma_days, min_periods=sma_days).mean()
    return (index_close > sma).where(sma.notna())


def baseline_targets(above, days):
    out = {}
    for d in days:
        a = above.get(d)
        if a is None or (isinstance(a, float) and math.isnan(a)):
            continue
        out[d] = {ETF: 1.0} if bool(a) else {ETF: REDUCED}
    return out


def thrust_targets(above, days, events, hold=THRUST_HOLD):
    base = baseline_targets(above, days)
    until = -1
    ev = set(events)
    out = {}
    for i, d in enumerate(days):
        if d in ev:
            until = i + hold - 1
        if d not in base:
            continue
        out[d] = {ETF: 1.0} if i <= until else base[d]
    return out


def lowvol_pick(vol_row, member_row, n=LOWVOL_N):
    v = vol_row[member_row.reindex(vol_row.index).fillna(False).astype(bool)].dropna()
    v = v[v > 0]
    return sorted(v.nsmallest(n).index)


def lowvol_targets(above, days, vol, member, n=LOWVOL_N):
    """Above -> 100% SPY. Below -> n lowest-vol members, equal weight; basket chosen on
    the day the regime turns down and re-chosen on the first trading day of each month."""
    out, basket, prev_above, prev_month = {}, None, None, None
    for d in days:
        a = above.get(d)
        if a is None or (isinstance(a, float) and math.isnan(a)):
            continue
        a = bool(a)
        if a:
            out[d] = {ETF: 1.0}
            basket = None
        else:
            new_month = prev_month is not None and d.month != prev_month
            if basket is None or new_month or prev_above:
                picks = lowvol_pick(vol.loc[d], member.loc[d], n)
                if picks:
                    # "_m" makes every monthly re-pick a new order, so weights are re-equalised
                    # even when the names did not change (the simulator ignores "_" keys)
                    basket = {t: 1.0 / len(picks) for t in picks}
                    basket["_m"] = d.strftime("%Y-%m")
            out[d] = dict(basket) if basket else {ETF: REDUCED}
        prev_above, prev_month = a, d.month
    return out


def orders_from_targets(targets, start, end):
    """Order on the first day of the window and whenever the target changes."""
    orders, prev = {}, None
    for d in sorted(targets):
        if d < start or d > end:
            continue
        t = targets[d]
        if prev is None or t != prev:
            orders[d] = t
        prev = t
    return orders


ETF_LIKE = {ETF, "SSO"}                # index funds: ETF slippage (SSO used by leverage_exposure_research)


def slippage_of(asset):
    return SLIP_ETF if asset in ETF_LIKE else SLIP_STOCK


def trade_cost(notional, asset):
    n = abs(notional)
    return max(FEE_PCT * n, FEE_MIN) + slippage_of(asset) * n


def rebalance(pos, desired):
    """pos: {asset: value} incl. CASH (mutated). desired: {asset: value} for non-cash
    assets; anything held and not in desired is sold. Returns a list of trades."""
    trades = []
    assets = (set(pos) | set(desired)) - {CASH}
    for a in sorted(assets, key=lambda x: desired.get(x, 0.0) - pos.get(x, 0.0)):  # sells first
        cur, new = pos.get(a, 0.0), max(desired.get(a, 0.0), 0.0)
        delta = new - cur
        closing = new == 0.0 and cur > 0
        if not closing and abs(delta) < MIN_TRADE:
            continue
        cost = trade_cost(delta, a)
        pos[CASH] = pos.get(CASH, 0.0) - delta - cost
        if closing:
            pos.pop(a, None)
        else:
            pos[a] = new
        trades.append({"asset": a, "side": "buy" if delta > 0 else "sell",
                       "notional": round(abs(delta), 2), "cost": round(cost, 2)})
    return trades


def simulate(orders, rets, start, end, initial=INITIAL):
    """orders: {signal_day: weights dict | callable(pos, equity) -> desired values}.
    Each order executes at the close of the NEXT trading day. rets: DataFrame of daily
    returns incl. a CASH column. Returns (equity Series, trades list, n_rebalances)."""
    days = rets.index[(rets.index >= start) & (rets.index <= end)]
    cols = {c: i for i, c in enumerate(rets.columns)}
    R = rets.values
    pos = {CASH: initial}
    eq, trades, n_reb, pending = [], [], 0, None
    for k, d in enumerate(days):
        if k > 0:
            row = R[rets.index.get_loc(d)]
            for a in list(pos):
                r = row[cols[a]] if a in cols else 0.0
                if np.isfinite(r):
                    pos[a] *= 1.0 + r
        if pending is not None:
            equity = sum(pos.values())
            spec = pending
            desired = spec(pos, equity) if callable(spec) else {
                a: w * equity for a, w in spec.items() if a != CASH and not a.startswith("_")}
            tr = rebalance(pos, desired)
            for t in tr:
                t["date"] = d.strftime("%Y-%m-%d")
                t["kind"] = getattr(spec, "kind", "target")
            trades += tr
            n_reb += 1
            pending = None
        if d in orders:
            pending = orders[d]
        eq.append(sum(pos.values()))
    return pd.Series(eq, index=days), trades, n_reb


def metrics(eq, cash_ret):
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return {"cagr": None, "max_dd": None, "sharpe": None}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else None
    dd = float((eq / eq.cummax() - 1).min())
    r = eq.pct_change().dropna()
    ex = r - cash_ret.reindex(r.index).fillna(0.0)
    sharpe = float(ex.mean() / ex.std() * math.sqrt(252)) if ex.std() > 0 else None
    return {"cagr": float(cagr) if cagr is not None else None, "max_dd": dd, "sharpe": sharpe}


def decade_table(eq, cash_ret):
    out = {}
    for name, a, b in DECADES:
        s = eq[(eq.index >= a) & (eq.index <= b)]
        if len(s) > 20:
            out[name] = metrics(s, cash_ret)
    return out


def verdict_h1(h1, base):
    ok = h1["cagr"] > base["cagr"] and abs(h1["max_dd"]) <= abs(base["max_dd"])
    return ok


def verdict_h2(h2, base):
    ok = h2["cagr"] > base["cagr"] and abs(h2["max_dd"]) <= abs(base["max_dd"]) + 0.02
    return ok


def pct(x, d=1):
    return "—" if x is None else f"{x * 100:.{d}f}%"


def num(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


# ---------------------------------------------------------------- shared data bundle (also used by strong_stocks_research)
def build_dataset():
    log("members (fja05680/sp500 - community dataset)")
    rows = load_members()
    rows = [(d, s) for d, s in rows if d >= pd.Timestamp("1998-01-01")]
    ever = sorted(set().union(*[s for _, s in rows]))
    log(f"{len(ever)} tickers were members at some point since 1998")
    gspc, spy, irx = load_market()
    adj = download_adj(ever)
    days = gspc.index.intersection(spy.index)
    days = days[days >= pd.Timestamp(PRICE_START)]
    adj = adj.reindex(days)
    member = membership_matrix(rows, days, list(adj.columns))
    stock_rets, n_bad = clean_returns(adj)
    cash_ret = (irx.reindex(days).ffill() / 100 / 252).shift(1).fillna(0.0)
    rets = stock_rets.copy()
    rets[ETF] = spy.reindex(days).pct_change(fill_method=None)
    rets[CASH] = cash_ret
    # coverage: how many members on each day have prices (the rest = mostly delisted)
    full_member = membership_matrix(rows, days, ever)
    n_members = full_member.sum(axis=1)
    n_with_data = (member & adj.notna()).sum(axis=1)
    return {"rows": rows, "ever": ever, "gspc": gspc.reindex(days), "adj": adj, "member": member,
            "rets": rets, "cash_ret": cash_ret, "n_bad": n_bad, "days": days,
            "coverage": (n_with_data / n_members.replace(0, np.nan))}


def window(holdout):
    if holdout:
        return pd.Timestamp(HOLDOUT_START), pd.Timestamp("2100-01-01")
    return pd.Timestamp(TEST_START), pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)


def forward_returns(spy_px, events, horizons=(63, 126)):
    rows = []
    idx = spy_px.index
    for e in events:
        i = idx.get_loc(e) + 1                         # entry at next close, like the strategy
        row = {"event": e.strftime("%Y-%m-%d")}
        for h in horizons:
            row[f"fwd_{h}"] = float(spy_px.iloc[i + h] / spy_px.iloc[i] - 1) if i + h < len(idx) else None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="run the 5-year holdout ONCE, at the very end")
    args = ap.parse_args()
    start, end = window(args.holdout)
    tag = "holdout" if args.holdout else "test"
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    D = build_dataset()
    days, rets, cash_ret = D["days"], D["rets"], D["cash_ret"]
    end = min(end, days[-1])
    above = regime_above(D["gspc"])

    log("breadth")
    breadth, n_breadth = breadth_series(D["adj"], D["member"])
    events_all = thrust_events(breadth)
    events = [e for e in events_all if start <= e <= end]

    log("low-vol")
    vol = D["rets"].drop(columns=[ETF, CASH]).rolling(LOWVOL_DAYS, min_periods=LOWVOL_DAYS).std()
    member = D["member"]

    targets = {
        "baseline": baseline_targets(above, days),
        "h1_breadth_thrust": thrust_targets(above, days, events_all),
        "h2_lowvol": lowvol_targets(above, days, vol, member),
        "spy_buy_hold": {d: {ETF: 1.0} for d in days},
    }
    results, curves = {}, {}
    for name, tg in targets.items():
        log(f"simulate {name}")
        eq, trades, n_reb = simulate(orders_from_targets(tg, start, end), rets, start, end)
        curves[name] = eq
        m = metrics(eq, cash_ret)
        m.update({"trades": len(trades), "rebalances": n_reb,
                  "costs": round(sum(t["cost"] for t in trades), 2),
                  "final_value": round(float(eq.iloc[-1]), 2),
                  "decades": decade_table(eq, cash_ret) if not args.holdout else {}})
        results[name] = m
        pd.DataFrame(trades).to_csv(out_dir / f"trades_{name}.csv", index=False)

    base = results["baseline"]
    v1 = verdict_h1(results["h1_breadth_thrust"], base)
    v2 = verdict_h2(results["h2_lowvol"], base)

    spy_px = (1 + rets[ETF].fillna(0)).cumprod()
    fwd = forward_returns(spy_px, events)
    in_win = spy_px[(spy_px.index >= start) & (spy_px.index <= end)]
    uncond = {h: float((in_win.shift(-h) / in_win - 1).dropna().mean()) for h in (63, 126)}
    pd.DataFrame(fwd).to_csv(out_dir / "h1_events.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / "equity_curves.csv")
    breadth.to_frame("breadth").assign(n=n_breadth).to_csv(out_dir / "breadth.csv")

    cov = D["coverage"][(D["coverage"].index >= start) & (D["coverage"].index <= end)]
    cov_by_year = cov.groupby(cov.index.year).mean().round(3).to_dict()
    summary = {"period": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")], "run": tag,
               "results": results, "h1_pass": bool(v1), "h2_pass": bool(v2),
               "h1_events": fwd, "h1_unconditional_fwd": uncond,
               "member_price_coverage_by_year": {str(k): v for k, v in cov_by_year.items()},
               "bad_daily_returns_masked": D["n_bad"],
               "tickers_ever": len(D["ever"]), "tickers_with_prices": int(D["adj"].shape[1])}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "SUMMARY.md").write_text(render_md(summary))
    log(f"done: H1 {'PASS' if v1 else 'FAIL'} | H2 {'PASS' if v2 else 'FAIL'}")


def render_md(s):
    R = s["results"]
    names = {"baseline": "כלל החשיפה הנוכחי (בסיס)", "h1_breadth_thrust": "השערה 1: התאוששות רוחב",
             "h2_lowvol": "השערה 2: תנודתיות נמוכה", "spy_buy_hold": "SPY קנה והחזק (לעיון)"}
    L = [f"# תוצאות ({'תקופת בקרה' if s['run'] == 'holdout' else 'תקופת בדיקה'}) {s['period'][0]} עד {s['period'][1]}", "",
         "| אסטרטגיה | תשואה שנתית | ירידה מקסימלית | Sharpe | עסקאות | איזונים | עלויות $ |",
         "|---|---|---|---|---|---|---|"]
    for k, n in names.items():
        r = R[k]
        L.append(f"| {n} | {pct(r['cagr'])} | {pct(r['max_dd'])} | {num(r['sharpe'])} | {r['trades']} | "
                 f"{r['rebalances']} | {r['costs']:,.0f} |")
    L += ["", f"**השערה 1:** {'✅ עומדת ברף' if s['h1_pass'] else '❌ לא עומדת ברף'} "
          f"(אותות בתקופה: {len(s['h1_events'])})",
          f"**השערה 2:** {'✅ עומדת ברף' if s['h2_pass'] else '❌ לא עומדת ברף'}", ""]
    if any(R[k].get("decades") for k in names):
        L += ["## לפי עשור", "", "| אסטרטגיה | עשור | תשואה שנתית | ירידה מקסימלית | Sharpe |", "|---|---|---|---|---|"]
        for k, n in names.items():
            for dec, m in R[k]["decades"].items():
                L.append(f"| {n} | {dec} | {pct(m['cagr'])} | {pct(m['max_dd'])} | {num(m['sharpe'])} |")
        L.append("")
    L += ["## השערה 1: תשואת SPY אחרי כל אות (לעיון, לא חלק מהרף)", "",
          "| אות | 3 חודשים | 6 חודשים |", "|---|---|---|"]
    for e in s["h1_events"]:
        L.append(f"| {e['event']} | {pct(e.get('fwd_63'))} | {pct(e.get('fwd_126'))} |")
    u = s["h1_unconditional_fwd"]
    L += [f"| ממוצע כל הימים | {pct(u.get(63))} | {pct(u.get(126))} |", "",
          "## כיסוי נתונים", "",
          f"מניות שהיו במדד מאז 1998: {s['tickers_ever']}, עם מחירים ב-yfinance: {s['tickers_with_prices']}. "
          f"תשואות יומיות חריגות שסוננו כשגיאת נתונים: {s['bad_daily_returns_masked']}.", "",
          "| שנה | % מחברי המדד עם מחירים |", "|---|---|"]
    for y, v in s["member_price_coverage_by_year"].items():
        L.append(f"| {y} | {pct(v, 0)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
