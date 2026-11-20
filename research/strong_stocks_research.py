"""
🔬 מחקר: תיק "מניות חזקות" עם מעט עסקאות (השערה 3).
אופליין בלבד. מופעל רק דרך .github/workflows/research.yml (הפעלה ידנית).
לא נוגע ב-stock_alerts.py, ב-index.html או בשום קובץ של האפליקציה החיה.

אושר ע"י תומר ב-20.11.2026. הכללים ורף ההצלחה נקבעו מראש
(research/2026-11-20-strong-stocks.md) ואסור לשנות אותם אחרי שרואים תוצאות.

Rule (checked on the last trading day of each month, executed at the next close):
  universe   S&P 500 members on that day (fja05680/sp500, point-in-time)
  rank       12-1 momentum: return from 252 to 21 trading days ago
  filter     from 2010: profitable at the last SEC filing known then
             (TTM operating income > 0, or TTM net income > 0 with positive equity).
             Before 2010 SEC XBRL data does not exist -> momentum only (flagged).
  portfolio  15 names; a held name is sold only when it drops out of the top 45
             eligible names; free slots are filled from the top of the ranking.
             New names get 1/15 of the (exposure-scaled) equity; held names are not
             re-equalised (fewer trades - that is the point of the rule).
  variants   plain (always invested) and exposure (50% cash while ^GSPC is below
             its 200-day average - every position scaled by 0.5 / 2 on a switch).
Costs, cash, start capital, test/holdout split: identical to breadth_lowvol_research.

Usage:  python research/strong_stocks_research.py            # test period only
        python research/strong_stocks_research.py --holdout  # holdout only - ONE run, at the end
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_lowvol_research as B  # noqa: E402  (shared data, costs, simulator, metrics)

OUTPUT_DIR = HERE / "output_strong_stocks"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# ---------------------------------------------------------------- pre-registered config (do NOT tune)
N_HOLD = 15
BUFFER = 45
MOM_LONG = 252
MOM_SKIP = 21
FUND_START = pd.Timestamp("2010-01-01")
FUND_MAX_AGE_DAYS = 460               # latest TTM period older than this = unknown = not eligible
SUB_WINDOW_START = "2010-01-01"       # the bar must also hold on 2010-2021 alone (fundamentals era)

BAR_EXCESS_CAGR = 0.02
BAR_EXTRA_DD = 0.05
BAR_TRADES_PER_MONTH = 2.0


# ---------------------------------------------------------------- pure helpers (tested)
def month_ends(days):
    s = pd.Series(days, index=days)
    return set(s.groupby([days.year, days.month]).max())


def momentum_12_1(adj):
    return adj.shift(MOM_SKIP) / adj.shift(MOM_LONG) - 1


def select_with_buffer(ranked_eligible, held, n=N_HOLD, buffer=BUFFER):
    """ranked_eligible: tickers best-first (at least `buffer` long when available).
    Keep held names still in the top `buffer`; fill the rest from the top."""
    top = set(ranked_eligible[:buffer])
    keep = sorted(t for t in held if t in top)
    new = [t for t in ranked_eligible if t not in keep][:max(0, n - len(keep))]
    return keep, new


def ranked_eligible_at(d, mom_row, member_row, is_profitable, need=BUFFER):
    """Momentum ranking of members with a value, filtered lazily by profitability
    (only as deep as needed to find `need` eligible names)."""
    m = mom_row[member_row.reindex(mom_row.index).fillna(False).astype(bool)].dropna()
    out = []
    for t in m.sort_values(ascending=False).index:
        if d >= FUND_START and not is_profitable(t, d):
            continue
        out.append(t)
        if len(out) >= need:
            break
    return out


def make_select_spec(keep, new, exposure, rescale):
    def spec(pos, equity):
        desired = {t: pos.get(t, 0.0) * rescale for t in keep}
        if new:
            slot = exposure * equity / N_HOLD
            avail = max(exposure * equity - sum(desired.values()), 0.0)
            per = min(slot, avail / len(new))
            for t in new:
                desired[t] = per
        return desired
    spec.kind = "select"
    return spec


def make_scale_spec(factor):
    def spec(pos, equity):
        return {a: v * factor for a, v in pos.items() if a != B.CASH}
    spec.kind = "scale"
    return spec


def build_orders(days, start, end, rank_fn, above=None):
    """Returns (orders, selection_log). above=None -> plain variant (always 100%)."""
    me = month_ends(days)
    held, orders, log_rows = set(), {}, []
    prev_e = None
    for d in days:
        if d < start or d > end:
            continue
        a = True if above is None else above.get(d)
        a = True if a is None or (isinstance(a, float) and np.isnan(a)) else bool(a)
        e = 1.0 if a else B.REDUCED
        factor = (e / prev_e) if (prev_e is not None and e != prev_e and held) else 1.0
        if d in me:
            keep, new = select_with_buffer(rank_fn(d), held)
            sold = sorted(held - set(keep))
            held = set(keep) | set(new)
            orders[d] = make_select_spec(keep, new, e, factor)
            log_rows.append({"date": d.strftime("%Y-%m-%d"), "sold": " ".join(sold),
                             "bought": " ".join(new), "held": len(held), "exposure": e})
        elif factor != 1.0:
            orders[d] = make_scale_spec(factor)
        prev_e = e
    return orders, log_rows


def trades_per_month(trades, start, end):
    months = max(1.0, (end - start).days / 30.4375)
    return len(trades) / months


def verdict(strat, spy):
    return (strat["cagr"] >= spy["cagr"] + BAR_EXCESS_CAGR
            and abs(strat["max_dd"]) <= abs(spy["max_dd"]) + BAR_EXTRA_DD
            and strat["trades_per_month"] <= BAR_TRADES_PER_MONTH)


# ---------------------------------------------------------------- SEC fundamentals (network)
def load_profitability(tickers):
    """Returns is_profitable(ticker, date) using only filings made before `date`."""
    import requests
    import winners_research as W
    session = requests.Session()
    session.headers.update({"User-Agent": W.SEC_UA, "Accept-Encoding": "gzip, deflate"})
    mode, diag = W.probe_sec(session)
    if mode is None:
        raise RuntimeError(f"SEC refused: {diag}")
    r = session.get(SEC_TICKERS_URL, timeout=30)
    r.raise_for_status()
    cik_of = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}
    todo = [t for t in sorted(tickers) if t in cik_of]
    B.log(f"SEC: {len(todo)}/{len(tickers)} tickers have a CIK today (rest = delisted/renamed -> never eligible from 2010)")
    parsed = {}
    if mode == "bulk":
        raw = W.load_bulk(session, [cik_of[t] for t in todo])
        for t in todo:
            if cik_of[t] in raw:
                parsed[t] = W.parse_company(raw[cik_of[t]])
        del raw
    else:
        misses = 0
        for k, t in enumerate(todo):
            f = W.fetch_facts(cik_of[t], session)
            if f:
                parsed[t] = W.parse_company(f)
                misses = 0
            else:
                misses += 1
                if misses >= 15:
                    raise RuntimeError(f"SEC stopped answering: {W.FETCH_ERRORS}")
            time.sleep(0.12)
            if (k + 1) % 100 == 0:
                B.log(f"SEC facts {k + 1}/{len(todo)} ({len(parsed)} ok)")
    cache = {}

    def is_profitable(t, d):
        key = (t, d.year, d.month)
        if key in cache:
            return cache[key]
        p, ok = parsed.get(t), False
        if p:
            asof = d.strftime("%Y-%m-%d")
            ebit, e_end = W.ttm(p.get("ebit", []), asof)
            ni, n_end = W.ttm(p.get("net_income", []), asof)
            eq = W.latest_instant(p.get("equity", []), asof)
            fresh = lambda end: end is not None and W._days(end, asof) <= FUND_MAX_AGE_DAYS  # noqa: E731
            if ebit is not None and fresh(e_end) and ebit > 0:
                ok = True
            elif ni is not None and fresh(n_end) and ni > 0 and eq and eq[0] > 0:
                ok = True
        cache[key] = ok
        return ok
    return is_profitable, {"mode": mode, "with_cik": len(todo), "with_facts": len(parsed), "diag": diag}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="run the 5-year holdout ONCE, at the very end")
    args = ap.parse_args()
    start, end = B.window(args.holdout)
    tag = "holdout" if args.holdout else "test"
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    D = B.build_dataset()
    days, rets, cash_ret, member = D["days"], D["rets"], D["cash_ret"], D["member"]
    end = min(end, days[-1])
    above = B.regime_above(D["gspc"])
    mom = momentum_12_1(D["adj"])

    in_era = member.loc[member.index >= FUND_START].any(axis=0)
    fund_tickers = sorted(in_era[in_era].index)
    try:
        is_profitable, fund_info = load_profitability(fund_tickers)
    except Exception as e:
        (out_dir / "SUMMARY.md").write_text(f"# השערה 3 - לא רצה\n\nאין גישה לנתוני SEC: `{e}`\n")
        B.log(f"SEC failed: {e}")
        sys.exit(1)

    rank_cache = {}

    def rank_fn(d):
        if d not in rank_cache:
            rank_cache[d] = ranked_eligible_at(d, mom.loc[d], member.loc[d], is_profitable)
        return rank_cache[d]

    results, curves, logs = {}, {}, {}
    variants = {"h3_plain": None, "h3_exposure": above}
    for name, ab in variants.items():
        B.log(f"simulate {name}")
        orders, sel_log = build_orders(days, start, end, rank_fn, ab)
        eq, trades, n_reb = B.simulate(orders, rets, start, end)
        curves[name], logs[name] = eq, sel_log
        pd.DataFrame(trades).to_csv(out_dir / f"trades_{name}.csv", index=False)
        pd.DataFrame(sel_log).to_csv(out_dir / f"selection_{name}.csv", index=False)
        results[name] = summarize(eq, trades, cash_ret, start, end, n_reb)
        if not args.holdout:
            sub = pd.Timestamp(SUB_WINDOW_START)
            results[name]["sub_2010"] = summarize(eq[eq.index >= sub], [t for t in trades if t["date"] >= SUB_WINDOW_START],
                                                  cash_ret, sub, end, None)
    for name, tg in {"spy_buy_hold": {d: {B.ETF: 1.0} for d in days},
                     "baseline": B.baseline_targets(above, days)}.items():
        eq, trades, n_reb = B.simulate(B.orders_from_targets(tg, start, end), rets, start, end)
        curves[name] = eq
        results[name] = summarize(eq, trades, cash_ret, start, end, n_reb)
        if not args.holdout:
            sub = pd.Timestamp(SUB_WINDOW_START)
            results[name]["sub_2010"] = summarize(eq[eq.index >= sub], [t for t in trades if t["date"] >= SUB_WINDOW_START],
                                                  cash_ret, sub, end, None)

    spy = results["spy_buy_hold"]
    passes = {}
    for v in variants:
        ok = verdict(results[v], spy)
        if not args.holdout:
            ok = ok and verdict(results[v]["sub_2010"], spy["sub_2010"])
        passes[v] = bool(ok)
    pd.DataFrame(curves).to_csv(out_dir / "equity_curves.csv")
    summary = {"period": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")], "run": tag,
               "results": results, "pass": passes, "sec": fund_info,
               "tickers_ever": len(D["ever"]), "tickers_with_prices": int(D["adj"].shape[1]),
               "bad_daily_returns_masked": D["n_bad"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "SUMMARY.md").write_text(render_md(summary))
    B.log("done: " + " | ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in passes.items()))


def summarize(eq, trades, cash_ret, start, end, n_reb):
    m = B.metrics(eq, cash_ret)
    kinds = pd.Series([t.get("kind") for t in trades], dtype=object)
    m.update({"trades": len(trades), "trades_per_month": trades_per_month(trades, start, end),
              "scale_trades": int((kinds == "scale").sum()), "rebalances": n_reb,
              "costs": round(sum(t["cost"] for t in trades), 2),
              "decades": B.decade_table(eq, cash_ret)})
    return m


def render_md(s):
    R = s["results"]
    names = {"h3_plain": "השערה 3: מניות חזקות", "h3_exposure": "השערה 3 + כלל חשיפה",
             "spy_buy_hold": "S&P 500 (SPY)", "baseline": "כלל החשיפה הנוכחי"}
    L = ["> ⚠️ הרכב המדד ההיסטורי ממאגר קהילתי (fja05680), לא ממקור רשמי. מניות שנמחקו חסרות ב-yfinance, "
         "ולכן התוצאה כנראה טובה מדי. לפני 2010 אין סינון רווחיות (אין נתוני SEC).", "",
         f"# תוצאות ({'תקופת בקרה' if s['run'] == 'holdout' else 'תקופת בדיקה'}) {s['period'][0]} עד {s['period'][1]}", "",
         "| אסטרטגיה | תשואה שנתית | ירידה מקסימלית | Sharpe | עסקאות | עסקאות לחודש | עלויות $ |",
         "|---|---|---|---|---|---|---|"]
    for k, n in names.items():
        r = R[k]
        L.append(f"| {n} | {B.pct(r['cagr'])} | {B.pct(r['max_dd'])} | {B.num(r['sharpe'])} | {r['trades']} | "
                 f"{r['trades_per_month']:.2f} | {r['costs']:,.0f} |")
    if "sub_2010" in R["h3_plain"]:
        L += ["", "## רק 2010 ואילך (עם סינון רווחיות)", "",
              "| אסטרטגיה | תשואה שנתית | ירידה מקסימלית | Sharpe | עסקאות לחודש |", "|---|---|---|---|---|"]
        for k, n in names.items():
            r = R[k]["sub_2010"]
            L.append(f"| {n} | {B.pct(r['cagr'])} | {B.pct(r['max_dd'])} | {B.num(r['sharpe'])} | {r['trades_per_month']:.2f} |")
    L += ["", "## לפי עשור", "", "| אסטרטגיה | עשור | תשואה שנתית | ירידה מקסימלית |", "|---|---|---|---|"]
    for k, n in names.items():
        for dec, m in R[k]["decades"].items():
            L.append(f"| {n} | {dec} | {B.pct(m['cagr'])} | {B.pct(m['max_dd'])} |")
    L += ["", "## רף", ""]
    for k in ("h3_plain", "h3_exposure"):
        L.append(f"- {names[k]}: {'✅ עומדת ברף' if s['pass'][k] else '❌ לא עומדת ברף'}")
    sec = s["sec"]
    L += ["", f"SEC: {sec['with_facts']}/{sec['with_cik']} מניות עם נתונים (מצב {sec['mode']}). "
          f"מניות שהיו במדד: {s['tickers_ever']}, עם מחירים: {s['tickers_with_prices']}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
