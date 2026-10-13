"""
🏆 מחקר "מניות מנצחות" - שלוש עדשות תמחור על S&P 500 (אופליין בלבד)
מופעל רק דרך .github/workflows/winners_research.yml (workflow_dispatch).
לא כותב לשום קובץ של האפליקציה החיה - התוצאות נשמרות כ-artifact.

Agreed with Tomer 11-13.10.2026, after the Yinon Arieli value webinar:
pick 20 stocks, hold ~1 year, compare to the S&P 500. Three lenses:

  L1  cheap vs its INDUSTRY  - EV/EBIT relative to the GICS sub-industry
      median (sector when the sub-industry has < MIN_GROUP names that day)
  L2  cheap vs ITSELF        - EV/EBIT relative to its own median over the
      previous 5 years (the "Meta Q3-2022" lens)
  L3  HEALTHY business       - filter: net income > 0, free cash flow > 0,
      net debt < 4x EBIT, and EBIT NOT at a cyclical peak (trailing EBIT
      <= 1.5x its own 5-year average). The last check exists because an
      EV/EBIT of 2-3 (the webinar's "winning stock" bars) is typical of a
      commodity producer at peak earnings - exactly when it's a trap.

Variants tested (each picks the 20 best-ranked, equal weight):
  raw      lowest EV/EBIT in the market (classic value, the baseline)
  L1, L2, L1+L3, L2+L3, L1+L2, L1+L2+L3 (webinar method + our filter)
  graham   P/E x P/B < 22.5, five years of positive earnings (Graham number)
Each with two exits: HOLD 12 months, or REVERT - sell at a quarterly check
once the stock is no longer cheap vs its industry (EV/EBIT >= industry
median) and park the money in the S&P 500 for the rest of the year.

Point in time: every fundamental comes from SEC XBRL "companyfacts" and is
used only from the day after it was FILED. EV uses the split-adjusted close
(not dividend-adjusted) x reported shares x later splits; returns use the
dividend-adjusted close for stocks and SPY alike (total return both sides).

Known limits, also written into the report:
  * Survivorship: today's S&P 500 list (delisted/merged losers missing,
    later winners present) - leans optimistic, so the bar is strict.
  * Financials and Real Estate are excluded from the EV/EBIT lenses (EV/EBIT
    is not meaningful for banks/insurers/REITs); Graham covers them.
  * XBRL coverage starts ~2009-2011, so the test runs 2012 onward.

Decision rule (fixed before running), per variant:
  PASS = average 12-month excess over SPY > 0, t >= 2 on the independent
  (non-overlapping, January-start) yearly results, and positive excess in
  the holdout (2019 onward) - after 0.1% per side, full turnover yearly.
"""
import json
import math
import os
import sys
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output_winners_research"
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_UA = os.environ.get("SEC_USER_AGENT", "TomStockAlert research tomtom0506@users.noreply.github.com")
START = "2012-01-01"
HOLDOUT = "2019-01-01"
N_PICKS = 20
MIN_GROUP = 5
HOLD_DAYS = 365
FEE = 0.001
OWN_HISTORY_YEARS = 5
OWN_HISTORY_MIN_POINTS = 8
PEAK_EBIT_MULT = 1.5
MAX_NETDEBT_EBIT = 4.0
MIN_INDEPENDENT_YEARS = 8        # fewer January portfolios than this = not enough evidence
EXCLUDED_SECTORS = {"Financials", "Real Estate"}

TAGS = {
    "ebit": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "lt_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "lt_debt_cur": ["LongTermDebtCurrent", "DebtCurrent"],
    "st_debt": ["ShortTermBorrowings", "CommercialPaper"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
}
SHARE_TAGS = [("dei", "EntityCommonStockSharesOutstanding"),
              ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
              ("us-gaap", "CommonStockSharesOutstanding")]
FLOW = {"ebit", "net_income", "cfo", "capex"}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- SEC data
def fetch_facts(cik, session):
    for attempt in range(3):
        try:
            r = session.get(SEC_FACTS_URL.format(cik=int(cik)), timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 + attempt * 3)
    return None


def _facts_for(facts, taxonomy, tag):
    node = ((facts.get("facts") or {}).get(taxonomy) or {}).get(tag) or {}
    out = []
    for unit, rows in (node.get("units") or {}).items():
        if unit not in ("USD", "shares"):
            continue
        for r in rows:
            if r.get("val") is None or not r.get("filed") or not r.get("end"):
                continue
            out.append({"start": r.get("start"), "end": r["end"], "filed": r["filed"],
                        "val": float(r["val"]), "form": r.get("form", "")})
    return out


def parse_company(facts):
    """{field: list of facts} with the first tag that has data."""
    parsed = {}
    for field, tags in TAGS.items():
        for tag in tags:
            rows = _facts_for(facts, "us-gaap", tag)
            if rows:
                parsed[field] = rows
                break
    for tax, tag in SHARE_TAGS:
        rows = _facts_for(facts, tax, tag)
        if rows:
            parsed["shares"] = rows
            break
    return parsed


def _days(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def known(rows, asof):
    """Facts filed strictly before `asof` (one-day lag)."""
    return [r for r in rows if r["filed"] < asof]


def latest_instant(rows, asof):
    k = known(rows, asof)
    if not k:
        return None
    end = max(r["end"] for r in k)
    cands = [r for r in k if r["end"] == end]
    return max(cands, key=lambda r: r["filed"])["val"], end


def annual_values(rows, asof):
    """{fiscal-year end: value} for ~1-year duration facts known at asof."""
    out = {}
    for r in known(rows, asof):
        if r.get("start") and 340 <= _days(r["start"], r["end"]) <= 380:
            if r["end"] not in out or r["filed"] > out[r["end"]][1]:
                out[r["end"]] = (r["val"], r["filed"])
    return {k: v[0] for k, v in out.items()}


def ttm(rows, asof):
    """Trailing twelve months: the latest annual figure, or latest FY +
    current year-to-date - prior-year year-to-date when a newer quarter
    exists. Returns (value, period_end) or (None, None)."""
    k = known(rows, asof)
    if not k:
        return None, None
    ann = annual_values(rows, asof)
    best_ann_end = max(ann) if ann else None
    ytd = {}
    for r in k:
        if not r.get("start"):
            continue
        d = _days(r["start"], r["end"])
        if 80 <= d <= 290:
            key = (r["start"], r["end"])
            if key not in ytd or r["filed"] > ytd[key]["filed"]:
                ytd[key] = r
    if ytd:
        cur = max(ytd.values(), key=lambda r: r["end"])
        if best_ann_end is None or cur["end"] > best_ann_end:
            dur = _days(cur["start"], cur["end"])
            prev_end = (date.fromisoformat(cur["end"]) - timedelta(days=364)).isoformat()
            prev = [r for r in ytd.values()
                    if abs(_days(r["end"], prev_end)) <= 10 and abs(_days(r["start"], r["end"]) - dur) <= 10]
            fy_end = (date.fromisoformat(cur["start"]) - timedelta(days=1)).isoformat()
            fy = [v for e, v in ann.items() if abs(_days(e, fy_end)) <= 10]
            if prev and fy:
                return fy[0] + cur["val"] - prev[0]["val"], cur["end"]
    if best_ann_end:
        return ann[best_ann_end], best_ann_end
    return None, None


def split_factor_after(splits, d):
    """Product of split ratios strictly after date d (yfinance splits)."""
    f = 1.0
    for sd, ratio in splits:
        if sd > d and ratio and ratio > 0:
            f *= ratio
    return f


def fundamentals_at(p, asof, splits):
    """Point-in-time snapshot for one company, or None."""
    ebit, ebit_end = ttm(p.get("ebit", []), asof)
    ni, _ = ttm(p.get("net_income", []), asof)
    cfo, _ = ttm(p.get("cfo", []), asof)
    capex, _ = ttm(p.get("capex", []), asof)
    sh = latest_instant(p.get("shares", []), asof)
    if ebit is None or sh is None:
        return None
    if _days(ebit_end, asof) > 200 or _days(sh[1], asof) > 400:
        return None                      # stale filings - company not reporting normally
    debt = 0.0
    for f in ("lt_debt", "lt_debt_cur", "st_debt"):
        v = latest_instant(p.get(f, []), asof)
        if v and _days(v[1], asof) <= 400:
            debt += max(v[0], 0.0)
    cash_v = latest_instant(p.get("cash", []), asof)
    cash = cash_v[0] if cash_v and _days(cash_v[1], asof) <= 400 else 0.0
    eq_v = latest_instant(p.get("equity", []), asof)
    ann_ebit = annual_values(p.get("ebit", []), asof)
    ann_ni = annual_values(p.get("net_income", []), asof)
    cutoff = (date.fromisoformat(asof) - timedelta(days=365 * OWN_HISTORY_YEARS + 60)).isoformat()
    hist_ebit = [v for e, v in ann_ebit.items() if e >= cutoff]
    hist_ni = [v for e, v in sorted(ann_ni.items()) if e >= cutoff]
    return {
        "ebit": ebit, "net_income": ni, "fcf": (cfo - capex) if cfo is not None and capex is not None else None,
        "shares_adj": sh[0] * split_factor_after(splits, sh[1]), "debt": debt, "cash": cash,
        "equity": eq_v[0] if eq_v and _days(eq_v[1], asof) <= 400 else None,
        "ebit_5y_avg": (sum(hist_ebit) / len(hist_ebit)) if len(hist_ebit) >= 3 else None,
        "ni_5y_all_positive": len(hist_ni) >= 5 and all(v > 0 for v in hist_ni[-5:]),
    }


# ---------------------------------------------------------------- lenses
def healthy(f):
    if f["net_income"] is None or f["net_income"] <= 0 or f["ebit"] <= 0:
        return False
    if f["fcf"] is None or f["fcf"] <= 0:
        return False
    if (f["debt"] - f["cash"]) / f["ebit"] > MAX_NETDEBT_EBIT:
        return False
    if f["ebit_5y_avg"] is None or f["ebit_5y_avg"] <= 0 or f["ebit"] > PEAK_EBIT_MULT * f["ebit_5y_avg"]:
        return False
    return True


def score_date(snap, own_hist, meta):
    """snap: {ticker: fundamentals + price}; own_hist: {ticker: [ev_ebit...]}
    (previous quarterly values, oldest first). Adds ev_ebit, l1, l2, l3,
    graham and returns the rows that have an EV/EBIT."""
    rows = []
    for t, f in snap.items():
        mcap = f["price"] * f["shares_adj"]
        ev = mcap + f["debt"] - f["cash"]
        r = dict(f, ticker=t, mcap=mcap, ev=ev, sector=meta[t]["sector"], sub=meta[t]["sub"])
        r["ev_ebit"] = ev / f["ebit"] if f["ebit"] > 0 and ev > 0 else None
        pe = mcap / f["net_income"] if f["net_income"] and f["net_income"] > 0 else None
        pb = mcap / f["equity"] if f["equity"] and f["equity"] > 0 else None
        r["graham"] = pe * pb if pe and pb else None
        r["graham_ok"] = bool(r["graham"] is not None and r["graham"] < 22.5 and f["ni_5y_all_positive"])
        r["l3"] = healthy(f)
        rows.append(r)
    valid = [r for r in rows if r["ev_ebit"] is not None and r["sector"] not in EXCLUDED_SECTORS]
    subs, secs = {}, {}
    for r in valid:
        subs.setdefault(r["sub"], []).append(r["ev_ebit"])
        secs.setdefault(r["sector"], []).append(r["ev_ebit"])
    for r in valid:
        grp = subs[r["sub"]] if len(subs[r["sub"]]) >= MIN_GROUP else secs[r["sector"]]
        r["l1"] = r["ev_ebit"] / float(np.median(grp)) if len(grp) >= MIN_GROUP else None
        h = [v for v in own_hist.get(r["ticker"], []) if v is not None and v > 0]
        r["l2"] = r["ev_ebit"] / float(np.median(h)) if len(h) >= OWN_HISTORY_MIN_POINTS else None
    return rows, valid


def _rank(rows, key):
    """Percentile rank (0 = cheapest) among rows that have `key`."""
    have = sorted([r for r in rows if r.get(key) is not None], key=lambda r: r[key])
    n = len(have)
    return {r["ticker"]: (i / max(n - 1, 1)) for i, r in enumerate(have)}


def pick(variant, rows, valid):
    if variant == "graham":
        c = sorted([r for r in rows if r["graham_ok"]], key=lambda r: r["graham"])
        return [r["ticker"] for r in c[:N_PICKS]]
    pool = valid
    if variant.endswith("+L3"):
        pool = [r for r in valid if r["l3"]]
    keys = {"raw": ["ev_ebit"], "L1": ["l1"], "L2": ["l2"], "L1+L3": ["l1"], "L2+L3": ["l2"],
            "L1+L2": ["l1", "l2"], "L1+L2+L3": ["l1", "l2"]}[variant]
    pool = [r for r in pool if all(r.get(k) is not None for k in keys)]
    if len(keys) == 1:
        c = sorted(pool, key=lambda r: r[keys[0]])
    else:
        ranks = [_rank(pool, k) for k in keys]
        c = sorted(pool, key=lambda r: sum(rk[r["ticker"]] for rk in ranks))
    return [r["ticker"] for r in c[:N_PICKS]]


VARIANTS = ["raw", "L1", "L2", "L1+L3", "L2+L3", "L1+L2", "L1+L2+L3", "graham"]


# ---------------------------------------------------------------- returns
def cohort_return(tickers, start, end, adj, spy, exits=None):
    """Equal-weight buy at the first close on/after start, sell at the last
    close on/before end. exits: {ticker: date} -> sold then, money in SPY."""
    def px(s, d, after=True):
        s = s.dropna()
        if after:
            s2 = s[s.index >= d]
            return (s2.index[0], float(s2.iloc[0])) if len(s2) else (None, None)
        s2 = s[s.index <= d]
        return (s2.index[-1], float(s2.iloc[-1])) if len(s2) else (None, None)
    s0d, s0 = px(spy, start)
    s1d, s1 = px(spy, end, after=False)
    if s0 is None or s1 is None or s1d <= s0d:
        return None
    rets = []
    for t in tickers:
        if t not in adj.columns:
            continue
        d0, p0 = px(adj[t], start)
        if p0 is None or d0 > s0d + pd.Timedelta(days=7):
            continue
        ex = (exits or {}).get(t)
        if ex is not None and ex < end:
            dx, px_ = px(adj[t], ex, after=False)
            _, sx = px(spy, ex, after=False)
            if px_ is None or sx is None:
                continue
            r = (px_ / p0) * (1 - FEE) ** 2 * (s1 / sx) - 1
        else:
            d1, p1 = px(adj[t], end, after=False)
            if p1 is None or d1 < s1d - pd.Timedelta(days=10):
                d1, p1 = px(adj[t], end, after=False)   # stopped trading early: take the last price
                if p1 is None:
                    continue
            r = (p1 / p0) * (1 - FEE) ** 2 - 1
        rets.append(r)
    if not rets:
        return None
    return {"ret": float(np.mean(rets)), "spy": s1 / s0 - 1, "n": len(rets)}


def tstat(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if len(x) < 3:
        return None
    s = np.std(x, ddof=1)
    return round(float(np.mean(x) / (s / math.sqrt(len(x)))), 2) if s > 0 else None


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_df = pd.read_csv(SP500_CSV_URL)
    meta_df["Symbol"] = meta_df["Symbol"].str.replace(".", "-", regex=False)
    meta = {row["Symbol"]: {"cik": int(row["CIK"]), "sector": row["GICS Sector"], "sub": row["GICS Sub-Industry"]}
            for _, row in meta_df.iterrows() if pd.notna(row.get("CIK"))}
    tickers = sorted(meta)
    log(f"universe {len(tickers)}")

    # prices: split-adjusted close (for EV) + dividend-adjusted close (returns) + splits
    close, adj, splits = {}, {}, {}
    for i in range(0, len(tickers) + 1, 100):
        part = tickers[i:i + 100] + (["SPY"] if i == 0 else [])
        if not part:
            continue
        log(f"prices {i + 1}-{i + len(part)}")
        data = yf.download(" ".join(part), start="2010-01-01", auto_adjust=False, actions=True,
                           group_by="ticker", threads=True, progress=False)
        for t in part:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
                if df["Close"].dropna().empty:
                    continue
                close[t], adj[t] = df["Close"], df["Adj Close"]
                sp = df["Stock Splits"] if "Stock Splits" in df else pd.Series(dtype=float)
                splits[t] = [(d.strftime("%Y-%m-%d"), float(v)) for d, v in sp[sp > 0].items()]
            except Exception:
                continue
        time.sleep(2)
    close, adj = pd.DataFrame(close), pd.DataFrame(adj)
    spy = adj.pop("SPY")
    close = close.drop(columns=["SPY"], errors="ignore")
    log(f"prices for {close.shape[1]} tickers")

    # fundamentals
    session = requests.Session()
    session.headers.update({"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"})
    parsed = {}
    for k, t in enumerate(tickers):
        f = fetch_facts(meta[t]["cik"], session)
        if f:
            parsed[t] = parse_company(f)
        time.sleep(0.12)                 # SEC fair-access limit: 10 requests/second
        if (k + 1) % 50 == 0:
            log(f"SEC facts {k + 1}/{len(tickers)} ({len(parsed)} ok)")
    log(f"SEC facts for {len(parsed)}/{len(tickers)}")

    # quarterly rebalance dates
    qdates = [d for d in pd.date_range("2010-01-01", date.today(), freq="QS") if d.strftime("%Y-%m-%d") >= "2010-01-01"]
    own_hist = {}
    by_date = {}
    for d in qdates:
        asof = d.strftime("%Y-%m-%d")
        s = close[close.index <= d]
        if s.empty:
            continue
        last_px = s.iloc[-1]
        snap = {}
        for t, p in parsed.items():
            if t not in last_px.index or not np.isfinite(last_px[t]):
                continue
            f = fundamentals_at(p, asof, splits.get(t, []))
            if f:
                f["price"] = float(last_px[t])
                snap[t] = f
        rows, valid = score_date(snap, own_hist, meta)
        for r in rows:
            own_hist.setdefault(r["ticker"], []).append(r["ev_ebit"])
            own_hist[r["ticker"]] = own_hist[r["ticker"]][-4 * OWN_HISTORY_YEARS:]
        by_date[asof] = (rows, valid)
        log(f"{asof}: {len(rows)} companies, {len(valid)} with EV/EBIT, {sum(r['l3'] for r in rows)} healthy")

    test_dates = [d for d in sorted(by_date) if d >= START]
    # equal-weight universe per cohort - separates "cheap stocks beat other
    # S&P stocks" from "equal weight beat the cap-weighted index"
    univ = {}
    for d in test_dates:
        start = pd.Timestamp(d)
        end = start + pd.Timedelta(days=HOLD_DAYS)
        if end <= close.index[-1]:
            cr = cohort_return([r["ticker"] for r in by_date[d][0]], start, end, adj, spy)
            if cr:
                univ[d] = cr["ret"]
    results = {}
    current = {}
    for v in VARIANTS:
        for exit_rule in ("HOLD", "REVERT"):
            name = f"{v} · {exit_rule}"
            cohorts = []
            for d in test_dates:
                rows, valid = by_date[d]
                picks = pick(v, rows, valid)
                if d == test_dates[-1] and exit_rule == "HOLD":
                    current[v] = [{k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k))
                                   for k in ("ticker", "sector", "sub", "ev_ebit", "l1", "l2", "l3", "graham")}
                                  for r in rows if r["ticker"] in picks]
                if len(picks) < 10:
                    continue
                start = pd.Timestamp(d)
                end = start + pd.Timedelta(days=HOLD_DAYS)
                if end > close.index[-1]:
                    continue
                exits = None
                if exit_rule == "REVERT":
                    exits = {}
                    for q in [x for x in sorted(by_date) if d < x < end.strftime("%Y-%m-%d")]:
                        rq = {r["ticker"]: r for r in by_date[q][1]}
                        for t in picks:
                            if t not in exits and t in rq and rq[t].get("l1") is not None and rq[t]["l1"] >= 1.0:
                                exits[t] = pd.Timestamp(q)
                cr = cohort_return(picks, start, end, adj, spy, exits)
                if cr:
                    cohorts.append({"date": d, **cr, "excess": cr["ret"] - cr["spy"],
                                    "vs_universe": (cr["ret"] - univ[d]) if d in univ else None})
            if not cohorts:
                results[name] = {"n": 0}
                continue
            jan = [c for c in cohorts if c["date"][5:7] == "01"]
            hold = [c for c in cohorts if c["date"] >= HOLDOUT]
            chain = 1.0
            spy_chain = 1.0
            for c in jan:
                chain *= 1 + c["ret"]
                spy_chain *= 1 + c["spy"]
            yrs = len(jan)
            res = {
                "n_cohorts": len(cohorts), "independent_years": yrs,
                "avg_return_pct": round(np.mean([c["ret"] for c in cohorts]) * 100, 2),
                "avg_spy_pct": round(np.mean([c["spy"] for c in cohorts]) * 100, 2),
                "avg_excess_pct": round(np.mean([c["excess"] for c in cohorts]) * 100, 2),
                "median_excess_pct": round(float(np.median([c["excess"] for c in cohorts])) * 100, 2),
                "beat_spy_pct": round(np.mean([c["excess"] > 0 for c in cohorts]) * 100, 1),
                "t_independent": tstat([c["excess"] for c in jan]),
                "holdout_avg_excess_pct": round(np.mean([c["excess"] for c in hold]) * 100, 2) if hold else None,
                "holdout_t_independent": tstat([c["excess"] for c in hold if c["date"][5:7] == "01"]),
                "cagr_pct": round((chain ** (1 / yrs) - 1) * 100, 2) if yrs else None,
                "spy_cagr_pct": round((spy_chain ** (1 / yrs) - 1) * 100, 2) if yrs else None,
                "by_year": {c["date"][:4]: round(c["excess"] * 100, 1) for c in jan},
                "avg_names": round(np.mean([c["n"] for c in cohorts]), 1),
                "avg_vs_equal_weight_universe_pct": round(np.mean([c["vs_universe"] for c in cohorts if c["vs_universe"] is not None]) * 100, 2)
                if any(c["vs_universe"] is not None for c in cohorts) else None,
            }
            res["verdict"] = "PASS" if (res["avg_excess_pct"] > 0 and (res["t_independent"] or 0) >= 2
                                        and (res["holdout_avg_excess_pct"] or 0) > 0
                                        and yrs >= MIN_INDEPENDENT_YEARS) else "FAIL"
            results[name] = res
            log(f"{name}: excess {res['avg_excess_pct']}% t={res['t_independent']} holdout {res['holdout_avg_excess_pct']}% -> {res['verdict']}")

    ranked = sorted([k for k, v in results.items() if v.get("n_cohorts")],
                    key=lambda k: (results[k]["verdict"] == "PASS", results[k]["t_independent"] or -9,
                                   results[k]["avg_excess_pct"]), reverse=True)
    best = ranked[0] if ranked else None
    meta_out = {"generated_at": datetime.now(timezone.utc).isoformat(), "universe": len(tickers),
                "with_fundamentals": len(parsed), "first_test_date": test_dates[0] if test_dates else None,
                "last_test_date": test_dates[-1] if test_dates else None, "runtime_sec": round(time.time() - t0, 1),
                "best": best, "best_passes": bool(best and results[best]["verdict"] == "PASS"),
                "coverage_latest": {"companies": len(by_date[test_dates[-1]][0]), "with_ev_ebit": len(by_date[test_dates[-1]][1]),
                                    "healthy": sum(r["l3"] for r in by_date[test_dates[-1]][0])} if test_dates else None}
    out = {"meta": meta_out, "results": results, "ranking": ranked,
           "current_picks": current, "current_picks_date": test_dates[-1] if test_dates else None}
    (OUTPUT_DIR / "winners_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))

    L = ["# 🏆 מחקר מניות מנצחות - שלוש עדשות על S&P 500", "",
         f"{len(parsed)} חברות עם דוחות SEC · תיקים כל רבעון מ-{meta_out['first_test_date']} · החזקה שנה · 20 מניות · "
         f"עמלה 0.1% לצד · {meta_out['runtime_sec']} ש'",
         (f"כיסוי בתאריך האחרון: {meta_out['coverage_latest']['companies']} חברות, {meta_out['coverage_latest']['with_ev_ebit']} עם EV/EBIT, "
          f"{meta_out['coverage_latest']['healthy']} עוברות את עדשת הבריאות") if meta_out.get("coverage_latest") else "", "",
         "**כלל ההחלטה (נקבע מראש):** עודף ממוצע על ה-S&P 500 מעל 0, t ≥ 2 על השנים הבלתי תלויות (תיקי ינואר), "
         "וחיובי בתקופת הבקרה (2019 והלאה).", "",
         "| שיטה | עודף ממוצע | חציון | ניצחה את S&P | t | שנים | בקרה 2019+ | תשואה שנתית מול S&P | מול ממוצע שווה של כל המניות | |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ranked:
        r = results[k]
        L.append(f"| {k} | {r['avg_excess_pct']:+.1f}% | {r['median_excess_pct']:+.1f}% | {r['beat_spy_pct']}% | "
                 f"{r['t_independent']} | {r['independent_years']} | {(r['holdout_avg_excess_pct'] or 0):+.1f}% | {r['cagr_pct']}% מול {r['spy_cagr_pct']}% | {(r['avg_vs_equal_weight_universe_pct'] or 0):+.1f}% | "
                 f"{'✅' if r['verdict'] == 'PASS' else '❌'} |")
    if best:
        L += ["", f"## הטובה ביותר: {best} - {'✅ עוברת' if results[best]['verdict'] == 'PASS' else '❌ אף שיטה לא עברה - תושק כ-🧪'}",
              "עודף לפי שנה (תיק ינואר): " + " · ".join(f"{y}: {v:+.1f}%" for y, v in results[best]["by_year"].items())]
        v = best.split(" · ")[0]
        L += ["", f"## 20 המניות של {v} נכון ל-{out['current_picks_date']}",
              " · ".join(f"{p['ticker']} (EV/EBIT {p['ev_ebit']})" for p in current.get(v, []))]
    L += ["", "## מגבלות",
          "- הטיית שורדים: רשימת ה-S&P 500 של היום. חברות שנמחקו חסרות, וחברות שהצליחו ונכנסו מאוחר יותר - נמצאות. זה מייפה את התוצאות.",
          "- פיננסים ונדל\"ן לא נכללים בעדשות EV/EBIT (המדד לא משמעותי לבנקים, ביטוח וקרנות נדל\"ן). שיטת גרהם כן כוללת אותם.",
          "- הדוחות מ-SEC זמינים בעיקר מ-2010, ולכן הבדיקה מתחילה ב-2012 (צריך שנתיים של היסטוריה לעדשה 2).",
          "- 'תשואה שנתית' = שרשור תיקי ינואר בלבד; העודף הממוצע = ממוצע של כל התיקים הרבעוניים (חופפים)."]
    (OUTPUT_DIR / "winners_report.md").write_text("\n".join(L))
    log(f"Done. best: {best}")


if __name__ == "__main__":
    main()
