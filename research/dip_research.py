"""
📉 מחקר "חברה חזקה בירידה" - השיטה של תומר (אופליין בלבד)
מופעל רק דרך .github/workflows/dip_research.yml (workflow_dispatch).
לא כותב לשום קובץ של האפליקציה החיה - התוצאות נשמרות כ-artifact.

Tomer, 13.10.2026: "I buy strong companies that sell a lot, when they're
cheap because of the market or a temporary crisis; they rise tens of
percent, I sell and wait for the next drop." The valuation research the
same day found nothing beyond survivorship, so this tests HIS rule directly
- and, as he asked, the combination with the valuation lenses.

Fixes the bias that sank the last test: membership is POINT-IN-TIME
(fja05680/sp500 historical components) - a stock counts only while it was
in the S&P 500, so future winners aren't bought before they got in, and
companies that later fell out ARE bought (if Yahoo still has their prices;
the share that's missing is reported, those are mostly the dead ones).

Signals, checked every Friday, one open position per stock:
  DIP     member now, closed >= D below its 52-week high (D = 25% / 35%)
  DIP+Q   + strong business at the last SEC filing known then: revenue not
          shrinking year over year, net income > 0, operating profit > 0
  DIP+Q+V + cheap: EV/EBIT below its industry median (SEC SIC 2-digit,
          >= 5 members) AND below its own 5-year median
Exits (buy at the next open, 0.1% per side):
  TP30    sell at +30% (his "tens of percent"), else after 2 years
  HIGH    sell when it gets back to the pre-drop 52-week high, else 2 years
  HOLD12  hold exactly 1 year (comparable to the valuation test)

Benchmarks for every trade, same days: SPY, and the equal-weight average
of all S&P members at the time (the honest one - it's what "just buy any
S&P stock" would have done).

Decision rule (fixed before running), per variant: average excess over the
equal-weight members > 0, t >= 2 across entry years (each year = one
observation, so overlapping trades are not counted as independent), and
positive in the holdout (entries 2019+). Price-only DIP also runs from 2001
(incl. 2008); fundamentals variants from 2011 (SEC XBRL).
"""
import json
import math
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
sys.path.insert(0, str(HERE))
import winners_research as W   # noqa: E402  (SEC parsing / point-in-time helpers)

OUTPUT_DIR = HERE / "output_dip_research"
MEMBERS_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
PRICE_START = "2000-01-01"
START_PRICE_ONLY = "2001-01-01"
START_FUNDAMENTALS = "2011-01-01"
HOLDOUT = "2019-01-01"
DRAWDOWNS = (0.25, 0.35)
TP = 0.30
MAX_HOLD = 504
HOLD12 = 252
FEE = 0.001
MIN_GROUP = 5
REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax"]


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def tstat(x):
    return W.tstat(x)


# ---------------------------------------------------------------- data
def load_members():
    df = pd.read_csv(MEMBERS_URL)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    rows = [(d, {t.strip().replace(".", "-") for t in str(s).split(",") if t.strip()})
            for d, s in zip(df["date"], df["tickers"])]
    return rows


def members_at(rows, d):
    lo, hi, best = 0, len(rows) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if rows[mid][0] <= d:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return rows[best][1] if best is not None else set()


def download_prices(tickers):
    O, C, A, SPL = {}, {}, {}, {}
    tickers = sorted(tickers)
    for i in range(0, len(tickers), 100):
        part = tickers[i:i + 100]
        log(f"prices {i + 1}-{i + len(part)} / {len(tickers)}")
        try:
            data = yf.download(" ".join(part), start=PRICE_START, auto_adjust=False, actions=True,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:
            log(f"  chunk failed: {e}")
            continue
        for t in part:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
                if df["Close"].dropna().shape[0] < 260:
                    continue
                O[t], C[t], A[t] = df["Open"], df["Close"], df["Adj Close"]
                sp = df["Stock Splits"] if "Stock Splits" in df else pd.Series(dtype=float)
                SPL[t] = [(d.strftime("%Y-%m-%d"), float(v)) for d, v in sp[sp > 0].items()]
            except Exception:
                continue
        time.sleep(2)
    return pd.DataFrame(O), pd.DataFrame(C), pd.DataFrame(A), SPL


def sec_session():
    s = requests.Session()
    s.headers.update({"User-Agent": W.SEC_UA, "Accept-Encoding": "gzip, deflate"})
    return s


def load_fundamentals(tickers, session):
    """{ticker: {"facts": parsed, "rev": rows, "sic2": str}} for tickers with a CIK."""
    r = session.get(SEC_TICKERS_URL, timeout=30)
    r.raise_for_status()
    cik_of = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}
    out, misses = {}, 0
    todo = [t for t in sorted(tickers) if t in cik_of]
    log(f"SEC: {len(todo)}/{len(tickers)} tickers have a CIK today (the rest were delisted/merged)")
    for k, t in enumerate(todo):
        cik = cik_of[t]
        f = W.fetch_facts(cik, session)
        if not f:
            misses += 1
            if misses >= 15:
                log(f"15 SEC failures in a row - stopping fundamentals. {W.FETCH_ERRORS}")
                break
            continue
        misses = 0
        rev = []
        for tag in REV_TAGS:
            rev = W._facts_for(f, "us-gaap", tag)
            if rev:
                break
        sic2 = None
        try:
            sub = session.get(SEC_SUBMISSIONS_URL.format(cik=cik), timeout=20)
            if sub.status_code == 200:
                sic = str(sub.json().get("sic") or "")
                sic2 = sic[:2] if sic else None
        except Exception:
            pass
        out[t] = {"facts": W.parse_company(f), "rev": rev, "sic2": sic2}
        time.sleep(0.25)              # two requests per company, stay under 10/s
        if (k + 1) % 100 == 0:
            log(f"SEC {k + 1}/{len(todo)} ({len(out)} ok)")
    return out


# ---------------------------------------------------------------- lenses
def quality(fund, asof):
    """Strong business: revenue not shrinking y/y, net income > 0, EBIT > 0."""
    p = fund["facts"]
    rev_now, rev_end = W.ttm(fund["rev"], asof)
    if rev_now is None or rev_now <= 0 or W._days(rev_end, asof) > 200:
        return False
    prev_asof = (date.fromisoformat(asof) - timedelta(days=365)).isoformat()
    rev_prev, _ = W.ttm(fund["rev"], prev_asof)
    if rev_prev is None or rev_now < rev_prev:
        return False
    ni, _ = W.ttm(p.get("net_income", []), asof)
    ebit, _ = W.ttm(p.get("ebit", []), asof)
    return bool(ni and ni > 0 and ebit and ebit > 0)


def quarterly_snapshots(funds, close, members, qdates, splits):
    """{quarter date: {ticker: {"q": bool, "ev_ebit": x, "l1": x, "l2": x}}}"""
    out, own = {}, {}
    for d in qdates:
        asof = d.strftime("%Y-%m-%d")
        mem = members_at(members, d)
        px = close[close.index <= d]
        if px.empty:
            continue
        last = px.iloc[-1]
        snap = {}
        for t in mem:
            if t not in funds or t not in last.index or not np.isfinite(last[t]):
                continue
            f = W.fundamentals_at(funds[t]["facts"], asof, splits.get(t, []))
            row = {"q": quality(funds[t], asof), "ev_ebit": None, "sic2": funds[t]["sic2"]}
            if f and f["ebit"] > 0:
                # split-adjusted close x reported shares x splits after the report date
                ev = float(last[t]) * f["shares_adj"] + f["debt"] - f["cash"]
                row["ev_ebit"] = ev / f["ebit"] if ev > 0 else None
            snap[t] = row
        groups = {}
        for t, r in snap.items():
            if r["ev_ebit"] is not None and r["sic2"]:
                groups.setdefault(r["sic2"], []).append(r["ev_ebit"])
        for t, r in snap.items():
            g = groups.get(r["sic2"]) or []
            r["l1"] = r["ev_ebit"] / float(np.median(g)) if r["ev_ebit"] and len(g) >= MIN_GROUP else None
            h = [v for v in own.get(t, []) if v]
            r["l2"] = r["ev_ebit"] / float(np.median(h)) if r["ev_ebit"] and len(h) >= 8 else None
        for t, r in snap.items():
            own.setdefault(t, []).append(r["ev_ebit"])
            own[t] = own[t][-20:]
        out[d] = snap
        log(f"{asof}: {len(mem)} members, {len(snap)} with fundamentals, {sum(r['q'] for r in snap.values())} strong")
    return out


# ---------------------------------------------------------------- simulation
def simulate(variant, dd, exit_rule, O, C, A, spy, ew, members, snaps, start):
    """Event-driven, one open position per ticker. Returns trade list."""
    idx = C.index
    fridays = [d for d in idx if d.weekday() == 4 and d >= pd.Timestamp(start)]
    hi252 = C.rolling(252, min_periods=200).max()
    draw = C / hi252 - 1
    qkeys = sorted(snaps)
    busy_until = {}
    trades = []
    pos = {d: i for i, d in enumerate(idx)}
    A = A.reindex(columns=C.columns)
    O = O.reindex(columns=C.columns)
    Av, Ov, Cv = A.values, O.values, C.values
    col = {t: j for j, t in enumerate(C.columns)}
    fin = np.isfinite(Av)
    last_valid_of = {j: (int(np.max(np.where(fin[:, j])[0])) if fin[:, j].any() else -1) for j in range(Av.shape[1])}
    hiv = hi252.values
    for d in fridays:
        i = pos[d]
        if i + 2 >= len(idx):
            break
        mem = members_at(members, d)
        row = draw.iloc[i]
        cands = [t for t in mem if t in col and np.isfinite(row.get(t, np.nan)) and row[t] <= -dd]
        if not cands:
            continue
        snap = None
        if variant != "DIP":
            qk = [q for q in qkeys if q <= d]
            if not qk:
                continue
            snap = snaps[qk[-1]]
        for t in cands:
            if busy_until.get(t, -1) >= i:
                continue
            if snap is not None:
                s = snap.get(t)
                if not s or not s["q"]:
                    continue
                if variant == "DIP+Q+V" and not (s.get("l1") is not None and s["l1"] <= 1.0
                                                 and s.get("l2") is not None and s["l2"] <= 1.0):
                    continue
            j = col[t]
            e = i + 1
            if not (np.isfinite(Ov[e, j]) and Ov[e, j] > 0 and np.isfinite(Av[e, j])):
                continue
            # adjusted-open entry: scale open by the adj/close ratio of that day
            ratio = Av[e, j] / Cv[e, j] if Cv[e, j] else np.nan
            entry = Ov[e, j] * ratio
            if not np.isfinite(entry) or entry <= 0:
                continue
            peak_adj = hiv[i, j] * (Av[i, j] / Cv[i, j])
            last_valid = last_valid_of[j]
            limit = min(e + (HOLD12 if exit_rule == "HOLD12" else MAX_HOLD), len(idx) - 1, last_valid)
            x, why = limit, ("time" if limit < last_valid else "data_end")
            if exit_rule in ("TP30", "HIGH"):
                target = entry * (1 + TP) if exit_rule == "TP30" else peak_adj
                path = Av[e:limit + 1, j]
                hit = np.where(np.isfinite(path) & (path >= target))[0]
                if len(hit):
                    x, why = e + int(hit[0]), "target"
            if x <= e:
                continue
            px = Av[x, j]
            ret = (px / entry) * (1 - FEE) ** 2 - 1
            s0, s1 = spy.iloc[e - 1], spy.iloc[x]
            w0, w1 = ew.iloc[e - 1], ew.iloc[x]
            trades.append({"ticker": t, "signal": d.strftime("%Y-%m-%d"), "entry": idx[e].strftime("%Y-%m-%d"),
                           "exit": idx[x].strftime("%Y-%m-%d"), "days": int(x - e), "ret": float(ret),
                           "spy": float(s1 / s0 - 1), "ew": float(w1 / w0 - 1), "why": why,
                           "drawdown": round(float(row[t]) * 100, 1)})
            busy_until[t] = x
    return trades


def summarize(trades):
    if not trades:
        return {"n": 0}
    df = pd.DataFrame(trades)
    df["x_ew"] = df["ret"] - df["ew"]
    df["x_spy"] = df["ret"] - df["spy"]
    df["year"] = df["entry"].str[:4]
    by_year = df.groupby("year")["x_ew"].mean()
    hold = df[df["entry"] >= HOLDOUT]
    yrs = df["days"].mean() / 252
    res = {
        "n": int(len(df)), "avg_ret_pct": round(df["ret"].mean() * 100, 2),
        "median_ret_pct": round(df["ret"].median() * 100, 2), "win_pct": round((df["ret"] > 0).mean() * 100, 1),
        "hit_target_pct": round((df["why"] == "target").mean() * 100, 1),
        "avg_days": round(df["days"].mean(), 0),
        "avg_x_spy_pct": round(df["x_spy"].mean() * 100, 2), "avg_x_ew_pct": round(df["x_ew"].mean() * 100, 2),
        "x_ew_per_year_pct": round(df["x_ew"].mean() / yrs * 100, 2) if yrs > 0 else None,
        "beat_ew_pct": round((df["x_ew"] > 0).mean() * 100, 1),
        "t_years": tstat(list(by_year.values)), "entry_years": int(len(by_year)),
        "holdout_x_ew_pct": round(hold["x_ew"].mean() * 100, 2) if len(hold) else None,
        "by_year_x_ew_pct": {y: round(v * 100, 1) for y, v in by_year.items()},
        "worst": df.nsmallest(8, "ret")[["ticker", "entry", "exit", "ret"]].assign(
            ret=lambda x: (x["ret"] * 100).round(1)).to_dict("records"),
        "best": df.nlargest(5, "ret")[["ticker", "entry", "exit", "ret"]].assign(
            ret=lambda x: (x["ret"] * 100).round(1)).to_dict("records"),
        "data_end_exits": int((df["why"] == "data_end").sum()),
    }
    res["verdict"] = "PASS" if (res["avg_x_ew_pct"] > 0 and (res["t_years"] or 0) >= 2
                                and (res["holdout_x_ew_pct"] or 0) > 0) else "FAIL"
    return res


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    members = load_members()
    universe = set()
    for d, s in members:
        if d >= pd.Timestamp(START_PRICE_ONLY):
            universe |= s
    log(f"point-in-time members since {START_PRICE_ONLY}: {len(universe)} tickers")
    O, C, A, splits = download_prices(universe | {"SPY"})
    spy = A.pop("SPY")
    O, C = O.drop(columns=["SPY"], errors="ignore"), C.drop(columns=["SPY"], errors="ignore")
    have = set(C.columns)
    log(f"prices for {len(have)}/{len(universe)} (missing = mostly delisted)")

    # equal-weight index of point-in-time members with prices
    rets = A.pct_change().clip(-0.9, 3.0)
    ew_r = []
    for d in A.index:
        mem = [t for t in members_at(members, d) if t in rets.columns]
        v = rets.loc[d, mem]
        ew_r.append(float(v.mean()) if v.notna().any() else 0.0)
    ew = (1 + pd.Series(ew_r, index=A.index).fillna(0)).cumprod()
    coverage = []
    for y in range(2001, date.today().year + 1):
        mem = members_at(members, pd.Timestamp(f"{y}-01-02"))
        coverage.append((y, len(mem), sum(1 for t in mem if t in have)))

    session = sec_session()
    mode, diag = W.probe_sec(session)
    funds = load_fundamentals(have, session) if mode == "api" else {}
    log(f"fundamentals for {len(funds)} tickers (SEC mode: {mode})")
    qdates = list(pd.date_range("2010-01-01", date.today(), freq="QS"))
    snaps = quarterly_snapshots(funds, C, members, qdates, splits) if funds else {}

    results = {}
    for variant in ("DIP", "DIP+Q", "DIP+Q+V"):
        if variant != "DIP" and not snaps:
            continue
        for dd in DRAWDOWNS:
            for ex in ("TP30", "HIGH", "HOLD12"):
                starts = [START_PRICE_ONLY, START_FUNDAMENTALS] if variant == "DIP" else [START_FUNDAMENTALS]
                for st in starts:
                    name = f"{variant} · {int(dd * 100)}% · {ex}" + (" · 2001+" if st == START_PRICE_ONLY else "")
                    tr = simulate(variant, dd, ex, O, C, A, spy, ew, members, snaps, st)
                    results[name] = summarize(tr)
                    r = results[name]
                    log(f"{name}: n={r.get('n')} x_ew={r.get('avg_x_ew_pct')} t={r.get('t_years')} -> {r.get('verdict')}")

    # what qualifies right now
    now = {}
    last_i = len(C.index) - 1
    draw_now = (C / C.rolling(252, min_periods=200).max() - 1).iloc[last_i]
    mem_now = members_at(members, C.index[-1])
    snap_now = snaps[max(snaps)] if snaps else {}
    for dd in DRAWDOWNS:
        lst = []
        for t in sorted(mem_now):
            if t in draw_now.index and np.isfinite(draw_now[t]) and draw_now[t] <= -dd:
                s = snap_now.get(t) or {}
                lst.append({"ticker": t, "drawdown_pct": round(float(draw_now[t]) * 100, 1), "strong": bool(s.get("q")),
                            "cheap_vs_industry": s.get("l1") is not None and s["l1"] <= 1.0,
                            "cheap_vs_itself": s.get("l2") is not None and s["l2"] <= 1.0})
        now[f"{int(dd * 100)}%"] = sorted(lst, key=lambda r: r["drawdown_pct"])

    # Multiple testing: 24 variants means ~1 will clear t >= 2 by luck alone
    # (it did on random data in testing). A rule only PASSES if the same rule
    # at the OTHER drawdown depth also holds up (positive, t >= 1.5) - a real
    # effect shouldn't vanish between 25% and 35%.
    for k, r in results.items():
        if r.get("verdict") != "PASS":
            continue
        other = k.replace("25%", "§").replace("35%", "25%").replace("§", "35%")
        o = results.get(other) or {}
        if not (o.get("n") and o["avg_x_ew_pct"] > 0 and (o.get("t_years") or 0) >= 1.5):
            r["verdict"] = "FAIL"
            r["failed_robustness"] = f"same rule at the other depth: x_ew {o.get('avg_x_ew_pct')}%, t {o.get('t_years')}"
    ranked = sorted([k for k, v in results.items() if v.get("n")],
                    key=lambda k: (results[k]["verdict"] == "PASS", results[k]["t_years"] or -9, results[k]["avg_x_ew_pct"]),
                    reverse=True)
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "members_since_2001": len(universe),
            "with_prices": len(have), "with_fundamentals": len(funds), "sec": diag,
            "coverage_by_year": coverage, "runtime_sec": round(time.time() - t0, 1), "best": ranked[0] if ranked else None}
    (OUTPUT_DIR / "dip_summary.json").write_text(json.dumps(
        {"meta": meta, "results": results, "ranking": ranked, "current": now}, ensure_ascii=False, indent=1, default=str))

    L = ["# 📉 מחקר 'חברה חזקה בירידה' - השיטה של תומר", "",
         f"חברות S&P 500 לפי ההרכב בכל תאריך (לא הרשימה של היום) · {len(have)}/{len(universe)} עם נתוני מחיר · "
         f"{len(funds)} עם דוחות SEC · {meta['runtime_sec']} ש'",
         "כיסוי (מניות במדד בתחילת שנה / עם מחיר): " + " · ".join(f"{y}: {h}/{m}" for y, m, h in coverage[::3]), "",
         "**כלל ההחלטה (נקבע מראש):** עודף ממוצע לעסקה מעל ממוצע שווה של כל מניות המדד באותם ימים > 0, "
         "t ≥ 2 על שנות הכניסה, וחיובי בבקרה (כניסות מ-2019). בגלל שנבדקות 24 וריאציות, כלל עובר רק אם גם אותו כלל "
         "בעומק הירידה השני (25%/35%) חיובי עם t ≥ 1.5 - אחרת ⚠️ (מקרי).", "",
         "| שיטה | עסקאות | תשואה ממוצעת | ברווח | הגיעה ליעד | ימים | מעל S&P | מעל ממוצע המניות | לשנה | t | בקרה | |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in ranked:
        r = results[k]
        L.append(f"| {k} | {r['n']} | {r['avg_ret_pct']:+.1f}% | {r['win_pct']}% | {r['hit_target_pct']}% | {int(r['avg_days'])} | "
                 f"{r['avg_x_spy_pct']:+.1f}% | {r['avg_x_ew_pct']:+.1f}% | {(r['x_ew_per_year_pct'] or 0):+.1f}% | {r['t_years']} | "
                 f"{(r['holdout_x_ew_pct'] or 0):+.1f}% | {'✅' if r['verdict'] == 'PASS' else ('⚠️' if r.get('failed_robustness') else '❌')} |")
    if ranked:
        b = results[ranked[0]]
        L += ["", f"## הטובה ביותר: {ranked[0]}",
              "עודף מעל ממוצע המניות לפי שנת כניסה: " + " · ".join(f"{y}: {v:+.1f}%" for y, v in b["by_year_x_ew_pct"].items()),
              "העסקאות הגרועות: " + " · ".join(f"{w['ticker']} {w['entry'][:7]} {w['ret']:+.0f}%" for w in b["worst"]),
              "הטובות: " + " · ".join(f"{w['ticker']} {w['entry'][:7]} {w['ret']:+.0f}%" for w in b["best"])]
    L += ["", "## מה עומד בתנאים עכשיו (ירידה של 25%+ מהשיא השנתי)",
          " · ".join(f"{r['ticker']} {r['drawdown_pct']:.0f}%{' 💪' if r['strong'] else ''}{' 💲' if r['cheap_vs_industry'] and r['cheap_vs_itself'] else ''}"
                     for r in now.get("25%", [])[:40]) or "אין", "💪 = עסק חזק · 💲 = זולה גם מול הענף וגם מול עצמה", "",
          "## מגבלות",
          "- מניות שנמחקו בלי נתוני מחיר ב-Yahoo חסרות (ראו כיסוי). אלה בעיקר חברות שקרסו או נרכשו - כלומר נשארת הטיה קטנה לטובת השיטה.",
          "- דוחות SEC רק לחברות שעדיין מדווחות היום, ולכן הגרסאות עם 💪/💲 עדיין מוטות מעט לטובת שורדים.",
          "- ענף = קוד SIC דו-ספרתי מה-SEC (לא GICS), כדי שכל החברות ימדדו באותו סרגל.",
          "- כל עסקה נמדדת מול אותם ימים בדיוק, אבל עסקאות חופפות - לכן ה-t מחושב על שנות כניסה ולא על עסקאות."]
    (OUTPUT_DIR / "dip_report.md").write_text("\n".join(L))
    log(f"Done. best: {meta['best']}")


if __name__ == "__main__":
    main()
