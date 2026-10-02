"""
🧪 מעבדת תיאוריות - שלב 4 - אופליין בלבד.
מופעל רק דרך .github/workflows/theory_lab_stage4.yml (workflow_dispatch).

Three questions, same honesty rules as stages 1-3 (costs, no look-ahead,
equal-weight benchmark of the same list, consistency across 5 periods):

PART 1 - "I already hold a stock: keep it or sell, and when?"
  Holding episodes: every S&P 500 stock bought at the first session of each
  year 2002-2025 and held for up to one year (252 sessions). Each exit rule
  is compared with simply holding the SAME stock over the SAME year:
    E1 trend 200     - out when the close is below its 200-day average, back in above
    E2 trend 50      - same with the 50-day average (faster)
    E3 trailing stop - sell after -10/-15/-20/-25% from the peak since purchase
    E4 profit take   - sell after +20% / +50%
    E5 trend 200 + market rule - in only while the stock AND the S&P 500 are above their 200-day averages
  Two separate verdicts per rule, because "sell" can help in two ways:
    return     - beats holding by >0 with t>=3 and in >=4/5 periods
    protection - cuts the average worst drop of an episode by >=25%,
                 costs <=2% of return per episode, and does so in >=4/5 periods

PART 2 - "I'm thinking of buying a stock with no signal: now, or wait?"
  Entry conditions vs the same-day equal-weight benchmark, hold 60 sessions:
    N1 pullback to the 50-day average inside an uptrend
    N2 RSI(14) < 40 inside an uptrend
    N3 a new 52-week high (breakout)
  Judged like stage 2 (5 periods, t, placebo).

PART 3 - lead-lag: does a stock's OWN move predict a related stock days later?
  * Own move = daily return minus its sector's equal-weight return (removes
    the "everything moves together" effect - market AND sector).
  * Only pairs inside the same GICS sector (economic link; far fewer tests).
  * Pairs are DISCOVERED on 2001-2012 only (lag 1-5 days, very strict t>=4),
    then TRADED once on 2013+: when the leader's own move is > 2 standard
    deviations, buy the follower at the next open and hold for the lag.
  * Placebo: the same rule on 50 sets of random same-sector pairs.
  * Sector version: do the sector's biggest names (by dollar volume) lead
    its smallest names by a day?
"""
import io
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import theory_lab as S1  # noqa: E402
import theory_lab_stage2 as S2  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "output_theory_lab_stage4"
COST = S1.COST_PER_SIDE
EPISODE = 252
DISCOVERY_END = "2013-01-01"
PAIR_T_MIN = 4.0
MAX_PAIRS = 60
LEADER_Z = 2.0
RANDOM_PAIR_SETS = 50


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def get_sectors():
    try:
        r = requests.get(S2.CONSTITUENTS_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        col = next(c for c in df.columns if "Sector" in c)
        return {str(s).strip().replace(".", "-"): str(sec) for s, sec in zip(df["Symbol"], df[col])}
    except Exception as e:
        log(f"sector list failed ({e}) - one sector for all")
        return {}


def period_of(year):
    for a, b in S2.PERIODS:
        if int(a[:4]) <= year < int(b[:4]):
            return f"{a[:4]}-{str(int(b[:4]) - 1) if b < '2030' else 'היום'}"
    return None


# ---------------------------------------------------------------------------
# PART 1 - exit rules on holding episodes
# ---------------------------------------------------------------------------
def episode_paths(lab):
    """[(ticker_j, start_i, end_i)] - first session of each year, 252 sessions."""
    idx = lab.idx
    years = sorted(set(idx.year))
    out = []
    C = lab.C.values
    for y in years:
        if y < 2002 or y > 2025:
            continue
        pos = np.flatnonzero(idx.year == y)
        if not len(pos):
            continue
        s = pos[0]
        e = min(s + EPISODE, len(idx) - 1)
        if e - s < 120:
            continue
        for j in range(C.shape[1]):
            if np.isfinite(C[s, j]) and np.isfinite(C[e, j]) and C[s, j] > 0:
                out.append((j, s, e))
    return out


def run_exit_rule(name, lab, episodes, position_fn, note=""):
    """position_fn(path_close, sma200, sma50, spy_ok) -> 0/1 array (decided at
    close t, applied to the return of t+1). Cash earns 0. COST per switch."""
    C = lab.C.values
    s200, s50 = lab.sma200.values, lab.sma50.values
    rows = []
    for j, s, e in episodes:
        p = C[s:e + 1, j]
        if not np.all(np.isfinite(p)):
            p = pd.Series(p).ffill().values
        r = p[1:] / p[:-1] - 1
        pos = position_fn(p, s200[s:e + 1, j], s50[s:e + 1, j], lab.spy_ok[s:e + 1])[:-1].astype(float)
        switches = np.abs(np.diff(np.r_[1.0, pos]))
        strat = pos * r - switches * COST
        eq_s, eq_h = np.cumprod(1 + strat), np.cumprod(1 + r)
        dd_s = float((eq_s / np.maximum.accumulate(eq_s) - 1).min())
        dd_h = float((eq_h / np.maximum.accumulate(eq_h) - 1).min())
        rows.append((lab.idx[s].year, eq_s[-1] - 1, eq_h[-1] - 1, dd_s, dd_h, float(switches.sum())))
    df = pd.DataFrame(rows, columns=["year", "rule", "hold", "dd_rule", "dd_hold", "switches"])
    df["excess"] = df["rule"] - df["hold"]
    df["period"] = df["year"].map(period_of)
    per = []
    for pname, g in df.groupby("period", sort=True):
        dd_cut = 1 - g["dd_rule"].mean() / g["dd_hold"].mean() if g["dd_hold"].mean() < 0 else 0
        per.append({"period": pname, "n": int(len(g)), "avg_excess_pct": round(g["excess"].mean() * 100, 2),
                    "dd_cut_pct": round(dd_cut * 100, 1)})
    t = S1.tstat(df["excess"].values)
    dd_cut = 1 - df["dd_rule"].mean() / df["dd_hold"].mean()
    pos_ret = sum(1 for p in per if p["avg_excess_pct"] > 0)
    pos_prot = sum(1 for p in per if p["dd_cut_pct"] >= 25 and p["avg_excess_pct"] >= -2)
    v_ret = "✅ מגדיל תשואה" if (t or 0) >= 3 and df["excess"].mean() > 0 and pos_ret >= 4 else (
        "❌ מקטין תשואה" if df["excess"].mean() < 0 else "🟡 לא מובהק")
    v_prot = "✅ מגן" if dd_cut >= 0.25 and df["excess"].mean() >= -0.02 and pos_prot >= 4 else "❌ לא מגן מספיק"
    return {"rule": name, "note": note, "episodes": int(len(df)),
            "avg_rule_pct": round(df["rule"].mean() * 100, 2), "avg_hold_pct": round(df["hold"].mean() * 100, 2),
            "avg_excess_pct": round(df["excess"].mean() * 100, 2), "t": round(t, 2) if t is not None else None,
            "better_than_hold_pct": round(float((df["excess"] > 0).mean() * 100), 1),
            "avg_dd_rule_pct": round(df["dd_rule"].mean() * 100, 1), "avg_dd_hold_pct": round(df["dd_hold"].mean() * 100, 1),
            "dd_cut_pct": round(dd_cut * 100, 1), "avg_switches": round(df["switches"].mean(), 1),
            "periods": per, "verdict_return": v_ret, "verdict_protection": v_prot}


def trend_rule(which):
    def f(p, s200, s50, spy_ok):
        m = s200 if which == 200 else s50
        return np.where(np.isfinite(m), (p > m).astype(float), 1.0)
    return f


def trend_market_rule(p, s200, s50, spy_ok):
    stock = np.where(np.isfinite(s200), (p > s200).astype(float), 1.0)
    return stock * spy_ok.astype(float)


def trailing_stop(x):
    def f(p, s200, s50, spy_ok):
        peak = np.maximum.accumulate(p)
        hit = np.flatnonzero(p <= peak * (1 - x))
        pos = np.ones(len(p))
        if len(hit):
            pos[hit[0]:] = 0.0
        return pos
    return f


def profit_take(x):
    def f(p, s200, s50, spy_ok):
        hit = np.flatnonzero(p >= p[0] * (1 + x))
        pos = np.ones(len(p))
        if len(hit):
            pos[hit[0]:] = 0.0
        return pos
    return f


# ---------------------------------------------------------------------------
# PART 3 - lead-lag
# ---------------------------------------------------------------------------
def residual_returns(lab, sectors):
    r = lab.C.pct_change()
    sec = pd.Series({t: sectors.get(t, "all") for t in lab.u})
    res = r.copy()
    for s_name, cols in sec.groupby(sec).groups.items():
        cols = list(cols)
        res[cols] = r[cols].sub(r[cols].mean(axis=1), axis=0)
    return res, sec


def discover_pairs(res, sec):
    """Lagged correlation of own-moves, same sector only, discovery years only."""
    disc = res[res.index < DISCOVERY_END]
    found = []
    for s_name, cols in sec.groupby(sec).groups.items():
        cols = [c for c in cols if disc[c].notna().sum() > 1000]
        if len(cols) < 4:
            continue
        X = disc[cols]
        for lag in range(1, 6):
            A, B = X.iloc[:-lag], X.iloc[lag:]
            A = (A - A.mean()) / A.std()
            B = (B - B.mean()) / B.std()
            Av, Bv = A.fillna(0).values, B.fillna(0).values
            n = min(A.notna().sum().min(), B.notna().sum().min())
            corr = Av.T @ Bv / max(len(Av) - 1, 1)   # corr[i, k]: leader i today vs follower k lag days later
            np.fill_diagonal(corr, 0)
            tmat = corr * np.sqrt(n)
            for i, k in zip(*np.where(tmat >= PAIR_T_MIN)):
                found.append({"leader": cols[i], "follower": cols[k], "lag": lag, "sector": s_name,
                              "corr": round(float(corr[i, k]), 4), "t": round(float(tmat[i, k]), 2)})
    found.sort(key=lambda x: -x["t"])
    return found


def trade_pairs(lab, res, pairs, start=DISCOVERY_END):
    """Leader own-move > LEADER_Z sd (60-day sd) on day t -> buy the follower
    at the open of t+1, hold `lag` sessions. Only trades on/after `start`."""
    z = res / res.rolling(60).std()
    trades = []
    for p in pairs:
        if p["leader"] not in z or p["follower"] not in lab.C:
            continue
        sig = pd.DataFrame(False, index=lab.idx, columns=lab.u)
        s = (z[p["leader"]] > LEADER_Z) & (z.index >= start)
        sig[p["follower"]] = s.reindex(lab.idx).fillna(False)
        df = lab.trades(sig, p["lag"])
        if len(df):
            trades.append(df)
    return pd.concat(trades) if trades else pd.DataFrame(columns=["ret", "excess", "days"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timing, t0 = {}, time.time()
    universe, src = S2.get_universe()
    sectors = get_sectors()
    frames = S2.download_chunked(universe + ["SPY"])
    timing["download_sec"] = round(time.time() - t0, 1)
    lab = S2.Lab2(frames, [t for t in universe if t in frames])
    spy = frames["SPY"]["Close"].reindex(lab.idx).ffill()
    lab.spy_ok = ((spy > spy.rolling(200).mean()) | spy.rolling(200).mean().isna()).values
    log(f"{len(lab.u)} tickers, {len(set(sectors.get(t, 'all') for t in lab.u))} sectors")

    # PART 1
    t1 = time.time()
    eps = episode_paths(lab)
    log(f"Part 1: {len(eps)} holding episodes")
    exits = [run_exit_rule("E1 · מכירה מתחת לממוצע 200, חזרה מעליו", lab, eps, trend_rule(200)),
             run_exit_rule("E2 · מכירה מתחת לממוצע 50, חזרה מעליו", lab, eps, trend_rule(50))]
    for x in (0.10, 0.15, 0.20, 0.25):
        exits.append(run_exit_rule(f"E3 · סטופ נגרר {int(x*100)}% מהשיא", lab, eps, trailing_stop(x)))
    for x in (0.20, 0.50):
        exits.append(run_exit_rule(f"E4 · מימוש רווח ב-+{int(x*100)}%", lab, eps, profit_take(x)))
    exits.append(run_exit_rule("E5 · מניה וגם S&P מעל ממוצע 200", lab, eps, trend_market_rule))
    for r in exits:
        log(f"  {r['rule']}: {r['verdict_return']} / {r['verdict_protection']}")
    timing["part1_sec"] = round(time.time() - t1, 1)

    # PART 2
    t2 = time.time()
    C = lab.C
    up = (C > lab.sma200) & (lab.sma50 > lab.sma200)
    entries = []
    near50 = (C <= lab.sma50 * 1.02) & (C >= lab.sma50 * 0.98) & up
    entries.append(S2.trade_result(lab, "N1 · נסיגה לממוצע 50 במגמה עולה · החזקה 60", lab.trades(near50, 60), ""))
    r14 = S1.rsi(C, 14)
    entries.append(S2.trade_result(lab, "N2 · RSI(14) מתחת ל-40 במגמה עולה · החזקה 60", lab.trades((r14 < 40) & up, 60), ""))
    hi = C.rolling(252, min_periods=200).max()
    entries.append(S2.trade_result(lab, "N3 · שיא חדש של 52 שבועות · החזקה 60", lab.trades((C >= hi) & (C.shift(1) < hi.shift(1)), 60), ""))
    for r in entries:
        log(f"  {r['theory']}: {r['verdict']}")
    timing["part2_sec"] = round(time.time() - t2, 1)

    # PART 3
    t3 = time.time()
    res, sec = residual_returns(lab, sectors)
    pairs = discover_pairs(res, sec)
    log(f"Part 3: {len(pairs)} pairs passed discovery t>={PAIR_T_MIN}")
    chosen = pairs[:MAX_PAIRS]
    real = trade_pairs(lab, res, chosen)
    lead = {"pairs_found": len(pairs), "pairs_traded": len(chosen), "top_pairs": chosen[:20]}
    if len(real):
        t_real = S1.tstat(real["excess"].values)
        rng = np.random.default_rng(5)
        rand_means = []
        by_sector = {s_name: list(cols) for s_name, cols in sec.groupby(sec).groups.items()}
        for _ in range(RANDOM_PAIR_SETS):
            rp = []
            for p in chosen:
                cols = by_sector.get(p["sector"], lab.u)
                a, b = rng.choice(cols, 2, replace=False)
                rp.append({"leader": a, "follower": b, "lag": p["lag"], "sector": p["sector"]})
            rdf = trade_pairs(lab, res, rp)
            if len(rdf):
                rand_means.append(rdf["excess"].mean())
        pct = round(float(np.mean(np.array(rand_means) < real["excess"].mean()) * 100), 1) if rand_means else None
        periods = S2.period_table(real["excess"])
        lead.update({"holdout_trades": int(len(real)), "win_rate_pct": round(float((real["ret"] > 0).mean() * 100), 1),
                     "avg_excess_pct": round(float(real["excess"].mean() * 100), 3),
                     "t": round(t_real, 2) if t_real is not None else None,
                     "random_pairs_beaten_pct": pct, "periods": periods})
        ok = (t_real or 0) >= 2 and real["excess"].mean() > 0 and (pct or 0) >= 95
        lead["verdict"] = "✅ עובר" if ok else ("🟡 כיוון חיובי, לא מובהק" if real["excess"].mean() > 0 else "❌ נכשל")
    else:
        lead["verdict"] = "אין מספיק נתונים"
    # sector version: biggest names lead the smallest by one day
    dv = (lab.C * lab.V).rolling(60).mean()
    r = lab.C.pct_change()
    mkt = r.mean(axis=1)
    ev = []
    for s_name, cols in sec.groupby(sec).groups.items():
        cols = list(cols)
        if len(cols) < 10:
            continue
        rank = dv[cols].rank(axis=1, pct=True)
        big = r[cols].where(rank >= 0.8).mean(axis=1) - mkt
        small = r[cols].where(rank <= 0.2).mean(axis=1) - mkt
        zb = big / big.rolling(60).std()
        nxt = small.shift(-1)
        for d in zb.index[(zb > LEADER_Z).values]:
            v = nxt.get(d)
            if v is not None and np.isfinite(v):
                ev.append((d, v - 2 * COST))
    sec_df = pd.DataFrame(ev, columns=["date", "excess"]).set_index("date").sort_index()
    sector_lead = {"events": int(len(sec_df))}
    if len(sec_df):
        sector_lead.update({"avg_next_day_excess_pct": round(float(sec_df["excess"].mean() * 100), 3),
                            "t": round(S1.tstat(sec_df["excess"].values) or 0, 2),
                            "t_2013": round(S1.tstat(sec_df[sec_df.index >= DISCOVERY_END]["excess"].values) or 0, 2),
                            "periods": S2.period_table(sec_df["excess"])})
        sector_lead["verdict"] = "✅ עובר" if sector_lead["t"] >= 3 and sector_lead["t_2013"] >= 2 else "❌ נכשל"
    timing["part3_sec"] = round(time.time() - t3, 1)
    timing["total_sec"] = round(time.time() - t0, 1)

    out = {"meta": {"generated_at": datetime.now(timezone.utc).isoformat(), "universe_source": src,
                    "usable_tickers": len(lab.u), "first_date": str(lab.idx[0].date()), "last_date": str(lab.idx[-1].date()),
                    "sectors": int(sec.nunique()), "timing": timing, "cost_per_side_pct": COST * 100},
           "part1_exit_rules": exits, "part2_entry_rules": entries,
           "part3_lead_lag_pairs": lead, "part3_sector_lead": sector_lead}
    (OUTPUT_DIR / "stage4_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    lines = ["# 🧪 שלב 4", "", f"זמנים: {timing}", "", "## כללי יציאה", "| כלל | תשואה | הגנה | עודף לשנה | חיתוך נפילה |", "|---|---|---|---|---|"]
    for r in exits:
        lines.append(f"| {r['rule']} | {r['verdict_return']} | {r['verdict_protection']} | {r['avg_excess_pct']}% | {r['dd_cut_pct']}% |")
    lines += ["", "## כללי כניסה"] + [f"- {r['theory']}: {r['verdict']}" for r in entries]
    lines += ["", f"## lead-lag: {lead.get('verdict')} · ענפים: {sector_lead.get('verdict')}"]
    (OUTPUT_DIR / "stage4_report.md").write_text("\n".join(lines))
    log("Done")


if __name__ == "__main__":
    main()
