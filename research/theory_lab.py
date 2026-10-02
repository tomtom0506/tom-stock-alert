"""
🧪 מעבדת תיאוריות (מחקר v4) - אופליין בלבד, לא נוגע בקוד החי.
מופעל רק דרך .github/workflows/theory_lab.yml (workflow_dispatch).

Tests ~20 market theories on ~26 years of daily data, with the same honest
harness for all of them:

  * DISCOVERY 2001-2016 / HOLDOUT 2017-today. Parameters are fixed in this
    file before anything runs; the holdout is reported separately and a
    theory only "passes" if it works in BOTH periods.
  * Costs: 0.15% per side (0.1% commission + 0.05% slippage), like the live
    paper portfolio.
  * No look-ahead: signals use data up to the close of day t; trades enter
    at the OPEN of day t+1.
  * Every trade is measured against the equal-weight universe over the same
    days ("excess"), so a theory can't look good just because the market
    went up.
  * Placebo: each stock-level theory is compared with 200 random-entry
    versions (same number of trades, same holding time). A real edge should
    beat ~95% of random.
  * Multiple-testing bar: because ~20 theories are tested, a holdout
    t-stat of 2 isn't enough - PASS needs t >= 3 in discovery AND a
    positive, t >= 2 holdout AND placebo >= 95%.

Known limits (written into the report): survivorship bias (the universe is
chosen today - delisted names have no Yahoo history), Yahoo data quality,
no earnings calendar before ~2018 (theory T8 uses a price/volume proxy).
"""
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from vectorbt_regime_momentum_research import RESEARCH_UNIVERSE as BASE_UNIVERSE
except Exception:  # keep the lab runnable on its own
    BASE_UNIVERSE = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "PG", "KO", "WMT"]

OUTPUT_DIR = Path(__file__).resolve().parent / "output_theory_lab"
LOOKBACK_YEARS = 26
SPLIT_DATE = "2017-01-01"
COST_PER_SIDE = 0.0015
PLACEBO_RUNS = 200
PASS_T_DISCOVERY = 3.0
PASS_T_HOLDOUT = 2.0
PASS_PLACEBO_PCT = 95.0
MIN_TRADES = 30

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]  # all since Dec-1998
MACRO = ["^VIX", "^TNX", "^IRX", "IYT"]
DUAL_LISTED = {"TEVA": "TEVA.TA", "NICE": "NICE.TA", "ESLT": "ESLT.TA", "TSEM": "TSEM.TA", "ICL": "ICL.TA"}


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def download(tickers):
    data = yf.download(tickers=" ".join(tickers), period=f"{LOOKBACK_YEARS}y", group_by="ticker",
                       threads=True, progress=False, auto_adjust=True)
    out = {}
    for t in tickers:
        try:
            df = data[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if len(df) > 300:
                out[t] = df
        except Exception:
            pass
    return out


def panel(frames, field):
    return pd.DataFrame({t: f[field] for t, f in frames.items()}).sort_index()


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def tstat(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def series_stats(r, periods_per_year):
    """r: pd.Series of period returns (fractions)."""
    r = r.dropna()
    if len(r) < 12:
        return None
    eq = (1 + r).cumprod()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else None
    dd = float((eq / eq.cummax() - 1).min())
    vol = r.std() * math.sqrt(periods_per_year)
    return {"n": int(len(r)), "cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
            "max_dd_pct": round(dd * 100, 1), "sharpe": round(r.mean() * periods_per_year / vol, 2) if vol else None}


def split(df_or_s):
    return df_or_s[df_or_s.index < SPLIT_DATE], df_or_s[df_or_s.index >= SPLIT_DATE]


# ---------------------------------------------------------------------------
# event (trade) engine - stock-level theories
# ---------------------------------------------------------------------------
class Lab:
    def __init__(self, frames, universe):
        self.u = [t for t in universe if t in frames]
        self.O = panel({t: frames[t] for t in self.u}, "Open")
        self.C = panel({t: frames[t] for t in self.u}, "Close")
        self.H = panel({t: frames[t] for t in self.u}, "High")
        self.L = panel({t: frames[t] for t in self.u}, "Low")
        self.V = panel({t: frames[t] for t in self.u}, "Volume")
        self.idx = self.C.index
        # equal-weight universe index (daily close-to-close), the benchmark for "excess"
        self.ew = self.C.pct_change().mean(axis=1).fillna(0)
        self.ew_cum = (1 + self.ew).cumprod()
        self.sma200 = self.C.rolling(200).mean()
        self.sma50 = self.C.rolling(50).mean()
        self.sma5 = self.C.rolling(5).mean()
        self.ret1 = self.C.pct_change()

    def run(self, name, signal, hold=None, exit_fn=None, max_hold=20, note=""):
        """signal: DataFrame(bool) - True at close of day t. Entry: open t+1.
        Exit: close of t+hold, or first day exit_fn(t_idx, col) is True (max_hold cap)."""
        C, O, idx = self.C.values, self.O.values, self.idx
        sig = signal.reindex(index=idx, columns=self.u).fillna(False).values
        n_days = len(idx)
        trades = []
        for j, t in enumerate(self.u):
            i = 0
            rows = np.flatnonzero(sig[:, j])
            next_free = -1
            for i in rows:
                if i <= next_free or i + 2 >= n_days:
                    continue
                e = i + 1
                if not np.isfinite(O[e, j]) or O[e, j] <= 0:
                    continue
                if exit_fn is None:
                    x = min(e + hold - 1, n_days - 1)
                else:
                    x = None
                    for k in range(e, min(e + max_hold, n_days)):
                        if exit_fn(k, j):
                            x = k
                            break
                    if x is None:
                        x = min(e + max_hold - 1, n_days - 1)
                if not np.isfinite(C[x, j]):
                    continue
                gross = C[x, j] / O[e, j] - 1
                net = (1 + gross) * (1 - COST_PER_SIDE) ** 2 - 1
                bench = self.ew_cum.iloc[x] / self.ew_cum.iloc[i] - 1
                trades.append((idx[i], t, net, net - bench, x - e + 1))
                next_free = x
        return self.summarize(name, trades, note)

    def summarize(self, name, trades, note):
        if not trades:
            return {"theory": name, "type": "trades", "note": note, "n": 0, "verdict": "אין מספיק נתונים"}
        df = pd.DataFrame(trades, columns=["date", "ticker", "ret", "excess", "days"]).set_index("date")
        out = {"theory": name, "type": "trades", "note": note, "n": int(len(df)),
               "avg_hold_days": round(float(df["days"].mean()), 1)}
        for label, part in zip(("discovery", "holdout"), split(df)):
            if len(part) == 0:
                out[label] = None
                continue
            wins = part[part["ret"] > 0]
            losses = part[part["ret"] <= 0]
            out[label] = {
                "n": int(len(part)),
                "win_rate_pct": round(len(wins) / len(part) * 100, 1),
                "avg_win_pct": round(wins["ret"].mean() * 100, 2) if len(wins) else None,
                "avg_loss_pct": round(losses["ret"].mean() * 100, 2) if len(losses) else None,
                "avg_trade_pct": round(part["ret"].mean() * 100, 3),
                "avg_excess_pct": round(part["excess"].mean() * 100, 3),
                "t_excess": round(tstat(part["excess"].values), 2) if tstat(part["excess"].values) is not None else None,
            }
        out["placebo_pct"] = self.placebo(df)
        out["verdict"] = verdict(out)
        self._last_trades = df
        return out

    def placebo(self, df):
        """Share of random-entry versions (same trade count & holding times,
        random tickers/days with data) whose mean excess is BELOW the real one.
        Vectorized: draws all random (day, ticker) pairs for a run at once."""
        if len(df) < MIN_TRADES:
            return None
        rng = np.random.default_rng(7)
        C, O = self.C.values, self.O.values
        ewc = self.ew_cum.values
        n_days, n_t = C.shape
        holds = df["days"].values.astype(int)
        real = df["excess"].mean()
        means = []
        for _ in range(PLACEBO_RUNS):
            h = np.repeat(holds, 3)                      # oversample, keep the first valid ones
            i = rng.integers(200, n_days - holds.max() - 2, size=len(h))
            j = rng.integers(0, n_t, size=len(h))
            e, x = i + 1, i + h
            o, c = O[e, j], C[x, j]
            ok = np.isfinite(o) & np.isfinite(c) & (o > 0)
            if ok.sum() == 0:
                continue
            net = (c[ok] / o[ok]) * (1 - COST_PER_SIDE) ** 2 - 1
            ex = net - (ewc[x[ok]] / ewc[i[ok]] - 1)
            means.append(ex[:len(holds)].mean())
        return round(float(np.mean(np.array(means) < real) * 100), 1) if means else None


def verdict(o, need_placebo=True):
    d, h = o.get("discovery") or {}, o.get("holdout") or {}
    if (o.get("n") or 0) < MIN_TRADES or not d or not h:
        return "אין מספיק נתונים"
    ok_d = (d.get("t_excess") or 0) >= PASS_T_DISCOVERY and (d.get("avg_excess_pct") or 0) > 0
    ok_h = (h.get("t_excess") or 0) >= PASS_T_HOLDOUT and (h.get("avg_excess_pct") or 0) > 0
    ok_p = (not need_placebo) or (o.get("placebo_pct") or 0) >= PASS_PLACEBO_PCT
    if ok_d and ok_h and ok_p:
        return "✅ עובר"
    if ok_d and not ok_h:
        return "❌ עבד בעבר, נכשל ב-2017+"
    if (d.get("avg_excess_pct") or 0) > 0 and (h.get("avg_excess_pct") or 0) > 0:
        return "🟡 כיוון חיובי, לא מובהק"
    return "❌ נכשל"


# ---------------------------------------------------------------------------
# monthly ranking engine - cross-sectional theories
# ---------------------------------------------------------------------------
def monthly_rank_theory(name, close, score_df, top_frac=0.1, higher_better=True, min_names=8, note=""):
    me = close.resample("ME").last()
    sc = score_df.reindex(me.index, method="ffill")
    fwd = me.pct_change().shift(-1)
    ew = fwd.mean(axis=1)
    rets, prev = [], set()
    for d in me.index[:-1]:
        s = sc.loc[d].dropna()
        s = s[me.loc[d, s.index].notna()]
        if len(s) < min_names * 3:
            rets.append((d, np.nan, np.nan))
            continue
        k = max(min_names, int(len(s) * top_frac))
        pick = set((s.sort_values(ascending=not higher_better)).index[:k])
        turnover = len(pick ^ prev) / (2 * k) if prev else 1.0
        r = fwd.loc[d, list(pick)].mean() - 2 * COST_PER_SIDE * turnover
        rets.append((d, r, ew.loc[d]))
        prev = pick
    df = pd.DataFrame(rets, columns=["date", "r", "ew"]).set_index("date").dropna()
    out = {"theory": name, "type": "monthly", "note": note, "n": int(len(df))}
    for label, part in zip(("discovery", "holdout"), split(df)):
        ex = part["r"] - part["ew"]
        st = series_stats(part["r"], 12) or {}
        bst = series_stats(part["ew"], 12) or {}
        out[label] = {**st, "bench_cagr_pct": bst.get("cagr_pct"),
                      "months_beating_pct": round(float((ex > 0).mean() * 100), 1) if len(ex) else None,
                      "avg_excess_pct": round(float(ex.mean() * 100), 3) if len(ex) else None,
                      "t_excess": round(tstat(ex.values), 2) if tstat(ex.values) is not None else None}
    out["placebo_pct"] = None
    d, h = out.get("discovery") or {}, out.get("holdout") or {}
    out["verdict"] = ("✅ עובר" if (d.get("t_excess") or 0) >= PASS_T_DISCOVERY and (h.get("t_excess") or 0) >= PASS_T_HOLDOUT
                      else ("❌ עבד בעבר, נכשל ב-2017+" if (d.get("t_excess") or 0) >= PASS_T_DISCOVERY
                            else ("🟡 כיוון חיובי, לא מובהק" if (d.get("avg_excess_pct") or 0) > 0 and (h.get("avg_excess_pct") or 0) > 0
                                  else "❌ נכשל")))
    return out


# ---------------------------------------------------------------------------
# index (timing) engine - market-level theories on SPY
# ---------------------------------------------------------------------------
def timing_theory(name, spy, position, note=""):
    """position: Series 0..1 decided at close t, applied to return t+1."""
    r = spy.pct_change()
    pos = position.reindex(spy.index).ffill().fillna(1).shift(1).fillna(1)
    switches = pos.diff().abs().fillna(0)
    strat = pos * r - switches * COST_PER_SIDE
    out = {"theory": name, "type": "timing", "note": note, "n": int(strat.notna().sum()),
           "time_invested_pct": round(float(pos.mean() * 100), 1),
           "switches_per_year": round(float(switches.sum() / (len(spy) / 252)), 1)}
    for label, (sp, bp) in zip(("discovery", "holdout"), zip(split(strat), split(r))):
        st = series_stats(sp, 252) or {}
        bst = series_stats(bp, 252) or {}
        out[label] = {**st, "bench_cagr_pct": bst.get("cagr_pct"), "bench_max_dd_pct": bst.get("max_dd_pct"),
                      "bench_sharpe": bst.get("sharpe")}
    d, h = out["discovery"], out["holdout"]
    better = lambda p: (p.get("sharpe") or -9) > (p.get("bench_sharpe") or 9) or \
        ((p.get("max_dd_pct") or -100) > (p.get("bench_max_dd_pct") or 0) * 0.7 and (p.get("cagr_pct") or -9) >= (p.get("bench_cagr_pct") or 0) - 2)
    out["placebo_pct"] = None
    out["verdict"] = "✅ עובר" if better(d) and better(h) else ("❌ עבד בעבר, נכשל ב-2017+" if better(d) else "❌ נכשל")
    return out


# ---------------------------------------------------------------------------
# descriptive: "after a drop of X% from the 52-week high, does it come back?"
# ---------------------------------------------------------------------------
def recovery_study(lab, thresholds=(0.2, 0.3, 0.4), horizons=(60, 120, 250)):
    C = lab.C
    hi = C.rolling(252, min_periods=200).max()
    dd = C / hi - 1
    rows = []
    for th in thresholds:
        hit = (dd <= -th) & (dd.shift(1) > -th)  # first day crossing the threshold
        events = [(i, j) for j in range(C.shape[1]) for i in np.flatnonzero(hit.values[:, j])]
        stats = {"threshold_pct": int(th * 100), "events": len(events)}
        for hz in horizons:
            back, fwd_r = 0, []
            n = 0
            for i, j in events:
                if i + hz >= len(C):
                    continue
                n += 1
                prev_high = hi.values[i, j]
                window = C.values[i + 1:i + hz + 1, j]
                if np.nanmax(window) >= prev_high:
                    back += 1
                fwd_r.append(C.values[i + hz, j] / C.values[i, j] - 1)
            stats[f"recovered_to_high_in_{hz}d_pct"] = round(back / n * 100, 1) if n else None
            stats[f"median_return_{hz}d_pct"] = round(float(np.nanmedian(fwd_r)) * 100, 1) if fwd_r else None
        rows.append(stats)
    return rows


# ---------------------------------------------------------------------------
# theories
# ---------------------------------------------------------------------------
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def run_all():
    universe = list(dict.fromkeys(BASE_UNIVERSE))
    all_t = universe + ["SPY"] + SECTOR_ETFS + MACRO + list(DUAL_LISTED) + list(DUAL_LISTED.values())
    log(f"Downloading {len(all_t)} tickers ({LOOKBACK_YEARS}y)...")
    frames = download(list(dict.fromkeys(all_t)))
    log(f"Loaded {len(frames)} tickers")
    lab = Lab(frames, universe)
    C, V, O, H, L = lab.C, lab.V, lab.O, lab.H, lab.L
    uptrend = C > lab.sma200
    results = []

    def add(r):
        log(f"  {r['theory']}: {r.get('verdict')}")
        results.append(r)

    # --- mean reversion family (the "it always comes back" idea) ---
    r2 = rsi(C, 2)
    add(lab.run("T1 · RSI(2)<10 במגמה עולה", (r2 < 10) & uptrend,
                exit_fn=lambda k, j: lab.C.values[k, j] > lab.sma5.values[k, j], max_hold=10,
                note="ירידה חדה וקצרה של מניה שמעל ממוצע 200; יציאה כשסוגרת מעל ממוצע 5"))
    drop5 = C / C.shift(5) - 1
    for x in (0.08, 0.12):
        for hold in (5, 10, 20):
            add(lab.run(f"T2 · ירידה של {int(x*100)}%+ ב-5 ימים במגמה עולה · החזקה {hold}", (drop5 <= -x) & uptrend, hold=hold,
                        note="מניה שירדה חד בשבוע בזמן שהמגמה הארוכה עולה"))
    down_streak = (lab.ret1 < 0).astype(int).rolling(4).sum() == 4
    add(lab.run("T10 · 4 ימי ירידה ברצף במגמה עולה · החזקה 5", down_streak & uptrend, hold=5))
    hi52 = C.rolling(252, min_periods=200).max()
    dd52 = C / hi52 - 1
    for th in (0.2, 0.3):
        cross = (dd52 <= -th) & (dd52.shift(1) > -th)
        add(lab.run(f"T3 · ירידה של {int(th*100)}% מהשיא השנתי · החזקה 120", cross, hold=120,
                    note="קנייה ברגע שהמניה חוצה ירידה של X% מהשיא"))
    vol20 = V.rolling(20).mean()
    quiet_pullback = (lab.ret1 < 0) & (lab.ret1.shift(1) < 0) & (lab.ret1.shift(2) < 0) & (V < vol20) & (C > lab.sma50) & uptrend
    add(lab.run("T9 · נסיגה שקטה (3 ימי ירידה בווליום נמוך) במגמה · החזקה 10", quiet_pullback, hold=10))

    # --- event / news proxy ---
    gap = O / C.shift(1) - 1
    rng_ = (H - L).replace(0, np.nan)
    strong_close = (C - L) / rng_ > 0.5
    gap_go = (gap >= 0.05) & (V >= 3 * vol20) & strong_close
    for hold in (20, 60):
        add(lab.run(f"T8 · קפיצת פתיחה 5%+ בווליום פי 3 · החזקה {hold}", gap_go, hold=hold,
                    note="תחליף לדוח כספי מפתיע: פער פתיחה גדול, ווליום חריג, סגירה חזקה"))
    gap_down = (gap <= -0.05) & (V >= 3 * vol20) & uptrend
    add(lab.run("T8b · צניחת פתיחה 5%+ במגמה עולה · החזקה 20", gap_down, hold=20, note="האם השוק מגזים בתגובה לחדשות רעות?"))

    # --- cross-sectional monthly ---
    mom = C.shift(21) / C.shift(252) - 1
    add(monthly_rank_theory("T4 · מומנטום 12-1 (העשירון העליון)", C, mom))
    add(monthly_rank_theory("T5 · קרבה לשיא השנתי", C, C / hi52))
    vol60 = lab.ret1.rolling(60).std()
    add(monthly_rank_theory("T6 · תנודתיות נמוכה", C, vol60, higher_better=False))
    mx = lab.ret1.rolling(21).max()
    add(monthly_rank_theory("T7 · הימנעות מ'מניות לוטו' (תשואה יומית מקסימלית נמוכה)", C, mx, higher_better=False))
    rev1m = C / C.shift(21) - 1
    add(monthly_rank_theory("T4b · היפוך חודשי (המפסידות של החודש)", C, rev1m, higher_better=False))
    combo = mom.rank(axis=1, pct=True) + (-vol60).rank(axis=1, pct=True) + (C / hi52).rank(axis=1, pct=True)
    add(monthly_rank_theory("T14 · שילוב: מומנטום + תנודתיות נמוכה + קרבה לשיא", C, combo))
    sec = panel({t: frames[t] for t in SECTOR_ETFS if t in frames}, "Close")
    if sec.shape[1] >= 6:
        sec_mom = sec / sec.shift(126) - 1
        add(monthly_rank_theory("T11 · רוטציית סקטורים (3 החזקים ב-6 חודשים)", sec, sec_mom, top_frac=0.34, min_names=3))

    # --- market timing on SPY ---
    spy = frames["SPY"]["Close"]
    spy_sma = spy.rolling(200).mean()
    add(timing_theory("I6 · S&P מעל ממוצע 200 (הכלל הקיים)", spy, (spy > spy_sma).astype(float).where(spy_sma.notna(), 1)))
    tom = pd.Series(0.0, index=spy.index)
    month_id = spy.index.to_period("M")
    pos_in_month = pd.Series(range(len(spy)), index=spy.index).groupby(month_id).cumcount()
    days_in_month = pd.Series(1, index=spy.index).groupby(month_id).transform("count")
    tom[(pos_in_month >= days_in_month - 2) | (pos_in_month <= 2)] = 1.0
    add(timing_theory("I1 · תחילת/סוף חודש בלבד", spy, tom, note="מושקע רק ביום המסחר האחרון ו-3 הראשונים בכל חודש"))
    halloween = pd.Series(np.where(spy.index.month.isin([11, 12, 1, 2, 3, 4]), 1.0, 0.0), index=spy.index)
    add(timing_theory("I2 · 'מכור במאי' (נוב'-אפר' בלבד)", spy, halloween))
    if "^VIX" in frames:
        vix = frames["^VIX"]["Close"].reindex(spy.index).ffill()
        spike = vix > 1.3 * vix.rolling(20).mean()
        pos = pd.Series(np.nan, index=spy.index)
        pos[spike] = 1.0
        pos = pos.ffill(limit=20).fillna(0.0)
        add(timing_theory("I3 · קנייה אחרי זינוק פחד (VIX) ל-20 יום", spy, pos, note="מושקע רק 20 ימים אחרי ש-VIX קופץ 30% מעל הממוצע"))
        calm = (vix < vix.rolling(252).quantile(0.8)).astype(float)
        add(timing_theory("I3b · יציאה כשהפחד בעשירון העליון", spy, (calm.astype(bool) | (spy > spy_sma)).astype(float)))
    if "^TNX" in frames and "^IRX" in frames:
        curve = (frames["^TNX"]["Close"] - frames["^IRX"]["Close"]).reindex(spy.index).ffill()
        add(timing_theory("I4 · יציאה כשעקום התשואות הפוך", spy, ((curve > 0) | (spy > spy_sma)).astype(float),
                          note="במזומן רק כשהעקום הפוך וגם המדד מתחת לממוצע 200"))
    breadth = (C > lab.sma200).sum(axis=1) / C.notna().sum(axis=1)
    washout = pd.Series(np.nan, index=breadth.index)
    washout[breadth < 0.3] = 1.0
    i6 = (spy > spy_sma).astype(float).reindex(breadth.index).fillna(1)
    add(timing_theory("I5 · כלל 200 + חזרה מוקדמת כשהשוק 'נשטף' (רוחב<30%)", spy,
                      (i6.astype(bool) | washout.ffill(limit=40).fillna(0).astype(bool)).astype(float)))
    if "IYT" in frames:
        iyt = frames["IYT"]["Close"].reindex(spy.index).ffill()
        dow = ((spy > spy_sma) | (iyt > iyt.rolling(200).mean())).astype(float).where(iyt.rolling(200).mean().notna(), 1)
        add(timing_theory("I7 · תאוריית דאו (S&P או התחבורה מעל 200)", spy, dow))
    o_spy = frames["SPY"]["Open"]
    overnight = (o_spy / spy.shift(1) - 1)
    intraday = (spy / o_spy - 1)
    on = {"theory": "I8 · מושקע רק בלילה (סגירה→פתיחה)", "type": "info", "note": "ללא עמלות - בפועל 2 עסקאות ביום",
          "discovery": {"overnight_cagr_pct": (series_stats(split(overnight)[0], 252) or {}).get("cagr_pct"),
                        "intraday_cagr_pct": (series_stats(split(intraday)[0], 252) or {}).get("cagr_pct")},
          "holdout": {"overnight_cagr_pct": (series_stats(split(overnight)[1], 252) or {}).get("cagr_pct"),
                      "intraday_cagr_pct": (series_stats(split(intraday)[1], 252) or {}).get("cagr_pct")},
          "verdict": "ℹ️ מידע (לא ישים בעמלות)"}
    add(on)

    # --- Israel: US -> Tel Aviv lag on dual-listed stocks ---
    trades = []
    for us, ta in DUAL_LISTED.items():
        if us not in frames or ta not in frames:
            continue
        u = frames[us]["Close"].pct_change()
        tdf = frames[ta]
        # US close of day d is AFTER the TA close of day d. Signal: US move on d minus TA move on d.
        diff = (u - tdf["Close"].pct_change()).reindex(tdf.index)
        for i in np.flatnonzero((diff > 0.02).values):
            if i + 1 >= len(tdf):
                continue
            o, c = tdf["Open"].iloc[i + 1], tdf["Close"].iloc[i + 1]
            if o > 0 and np.isfinite(c):
                net = (c / o) * (1 - COST_PER_SIDE) ** 2 - 1
                trades.append((tdf.index[i], ta, net, net, 1))
    if trades:
        r = lab.summarize("T12 · פיגור ת\"א אחרי ארה\"ב (מניות דואליות)", trades,
                          "כשהמניה עלתה בנאסד\"ק 2%+ יותר מבת\"א - קנייה בפתיחה בת\"א, מכירה בסגירה")
        r["placebo_pct"] = None  # the US-universe placebo doesn't apply to a TA-only same-day trade
        r["verdict"] = verdict(r, need_placebo=False) if r.get("n", 0) >= MIN_TRADES else "אין מספיק נתונים"
        add(r)

    # --- descriptive recovery study ---
    recovery = recovery_study(lab)
    return results, recovery, lab, frames


def to_markdown(results, recovery, meta):
    lines = ["# 🧪 מעבדת תיאוריות - תוצאות", "",
             f"נוצר: {meta['generated_at']} · {meta['usable_tickers']} מניות · {meta['first_date']} עד {meta['last_date']}",
             f"גילוי: עד {SPLIT_DATE} · מבחן: מ-{SPLIT_DATE} · עלות {COST_PER_SIDE*100:.2f}% לכל צד", "",
             "| תיאוריה | פסיקה | גילוי | מבחן 2017+ |", "|---|---|---|---|"]
    for r in results:
        def cell(p):
            if not p:
                return "—"
            if r["type"] == "trades":
                return f"{p['n']} עסק' · הצלחה {p['win_rate_pct']}% · עודף {p['avg_excess_pct']}% · t={p['t_excess']}"
            if r["type"] == "monthly":
                return f"CAGR {p.get('cagr_pct')}% מול {p.get('bench_cagr_pct')}% · t={p.get('t_excess')}"
            if r["type"] == "timing":
                return f"CAGR {p.get('cagr_pct')}% DD {p.get('max_dd_pct')}% מול {p.get('bench_cagr_pct')}%/{p.get('bench_max_dd_pct')}%"
            return json.dumps(p, ensure_ascii=False)
        lines.append(f"| {r['theory']} | {r.get('verdict')} | {cell(r.get('discovery'))} | {cell(r.get('holdout'))} |")
    lines += ["", "## אחרי ירידה מהשיא - כמה חזרו?", "", "| ירידה | אירועים | חזרו לשיא תוך 60/120/250 ימים | תשואה חציונית 60/120/250 |", "|---|---|---|---|"]
    for r in recovery:
        lines.append(f"| {r['threshold_pct']}% | {r['events']} | {r.get('recovered_to_high_in_60d_pct')}% / {r.get('recovered_to_high_in_120d_pct')}% / {r.get('recovered_to_high_in_250d_pct')}% | "
                     f"{r.get('median_return_60d_pct')}% / {r.get('median_return_120d_pct')}% / {r.get('median_return_250d_pct')}% |")
    lines += ["", "⚠️ הטיית שורדים: המניות נבחרו היום. מניות שירדו ולא חזרו (פשיטות רגל, מחיקות) חסרות בנתונים, ולכן שיעורי ההתאוששות כאן גבוהים מהמציאות."]
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results, recovery, lab, frames = run_all()
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "lookback_years": LOOKBACK_YEARS,
            "split_date": SPLIT_DATE, "cost_per_side_pct": COST_PER_SIDE * 100, "placebo_runs": PLACEBO_RUNS,
            "pass_rules": {"t_discovery": PASS_T_DISCOVERY, "t_holdout": PASS_T_HOLDOUT, "placebo_pct": PASS_PLACEBO_PCT,
                           "min_trades": MIN_TRADES},
            "usable_tickers": len(lab.u), "loaded": sorted(frames), "first_date": str(lab.idx[0].date()),
            "last_date": str(lab.idx[-1].date()),
            "known_limits": ["survivorship bias (universe chosen today)", "Yahoo data quality",
                             "no historical earnings calendar - T8 uses a price/volume proxy",
                             "cash earns 0 in timing theories"]}
    (OUTPUT_DIR / "theory_lab_summary.json").write_text(
        json.dumps({"meta": meta, "results": results, "recovery": recovery}, ensure_ascii=False, indent=1, default=str))
    (OUTPUT_DIR / "theory_lab_report.md").write_text(to_markdown(results, recovery, meta))
    log(f"Done: {len(results)} theories -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
