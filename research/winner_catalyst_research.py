#!/usr/bin/env python3
"""
research/winner_catalyst_research.py  -  מחקר OFFLINE בלבד
=============================================================
לא רץ בקרון החי ולא נוגע ב-stock_alerts.py / predictions.json.
מופעל ידנית בלבד (workflow_dispatch) ומפיק קבצי תוצאות ל-research/output/.

שני מחקרים (ראה research/2026-10-16-winner-fingerprint.md ו-2026-10-16-catalyst-calendar.md):

  1. fingerprint  - "טביעת אצבע של המנצחות": מה היה נכון לגבי 25 המניות שעלו הכי הרבה
                    בחודש הבא, *לפני* שעלו. התכונות נבחרות רק על התקופה שלפני SPLIT
                    ונבדקות רק על התקופה שאחריו.
  2. catalyst     - "לוח קטליזטורים" לטכנולוגיה ותרופות: event study סביב אירועים
                    (ניסויי שלב 3, אישורי FDA, דוחות, הסכמים מהותיים 8-K 1.01, חוזים
                    ממשלתיים, אירועי השקה ידניים) + בדיקה חודשית "קטליזטור ב-30 הימים הבאים".

סודות: נקרא רק ממשתני סביבה (SEC_USER_AGENT). אין מפתחות בקוד - הריפו ציבורי.

הרצה:
  python research/winner_catalyst_research.py --part all
  python research/winner_catalyst_research.py --part fingerprint --max-tickers 80   # בדיקה מהירה
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

RESEARCH_VERSION = "1.0.1"

# ---------------------------------------------------------------- הגדרות
TOP_N = 25                    # כמה "מנצחות"/"מפסידות" בכל חודש
START = "2000-01-01"          # כולל 2000-02 ו-2008-09
SPLIT = "2016-01-01"          # לפני = מציאת דפוסים, אחרי = בדיקה בלבד
COST_PER_SIDE = 0.001         # 0.1% לכל צד (קנייה/מכירה)
MIN_LIFT = 1.3                # חמישון נחשב "תכונה" רק אם מעלה את סיכוי ה-TOP פי 1.3 לפחות
MIN_DIRECTION = 1.2           # ...וגם מעלה את סיכוי ה-TOP יותר משהוא מעלה את סיכוי ה-BOTTOM
PLACEBO_RUNS = 300            # כמה תיקים אקראיים להשוואה
CATALYST_WINDOW_DAYS = 31     # "קטליזטור בחודש הבא"
BIG_GOV_AWARD = 100_000_000   # חוזה ממשלתי נחשב רק מ-100 מיליון דולר

OUT_DIR = os.path.join("research", "output")
CACHE_DIR = os.path.join("research", "cache")
MANUAL_TECH_EVENTS = os.path.join("research", "tech_events_manual.csv")

FEATURES = ["ret_1m", "ret_3m", "ret_12_1", "dist_high", "dist_low",
            "vol_60", "beta_252", "rsi_14", "vol_surge", "log_dollar_vol"]

FEATURE_LABELS_HE = {
    "ret_1m": "תשואה חודש אחרון", "ret_3m": "תשואה 3 חודשים",
    "ret_12_1": "מומנטום 12-1", "dist_high": "מרחק משיא 52ש'",
    "dist_low": "מרחק משפל 52ש'", "vol_60": "תנודתיות 60 יום",
    "beta_252": "בטא מול SPY", "rsi_14": "RSI 14",
    "vol_surge": "זינוק מחזור (20/120)", "log_dollar_vol": "גודל/נזילות (מחזור דולרי)",
}

# יקום גיבוי אם אין גישה לרשימת S&P 500 (מוטה-שורדים, רק לבדיקה)
FALLBACK = {
    "tech": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC", "QCOM",
             "TXN", "MU", "AMAT", "LRCX", "KLAC", "ADI", "CSCO", "IBM", "NOW", "INTU",
             "ANET", "PANW", "CRWD", "PLTR", "SNPS", "CDNS", "MCHP", "NXPI", "ON", "WDC",
             "STX", "HPQ", "DELL", "CIEN", "GOOGL", "META"],
    "pharma": ["LLY", "JNJ", "MRK", "PFE", "ABBV", "BMY", "AMGN", "GILD", "REGN", "VRTX",
               "BIIB", "MRNA", "ZTS", "INCY", "ALNY", "BMRN", "NBIX", "TECH", "UTHR", "EXEL"],
    "other": ["JPM", "BAC", "WFC", "GS", "XOM", "CVX", "COP", "PG", "KO", "PEP", "WMT",
              "COST", "HD", "LOW", "MCD", "NKE", "DIS", "CAT", "DE", "HON", "GE", "BA",
              "LMT", "RTX", "UNP", "UPS", "NEE", "DUK", "SO", "AMT", "PLD", "SPG", "MMM",
              "VZ", "T", "CMCSA", "TGT", "F", "GM", "VST", "CEG", "NRG"],
}


# ---------------------------------------------------------------- כלים כלליים
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def clean_company_name(name: str) -> str:
    """'Pfizer Inc.' -> 'Pfizer'; 'Bristol Myers Squibb' נשאר."""
    n = re.sub(r"\(.*?\)", "", str(name))
    n = re.sub(r"[,.]", " ", n)
    n = re.sub(r"\b(Inc|Corp|Corporation|Company|Co|Ltd|plc|PLC|Holdings|Group|Class [A-C]|N\.?V|SA|AG)\b",
               " ", n, flags=re.I)
    return re.sub(r"\s+", " ", n).strip()


def name_matches(candidate: str, company: str) -> bool:
    """התאמה שמרנית: כל מילות שם החברה המנוקה מופיעות בשם המועמד."""
    words = [w for w in clean_company_name(company).upper().split() if len(w) > 1]
    cand = re.sub(r"[^A-Z0-9 ]", " ", str(candidate).upper())
    return bool(words) and all(re.search(rf"\b{re.escape(w)}\b", cand) for w in words)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """יום המסחר האחרון בכל חודש (בלי aliases של resample - תואם pandas 2 ו-3)."""
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().values)


# ---------------------------------------------------------------- יקום
def load_universe(path: str | None) -> pd.DataFrame:
    """עמודות: ticker, name, sector, sub_industry, group (tech/pharma/other)."""
    df = None
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        log(f"יקום מקובץ: {path} ({len(df)})")
    else:
        try:
            import requests
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30).text
            t = pd.read_html(io.StringIO(html))[0]
            df = pd.DataFrame({
                "ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
                "name": t["Security"], "sector": t["GICS Sector"], "sub_industry": t["GICS Sub-Industry"],
            })
            log(f"יקום: S&P 500 הנוכחי ({len(df)}) - שים לב: מוטה-שורדים")
        except Exception as e:  # noqa: BLE001
            log(f"אין גישה לרשימת S&P 500 ({e}) - משתמש ביקום גיבוי")
    if df is None:
        rows = [{"ticker": t, "name": t, "sector": g, "sub_industry": g, "group": g}
                for g, ts in FALLBACK.items() for t in ts]
        return pd.DataFrame(rows)
    if "group" not in df.columns:
        def grp(r):
            s, si = str(r.get("sector", "")), str(r.get("sub_industry", ""))
            if s == "Information Technology" or si == "Interactive Media & Services":
                return "tech"
            if "Pharmaceutical" in si or "Biotechnology" in si:
                return "pharma"
            return "other"
        df["group"] = df.apply(grp, axis=1)
    return df.drop_duplicates("ticker").reset_index(drop=True)


# ---------------------------------------------------------------- מחירים
def download_prices(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    all_t = sorted(set(tickers) | {"SPY"})
    log(f"מוריד מחירים ל-{len(all_t)} טיקרים מ-{start}...")
    data = yf.download(all_t, start=start, auto_adjust=True, progress=False,
                       group_by="column", threads=True)
    close, vol = data["Close"].copy(), data["Volume"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    return close, vol


def clean_prices(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """לקח NFE: קפיצה יומית >200% או ירידה >90% = כנראה ארטיפקט פעולה תאגידית.
    מחזיר close + מסכת 'חודש חשוד' (True = לא להשתמש בחודש הזה לטיקר הזה)."""
    r = close.pct_change(fill_method=None)
    bad = (r > 2.0) | (r < -0.9)
    bad_month = bad.groupby(close.index.to_period("M")).max()
    n_bad = int(bad.values.sum())
    if n_bad:
        log(f"סומנו {n_bad} ימים חשודים (פעולה תאגידית/נתון שגוי) - החודשים שלהם מוחרגים")
    return close, bad_month


# ---------------------------------------------------------------- מחקר 1: טביעת אצבע
def build_panel(close: pd.DataFrame, vol: pd.DataFrame, universe: pd.DataFrame,
                bad_month: pd.DataFrame | None = None, top_n: int = TOP_N) -> pd.DataFrame:
    """פאנל חודשי (date, ticker): תכונות נכון לסוף חודש t + תשואת חודש t+1 + תוויות."""
    spy = close["SPY"]
    stocks = [c for c in close.columns if c != "SPY"]
    c, v = close[stocks], vol.reindex(columns=stocks)
    r, spy_r = c.pct_change(fill_method=None), spy.pct_change(fill_method=None)

    gain = c.diff().clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-c.diff().clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    w = 252
    cov = r.mul(spy_r, axis=0).rolling(w, min_periods=200).mean() - \
        r.rolling(w, min_periods=200).mean().mul(spy_r.rolling(w, min_periods=200).mean(), axis=0)
    beta = cov.div(spy_r.rolling(w, min_periods=200).var(ddof=0), axis=0)

    daily = {
        "ret_1m": c / c.shift(21) - 1,
        "ret_3m": c / c.shift(63) - 1,
        "ret_12_1": c.shift(21) / c.shift(252) - 1,
        "dist_high": c / c.rolling(252, min_periods=200).max() - 1,
        "dist_low": c / c.rolling(252, min_periods=200).min() - 1,
        "vol_60": r.rolling(60, min_periods=50).std() * math.sqrt(252),
        "beta_252": beta,
        "rsi_14": rsi,
        "vol_surge": v.rolling(20, min_periods=15).mean() / v.rolling(120, min_periods=90).mean(),
        "log_dollar_vol": np.log((c * v).rolling(60, min_periods=50).mean().replace(0, np.nan)),
    }

    me = month_end_dates(c.index)
    m_close = c.loc[me]
    fwd = m_close.shift(-1) / m_close - 1
    spy_m = spy.loc[me]
    spy_fwd = spy_m.shift(-1) / spy_m - 1

    parts = {k: df.loc[me].stack(future_stack=True) for k, df in daily.items()}
    panel = pd.DataFrame(parts)
    panel["fwd"] = fwd.stack(future_stack=True)
    panel.index.names = ["date", "ticker"]
    panel = panel.reset_index()

    if bad_month is not None:
        bm = bad_month.reindex(columns=stocks).fillna(False)
        # חוסם גם את חודש התכונה וגם את חודש התשואה הקדימה
        bm_shift = bm.shift(-1, fill_value=False)
        bm_any = (bm.astype(bool) | bm_shift.astype(bool))
        bm_any.index = [p.to_timestamp(how="end").normalize() for p in bm_any.index]
        key = panel["date"].dt.to_period("M")
        flag = bm_any.stack(future_stack=True)
        flag.index = pd.MultiIndex.from_tuples([(pd.Timestamp(d).to_period("M"), t) for d, t in flag.index])
        lookup = pd.Series(flag.values, index=flag.index)
        mi = pd.MultiIndex.from_arrays([key, panel["ticker"]])
        panel = panel[~lookup.reindex(mi).fillna(False).astype(bool).values]

    panel = panel.dropna(subset=["fwd"] + FEATURES)
    grp = universe.set_index("ticker")["group"]
    sec = universe.set_index("ticker")["sector"]
    panel["group"] = panel["ticker"].map(grp).fillna("other")
    panel["sector"] = panel["ticker"].map(sec).fillna("?")

    n = panel.groupby("date")["fwd"].transform("count")
    log(f"לפני סינון גודל: {len(panel):,} תצפיות, {panel['ticker'].nunique()} טיקרים עם נתונים, "
        f"עד {int(n.max()) if len(n) else 0} מניות בחודש (נדרש לפחות {4 * top_n})")
    panel = panel[n >= 4 * top_n].copy()            # מספיק מניות כדי ש-TOP_N יהיה משמעותי
    panel["n"] = panel.groupby("date")["fwd"].transform("count")
    rk_top = panel.groupby("date")["fwd"].rank(ascending=False, method="first")
    rk_bot = panel.groupby("date")["fwd"].rank(ascending=True, method="first")
    panel["top"] = rk_top <= top_n
    panel["bottom"] = rk_bot <= top_n
    panel["base"] = top_n / panel["n"]
    panel["spy_fwd"] = panel["date"].map(spy_fwd)
    return panel.reset_index(drop=True)


def add_quintiles(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    for f in FEATURES:
        p[f + "_q"] = p.groupby("date")[f].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if s.notna().sum() >= 5 else np.nan)
    return p


STATS_COLS = ["feature", "label", "quintile", "obs", "top_rate", "bottom_rate",
              "lift_top", "lift_bottom", "direction", "mean_fwd"]


def quintile_stats(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if p.empty:
        return pd.DataFrame(columns=STATS_COLS)
    base = p["base"].mean()
    for f in FEATURES:
        for q, g in p.groupby(f + "_q"):
            top_rate, bot_rate = g["top"].mean(), g["bottom"].mean()
            rows.append({"feature": f, "label": FEATURE_LABELS_HE[f], "quintile": int(q), "obs": len(g),
                         "top_rate": top_rate, "bottom_rate": bot_rate,
                         "lift_top": top_rate / base, "lift_bottom": bot_rate / base,
                         "direction": top_rate / bot_rate if bot_rate > 0 else np.nan,
                         "mean_fwd": g["fwd"].mean()})
    for col in ("group", "sector"):
        for k, g in p.groupby(col):
            top_rate, bot_rate = g["top"].mean(), g["bottom"].mean()
            rows.append({"feature": col, "label": col, "quintile": k, "obs": len(g),
                         "top_rate": top_rate, "bottom_rate": bot_rate,
                         "lift_top": top_rate / base, "lift_bottom": bot_rate / base,
                         "direction": top_rate / bot_rate if bot_rate > 0 else np.nan,
                         "mean_fwd": g["fwd"].mean()})
    return pd.DataFrame(rows, columns=STATS_COLS)


def select_rules(stats_in: pd.DataFrame) -> pd.DataFrame:
    """רק חמישונים שמעלים את סיכוי ה-TOP *וגם* מטים לכיוון מעלה (לא רק תנודתיות)."""
    s = stats_in[stats_in["feature"].isin(FEATURES)]
    return s[(s["lift_top"] >= MIN_LIFT) & (s["direction"] >= MIN_DIRECTION)].copy()


def score_panel(p: pd.DataFrame, rules: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=p.index)
    for _, r in rules.iterrows():
        score += (p[r["feature"] + "_q"] == r["quintile"]).astype(float) * (r["lift_top"] - 1)
    return score


def pick_portfolio(p: pd.DataFrame, score: pd.Series, top_n: int = TOP_N) -> pd.DataFrame:
    q = p.assign(score=score)
    q = q[q["score"] > 0].sort_values(["date", "score", "ticker"], ascending=[True, False, True])
    return q.groupby("date").head(top_n)


def evaluate_portfolio(picks: pd.DataFrame, p: pd.DataFrame, label: str) -> dict:
    if picks.empty:
        return {"period": label, "months": 0, "note": "אין בחירות"}
    monthly = picks.groupby("date").agg(ret=("fwd", "mean"), hit_top=("top", "mean"),
                                        hit_bottom=("bottom", "mean"), k=("ticker", "count"))
    # עלות: תחלופה חודשית (חלק מהתיק שהשתנה) * 2 צדדים
    sets = picks.groupby("date")["ticker"].apply(set)
    turnover, prev = [], None
    for d, s in sets.items():
        turnover.append(1.0 if prev is None else 1 - len(s & prev) / max(len(s), 1))
        prev = s
    monthly["cost"] = np.array(turnover) * 2 * COST_PER_SIDE
    monthly["net"] = monthly["ret"] - monthly["cost"]
    spy = p.drop_duplicates("date").set_index("date")["spy_fwd"].reindex(monthly.index)
    eq, eq_spy = (1 + monthly["net"]).cumprod(), (1 + spy.fillna(0)).cumprod()
    yrs = len(monthly) / 12
    dd = (eq / eq.cummax() - 1).min()
    return {
        "period": label, "months": len(monthly),
        "avg_picks": round(monthly["k"].mean(), 1),
        "hit_rate_top": round(monthly["hit_top"].mean(), 4),
        "hit_rate_bottom": round(monthly["hit_bottom"].mean(), 4),
        "base_rate": round(p[p["date"].isin(monthly.index)]["base"].mean(), 4),
        "cagr_net": round(eq.iloc[-1] ** (1 / yrs) - 1, 4) if yrs > 0 else None,
        "cagr_spy": round(eq_spy.iloc[-1] ** (1 / yrs) - 1, 4) if yrs > 0 else None,
        "max_drawdown": round(dd, 4),
        "months_beat_spy": round((monthly["net"] > spy).mean(), 4),
        "avg_monthly_turnover": round(float(np.mean(turnover)), 3),
    }


def placebo(p: pd.DataFrame, k_per_month: pd.Series, runs: int = PLACEBO_RUNS, seed: int = 42) -> np.ndarray:
    """תיקים אקראיים באותו גודל בכל חודש -> התפלגות hit_rate_top של 'מזל'."""
    rng = np.random.default_rng(seed)
    groups = {d: g["top"].to_numpy() for d, g in p.groupby("date") if d in k_per_month.index}
    out = np.empty(runs)
    for i in range(runs):
        rates = []
        for d, tops in groups.items():
            k = int(min(k_per_month[d], len(tops)))
            if k:
                rates.append(tops[rng.choice(len(tops), k, replace=False)].mean())
        out[i] = np.mean(rates) if rates else np.nan
    return out


def run_fingerprint(panel: pd.DataFrame, split: str = SPLIT) -> dict:
    p = add_quintiles(panel)
    split_ts = pd.Timestamp(split)
    p_in, p_out = p[p["date"] < split_ts], p[p["date"] >= split_ts]
    log(f"טביעת אצבע: {p_in['date'].nunique()} חודשים למציאה, {p_out['date'].nunique()} חודשים לבדיקה")

    stats_in, stats_out = quintile_stats(p_in), quintile_stats(p_out)
    rules = select_rules(stats_in)
    log(f"נבחרו {len(rules)} תכונות על תקופת המציאה")

    # האם כל תכונה שנבחרה 'החזיקה' מחוץ למדגם?
    if not rules.empty:
        chk = rules.merge(stats_out[["feature", "quintile", "lift_top", "direction"]],
                          on=["feature", "quintile"], how="left", suffixes=("_in", "_out"))
        chk["held_out_of_sample"] = (chk["lift_top_out"] >= 1.1) & (chk["direction_out"] >= 1.0)
    else:
        chk = pd.DataFrame()

    res = {"rules": chk, "stats_in": stats_in, "stats_out": stats_out}
    if not rules.empty:
        picks_in = pick_portfolio(p_in, score_panel(p_in, rules))
        picks_out = pick_portfolio(p_out, score_panel(p_out, rules))
        ev_in = evaluate_portfolio(picks_in, p_in, "מציאה (אופטימי)")
        ev_out = evaluate_portfolio(picks_out, p_out, "בדיקה מחוץ למדגם")
        if not picks_out.empty:
            k = picks_out.groupby("date")["ticker"].count()
            pl = placebo(p_out, k)
            ev_out["placebo_mean_hit_top"] = round(float(np.nanmean(pl)), 4)
            ev_out["placebo_95pct_hit_top"] = round(float(np.nanpercentile(pl, 95)), 4)
            ev_out["beats_placebo_95"] = bool(ev_out["hit_rate_top"] > ev_out["placebo_95pct_hit_top"])
        res["portfolio"] = pd.DataFrame([ev_in, ev_out])
    else:
        res["portfolio"] = pd.DataFrame([{"period": "-", "note": "אף תכונה לא עברה את הסף בתקופת המציאה"}])
    return res


# ---------------------------------------------------------------- מחקר 2: קטליזטורים
def _cache_get(key: str):
    path = os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _cache_put(key: str, value) -> None:
    path = os.path.join(CACHE_DIR, re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh)


def _http_json(url, params=None, headers=None, method="GET", body=None, retries=3):
    import requests
    for i in range(retries):
        try:
            if method == "POST":
                resp = requests.post(url, json=body, headers=headers, timeout=60)
            else:
                resp = requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                log(f"  שגיאה ב-{url}: {e}")
            time.sleep(2 * (i + 1))
    return None


def events_clinicaltrials(ticker: str, name: str) -> list[dict]:
    """סיום ראשוני (primary completion) של ניסויי שלב 3 שהחברה היא הספונסרית הראשית בהם.
    הערה: זה התאריך שבו הניסוי *הסתיים*, לא תאריך פרסום התוצאות - קירוב לקטליזטור."""
    key = f"ct_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    out, token = [], None
    for _ in range(20):
        params = {"query.spons": clean_company_name(name), "filter.advanced": "AREA[Phase]PHASE3",
                  "pageSize": 1000, "format": "json"}
        if token:
            params["pageToken"] = token
        js = _http_json("https://clinicaltrials.gov/api/v2/studies", params=params)
        if not js:
            break
        for st in js.get("studies", []):
            ps = st.get("protocolSection", {})
            sponsor = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")
            if not name_matches(sponsor, name):
                continue
            stm = ps.get("statusModule", {})
            d = stm.get("primaryCompletionDateStruct", {}).get("date")
            if d:
                out.append({"ticker": ticker, "date": d, "type": "phase3_completion", "scheduled": True,
                            "source": "ClinicalTrials.gov", "detail": ps.get("identificationModule", {}).get("nctId", "")})
        token = js.get("nextPageToken")
        if not token:
            break
        time.sleep(0.3)
    _cache_put(key, out)
    return out


def events_openfda(ticker: str, name: str) -> list[dict]:
    """אישורים מקוריים (ORIG / AP) של NDA/BLA בלבד - בלי גנריקה (ANDA)."""
    key = f"fda_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    word = clean_company_name(name).split()[0].upper() if clean_company_name(name) else ""
    out, skip = [], 0
    while word and skip < 5000:
        js = _http_json("https://api.fda.gov/drug/drugsfda.json",
                        params={"search": f"sponsor_name:{word}", "limit": 100, "skip": skip})
        if not js or not js.get("results"):
            break
        for r in js["results"]:
            app = str(r.get("application_number", ""))
            if not app.startswith(("NDA", "BLA")) or not name_matches(r.get("sponsor_name", ""), name):
                continue
            for s in r.get("submissions", []):
                if s.get("submission_type") == "ORIG" and s.get("submission_status") == "AP":
                    d = s.get("submission_status_date")
                    if d:
                        out.append({"ticker": ticker, "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}",
                                    "type": "fda_approval", "scheduled": True,
                                    "source": "openFDA Drugs@FDA", "detail": app})
        skip += 100
        time.sleep(0.3)
    _cache_put(key, out)
    return out


_SEC_TICKERS = None


def events_edgar_101(ticker: str, ua: str) -> list[dict]:
    """8-K עם סעיף 1.01 (הסכם מהותי). לא מתוזמן מראש -> רק לבדיקת תגובה/המשך אחרי האירוע."""
    global _SEC_TICKERS
    key = f"sec101_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    if _SEC_TICKERS is None:
        js = _http_json("https://www.sec.gov/files/company_tickers.json", headers=headers) or {}
        _SEC_TICKERS = {v["ticker"].upper(): int(v["cik_str"]) for v in js.values()}
    cik = _SEC_TICKERS.get(ticker.upper().replace("-", "."), _SEC_TICKERS.get(ticker.upper()))
    out = []
    if cik:
        sub = _http_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=headers)
        blocks = [sub.get("filings", {}).get("recent", {})] if sub else []
        for f in (sub or {}).get("filings", {}).get("files", []):
            time.sleep(0.15)
            more = _http_json(f"https://data.sec.gov/submissions/{f['name']}", headers=headers)
            if more:
                blocks.append(more)
        for b in blocks:
            for form, d, items in zip(b.get("form", []), b.get("filingDate", []), b.get("items", [])):
                if form == "8-K" and "1.01" in str(items).split(","):
                    out.append({"ticker": ticker, "date": d, "type": "material_agreement_8k101",
                                "scheduled": False, "source": "SEC EDGAR", "detail": items})
    time.sleep(0.15)
    _cache_put(key, out)
    return out


def events_usaspending(ticker: str, name: str) -> list[dict]:
    """חוזי ממשל פדרלי גדולים (מ-2008). לא מתוזמן מראש."""
    key = f"usa_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    body = {"filters": {"recipient_search_text": [clean_company_name(name)],
                        "award_type_codes": ["A", "B", "C", "D"],
                        "time_period": [{"start_date": "2007-10-01", "end_date": datetime.now().strftime("%Y-%m-%d")}],
                        "award_amounts": [{"lower_bound": BIG_GOV_AWARD}]},
            "fields": ["Award ID", "Recipient Name", "Start Date", "Award Amount"],
            "limit": 100, "page": 1, "sort": "Award Amount", "order": "desc"}
    out = []
    for page in range(1, 6):
        body["page"] = page
        js = _http_json("https://api.usaspending.gov/api/v2/search/spending_by_award/", method="POST", body=body)
        if not js:
            break
        for r in js.get("results", []):
            if name_matches(r.get("Recipient Name", ""), name) and r.get("Start Date"):
                out.append({"ticker": ticker, "date": r["Start Date"], "type": "gov_contract_100m",
                            "scheduled": False, "source": "USAspending.gov", "detail": r.get("Award ID", "")})
        if not js.get("page_metadata", {}).get("hasNext"):
            break
        time.sleep(0.5)
    _cache_put(key, out)
    return out


def events_earnings(ticker: str) -> list[dict]:
    key = f"earn_{ticker}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    out = []
    try:
        import yfinance as yf
        ed = yf.Ticker(ticker).get_earnings_dates(limit=100)
        if ed is not None:
            for d in ed.index:
                out.append({"ticker": ticker, "date": pd.Timestamp(d).tz_localize(None).strftime("%Y-%m-%d"),
                            "type": "earnings", "scheduled": True, "source": "yfinance", "detail": ""})
    except Exception as e:  # noqa: BLE001
        log(f"  דוחות {ticker}: {e}")
    _cache_put(key, out)
    return out


def events_manual_tech() -> list[dict]:
    """research/tech_events_manual.csv: ticker,date,event  (אירועי השקה קבועים, ממולא ידנית)."""
    if not os.path.exists(MANUAL_TECH_EVENTS):
        return []
    df = pd.read_csv(MANUAL_TECH_EVENTS, comment="#").dropna(subset=["ticker", "date"])
    return [{"ticker": r.ticker, "date": str(r.date), "type": "product_event", "scheduled": True,
             "source": "manual", "detail": str(getattr(r, "event", ""))} for r in df.itertuples()]


def collect_events(universe: pd.DataFrame, skip_apis: bool = False) -> pd.DataFrame:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    rows = events_manual_tech()
    if not skip_apis:
        if not ua:
            log("SEC_USER_AGENT לא מוגדר - מדלג על EDGAR (SEC חוסם בקשות אנונימיות)")
        sel = universe[universe["group"].isin(["tech", "pharma"])]
        for i, r in enumerate(sel.itertuples(), 1):
            log(f"קטליזטורים {i}/{len(sel)}: {r.ticker} ({r.group})")
            rows += events_earnings(r.ticker)
            if ua:
                rows += events_edgar_101(r.ticker, ua)
            if r.group == "pharma":
                rows += events_clinicaltrials(r.ticker, r.name)
                rows += events_openfda(r.ticker, r.name)
            else:
                rows += events_usaspending(r.ticker, r.name)
    ev = pd.DataFrame(rows, columns=["ticker", "date", "type", "scheduled", "source", "detail"])
    if ev.empty:
        return ev
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce", format="mixed")
    ev = ev.dropna(subset=["date"]).drop_duplicates(["ticker", "date", "type"])
    grp = universe.set_index("ticker")["group"]
    ev["group"] = ev["ticker"].map(grp).fillna("tech")
    return ev.sort_values(["ticker", "date"]).reset_index(drop=True)


def event_study(events: pd.DataFrame, close: pd.DataFrame, split: str = SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """תשואה עודפת מול SPY בחלונות: לפני (60/30 יום), תגובה (-1..+1), המשך (+1..+21)."""
    spy = close["SPY"]
    idx = close.index
    rows = []
    for e in events.itertuples():
        if e.ticker not in close.columns:
            continue
        pos = idx.searchsorted(e.date)
        if pos - 61 < 0 or pos + 21 >= len(idx):
            continue
        px = close[e.ticker]

        def ex(a, b):
            pa, pb, sa, sb = px.iloc[a], px.iloc[b], spy.iloc[a], spy.iloc[b]
            if any(pd.isna(x) or x == 0 for x in (pa, pb, sa, sb)):
                return np.nan
            return (pb / pa) - (sb / sa)

        rows.append({"ticker": e.ticker, "date": e.date, "type": e.type, "group": e.group,
                     "scheduled": e.scheduled,
                     "pre_60": ex(pos - 61, pos - 1), "pre_30": ex(pos - 31, pos - 1),
                     "reaction": ex(pos - 1, min(pos + 1, len(idx) - 1)),
                     "post_20": ex(pos + 1, pos + 21)})
    es = pd.DataFrame(rows)
    if es.empty:
        return es, pd.DataFrame()
    es["period"] = np.where(es["date"] < pd.Timestamp(split), "לפני " + split[:4], "מ-" + split[:4])
    es["abs_reaction"] = es["reaction"].abs()

    def agg(g):
        o = {"events": len(g)}
        for col in ("pre_60", "pre_30", "reaction", "post_20", "abs_reaction"):
            x = g[col].dropna()
            o[col + "_mean"] = x.mean()
            o[col + "_median"] = x.median()
            o[col + "_t"] = x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
        o["pre_30_hit"] = (g["pre_30"] > 0).mean()
        # אסטרטגיית "ריצה לקראת האירוע": כניסה 30 יום לפני, יציאה יום לפני - רק לאירועים מתוזמנים
        o["runup_net_mean"] = (g["pre_30"] - 2 * COST_PER_SIDE).mean() if g["scheduled"].all() else np.nan
        return pd.Series(o)

    summary = es.groupby(["type", "group", "period"]).apply(agg, include_groups=False).reset_index()
    return es, summary


def catalyst_monthly_flag(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """בסוף חודש t: האם יש למניה אירוע *מתוזמן* ב-31 הימים הבאים? משווים סיכוי TOP/BOTTOM.
    אירועים לא-מתוזמנים (8-K, חוזים) לא נכנסים כאן - זה היה מידע מהעתיד."""
    sch = events[events["scheduled"].astype(bool)]
    p = panel[panel["group"].isin(["tech", "pharma"])].copy()
    if p.empty or sch.empty:
        return pd.DataFrame()
    by_t = {t: np.sort(g["date"].values) for t, g in sch.groupby("ticker")}
    by_tt = {(t, ty): np.sort(g["date"].values) for (t, ty), g in sch.groupby(["ticker", "type"])}
    win = np.timedelta64(CATALYST_WINDOW_DAYS, "D")

    def has(dates, d):
        if dates is None or len(dates) == 0:
            return False
        d64 = np.datetime64(d)
        i = np.searchsorted(dates, d64, side="right")
        return i < len(dates) and dates[i] <= d64 + win

    p["cat_any"] = [has(by_t.get(t), d) for t, d in zip(p["ticker"], p["date"])]
    rows = []
    base = p["base"].mean()
    for ty in sorted(sch["type"].unique()):
        col = "cat_" + ty
        p[col] = [has(by_tt.get((t, ty)), d) for t, d in zip(p["ticker"], p["date"])]
    for col in ["cat_any"] + ["cat_" + ty for ty in sorted(sch["type"].unique())]:
        for grp_name, g in [("all", p)] + list(p.groupby("group")):
            for flag, h in g.groupby(col):
                rows.append({"catalyst": col.replace("cat_", ""), "group": grp_name, "has_catalyst_next_30d": bool(flag),
                             "obs": len(h), "top_rate": h["top"].mean(), "bottom_rate": h["bottom"].mean(),
                             "lift_top": h["top"].mean() / base, "lift_bottom": h["bottom"].mean() / base,
                             "mean_fwd": h["fwd"].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- סיכום
def pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def write_summary(fp: dict | None, cat: dict | None, meta: dict) -> str:
    L = [f"# תוצאות מחקר - מנצחות וקטליזטורים (v{RESEARCH_VERSION})",
         f"הורץ: {meta['run_at']} | יקום: {meta['universe_size']} מניות | {meta['start']} עד היום | "
         f"חלוקה: {SPLIT} | TOP_N={TOP_N}", ""]
    if fp:
        L += ["## 1. טביעת אצבע של המנצחות", ""]
        rules = fp["rules"]
        if rules is None or rules.empty:
            L.append("אף תכונה לא עברה את הסף בתקופת המציאה.")
        else:
            L.append("| תכונה | חמישון | פי-סיכוי TOP (מציאה) | פי-סיכוי TOP (בדיקה) | כיוון (בדיקה) | החזיקה? |")
            L.append("|---|---|---|---|---|---|")
            for r in rules.itertuples():
                L.append(f"| {r.label} | {r.quintile} | {r.lift_top_in:.2f} | {r.lift_top_out:.2f} | "
                         f"{r.direction_out:.2f} | {'✅' if r.held_out_of_sample else '❌'} |")
        L += ["", "### תיק מבוסס התכונות", ""]
        port = fp["portfolio"]
        for r in port.to_dict("records"):
            if "hit_rate_top" not in r:
                L.append(f"- {r.get('note', '')}")
                continue
            L.append(f"- **{r['period']}**: {r['months']} חודשים, ~{r['avg_picks']} מניות; "
                     f"סיכוי TOP {pct(r['hit_rate_top'])} מול בסיס {pct(r['base_rate'])}; "
                     f"BOTTOM {pct(r['hit_rate_bottom'])}; CAGR נטו {pct(r['cagr_net'])} מול SPY {pct(r['cagr_spy'])}; "
                     f"ירידה מקס' {pct(r['max_drawdown'])}")
            if pd.notna(r.get("placebo_95pct_hit_top", np.nan)):
                L.append(f"  - מול {PLACEBO_RUNS} תיקים אקראיים: ממוצע {pct(r['placebo_mean_hit_top'])}, "
                         f"אחוזון 95 {pct(r['placebo_95pct_hit_top'])} → "
                         f"{'עוקף מזל ✅' if r['beats_placebo_95'] else 'לא נבדל ממזל ❌'}")
        L.append("")
    if cat:
        L += ["## 2. קטליזטורים (טכנולוגיה + תרופות)", ""]
        cnt = cat.get("event_counts")
        if cnt is not None and not cnt.empty:
            L.append("אירועים שנאספו: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))
            L.append("")
        s = cat.get("event_summary")
        if s is not None and not s.empty:
            L.append("| סוג | מגזר | תקופה | אירועים | לפני 30 יום | t | תגובה | גודל תגובה | אחרי 20 יום | t | ריצה-לקראת נטו |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in s.itertuples():
                L.append(f"| {r.type} | {r.group} | {r.period} | {int(r.events)} | {pct(r.pre_30_mean)} | "
                         f"{r.pre_30_t:.1f} | {pct(r.reaction_mean)} | {pct(r.abs_reaction_mean)} | "
                         f"{pct(r.post_20_mean)} | {r.post_20_t:.1f} | {pct(r.runup_net_mean)} |")
            L.append("")
        m = cat.get("monthly")
        if m is not None and not m.empty:
            L += ["### קטליזטור מתוזמן ב-30 הימים הבאים → סיכוי להיות TOP / BOTTOM", "",
                  "| קטליזטור | מגזר | יש קטליזטור? | תצפיות | פי-סיכוי TOP | פי-סיכוי BOTTOM |", "|---|---|---|---|---|---|"]
            for r in m.itertuples():
                L.append(f"| {r.catalyst} | {r.group} | {'כן' if r.has_catalyst_next_30d else 'לא'} | {r.obs} | "
                         f"{r.lift_top:.2f} | {r.lift_bottom:.2f} |")
            L.append("")
    L += ["## איך לקרוא", "",
          "- **פי-סיכוי TOP** = כמה פעמים יותר מהבסיס המניה נכנסה ל-25 המובילות בחודש הבא (1.00 = כמו כולם).",
          "- אם גם TOP וגם BOTTOM גבוהים → התכונה מנבאת *תזוזה גדולה*, לא כיוון.",
          "- **t** מעל ~2 (בערך מוחלט) = לא סביר שזה מקרי. בתקופת הבדיקה זה מה שקובע.",
          "- תכונה ש'לא החזיקה' מחוץ למדגם = כנראה מקרה, לא דפוס.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    global TOP_N
    ap = argparse.ArgumentParser(description="מחקר offline: טביעת אצבע של מנצחות + קטליזטורים")
    ap.add_argument("--part", choices=["fingerprint", "catalyst", "all"], default="all")
    ap.add_argument("--universe", default=None, help="CSV עם ticker,name,sector,sub_industry[,group]")
    ap.add_argument("--start", default=START)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--max-tickers", type=int, default=0, help="להרצת בדיקה מהירה")
    ap.add_argument("--skip-apis", action="store_true", help="קטליזטורים רק מהקובץ הידני/מטמון")
    a = ap.parse_args(argv)
    TOP_N = a.top_n
    ensure_dirs()

    uni = load_universe(a.universe)
    if a.max_tickers:
        share = uni.groupby("group")["ticker"].transform("count")
        rank = uni.groupby("group").cumcount()
        uni = uni[rank < np.maximum(1, a.max_tickers * share // len(uni))].reset_index(drop=True)
    uni.to_csv(os.path.join(OUT_DIR, "universe_used.csv"), index=False)

    # יקום קטן (הרצת בדיקה): שומרים על אותו בסיס של ~5% במקום 25 קבוע
    if len(uni) < 20 * TOP_N:
        TOP_N = max(3, len(uni) // 20)
        log(f"יקום קטן ({len(uni)}) - TOP_N הותאם ל-{TOP_N} (בסיס ~5%, כמו בהרצה המלאה)")

    close, vol = download_prices(uni["ticker"].tolist(), a.start)
    close, bad_month = clean_prices(close)
    panel = build_panel(close, vol, uni, bad_month, top_n=TOP_N)
    log(f"פאנל: {len(panel):,} תצפיות מניה-חודש, {panel['date'].nunique()} חודשים")
    if panel.empty:
        log("שגיאה: הפאנל ריק - אין מספיק מניות עם נתונים. ראה שורת 'לפני סינון גודל' למעלה.")
        return 2

    fp = cat = None
    if a.part in ("fingerprint", "all"):
        fp = run_fingerprint(panel)
        fp["stats_in"].to_csv(os.path.join(OUT_DIR, "fingerprint_quintiles_insample.csv"), index=False)
        fp["stats_out"].to_csv(os.path.join(OUT_DIR, "fingerprint_quintiles_outsample.csv"), index=False)
        fp["rules"].to_csv(os.path.join(OUT_DIR, "fingerprint_rules.csv"), index=False)
        fp["portfolio"].to_csv(os.path.join(OUT_DIR, "fingerprint_portfolio.csv"), index=False)

    if a.part in ("catalyst", "all"):
        ev = collect_events(uni, skip_apis=a.skip_apis)
        ev.to_csv(os.path.join(OUT_DIR, "catalyst_events.csv"), index=False)
        log(f"נאספו {len(ev):,} אירועים")
        es, summ = event_study(ev, close) if not ev.empty else (pd.DataFrame(), pd.DataFrame())
        es.to_csv(os.path.join(OUT_DIR, "catalyst_event_returns.csv"), index=False)
        summ.to_csv(os.path.join(OUT_DIR, "catalyst_summary.csv"), index=False)
        mon = catalyst_monthly_flag(panel, ev) if not ev.empty else pd.DataFrame()
        mon.to_csv(os.path.join(OUT_DIR, "catalyst_monthly_flag.csv"), index=False)
        cat = {"event_counts": ev["type"].value_counts() if not ev.empty else None,
               "event_summary": summ, "monthly": mon}

    meta = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "universe_size": len(uni), "start": a.start}
    md = write_summary(fp, cat, meta)
    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
