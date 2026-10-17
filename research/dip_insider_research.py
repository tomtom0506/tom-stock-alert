#!/usr/bin/env python3
"""
research/dip_insider_research.py  -  מחקר OFFLINE בלבד
=========================================================
לא רץ בקרון החי ולא נוגע ב-stock_alerts.py / predictions.json. הפעלה ידנית בלבד.
ראה research/2026-10-16-quality-dip.md ו-research/2026-10-16-insider-cluster-buys.md

שלושה מחקרים:
  A. dip      - השיטה של Tomer בגרסה שיטתית: מניה חזקה שירדה 20-40% מהשיא, כניסה רק
                אחרי שהפסיקה לרדת, יציאה ב-+30% (+ עצירת הפסד/זמן לבדיקה). תיק מלא על
                26 שנה כולל 2000-2002 ו-2008. ספים נבחרים על 2000-2015, נבדקים על 2016+.
  B. insider  - אשכול קניות של אנשי פנים (2+ דירקטורים/נושאי משרה קנו בשוק הפתוח תוך
                30 יום). מקור: SEC Insider Transactions Data Sets (Form 4, מ-2006).
                האות נמדד מתאריך *הדיווח* (כשהמידע הפך לציבורי), לא מתאריך הקנייה.
  C. combo    - אשכול קניות אנשי פנים *אחרי* ירידה של 20%+ במניה חזקה, עם כללי היציאה של A.

יקום: S&P 500 + S&P 400 (בינוניות) + S&P 600 (קטנות) = ~1,500 מניות. מוטה-שורדים (הרכב נוכחי).
סודות: רק SEC_USER_AGENT ממשתנה סביבה. אין מפתחות בקוד - הריפו ציבורי.

הרצה:
  python research/dip_insider_research.py --part all
  python research/dip_insider_research.py --part all --max-tickers 150     # בדיקה מהירה
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

RESEARCH_VERSION = "1.2.0"

# ---------------------------------------------------------------- הגדרות
DOWNLOAD_START = "1995-01-01"     # היסטוריה ל"מניה חזקה" (3 שנים לפני שנה אחורה) כבר ב-2000
SIM_START = "2000-01-01"
SPLIT = "2016-01-01"
COST_PER_SIDE = 0.001
MAX_POSITIONS = 10
MIN_PRICE = 5.0                   # בלי מניות אגורה
MIN_DOLLAR_VOL = 2_000_000        # מחזור יומי ממוצע 60 יום
STRONG_3Y = 0.30                  # "חזקה": עלתה 30%+ בשלוש השנים שהסתיימו לפני שנה

# ברירת המחדל של Tomer - נקבעה *לפני* שראינו נתונים, לא עוברת אופטימיזציה
TOMER_DEFAULT = {"band": (-0.40, -0.20), "target": 0.30, "stop": None, "max_days": None}

GRID_BANDS = [(-0.30, -0.20), (-0.40, -0.30), (-0.40, -0.20), (-0.60, -0.40)]
GRID_TARGETS = [0.20, 0.30, 0.40]
GRID_STOPS = [None, -0.20, -0.30]
GRID_TIME = [252, None]
MIN_TRADES_IN_SAMPLE = 30
PLACEBO_SEEDS = 5

INSIDER_FIRST_YEAR = 2006
MIN_INSIDERS = 2
CLUSTER_DAYS = 30
MIN_TX_VALUE = 10_000
MIN_CLUSTER_VALUE = 100_000
COOLDOWN_DAYS = 90
INSIDER_HORIZONS = [21, 63, 126, 252]
INSIDER_HOLD = 126
INSIDER_MAX_POS = 20
COMBO_RECENT_DAYS = 10            # אשכול שדווח ב-10 ימי המסחר האחרונים

OUT_DIR = os.path.join("research", "output")
CACHE_DIR = os.path.join("research", "cache")

INDEX_PAGES = [
    ("large", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    ("mid", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
    ("small", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
]
FALLBACK = ["AAPL", "MSFT", "NVDA", "INTC", "DELL", "CSCO", "ORCL", "AMD", "MU", "QCOM",
            "PFE", "MRK", "BMY", "GILD", "BIIB", "JPM", "BAC", "C", "WFC", "XOM", "CVX",
            "HD", "LOW", "NKE", "DIS", "BA", "CAT", "GE", "F", "GM", "TGT", "WMT", "KO",
            "PEP", "MCD", "SBUX", "AMZN", "META", "NFLX", "ADBE", "CRM", "TXN", "AMAT",
            "LRCX", "KLAC", "WDC", "STX", "HPQ", "IBM", "T", "VZ", "CMCSA", "UPS", "FDX"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------- יקום
def _pick(cols, *cands):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_universe(path: str | None) -> pd.DataFrame:
    """עמודות: ticker, name, sector, tier (large/mid/small)."""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if "tier" not in df.columns:
            df["tier"] = "large"
        if "sub_industry" not in df.columns:
            df["sub_industry"] = "?"
        log(f"יקום מקובץ: {path} ({len(df)})")
        return df
    frames = []
    try:
        import requests
        for tier, url in INDEX_PAGES:
            html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30).text
            for t in pd.read_html(io.StringIO(html)):
                sym = _pick(t.columns, "Symbol", "Ticker symbol", "Ticker")
                name = _pick(t.columns, "Security", "Company")
                if sym and name and len(t) > 100:
                    sec = _pick(t.columns, "GICS Sector", "GICS sector")
                    subi = _pick(t.columns, "GICS Sub-Industry", "GICS sub-industry", "GICS Sub-industry")
                    frames.append(pd.DataFrame({
                        "ticker": t[sym].astype(str).str.strip().str.replace(".", "-", regex=False),
                        "name": t[name].astype(str), "sector": t[sec] if sec else "?",
                        "sub_industry": t[subi] if subi else "?", "tier": tier}))
                    log(f"  {tier}: {len(t)} מניות")
                    break
    except Exception as e:  # noqa: BLE001
        log(f"שגיאה בטעינת רשימות המדדים ({e})")
    if not frames:
        log("משתמש ביקום גיבוי (רק לבדיקה)")
        return pd.DataFrame({"ticker": FALLBACK, "name": FALLBACK, "sector": "?", "sub_industry": "?", "tier": "large"})
    df = pd.concat(frames).drop_duplicates("ticker")
    key = df["name"].str.replace(r"\s*\(Class [A-C]\)", "", regex=True).str.strip()
    df = df[~key.duplicated()].reset_index(drop=True)
    log(f"יקום: {len(df)} מניות (גדולות+בינוניות+קטנות) - מוטה-שורדים")
    return df


# ---------------------------------------------------------------- מחירים
def download_prices(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    all_t = sorted(set(tickers) | {"SPY"})
    closes, vols = [], []
    for i in range(0, len(all_t), 150):
        chunk = all_t[i:i + 150]
        log(f"מוריד מחירים {i + 1}-{i + len(chunk)} מתוך {len(all_t)}...")
        for attempt in range(3):
            try:
                d = yf.download(chunk, start=start, auto_adjust=True, progress=False,
                                group_by="column", threads=True)
                c, v = d["Close"], d["Volume"]
                if isinstance(c, pd.Series):
                    c, v = c.to_frame(chunk[0]), v.to_frame(chunk[0])
                closes.append(c)
                vols.append(v)
                break
            except Exception as e:  # noqa: BLE001
                log(f"  ניסיון {attempt + 1} נכשל: {e}")
                time.sleep(10)
    close = pd.concat(closes, axis=1)
    vol = pd.concat(vols, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    vol = vol.loc[:, ~vol.columns.duplicated()].reindex(columns=close.columns)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    close = close.dropna(axis=1, how="all")
    log(f"מחירים: {close.shape[1]} טיקרים עם נתונים, {len(close)} ימי מסחר")
    return close, vol.reindex(columns=close.columns)


class Market:
    """כל המטריצות שהסימולציה צריכה, כ-numpy, מחושבות פעם אחת. אין בהן מידע מהעתיד."""

    def __init__(self, close: pd.DataFrame, vol: pd.DataFrame):
        self.dates = close.index
        self.spy = close["SPY"].to_numpy(float)
        stocks = [c for c in close.columns if c != "SPY"]
        self.tickers = stocks
        c, v = close[stocks], vol[stocks]
        r = c.pct_change(fill_method=None)
        bad = (r > 2.0) | (r < -0.9)                   # לקח NFE: ארטיפקט פעולה תאגידית
        self.n_bad = int(bad.values.sum())
        self.bad = bad.to_numpy()
        recent_bad = bad.astype(float).rolling(260, min_periods=1).max().fillna(0).astype(bool)
        sma20 = c.rolling(20, min_periods=20).mean()
        above = c > sma20
        cross_up = above & ~above.shift(1, fill_value=False)
        high252 = c.rolling(252, min_periods=200).max()
        self.dd = (c / high252 - 1).to_numpy(float)
        strength = c.shift(252) / c.shift(1008) - 1
        self.strength = strength.to_numpy(float)
        self.strong = (strength > STRONG_3Y).to_numpy()
        liquid = (c >= MIN_PRICE) & ((c * v).rolling(60, min_periods=50).mean() >= MIN_DOLLAR_VOL)
        self.liquid = liquid.to_numpy()
        self.ok = self.liquid & ~recent_bad.to_numpy()
        self.cross_up = cross_up.to_numpy()
        spy_s = close["SPY"]
        self.regime = (spy_s > spy_s.rolling(200, min_periods=200).mean()).to_numpy()
        self.c = c.to_numpy(float)
        self.vol = v.to_numpy(float)

    def idx(self, date) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(date)))


# ---------------------------------------------------------------- סימולטור תיק
def simulate(m: Market, entry: np.ndarray, start: str, end: str | None, target: float | None,
             stop: float | None, max_days: int | None, max_pos: int = MAX_POSITIONS,
             use_regime: bool = False, rank: np.ndarray | None = None) -> dict:
    """כניסה בסגירה של היום *שאחרי* האות. יציאה בסגירה כשמתקיים תנאי.
    משקל שווה: כל עסקה = 1/max_pos מההון בזמן הכניסה (או מה שנשאר במזומן)."""
    i0 = max(m.idx(start), 1)
    i1 = len(m.dates) if end is None else m.idx(end)
    cash, pos, trades = 1.0, {}, []
    eq = np.empty(i1 - i0)
    invested = np.empty(i1 - i0)
    last_px = {}
    for k, t in enumerate(range(i0, i1)):
        row = m.c[t]
        # --- יציאות
        for j in list(pos):
            p = pos[j]
            px = row[j]
            reason = None
            if m.bad[t, j] or np.isnan(px):
                px, reason = last_px.get(j, p["entry"]), "data"
            elif target is not None and px >= p["entry"] * (1 + target):
                reason = "target"
            elif stop is not None and px <= p["entry"] * (1 + stop):
                reason = "stop"
            elif max_days is not None and t - p["t"] >= max_days:
                reason = "time"
            if reason:
                cash += p["sh"] * px * (1 - COST_PER_SIDE)
                trades.append({"ticker": m.tickers[j], "entry_date": m.dates[p["t"]], "exit_date": m.dates[t],
                               "ret": px / p["entry"] * (1 - COST_PER_SIDE) ** 2 - 1, "days": t - p["t"],
                               "reason": reason})
                del pos[j]
            else:
                last_px[j] = px
        # --- כניסות (האות של אתמול, מחיר היום)
        slots = max_pos - len(pos)
        if slots > 0 and (not use_regime or m.regime[t - 1]):
            cand = np.flatnonzero(entry[t - 1] & ~np.isnan(row))
            cand = [j for j in cand if j not in pos]
            if cand:
                if rank is not None:
                    cand.sort(key=lambda j: -np.nan_to_num(rank[t - 1, j], nan=-1e9))
                equity = cash + sum(p["sh"] * last_px.get(j, p["entry"]) for j, p in pos.items())
                for j in cand[:slots]:
                    alloc = min(equity / max_pos, cash)
                    if alloc <= 1e-9:
                        break
                    cash -= alloc
                    pos[j] = {"t": t, "entry": row[j], "sh": alloc * (1 - COST_PER_SIDE) / row[j]}
                    last_px[j] = row[j]
        val = sum(p["sh"] * last_px.get(j, p["entry"]) for j, p in pos.items())
        eq[k] = cash + val
        invested[k] = val / eq[k] if eq[k] > 0 else 0
    for j, p in pos.items():   # פוזיציות פתוחות בסוף - מסומנות, לא נמכרות
        trades.append({"ticker": m.tickers[j], "entry_date": m.dates[p["t"]], "exit_date": pd.NaT,
                       "ret": last_px.get(j, p["entry"]) / p["entry"] - 1, "days": i1 - 1 - p["t"],
                       "reason": "open"})
    return {"equity": pd.Series(eq, index=m.dates[i0:i1]), "invested": invested,
            "trades": pd.DataFrame(trades), "i0": i0, "i1": i1}


def metrics(m: Market, res: dict) -> dict:
    eq = res["equity"]
    if len(eq) < 2:
        return {}
    yrs = len(eq) / 252
    spy = m.spy[res["i0"]:res["i1"]]
    spy_eq = spy / spy[0]
    tr = res["trades"]
    closed = tr[tr["reason"] != "open"] if not tr.empty else tr

    def dd(x):
        x = np.asarray(x)
        return float((x / np.maximum.accumulate(x) - 1).min())

    out = {
        "cagr": eq.iloc[-1] ** (1 / yrs) - 1, "spy_cagr": spy_eq[-1] ** (1 / yrs) - 1,
        "max_dd": dd(eq.values), "spy_max_dd": dd(spy_eq),
        "trades": len(closed), "open_at_end": int((tr["reason"] == "open").sum()) if not tr.empty else 0,
        "avg_invested": float(np.mean(res["invested"])),
    }
    out["calmar"] = out["cagr"] / abs(out["max_dd"]) if out["max_dd"] < 0 else np.nan
    out["spy_calmar"] = out["spy_cagr"] / abs(out["spy_max_dd"]) if out["spy_max_dd"] < 0 else np.nan
    if len(closed):
        out.update({
            "win_rate": float((closed["ret"] > 0).mean()), "avg_trade": float(closed["ret"].mean()),
            "avg_days": float(closed["days"].mean()),
            "pct_target": float((closed["reason"] == "target").mean()),
            "pct_stop": float((closed["reason"] == "stop").mean()),
            "pct_time": float((closed["reason"] == "time").mean()),
            "worst_trade": float(closed["ret"].min()),
        })
    if not tr.empty:
        stuck = tr[(tr["reason"] == "open") & (tr["days"] > 504)]
        out["stuck_2y_plus"] = len(stuck)
    return out


# ---------------------------------------------------------------- A: קניית ירידות
def dip_entry(m: Market, band) -> np.ndarray:
    lo, hi = band
    with np.errstate(invalid="ignore"):
        in_band = (m.dd >= lo) & (m.dd <= hi)
    return m.strong & in_band & m.cross_up & m.ok


def run_dip(m: Market) -> dict:
    rows = []
    for band in GRID_BANDS:
        e = dip_entry(m, band)
        for tg in GRID_TARGETS:
            for st in GRID_STOPS:
                for md in GRID_TIME:
                    for period, (s, en) in {"in": (SIM_START, SPLIT), "out": (SPLIT, None)}.items():
                        r = simulate(m, e, s, en, tg, st, md, rank=m.strength)
                        mt = metrics(m, r)
                        rows.append({"band": f"{int(band[1] * 100)}..{int(band[0] * 100)}%", "target": tg,
                                     "stop": st, "max_days": md, "period": period, **mt})
        log(f"  רצועת ירידה {band} הושלמה")
    grid = pd.DataFrame(rows)
    ins = grid[(grid["period"] == "in") & (grid["trades"] >= MIN_TRADES_IN_SAMPLE)]
    best = ins.sort_values("calmar", ascending=False).head(1)
    sel = None
    if not best.empty:
        b = best.iloc[0]
        # מפתח שמטפל ב-None/NaN (ב-pandas 3, astype(str) משאיר NaN כחסר - זה היה הבאג ב-v1.0.0)
        def _k(x):
            return "none" if x is None or (isinstance(x, float) and np.isnan(x)) else str(float(x))
        key = grid["band"] + "|" + grid["target"].map(_k) + "|" + grid["stop"].map(_k) + "|" + grid["max_days"].map(_k)
        bk = f"{b['band']}|{_k(b['target'])}|{_k(b['stop'])}|{_k(b['max_days'])}"
        sel = grid[key == bk]
    # ברירת המחדל של Tomer - קבועה מראש
    e = dip_entry(m, TOMER_DEFAULT["band"])
    tom = []
    for period, (s, en) in {"in": (SIM_START, SPLIT), "out": (SPLIT, None)}.items():
        for reg in (False, True):
            r = simulate(m, e, s, en, TOMER_DEFAULT["target"], TOMER_DEFAULT["stop"],
                         TOMER_DEFAULT["max_days"], use_regime=reg, rank=m.strength)
            tom.append({"variant": "Tomer" + (" + מסנן שוק" if reg else ""), "period": period, **metrics(m, r)})
            if period == "out" and not reg:
                tom_trades = r["trades"]
    # פלצבו: אותן מניות חזקות, אותם כללי יציאה, אבל כניסה ב*זמן אקראי* (בלי תנאי ירידה).
    # זה מנטרל את הטיית השורדים: אם קניית ירידות לא עוקפת כניסה אקראית באותו יקום -
    # התשואה באה מזה שהמניות שרדו, לא מהתזמון.
    base_mask = m.strong & m.ok
    rate = e.sum() / max(base_mask.sum(), 1)
    plac = []
    for seed in range(PLACEBO_SEEDS):
        rnd = base_mask & (np.random.default_rng(seed).random(base_mask.shape) < rate)
        for period, (s, en) in {"in": (SIM_START, SPLIT), "out": (SPLIT, None)}.items():
            r = simulate(m, rnd, s, en, TOMER_DEFAULT["target"], TOMER_DEFAULT["stop"], TOMER_DEFAULT["max_days"],
                         rank=m.strength)
            plac.append({"seed": seed, "period": period, **metrics(m, r)})
    plac = pd.DataFrame(plac)
    pl = plac.groupby("period")[["cagr", "max_dd", "calmar", "win_rate", "avg_days"]].mean().reset_index()
    pl.insert(0, "variant", "פלצבו: חזקה, זמן אקראי")
    tom_df = pd.concat([pd.DataFrame(tom), pl], ignore_index=True)

    out_grid = grid[grid["period"] == "out"]
    robust = {
        "combos": len(out_grid),
        "beat_spy_cagr": int((out_grid["cagr"] > out_grid["spy_cagr"]).sum()),
        "beat_spy_calmar": int((out_grid["calmar"] > out_grid["spy_calmar"]).sum()),
    }
    # תקופות משבר בנפרד - האם קניית ירידות שרדה?
    crises = []
    for name, s, en in [("בועת הדוט-קום 2000-2003", "2000-01-01", "2003-12-31"),
                        ("המשבר הפיננסי 2007-2009", "2007-10-01", "2009-12-31"),
                        ("2022", "2022-01-01", "2022-12-31")]:
        r = simulate(m, e, s, en, TOMER_DEFAULT["target"], TOMER_DEFAULT["stop"], TOMER_DEFAULT["max_days"],
                     rank=m.strength)
        mt = metrics(m, r)
        eq = r["equity"]
        spy = m.spy[r["i0"]:r["i1"]]
        crises.append({"period": name, "total_ret": eq.iloc[-1] - 1 if len(eq) else np.nan,
                       "spy_total_ret": spy[-1] / spy[0] - 1 if len(spy) else np.nan,
                       "max_dd": mt.get("max_dd"), "spy_max_dd": mt.get("spy_max_dd")})
    return {"grid": grid, "selected": sel, "tomer": tom_df, "robust": robust,
            "crises": pd.DataFrame(crises), "tomer_trades": tom_trades}


# ---------------------------------------------------------------- B: אנשי פנים
def _read_tsv(z: zipfile.ZipFile, name: str) -> pd.DataFrame | None:
    for n in z.namelist():
        if n.upper().endswith(name.upper()):
            with z.open(n) as fh:
                df = pd.read_csv(fh, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                                 on_bad_lines="skip", low_memory=False, encoding_errors="replace")
            df.columns = [c.upper().strip() for c in df.columns]
            return df
    return None


def parse_insider_zip(raw: bytes) -> pd.DataFrame:
    """מחזיר רק קניות בשוק הפתוח (קוד P) של דירקטורים/נושאי משרה, מטופס 4."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    sub, own, tx = _read_tsv(z, "SUBMISSION.tsv"), _read_tsv(z, "REPORTINGOWNER.tsv"), _read_tsv(z, "NONDERIV_TRANS.tsv")
    need = {"sub": ["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"],
            "own": ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"],
            "tx": ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"]}
    for nm, df in (("sub", sub), ("own", own), ("tx", tx)):
        if df is None or any(c not in df.columns for c in need[nm]):
            log(f"  מבנה קובץ לא צפוי ({nm}): {None if df is None else list(df.columns)[:15]}")
            return pd.DataFrame()
    if "DOCUMENT_TYPE" in sub.columns:
        sub = sub[sub["DOCUMENT_TYPE"].astype(str).str.startswith("4")]
    tx = tx[tx["TRANS_CODE"].astype(str).str.strip() == "P"].copy()
    if "TRANS_ACQUIRED_DISP_CD" in tx.columns:
        tx = tx[tx["TRANS_ACQUIRED_DISP_CD"].astype(str).str.strip() == "A"]
    tx["value"] = pd.to_numeric(tx["TRANS_SHARES"], errors="coerce") * pd.to_numeric(tx["TRANS_PRICEPERSHARE"], errors="coerce")
    tx = tx.groupby("ACCESSION_NUMBER", as_index=False).agg(value=("value", "sum"))
    # v2 (17.10.2026, parity check): a joint filing lists several reporting
    # owners (a fund + its people). v1 kept the "first" row, but the TSV row
    # order isn't stable, so one group filing 3 times became 3 "insiders".
    # Now: keep ALL owners of the filing (owners_all); the filing qualifies if
    # any owner is a director/officer; canonical_owners() below then gives
    # filings of the same group one identity. Same rule as insider_live.py.
    own = own.copy()
    own["RPTOWNERCIK"] = own["RPTOWNERCIK"].astype(str).str.split(".").str[0].str.lstrip("0")
    own["_q"] = own["RPTOWNER_RELATIONSHIP"].astype(str).str.contains("Director|Officer", case=False, regex=True)
    if "RPTOWNER_TITLE" not in own.columns:
        own["RPTOWNER_TITLE"] = ""
    agg = own.groupby("ACCESSION_NUMBER").apply(lambda g: pd.Series({
        "qual": bool(g["_q"].any()),
        "owners_all": "|".join(sorted(set(g["RPTOWNERCIK"]))),
        "RPTOWNER_TITLE": " / ".join(t for t in g.loc[g["_q"], "RPTOWNER_TITLE"].fillna("").astype(str) if t),
    })).reset_index()
    agg = agg[agg["qual"]]
    df = tx.merge(sub[["ACCESSION_NUMBER", "FILING_DATE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]], on="ACCESSION_NUMBER") \
           .merge(agg, on="ACCESSION_NUMBER")
    df["RPTOWNERCIK"] = df["owners_all"].str.split("|").str[0]
    df = df[df["value"] >= MIN_TX_VALUE]
    out = pd.DataFrame({
        "cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"),
        "symbol": df["ISSUERTRADINGSYMBOL"].astype(str).str.upper().str.strip(),
        "owner": df["RPTOWNERCIK"].astype(str),
        "owners_all": df["owners_all"].astype(str),
        "filing_date": pd.to_datetime(df["FILING_DATE"], errors="coerce", format="mixed"),
        "value": df["value"],
        "title": df.get("RPTOWNER_TITLE", pd.Series("", index=df.index)).fillna("").astype(str),
    })
    return out.dropna(subset=["filing_date"])


def fetch_insider_purchases(ua: str, first_year: int = INSIDER_FIRST_YEAR) -> pd.DataFrame:
    import requests
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
    frames = []
    now = datetime.now()
    for y in range(first_year, now.year + 1):
        for q in range(1, 5):
            if y == now.year and (q - 1) * 3 + 1 > now.month:
                break
            cache = os.path.join(CACHE_DIR, f"insider_P_v2_{y}q{q}.csv")   # v2: joint filings kept whole
            if os.path.exists(cache):
                frames.append(pd.read_csv(cache, parse_dates=["filing_date"]))
                continue
            url = f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"
            try:
                resp = requests.get(url, headers=headers, timeout=180)
                if resp.status_code != 200:
                    log(f"  {y}q{q}: HTTP {resp.status_code} - מדלג")
                    continue
                df = parse_insider_zip(resp.content)
                df.to_csv(cache, index=False)
                frames.append(df)
                log(f"  {y}q{q}: {len(df):,} קניות של אנשי פנים")
            except Exception as e:  # noqa: BLE001
                log(f"  {y}q{q}: שגיאה {e}")
            time.sleep(0.5)
    if not frames:
        return pd.DataFrame()
    return canonical_owners(pd.concat(frames, ignore_index=True), key="cik")


def canonical_owners(df: pd.DataFrame, key: str = "cik", window_days: int = 30) -> pd.DataFrame:
    """One identity per buying group: per issuer, in filing-date order, a
    filing whose owners overlap a filing of the last `window_days` days takes
    that filing's identity; otherwise its identity is its lowest owner CIK.
    So a fund + director co-filing three times counts once, while two
    directors who each file alone stay two insiders. Shared with
    insider_live.py so live == research."""
    if df.empty or "owners_all" not in df.columns:
        return df
    df = df.sort_values([key, "filing_date"]).copy()
    ids = []
    last = {}            # (issuer, owner cik) -> (identity, date)
    for issuer, d, owners in zip(df[key], df["filing_date"], df["owners_all"].astype(str)):
        os_ = [o for o in owners.split("|") if o]
        hit = None
        for o in os_:
            prev = last.get((issuer, o))
            if prev and (d - prev[1]).days <= window_days:
                hit = prev[0] if hit is None else min(hit, prev[0])
        ident = hit if hit is not None else (min(os_, key=lambda x: int(x) if x.isdigit() else 0) if os_ else "")
        for o in os_:
            last[(issuer, o)] = (ident, d)
        ids.append(ident)
    df["owner"] = ids
    return df


def map_tickers(purch: pd.DataFrame, ua: str | None) -> pd.DataFrame:
    """CIK -> הטיקר הנוכחי (היקום בנוי מטיקרים נוכחיים); גיבוי: הסימול שדווח."""
    cikmap = {}
    if ua:
        try:
            import requests
            js = requests.get("https://www.sec.gov/files/company_tickers.json",
                              headers={"User-Agent": ua}, timeout=60).json()
            cikmap = {int(v["cik_str"]): v["ticker"].upper().replace(".", "-") for v in js.values()}
        except Exception as e:  # noqa: BLE001
            log(f"  מיפוי CIK נכשל ({e}) - משתמש בסימול המדווח")
    p = purch.copy()
    p["ticker"] = [cikmap.get(int(c), s) if pd.notna(c) else s for c, s in zip(p["cik"], p["symbol"])]
    return p


def find_clusters(purch: pd.DataFrame) -> pd.DataFrame:
    """אשכול: לפחות MIN_INSIDERS אנשי פנים שונים, סכום MIN_CLUSTER_VALUE+, תוך CLUSTER_DAYS
    (לפי תאריך דיווח). אחרי אות - שקט של COOLDOWN_DAYS לאותה מניה."""
    rows = []
    win, cool = pd.Timedelta(days=CLUSTER_DAYS), pd.Timedelta(days=COOLDOWN_DAYS)
    for t, g in purch.sort_values("filing_date").groupby("ticker"):
        last = None
        for i in range(len(g)):
            d = g["filing_date"].iloc[i]
            if last is not None and d - last < cool:
                continue
            w = g[(g["filing_date"] > d - win) & (g["filing_date"] <= d)]
            n_ins = w["owner"].nunique()
            if n_ins >= MIN_INSIDERS and w["value"].sum() >= MIN_CLUSTER_VALUE:
                titles = " ".join(w["title"].fillna("").astype(str)).upper()
                rows.append({"ticker": t, "signal_date": d, "insiders": n_ins, "value": w["value"].sum(),
                             "ceo_cfo": bool(re.search(r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL", titles))})
                last = d
    return pd.DataFrame(rows)


def insider_event_study(m: Market, clusters: pd.DataFrame, tiers: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    tix = {t: j for j, t in enumerate(m.tickers)}
    rows = []
    for e in clusters.itertuples():
        j = tix.get(e.ticker)
        if j is None:
            continue
        pos = int(m.dates.searchsorted(e.signal_date, side="right"))   # יום המסחר הראשון *אחרי* הדיווח
        if pos < 600 or pos >= len(m.dates):
            continue
        px = m.c[:, j]
        if np.isnan(px[pos]):
            continue
        rec = {"ticker": e.ticker, "signal_date": e.signal_date, "insiders": e.insiders, "value": e.value,
               "ceo_cfo": e.ceo_cfo, "tier": tiers.get(e.ticker, "?"),
               "after_drop_20": bool(m.dd[pos - 1, j] <= -0.20) if not np.isnan(m.dd[pos - 1, j]) else False,
               "strong": bool(m.strong[pos - 1, j])}
        for h in INSIDER_HORIZONS:
            if pos + h >= len(m.dates) or np.isnan(px[pos + h]):
                rec[f"ex_{h}"] = np.nan
                continue
            ex = px[pos + h] / px[pos] - m.spy[pos + h] / m.spy[pos]
            # בסיס: התשואה העודפת הרגילה של המניה באותו אורך חלון, בשנתיים שלפני (בלי עתיד)
            lo = max(0, pos - 1 - 504)
            a = px[lo:pos - h]
            b = px[lo + h:pos]
            sa, sb = m.spy[lo:pos - h], m.spy[lo + h:pos]
            base = np.nanmean(b / a - sb / sa) if len(a) > 60 else np.nan
            rec[f"ex_{h}"] = ex
            rec[f"adj_{h}"] = ex - base if not np.isnan(base) else np.nan
        rows.append(rec)
    es = pd.DataFrame(rows)
    if es.empty:
        return es, pd.DataFrame()
    es["period"] = np.where(es["signal_date"] < pd.Timestamp(SPLIT), f"{INSIDER_FIRST_YEAR}-2015", "2016+")
    cuts = [("הכל", lambda d: d.index == d.index), ("אחרי ירידה 20%+", lambda d: d["after_drop_20"]),
            ("בלי ירידה", lambda d: ~d["after_drop_20"]), ("מנכ\"ל/סמנכ\"ל כספים", lambda d: d["ceo_cfo"]),
            ("3+ אנשי פנים", lambda d: d["insiders"] >= 3),
            ("גדולות", lambda d: d["tier"] == "large"), ("בינוניות", lambda d: d["tier"] == "mid"),
            ("קטנות", lambda d: d["tier"] == "small")]
    out = []
    for period, g in [("הכל", es)] + list(es.groupby("period")):
        for name, f in cuts:
            h = g[f(g)]
            rec = {"period": period, "cut": name, "events": len(h)}
            for hz in INSIDER_HORIZONS:
                x = h.get(f"adj_{hz}", pd.Series(dtype=float)).dropna()
                rec[f"adj_{hz}_mean"] = x.mean() if len(x) else np.nan
                rec[f"adj_{hz}_t"] = x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
                rec[f"hit_{hz}"] = float((x > 0).mean()) if len(x) else np.nan
                y = h.get(f"ex_{hz}", pd.Series(dtype=float)).dropna()
                rec[f"ex_{hz}_mean"], rec[f"ex_{hz}_std"], rec[f"ex_{hz}_n"] = y.mean(), y.std(ddof=1), len(y)
            out.append(rec)
    return es, pd.DataFrame(out)


def placebo_clusters(clusters: pd.DataFrame, m: Market, k: int = 5, seed: int = 11) -> pd.DataFrame:
    """אותן מניות, תאריכים אקראיים באותה תקופה. משווים אליהם במקום ל'בסיס' של המניה:
    כך גם מניה שפשוט הייתה במגמה/שרדה לא נראית כמו אות."""
    rng = np.random.default_rng(seed)
    d = m.dates[(m.dates >= pd.Timestamp(f"{INSIDER_FIRST_YEAR}-01-01"))]
    rows = []
    split = pd.Timestamp(SPLIT)
    for e in clusters.itertuples():
        pool = d[d < split] if e.signal_date < split else d[d >= split]
        for dd in rng.choice(pool, k):
            rows.append({"ticker": e.ticker, "signal_date": pd.Timestamp(dd), "insiders": 0, "value": 0, "ceo_cfo": e.ceo_cfo})
    return pd.DataFrame(rows)


def vs_placebo(real: pd.DataFrame, plac: pd.DataFrame) -> pd.DataFrame:
    """הפרש התשואה העודפת (מול SPY) בין אירועים אמיתיים לאקראיים, לכל חיתוך, עם t של Welch.
    חיתוכי 'מנכ"ל' ו-'3+' בפלצבו = אותן מניות בזמן אקראי (אין שם אנשי פנים)."""
    pl = plac.set_index(["period", "cut"])
    alt = {"3+ אנשי פנים": "הכל"}
    out = real.copy()
    for hz in INSIDER_HORIZONS:
        diffs, ts = [], []
        for r in real.itertuples():
            key = (r.period, alt.get(r.cut, r.cut))
            if key not in pl.index:
                diffs.append(np.nan); ts.append(np.nan); continue
            q = pl.loc[key]
            m1, s1, n1 = getattr(r, f"ex_{hz}_mean"), getattr(r, f"ex_{hz}_std"), getattr(r, f"ex_{hz}_n")
            m2, s2, n2 = q[f"ex_{hz}_mean"], q[f"ex_{hz}_std"], q[f"ex_{hz}_n"]
            if n1 > 2 and n2 > 2:
                se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
                diffs.append(m1 - m2); ts.append((m1 - m2) / se if se > 0 else np.nan)
            else:
                diffs.append(np.nan); ts.append(np.nan)
        out[f"vsp_{hz}"], out[f"vsp_{hz}_t"] = diffs, ts
    return out


def insider_signal_matrix(m: Market, clusters: pd.DataFrame, recent_days: int = 1) -> np.ndarray:
    """True ביום המסחר הראשון שאחרי הדיווח (ו-recent_days-1 הימים שאחריו). הסימולטור נכנס למחרת."""
    sig = np.zeros_like(m.ok, dtype=bool)
    tix = {t: j for j, t in enumerate(m.tickers)}
    for e in clusters.itertuples():
        j = tix.get(e.ticker)
        if j is None:
            continue
        pos = int(m.dates.searchsorted(e.signal_date, side="right"))
        sig[pos:min(pos + recent_days, len(m.dates)), j] = True
    return sig


def run_insider_portfolios(m: Market, clusters: pd.DataFrame, dip_sel: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    start = f"{INSIDER_FIRST_YEAR}-06-01"
    # B: כל אשכול, החזקה 6 חודשים
    sig1 = insider_signal_matrix(m, clusters, 1) & m.ok
    # הסימולטור נכנס ביום t לפי entry[t-1]; האות מסומן ביום הראשון אחרי הדיווח -> כניסה ביום שאחריו
    for period, (s, en) in {"2006-2015": (start, SPLIT), "2016+": (SPLIT, None)}.items():
        r = simulate(m, sig1, s, en, None, None, INSIDER_HOLD, max_pos=INSIDER_MAX_POS)
        rows.append({"strategy": "B: כל אשכול, 6 חודשים", "period": period, **metrics(m, r)})
    # C: אשכול אחרי ירידה במניה חזקה + כללי היציאה של Tomer (ושל השילוב הנבחר ב-A אם יש)
    recent = insider_signal_matrix(m, clusters, COMBO_RECENT_DAYS)
    with np.errstate(invalid="ignore"):
        combo = recent & m.strong & (m.dd <= -0.20) & m.ok
    exits = [("C: אשכול + ירידה, יציאת Tomer (+30%)", TOMER_DEFAULT["target"], TOMER_DEFAULT["stop"], TOMER_DEFAULT["max_days"])]
    if dip_sel is not None and not dip_sel.empty:
        b = dip_sel.iloc[0]
        st = None if pd.isna(b["stop"]) else float(b["stop"])
        md = None if pd.isna(b["max_days"]) else int(b["max_days"])
        exits.append((f"C: אשכול + ירידה, יציאה נבחרת ({b['target']:.0%}/{st}/{md})", float(b["target"]), st, md))
    with np.errstate(invalid="ignore"):
        dip_only = m.strong & (m.dd <= -0.20) & (m.dd >= -0.60) & m.cross_up & m.ok
    for period, (s, en) in {"2006-2015": (start, SPLIT), "2016+": (SPLIT, None)}.items():
        r = simulate(m, dip_only, s, en, TOMER_DEFAULT["target"], TOMER_DEFAULT["stop"], TOMER_DEFAULT["max_days"],
                     rank=m.strength)
        rows.append({"strategy": "השוואה: ירידה בלי אנשי פנים, יציאת Tomer", "period": period, **metrics(m, r)})
    for name, tg, st, md in exits:
        for period, (s, en) in {"2006-2015": (start, SPLIT), "2016+": (SPLIT, None)}.items():
            r = simulate(m, combo, s, en, tg, st, md, rank=m.strength)
            rows.append({"strategy": name, "period": period, **metrics(m, r)})
    return pd.DataFrame(rows)




# ---------------------------------------------------------------- B2: תיק אנשי פנים משופר (הוצע *אחרי* ש-B נראה)
# ⚠️ פוסט-הוק: הכללים נבחרו אחרי שראינו את חיתוכי B (מנכ"ל/סמנכ"ל, 3+, שנה). לכן התוצאה ההיסטורית
# אופטימית מטבעה, והמבחן האמיתי הוא מעקב קדימה. החיתוכים עצמם הוגדרו מראש בקובץ הרעיון.
B2_HOLD = 252
B2_WINDOW = 5               # אות "חי" 5 ימי מסחר - כך פוזיציה שמתפנה תוך שבוע עדיין תופסת אותו
B2_MAX_POS = 20


def insider_score(c: pd.DataFrame) -> pd.Series:
    """ציון חוזק: מנכ"ל/סמנכ"ל כספים = 2, כל איש פנים = 1, ותוספת קטנה לפי גודל הקנייה."""
    return 2 * c["ceo_cfo"].astype(float) + c["insiders"].astype(float) + np.log10(c["value"].clip(lower=1)) / 10


def insider_rank_matrix(m: Market, clusters: pd.DataFrame, days: int) -> tuple[np.ndarray, np.ndarray]:
    sig = np.zeros_like(m.ok, dtype=bool)
    rank = np.full(m.ok.shape, np.nan)
    tix = {t: j for j, t in enumerate(m.tickers)}
    sc = insider_score(clusters)
    for e, v in zip(clusters.itertuples(), sc):
        j = tix.get(e.ticker)
        if j is None:
            continue
        pos = int(m.dates.searchsorted(e.signal_date, side="right"))
        sl = slice(pos, min(pos + days, len(m.dates)))
        sig[sl, j] = True
        rank[sl, j] = np.fmax(rank[sl, j], v)
    return sig & m.ok, rank


def run_b2(m: Market, clusters: pd.DataFrame) -> pd.DataFrame:
    rows = []
    start = f"{INSIDER_FIRST_YEAR}-06-01"
    strong = clusters[clusters["ceo_cfo"].astype(bool) | (clusters["insiders"] >= 3)]
    variants = [("B2: כל אשכול, עדיפות לחזקים, שנה", clusters),
                ("B2: רק חזקים (מנכ\"ל/סמנכ\"ל או 3+), שנה", strong)]
    for name, cl in variants:
        sig, rank = insider_rank_matrix(m, cl, B2_WINDOW)
        for period, (s, en) in {"2006-2015": (start, SPLIT), "2016+": (SPLIT, None)}.items():
            for reg in (False, True):
                r = simulate(m, sig, s, en, None, None, B2_HOLD, max_pos=B2_MAX_POS, use_regime=reg, rank=rank)
                rows.append({"strategy": name + (" + מסנן שוק" if reg else ""), "period": period, **metrics(m, r)})
    return pd.DataFrame(rows)


def live_signals(m: Market, clusters: pd.DataFrame, tiers: pd.Series, days: int = 45) -> pd.DataFrame:
    """אותות מהשבועות האחרונים - נקודת ההתחלה למעקב קדימה במעבדה (לא המלצה)."""
    if clusters.empty:
        return pd.DataFrame()
    cut = m.dates[-1] - pd.Timedelta(days=days)
    c = clusters[clusters["signal_date"] >= cut].copy()
    if c.empty:
        return c
    c["score"] = insider_score(c).round(2)
    c["strong"] = c["ceo_cfo"].astype(bool) | (c["insiders"] >= 3)
    c["tier"] = c["ticker"].map(tiers).fillna("מחוץ ליקום")
    tix = {t: j for j, t in enumerate(m.tickers)}
    c["dd_from_high"] = [m.dd[-1, tix[t]] if t in tix else np.nan for t in c["ticker"]]
    c["passes_liquidity"] = [bool(m.ok[-1, tix[t]]) if t in tix else False for t in c["ticker"]]
    return c.sort_values(["strong", "score"], ascending=False)

# ---------------------------------------------------------------- D: שרשרת אספקה / התפשטות בענף
LEADER_JUMP = 0.08          # D1: המובילה בתת-ענף קפצה 8%+ מעל SPY ביום אחד
NVDA_JUMP = 0.05            # D2: אנבידיה קפצה 5%+ מעל SPY
CHAIN_HORIZONS = [5, 10, 21]
CHAIN_COOLDOWN = 21         # ימי מסחר בין אירועים באותה קבוצה
CHAIN_PLACEBO_K = 5
MIN_FOLLOWERS = 3
# D2: מפה ידנית של שרשרת ה-AI. נבנתה *היום* - ולכן יש בה ידיעה בדיעבד מי הצליח.
# משמשת רק לשאלה אם יש *עיכוב* (הספקים זזים אחרי אנבידיה ולא באותו יום), מול ימים אקראיים.
AI_LEADER = "NVDA"
AI_SUPPLIERS = {
    "שבבים וזיכרון": ["TSM", "MU", "AVGO", "MRVL", "AMD"],
    "שרתים": ["SMCI", "DELL", "HPE"],
    "קירור וחשמל": ["VRT", "ETN", "CEG", "VST", "MPWR"],
    "תקשורת ואופטיקה": ["ANET", "COHR", "LITE", "CIEN", "APH"],
}
AI_EXTRA_TICKERS = sorted({AI_LEADER} | {t for v in AI_SUPPLIERS.values() for t in v})


def _basket_ex(m: Market, cols: list[int], a: int, b: int) -> float:
    """תשואה עודפת ממוצעת (משקל שווה) של סל מניות מסגירת a לסגירת b, מול SPY."""
    if b >= len(m.dates) or a < 0:
        return np.nan
    pa, pb = m.c[a, cols], m.c[b, cols]
    ok = ~np.isnan(pa) & ~np.isnan(pb) & (pa > 0) & ~m.bad[a:b + 1][:, cols].any(axis=0)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(pb[ok] / pa[ok] - 1) - (m.spy[b] / m.spy[a] - 1))


def _measure(m: Market, cols: list[int], t: int) -> dict:
    """t = יום הקפיצה של המובילה. כניסה בסגירה של t+1 (יום אחרי), כמו משקיע שקרא את החדשות."""
    rec = {"same_day": _basket_ex(m, cols, t - 1, t), "next_day": _basket_ex(m, cols, t, t + 1)}
    for h in CHAIN_HORIZONS:
        rec[f"after_{h}"] = _basket_ex(m, cols, t + 1, t + 1 + h)
    return rec


def chain_industry(m: Market, uni: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D1: בכל תת-ענף, המובילה = המחזור הדולרי הגבוה ביותר (60 יום, נכון לאתמול).
    כשהיא קופצת LEADER_JUMP+ מעל SPY - מה עושות שאר המניות בתת-הענף בימים שאחרי?"""
    tix = {t: j for j, t in enumerate(m.tickers)}
    dv = pd.DataFrame(m.c * np.nan_to_num(m.vol), index=m.dates).rolling(60, min_periods=50).mean().shift(1).to_numpy()
    r = np.vstack([np.full(m.c.shape[1], np.nan), m.c[1:] / m.c[:-1] - 1])
    spy_r = np.concatenate([[np.nan], m.spy[1:] / m.spy[:-1] - 1])
    rng = np.random.default_rng(21)
    real, plac = [], []
    split_i = m.idx(SPLIT)
    i_start = m.idx(SIM_START)
    for sub, g in uni.groupby("sub_industry"):
        cols = [tix[t] for t in g["ticker"] if t in tix]
        if len(cols) < MIN_FOLLOWERS + 1:
            continue
        cols = np.array(cols)
        sub_dv = np.where(np.isnan(dv[:, cols]), -1, dv[:, cols])
        leader_pos = sub_dv.argmax(axis=1)
        has = sub_dv.max(axis=1) > 0
        lead_r = r[np.arange(len(m.dates)), cols[leader_pos]]
        ab = lead_r - spy_r
        last = -10 ** 9
        for t in np.flatnonzero(has & (ab >= LEADER_JUMP)):
            if t < i_start or t + 1 + max(CHAIN_HORIZONS) >= len(m.dates) or t - last < CHAIN_COOLDOWN:
                continue
            if m.bad[t, cols[leader_pos[t]]]:
                continue
            fol = [int(c) for k, c in enumerate(cols) if k != leader_pos[t] and m.ok[t - 1, c]]
            if len(fol) < MIN_FOLLOWERS:
                continue
            last = t
            period = "2000-2015" if t < split_i else "2016+"
            real.append({"sub_industry": sub, "date": m.dates[t], "leader": m.tickers[cols[leader_pos[t]]],
                         "leader_jump": ab[t], "followers": len(fol), "period": period, **_measure(m, fol, t)})
            lo, hi = (i_start + 1, split_i) if t < split_i else (split_i, len(m.dates) - 2 - max(CHAIN_HORIZONS))
            for tt in rng.integers(lo, hi, CHAIN_PLACEBO_K):
                fol2 = [int(c) for k, c in enumerate(cols) if k != leader_pos[tt] and m.ok[tt - 1, c]]
                if len(fol2) >= MIN_FOLLOWERS:
                    plac.append({"period": period, **_measure(m, fol2, int(tt))})
    return pd.DataFrame(real), pd.DataFrame(plac)


def chain_ai(m: Market) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D2: ימים שבהם אנבידיה קפצה NVDA_JUMP+ מעל SPY -> סלי הספקים לפי קבוצה."""
    tix = {t: j for j, t in enumerate(m.tickers)}
    if AI_LEADER not in tix:
        return pd.DataFrame(), pd.DataFrame()
    j = tix[AI_LEADER]
    r = m.c[1:, j] / m.c[:-1, j] - 1
    ab = np.concatenate([[np.nan], r - (m.spy[1:] / m.spy[:-1] - 1)])
    groups = {k: [tix[t] for t in v if t in tix] for k, v in AI_SUPPLIERS.items()}
    groups["כל הספקים"] = sorted({c for v in groups.values() for c in v})
    rng = np.random.default_rng(31)
    real, plac, last = [], [], -10 ** 9
    split_i = m.idx(SPLIT)
    for t in np.flatnonzero(ab >= NVDA_JUMP):
        if t < m.idx(SIM_START) or t + 1 + max(CHAIN_HORIZONS) >= len(m.dates) or t - last < 5:
            continue
        last = t
        period = "2000-2015" if t < split_i else "2016+"
        lo, hi = (m.idx(SIM_START) + 1, split_i) if t < split_i else (split_i, len(m.dates) - 2 - max(CHAIN_HORIZONS))
        rand = rng.integers(lo, hi, CHAIN_PLACEBO_K)
        for gname, cols in groups.items():
            live = [c for c in cols if m.ok[t - 1, c]]
            if not live:
                continue
            real.append({"group": gname, "date": m.dates[t], "nvda_jump": ab[t], "period": period, **_measure(m, live, int(t))})
            for tt in rand:
                live2 = [c for c in cols if m.ok[tt - 1, c]]
                if live2:
                    plac.append({"group": gname, "period": period, **_measure(m, live2, int(tt))})
    return pd.DataFrame(real), pd.DataFrame(plac)


def chain_compare(real: pd.DataFrame, plac: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if real.empty:
        return pd.DataFrame()
    cols = ["same_day", "next_day"] + [f"after_{h}" for h in CHAIN_HORIZONS]
    out = []
    keys = [("הכל",) + tuple("הכל" for _ in by[1:])] if by else []
    for key, g in list(real.groupby(by)) + ([(("הכל",) * len(by), real)] if "period" in by and len(by) == 1 else []):
        key = key if isinstance(key, tuple) else (key,)
        q = plac
        for col, val in zip(by, key):
            if val != "הכל" and col in plac.columns:
                q = q[q[col] == val]
        rec = dict(zip(by, key))
        rec["events"] = len(g)
        for c in cols:
            x, y = g[c].dropna(), q[c].dropna() if c in q.columns else pd.Series(dtype=float)
            rec[c] = x.mean() if len(x) else np.nan
            if len(x) > 2 and len(y) > 2:
                se = math.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
                rec[c + "_vs_placebo"] = x.mean() - y.mean()
                rec[c + "_t"] = (x.mean() - y.mean()) / se if se > 0 else np.nan
            else:
                rec[c + "_vs_placebo"] = rec[c + "_t"] = np.nan
        out.append(rec)
    del keys
    return pd.DataFrame(out)

# ---------------------------------------------------------------- סיכום
def pct(x):
    try:
        return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def num(x, d=1):
    try:
        return "—" if x is None or pd.isna(x) else f"{x:.{d}f}"
    except (TypeError, ValueError):
        return "—"


def perf_rows(df: pd.DataFrame, label_col: str) -> list[str]:
    L = [f"| {label_col} | תקופה | CAGR | SPY | ירידה מקס' | SPY | עסקאות | הצלחה | ממוצע לעסקה | ימים | %עצירה | תקועות 2ש+ | מושקע |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.to_dict("records"):
        L.append(f"| {r.get(label_col, '')} | {r.get('period', '')} | {pct(r.get('cagr'))} | {pct(r.get('spy_cagr'))} | "
                 f"{pct(r.get('max_dd'))} | {pct(r.get('spy_max_dd'))} | {r.get('trades', '—')} | {pct(r.get('win_rate'))} | "
                 f"{pct(r.get('avg_trade'))} | {num(r.get('avg_days'), 0)} | {pct(r.get('pct_stop'))} | "
                 f"{r.get('stuck_2y_plus', '—')} | {pct(r.get('avg_invested'))} |")
    return L


def write_summary(meta, dip=None, ins=None, chain=None) -> str:
    L = [f"# תוצאות מחקר - קניית ירידות ואנשי פנים (v{RESEARCH_VERSION})",
         f"הורץ: {meta['run_at']} | יקום: {meta['universe']} | חלוקה: {SPLIT} | עד {MAX_POSITIONS} פוזיציות | "
         f"עמלה {COST_PER_SIDE:.1%} לצד", ""]
    if dip:
        L += ["## A. קניית ירידות (השיטה של Tomer, שיטתית)", "",
              f"מניה חזקה (+{STRONG_3Y:.0%} ב-3 שנים עד לפני שנה), ירידה מהשיא, כניסה בחציית ממוצע 20 יום כלפי מעלה, "
              f"מחיר ≥ ${MIN_PRICE:.0f}, מחזור ≥ ${MIN_DOLLAR_VOL / 1e6:.0f}M ליום.", "",
              "### ברירת המחדל של Tomer (נקבעה מראש: ירידה 20-40%, יציאה +30%, בלי עצירה)", "",
              "_השורה הקובעת: Tomer מול 'פלצבו'. היקום בנוי מחברות ששרדו עד היום, ולכן כל קנייה בהן נראית טוב בדיעבד. "
              "רק אם קניית הירידה עוקפת כניסה בזמן אקראי לאותן מניות - התזמון שווה משהו._", ""]
        L += perf_rows(dip["tomer"], "variant")
        L += ["", "### משברים", "", "| תקופה | תיק | SPY | ירידה מקס' | SPY |", "|---|---|---|---|---|"]
        for r in dip["crises"].to_dict("records"):
            L.append(f"| {r['period']} | {pct(r['total_ret'])} | {pct(r['spy_total_ret'])} | {pct(r['max_dd'])} | {pct(r['spy_max_dd'])} |")
        rb = dip["robust"]
        L += ["", f"### חוסן: מתוך {rb['combos']} שילובי ספים, בתקופת הבדיקה {rb['beat_spy_cagr']} ניצחו את SPY בתשואה, "
                  f"{rb['beat_spy_calmar']} בתשואה-לסיכון (Calmar).", ""]
        if dip["selected"] is not None and not dip["selected"].empty:
            s = dip["selected"].copy()
            s["combo"] = s["band"] + " / +" + (s["target"] * 100).round().astype(int).astype(str) + "% / עצירה " + \
                s["stop"].map(lambda x: "בלי" if pd.isna(x) else f"{x:.0%}") + " / זמן " + \
                s["max_days"].map(lambda x: "בלי" if pd.isna(x) else f"{int(x)} ימים")
            L += ["### השילוב שנבחר על 2000-2015 (לפי Calmar) ונבדק על 2016+", ""]
            L += perf_rows(s, "combo")
            L.append("")
    if ins:
        L += ["## B. אשכולות קניה של אנשי פנים", ""]
        L.append(f"קניות בשוק הפתוח שנאספו: {ins['n_purchases']:,} | אשכולות: {ins['n_clusters']:,} | "
                 f"בתוך היקום עם מחירים: {ins['n_events']:,}")
        L += ["", "_כמה יותר (מול SPY) הרוויחו מניות אחרי אשכול קניות, לעומת *אותן מניות בתאריכים אקראיים*. "
                  "כניסה ביום המסחר שאחרי הדיווח. 0% = אין אות._", "",
              "| תקופה | חיתוך | אירועים | חודש | t | 3 חודשים | t | 6 חודשים | t | שנה | t |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in ins["summary"].to_dict("records"):
            L.append(f"| {r['period']} | {r['cut']} | {r['events']} | {pct(r.get('vsp_21'))} | {num(r.get('vsp_21_t'))} | "
                     f"{pct(r.get('vsp_63'))} | {num(r.get('vsp_63_t'))} | {pct(r.get('vsp_126'))} | {num(r.get('vsp_126_t'))} | "
                     f"{pct(r.get('vsp_252'))} | {num(r.get('vsp_252_t'))} |")
        L += ["", "## B+C. תיקים", "",
              "_⚠️ שורות B2 הן פוסט-הוק: הכללים נבחרו אחרי שראינו את B. התוצאה ההיסטורית אופטימית; המבחן האמיתי = מעקב קדימה._", ""]
        L += perf_rows(ins["portfolios"], "strategy")
        L.append("")
        lv = ins.get("live")
        if lv is not None and not lv.empty:
            L += [f"### אותות מ-45 הימים האחרונים ({len(lv)}) - למעקב קדימה, לא המלצה", "",
                  "| טיקר | תאריך דיווח | אנשי פנים | מנכ\"ל/סמנכ\"ל | סכום | ציון | חזק | שכבה | מתחת לשיא | נזילות |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in lv.head(25).to_dict("records"):
                L.append(f"| {r['ticker']} | {pd.Timestamp(r['signal_date']):%Y-%m-%d} | {r['insiders']} | "
                         f"{'✓' if r['ceo_cfo'] else ''} | ${r['value'] / 1e6:.2f}M | {r['score']} | {'✓' if r['strong'] else ''} | "
                         f"{r['tier']} | {pct(r['dd_from_high'])} | {'✓' if r['passes_liquidity'] else '✗'} |")
            L.append("")
    if chain:
        def ctable(df, keycols):
            hdr = " | ".join(keycols)
            T = [f"| {hdr} | אירועים | באותו יום | t | למחרת | t | +5 ימים | t | +10 | t | +21 | t |",
                 "|" + "---|" * (len(keycols) + 11)]
            for r in df.to_dict("records"):
                k = " | ".join(str(r[c]) for c in keycols)
                T.append(f"| {k} | {r['events']} | {pct(r.get('same_day_vs_placebo'))} | {num(r.get('same_day_t'))} | "
                         f"{pct(r.get('next_day_vs_placebo'))} | {num(r.get('next_day_t'))} | "
                         f"{pct(r.get('after_5_vs_placebo'))} | {num(r.get('after_5_t'))} | "
                         f"{pct(r.get('after_10_vs_placebo'))} | {num(r.get('after_10_t'))} | "
                         f"{pct(r.get('after_21_vs_placebo'))} | {num(r.get('after_21_t'))} |")
            return T
        L += ["## D. שרשרת אספקה - האם 'התשתית' זזה אחרי המובילה?", "",
              "_כל הערכים: תשואה עודפת מול SPY *מעבר לימים אקראיים* באותה תקופה. 'באותו יום' ו'למחרת' = מה שכבר קרה "
              "לפני שאפשר להיכנס. '+5/+10/+21' = מכניסה בסגירה של היום שאחרי הקפיצה - זה מה שאפשר להרוויח._", "",
              f"### D1. התפשטות בתת-ענף (מובילה קפצה {LEADER_JUMP:.0%}+, {chain['n1']} אירועים)", ""]
        if not chain["industry"].empty:
            L += ctable(chain["industry"], ["period"])
        L += ["", f"### D2. שרשרת ה-AI (אנבידיה קפצה {NVDA_JUMP:.0%}+, {chain['n2']} ימים)", "",
              "_⚠️ המפה נבנתה היום, כשכבר ידוע מי הצליח. מה שתקף כאן הוא רק *העיכוב* מול ימים אקראיים, לא בחירת המניות._", ""]
        if not chain["ai"].empty:
            L += ctable(chain["ai"], ["group", "period"])
        L.append("")
    L += ["## איך לקרוא", "",
          "- **CAGR** = תשואה שנתית ממוצעת. **ירידה מקס'** = הנפילה הגדולה מהשיא - כמה כאב בדרך.",
          "- **מושקע** = איזה חלק מהכסף היה בפועל במניות (השאר מזומן בלי ריבית - הנחה שמרנית).",
          "- **תקועות 2ש+** = עסקאות שנכנסו ולא הגיעו ליעד גם אחרי שנתיים (הסיכון 'אינטל').",
          "- **t** מעל ~2 = לא סביר שמקרי. רק תקופת 2016+ קובעת; 2000-2015 שימשה לבחירת ספים.",
          "- **חוסן**: אם רק שילוב אחד מתוך 72 עובד - זה מזל. אם רובם - זה דפוס.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="מחקר offline: קניית ירידות + אנשי פנים")
    ap.add_argument("--part", default="all", help="dip / insider / chain / all, או כמה מופרדים בפסיק: insider,chain")
    ap.add_argument("--universe", default=None)
    ap.add_argument("--max-tickers", type=int, default=0)
    a = ap.parse_args(argv)
    parts = {"dip", "insider", "chain"} if a.part.strip() == "all" else {x.strip() for x in a.part.split(",")}
    bad = parts - {"dip", "insider", "chain"}
    if bad:
        ap.error(f"part לא מוכר: {bad}")
    ensure_dirs()

    uni = load_universe(a.universe)
    if a.max_tickers and len(uni) > a.max_tickers:
        uni = uni.iloc[np.linspace(0, len(uni) - 1, a.max_tickers).round().astype(int)].drop_duplicates("ticker")
        log(f"הרצת בדיקה: {len(uni)} מניות פזורות על פני כל היקום")
    for extra in ["DELL", "INTC"] + AI_EXTRA_TICKERS:   # דוגמאות מהשיחה + מפת ה-AI (D2)
        if extra not in set(uni["ticker"]):
            uni = pd.concat([uni, pd.DataFrame([{"ticker": extra, "name": extra, "sector": "?",
                                                 "sub_industry": "_extra", "tier": "large"}])])
    uni.to_csv(os.path.join(OUT_DIR, "universe_used.csv"), index=False)
    tiers = uni.set_index("ticker")["tier"]

    close, vol = download_prices(uni["ticker"].tolist(), DOWNLOAD_START)
    m = Market(close, vol)
    log(f"סומנו {m.n_bad} ימים חשודים (פעולה תאגידית/נתון שגוי) - מוחרגים")
    meta = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "universe": f"{len(m.tickers)} מניות ({', '.join(f'{k}={v}' for k, v in uni['tier'].value_counts().items())})"}

    dip = ins = None
    if "dip" in parts:
        log("A: קניית ירידות - רשת ספים...")
        dip = run_dip(m)
        dip["grid"].to_csv(os.path.join(OUT_DIR, "dip_grid.csv"), index=False)
        dip["tomer"].to_csv(os.path.join(OUT_DIR, "dip_tomer_default.csv"), index=False)
        dip["crises"].to_csv(os.path.join(OUT_DIR, "dip_crises.csv"), index=False)
        dip["tomer_trades"].to_csv(os.path.join(OUT_DIR, "dip_tomer_trades_2016plus.csv"), index=False)

    if "insider" in parts:
        ua = os.environ.get("SEC_USER_AGENT", "").strip()
        if not ua:
            log("SEC_USER_AGENT לא מוגדר - SEC חוסם בקשות אנונימיות. מדלג על B/C.")
        else:
            log("B: מוריד נתוני Form 4 מ-SEC (פעם ראשונה ארוכה; נשמר במטמון)...")
            purch = map_tickers(fetch_insider_purchases(ua), ua)
            clusters = find_clusters(purch) if not purch.empty else pd.DataFrame()
            clusters.to_csv(os.path.join(OUT_DIR, "insider_clusters.csv"), index=False)
            es, summ = insider_event_study(m, clusters, tiers) if not clusters.empty else (pd.DataFrame(), pd.DataFrame())
            if not summ.empty:
                log("  פלצבו: אותן מניות בתאריכים אקראיים...")
                _, psumm = insider_event_study(m, placebo_clusters(clusters, m), tiers)
                summ = vs_placebo(summ, psumm)
            es.to_csv(os.path.join(OUT_DIR, "insider_event_returns.csv"), index=False)
            summ.to_csv(os.path.join(OUT_DIR, "insider_summary.csv"), index=False)
            ports = run_insider_portfolios(m, clusters, dip["selected"] if dip else None) if not clusters.empty else pd.DataFrame()
            if not clusters.empty:
                log("B2: תיק משופר (עדיפות לאותות חזקים, החזקה שנה)...")
                ports = pd.concat([ports, run_b2(m, clusters)], ignore_index=True)
            live = live_signals(m, clusters, tiers)
            live.to_csv(os.path.join(OUT_DIR, "insider_live_signals.csv"), index=False)
            ports.to_csv(os.path.join(OUT_DIR, "insider_portfolios.csv"), index=False)
            ins = {"n_purchases": len(purch), "n_clusters": len(clusters), "n_events": len(es),
                   "summary": summ, "portfolios": ports, "live": live}

    chain = None
    if "chain" in parts:
        log("D1: התפשטות בתוך תת-ענף...")
        r1, p1 = chain_industry(m, uni[uni["sub_industry"] != "_extra"])
        r1.to_csv(os.path.join(OUT_DIR, "chain_industry_events.csv"), index=False)
        log(f"  {len(r1)} קפיצות של מובילות ענף")
        log("D2: שרשרת ה-AI סביב אנבידיה...")
        r2, p2 = chain_ai(m)
        r2.to_csv(os.path.join(OUT_DIR, "chain_ai_events.csv"), index=False)
        chain = {"industry": chain_compare(r1, p1, ["period"]) if not r1.empty else pd.DataFrame(),
                 "ai": chain_compare(r2, p2, ["group", "period"]) if not r2.empty else pd.DataFrame(),
                 "n1": len(r1), "n2": r2["date"].nunique() if not r2.empty else 0}
        chain["industry"].to_csv(os.path.join(OUT_DIR, "chain_industry_summary.csv"), index=False)
        chain["ai"].to_csv(os.path.join(OUT_DIR, "chain_ai_summary.csv"), index=False)

    md = write_summary(meta, dip, ins, chain)
    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
