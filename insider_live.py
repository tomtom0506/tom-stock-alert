"""
🧪 קניות אנשי פנים - מעקב קדימה במעבדה (handoff מצ'אט המחקר, 16.10.2026).

Signal rules are IMPORTED from research/dip_insider_research.py (find_clusters,
insider_score, the thresholds) - one definition, so live == research by
construction. Only the data path differs: the research used SEC's quarterly
insider data sets (they lag ~a quarter); live reads SEC EDGAR's DAILY index
and parses each Form 4 XML (ownershipDocument).

  purchase   transactionCode P, acquired (A), value >= $10,000, by a director
             or officer (10% owners alone don't count); joint filings: first
             reporting owner only (as in the research)
  cluster    >= 2 different insiders, >= $100,000 together, within 30 days by
             filing date; 90 quiet days per stock after a signal
  strong     CEO or CFO among them (officerTitle), or 3+ insiders
  universe   S&P 500 + 400 + 600, price >= $5, 60-day avg $ volume >= $2M

Paper portfolio "רק חזקים": $100,000, 20 slots of $5,000, buy at the close of
the session after the filing (a signal waits up to 5 sessions for a free slot,
best score first), sell after 252 sessions, no target / stop / market filter,
0.1% per side. Only signals filed on/after the day this went live enter it.

Runs from .github/workflows/winners.yml after winners_live.py. Writes
insider.json only.
"""
import json
import os
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "research"))
import dip_insider_research as R   # noqa: E402  (the research rules, unchanged)

STATE_FILE = BASE / "insider.json"
UA = os.environ.get("SEC_USER_AGENT") or "TomStockAlert research tomtom0506@users.noreply.github.com"
INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{d}.idx"
BACKFILL_DAYS = 35          # enough history for the 30-day cluster window on day one
KEEP_DAYS = 150             # 30-day window + 90-day cooldown + margin
SIGNAL_LOOKBACK_DAYS = 45
SLOTS, SLOT_USD, START_USD = 20, 5000.0, 100000.0
HOLD = 252
WAIT_SESSIONS = 5
FEE = 0.001
REQ_PAUSE = 0.12            # SEC fair access: < 10 requests / second
NOTE = "ניסוי. בהיסטוריה פחות ממחצית האותות ניצחו את SPY; הרווח הגיע ממיעוט מנצחות גדולות."


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    return s


# ---------------------------------------------------------------- Form 4 parsing
def _txt(node, path):
    el = node.find(path)
    return (el.text or "").strip() if el is not None and el.text else ""


def _flag(node, path):
    return _txt(node, path).lower() in ("1", "true")


def parse_form4(xml_text):
    """-> dict(owner, value, title, issuer_cik, symbol) for a qualifying
    purchase filing, else None. Mirrors research parse_insider_zip()."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    # Joint filings (a fund + its directors) list several reporting owners.
    # v5.24.1: pick the qualifying INDIVIDUAL - a director/officer who is not
    # also flagged as a 10% owner (the fund entity) - else the first
    # qualifying one. Parity run 16.10: taking the plain first owner chose
    # the fund for every filing, so 3 directors buying counted as 1 insider.
    owners = []
    for o in root.findall("reportingOwner"):
        r_ = o.find("reportingOwnerRelationship")
        if r_ is not None and (_flag(r_, "isDirector") or _flag(r_, "isOfficer")):
            owners.append((o, r_))
    if not owners:
        return None
    owner, rel = next(((o, r_) for o, r_ in owners if not _flag(r_, "isTenPercentOwner")), owners[0])
    value = 0.0
    for tx in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        if _txt(tx, "transactionCoding/transactionCode") != "P":
            continue
        if _txt(tx, "transactionAmounts/transactionAcquiredDisposedCode/value") not in ("", "A"):
            continue
        try:
            sh = float(_txt(tx, "transactionAmounts/transactionShares/value") or 0)
            px = float(_txt(tx, "transactionAmounts/transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        value += sh * px
    if value < R.MIN_TX_VALUE:
        return None
    return {"owner": _txt(owner, "reportingOwnerId/rptOwnerCik"), "value": round(value, 2),
            "title": _txt(rel, "officerTitle"), "issuer_cik": _txt(root, "issuer/issuerCik"),
            "symbol": _txt(root, "issuer/issuerTradingSymbol").upper()}


def extract_xml(full_txt):
    m = re.search(r"<XML>\s*(.*?)\s*</XML>", full_txt, re.S | re.I)
    return m.group(1) if m else None


def parse_index(text):
    """form.idx rows -> [(form, cik, date_filed, file_name)] for Form 4 / 4/A."""
    out = []
    for line in text.splitlines():
        if not (line.startswith("4 ") or line.startswith("4/A ")):
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 5:
            continue
        form, cik, filed, fname = parts[0], parts[-3], parts[-2], parts[-1]
        out.append((form, int(cik) if cik.isdigit() else None, filed, fname))
    return out


# ---------------------------------------------------------------- daily ingest
def universe(sess):
    uni = R.load_universe(None)
    js = sess.get("https://www.sec.gov/files/company_tickers.json", timeout=60).json()
    cik_of = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in js.values()}
    tick_of = {}
    for t in uni["ticker"]:
        c = cik_of.get(t)
        if c:
            tick_of[c] = t
    tiers = dict(zip(uni["ticker"], uni["tier"]))
    return tick_of, tiers


def ingest_day(sess, d, tick_of):
    q = (d.month - 1) // 3 + 1
    r = sess.get(INDEX_URL.format(y=d.year, q=q, d=d.strftime("%Y%m%d")), timeout=60)
    time.sleep(REQ_PAUSE)
    if r.status_code != 200:
        return None                                      # weekend / holiday / not published yet
    rows = parse_index(r.text)
    files = {}
    for form, cik, filed, fname in rows:
        if cik in tick_of:
            files[fname] = tick_of[cik]
    found = []
    for fname, ticker in files.items():
        try:
            fr = sess.get("https://www.sec.gov/Archives/" + fname, timeout=30)
        except Exception:
            continue
        time.sleep(REQ_PAUSE)
        if fr.status_code != 200:
            continue
        x = extract_xml(fr.text)
        p = parse_form4(x) if x else None
        if p:
            found.append({**p, "ticker": ticker, "filing_date": d.isoformat(), "accession": fname.rsplit("/", 1)[-1]})
    log(f"{d}: {len(rows)} Form 4 rows, {len(files)} in universe, {len(found)} qualifying purchases")
    return found


# ---------------------------------------------------------------- signals + portfolio
def clusters_from(purchases):
    if not purchases:
        return pd.DataFrame()
    df = pd.DataFrame(purchases)
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.drop_duplicates("accession")
    return R.find_clusters(df[["ticker", "owner", "filing_date", "value", "title"]])


def prices(tickers, start):
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    data = yf.download(" ".join(sorted(set(tickers))), start=start, auto_adjust=True, group_by="ticker",
                       threads=True, progress=False)
    C, V = {}, {}
    for t in sorted(set(tickers)):
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
            C[t], V[t] = df["Close"], df["Volume"]
        except Exception:
            continue
    return pd.DataFrame(C), pd.DataFrame(V)


def build(st, cl, tiers):
    today = date.today()
    sig = cl[cl["signal_date"] >= pd.Timestamp(today - timedelta(days=SIGNAL_LOOKBACK_DAYS))].copy() if len(cl) else cl
    pos_t = [p["ticker"] for p in st["positions"] + st["closed"]]
    tick = list(set((list(sig["ticker"]) if len(sig) else []) + pos_t + ["SPY"]))
    C, V = prices(tick, (today - timedelta(days=420)).isoformat())
    signals = []
    if len(sig):
        sig["score"] = R.insider_score(sig).round(2)
        sig["strong"] = sig["ceo_cfo"].astype(bool) | (sig["insiders"] >= 3)
        for e in sig.sort_values(["strong", "score"], ascending=False).itertuples():
            c = C[e.ticker].dropna() if e.ticker in C.columns else pd.Series(dtype=float)
            v = V[e.ticker].dropna() if e.ticker in V.columns else pd.Series(dtype=float)
            liq = bool(len(c) >= 60 and c.iloc[-1] >= R.MIN_PRICE and (c * v).iloc[-60:].mean() >= R.MIN_DOLLAR_VOL)
            dd = float(c.iloc[-1] / c.iloc[-252:].max() - 1) * 100 if len(c) else None
            signals.append({"ticker": e.ticker, "filing_date": e.signal_date.strftime("%Y-%m-%d"), "insiders": int(e.insiders),
                            "ceo_cfo": bool(e.ceo_cfo), "value": round(float(e.value)), "score": float(e.score),
                            "strong": bool(e.strong), "tier": tiers.get(e.ticker, "?"), "liquid": liq,
                            "dd_from_high_pct": None if dd is None else round(dd, 1)})
    st["signals"] = signals
    # ---- portfolio (forward only: signals filed on/after go-live)
    spy = C["SPY"].dropna() if "SPY" in C.columns else pd.Series(dtype=float)
    if spy.empty:
        return
    idx = spy.index
    held = {p["ticker"] for p in st["positions"]}
    queue = [s for s in signals if s["strong"] and s["liquid"] and s["filing_date"] >= st["started"]
             and s["ticker"] not in held and not any(p["ticker"] == s["ticker"] and p["filing_date"] == s["filing_date"]
                                                     for p in st["positions"] + st["closed"])]
    for s in sorted(queue, key=lambda x: -x["score"]):
        if len(st["positions"]) >= SLOTS:
            break
        after = idx[idx > pd.Timestamp(s["filing_date"])]
        if not len(after) or s["ticker"] not in C.columns:
            continue
        if len(after) > WAIT_SESSIONS:
            continue                                     # expired while waiting for a slot
        e = after[0]
        px = C.loc[e, s["ticker"]]
        if not np.isfinite(px):
            continue
        st["positions"].append({"ticker": s["ticker"], "filing_date": s["filing_date"], "entry_date": e.strftime("%Y-%m-%d"),
                                "entry_price": round(float(px), 4), "shares": SLOT_USD * (1 - FEE) / float(px),
                                "insiders": s["insiders"], "ceo_cfo": s["ceo_cfo"], "value": s["value"], "score": s["score"]})
    keep = []
    for p in st["positions"]:
        if p["ticker"] not in C.columns:
            keep.append(p)
            continue
        c = C[p["ticker"]].dropna()
        ei = idx.get_loc(pd.Timestamp(p["entry_date"])) if pd.Timestamp(p["entry_date"]) in idx else None
        if ei is not None and ei + HOLD <= len(idx) - 1:
            x = idx[ei + HOLD]
            xp = float(c[c.index <= x].iloc[-1])
            st["closed"].append({**p, "exit_date": x.strftime("%Y-%m-%d"), "exit_price": round(xp, 4),
                                 "ret_pct": round((xp * (1 - FEE) / (p["entry_price"] / (1 - FEE)) - 1) * 100, 2),
                                 "spy_pct": round((float(spy.loc[x]) / float(spy[spy.index <= pd.Timestamp(p["entry_date"])].iloc[-1]) - 1) * 100, 2)})
            continue
        p["last_price"] = round(float(c.iloc[-1]), 4)
        p["ret_pct"] = round((p["last_price"] / p["entry_price"] - 1) * 100, 2)
        keep.append(p)
    st["positions"] = keep
    invested = SLOT_USD * (len(st["positions"]) + len(st["closed"]))
    proceeds = sum(SLOT_USD * (1 + c["ret_pct"] / 100) for c in st["closed"])
    mark = sum(p["shares"] * p.get("last_price", p["entry_price"]) for p in st["positions"])
    value = START_USD - invested + proceeds + mark
    s0 = float(spy[spy.index <= pd.Timestamp(st["started"])].iloc[-1]) if (spy.index <= pd.Timestamp(st["started"])).any() else float(spy.iloc[0])
    st["portfolio"] = {"value": round(value, 2), "ret_pct": round((value / START_USD - 1) * 100, 2),
                       "spy_pct": round((float(spy.iloc[-1]) / s0 - 1) * 100, 2), "open": len(st["positions"]),
                       "closed": len(st["closed"]), "since": st["started"]}


def main():
    st = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.exists() else {
        "started": date.today().isoformat(), "purchases": [], "processed": [], "positions": [], "closed": [], "note": NOTE}
    sess = session()
    try:
        tick_of, tiers = universe(sess)
    except Exception as e:
        log(f"universe failed: {e}")
        return
    log(f"universe: {len(tick_of)} companies with a CIK")
    done = set(st["processed"])
    d = date.today() - timedelta(days=BACKFILL_DAYS)
    new = 0
    while d <= date.today():
        if d.weekday() < 5 and d.isoformat() not in done:
            try:
                rows = ingest_day(sess, d, tick_of)
            except Exception as e:
                log(f"{d}: failed ({e})")
                rows = None
            if rows is not None:
                st["purchases"] += rows
                done.add(d.isoformat())
                new += len(rows)
        d += timedelta(days=1)
    cut = (date.today() - timedelta(days=KEEP_DAYS)).isoformat()
    st["purchases"] = [p for p in st["purchases"] if p["filing_date"] >= cut]
    st["processed"] = sorted(x for x in done if x >= cut)
    cl = clusters_from(st["purchases"])
    build(st, cl, tiers)
    month = (date.today() - timedelta(days=30)).isoformat()
    n30 = sum(1 for s in st.get("signals", []) if s["strong"] and s["filing_date"] >= month)
    live_days = (date.today() - date.fromisoformat(st["started"])).days
    st["sanity"] = {"strong_last_30d": n30, "expected_per_month": 9,
                    "warning": (live_days >= 30 and (n30 == 0 or n30 > 40))}
    st["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"done: {new} new purchases, {len(st.get('signals', []))} signals, {n30} strong in 30 days, portfolio {st.get('portfolio')}")


if __name__ == "__main__":
    main()
