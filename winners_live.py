"""
🏆 מניות מנצחות 🧪 - תיק ניסוי חי (Tomer, 13.10.2026).

Rule (the best combination from research/dip_research.py, 13.10.2026 - NOT
statistically proven there: +10% per trade over the equal-weight index but
t=0.31, so this runs as a forward test that is recorded in advance):
  * S&P 500 member, closed >= 35% below its 52-week high
  * strong business at the last SEC filing: revenue not shrinking y/y,
    net income > 0, operating profit > 0
  * cheap on BOTH lenses: EV/EBIT below its industry median (SEC SIC 2-digit)
    and below its own 5-year median
  * up to 20 open positions, the cheapest (lowest industry x own-history
    ratio) first; nothing is filled in when fewer qualify
  * buy at the open of the session after the selection, hold 252 sessions
    (one year), sell at that close; 0.1% per side

Every pick also gets a short "why did it fall" explanation:
  1. vs the market and its sector ETF over the same drop -> 🌍 market /
     🏭 sector / 🏢 company
  2. its 3 worst days in the drop, 📊 when one was an earnings day
  3. up to 3 headlines from those days (Google News), keyword-tagged

Runs from .github/workflows/winners.yml: every weekday after the US close
(mark to market, fill/close positions), selection on Fridays or on a
manual run. Writes winners.json only - the main bot is not touched.
"""
import json
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
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "research"))
import winners_research as W   # noqa: E402
import dip_research as D       # noqa: E402

WINNERS_FILE = BASE / "winners.json"
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
MAX_POS = 20
DRAWDOWN = 0.35
HOLD = 252
FEE = 0.001
SECTOR_ETF = {"Information Technology": "XLK", "Financials": "XLF", "Health Care": "XLV",
              "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
              "Industrials": "XLI", "Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
              "Communication Services": "XLC"}
NEWS_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
RESEARCH_NOTE = ("במחקר 2011-2026 (רשימת S&P לפי ההרכב בכל תאריך): +10% לעסקה מעל ממוצע כל המניות באותם ימים, "
                 "אבל t=0.31 - כלומר לא מובהק, ורוב העודף הגיע מ-2025 ו-2016. לכן זה ניסוי שנמדד מהיום.")


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_state():
    if WINNERS_FILE.exists():
        return json.loads(WINNERS_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "started": None, "positions": [], "pending": [], "closed": [],
            "selection_log": [], "candidates": [], "rule": {"drawdown_pct": DRAWDOWN * 100, "max_positions": MAX_POS,
                                                            "hold_sessions": HOLD, "fee_per_side_pct": FEE * 100},
            "research_note": RESEARCH_NOTE}


def save_state(st):
    st["updated_at"] = datetime.now(timezone.utc).isoformat()
    WINNERS_FILE.write_text(json.dumps(st, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def telegram(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("(telegram not configured)\n" + text)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=20)
    except Exception as e:
        print(f"telegram failed: {e}")


def members_meta():
    df = pd.read_csv(SP500_CSV_URL)
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return {r["Symbol"]: {"name": r["Security"], "sector": r["GICS Sector"], "sub": r["GICS Sub-Industry"]}
            for _, r in df.iterrows()}


def download(tickers, start, actions=False):
    O, C, A, SPL = {}, {}, {}, {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), 100):
        part = tickers[i:i + 100]
        try:
            data = yf.download(" ".join(part), start=start, auto_adjust=False, actions=actions,
                               group_by="ticker", threads=True, progress=False)
        except Exception as e:
            log(f"download chunk failed: {e}")
            continue
        for t in part:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df.index = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
                if df["Close"].dropna().empty:
                    continue
                O[t], C[t], A[t] = df["Open"], df["Close"], df["Adj Close"]
                if actions and "Stock Splits" in df:
                    sp = df["Stock Splits"]
                    SPL[t] = [(d.strftime("%Y-%m-%d"), float(v)) for d, v in sp[sp > 0].items()]
            except Exception:
                continue
        time.sleep(1)
    return pd.DataFrame(O), pd.DataFrame(C), pd.DataFrame(A), SPL


# ---------------------------------------------------------------- explanation
def _news_on(ticker, name, day):
    d0 = (day - timedelta(days=1)).strftime("%Y-%m-%d")
    d1 = (day + timedelta(days=2)).strftime("%Y-%m-%d")
    short = (name or "").replace(",", " ").split(" Inc")[0].split(" Corp")[0].strip()
    q = f'("{short}" OR {ticker}) stock after:{d0} before:{d1}' if short else f"{ticker} stock after:{d0} before:{d1}"
    try:
        r = requests.get("https://news.google.com/rss/search",
                         params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}, headers=NEWS_UA, timeout=15)
        if r.status_code != 200:
            return []
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
        out = []
        for it in root.iter("item"):
            title = (it.findtext("title") or "").strip()
            src = (it.findtext("source") or "").strip()
            if src and title.endswith(" - " + src):
                title = title[: -len(" - " + src)]
            if title:
                out.append({"title": title[:180], "publisher": src or None, "url": (it.findtext("link") or "").strip() or None})
        return out[:3]
    except Exception:
        return []


TAG_RULES = [
    ("📉 דוח / תחזית", ["earnings", "results", "quarter", "guidance", "forecast", "outlook", "revenue", "misses",
                        "miss", "beat", "profit", "sales"]),
    ("👔 הנהלה", ["ceo", "cfo", "resigns", "steps down", "executive", "chairman", "board"]),
    ("⚖️ חקירה / משפט", ["probe", "investigation", "lawsuit", "sec ", "doj", "antitrust", "fraud", "settlement", "fine"]),
    ("🏷️ אנליסטים", ["downgrade", "cut to", "price target", "rating", "analyst"]),
    ("💵 הנפקה / חוב", ["offering", "dilution", "convertible", "debt", "credit rating"]),
    ("🌍 שוק / מאקרו", ["fed", "tariff", "inflation", "rates", "recession", "market sell", "selloff", "sell-off"]),
    ("🤖 תחרות / טכנולוגיה", ["ai ", "competition", "competitor", "disrupt", "market share"]),
]


def tag_headline(title):
    import re
    t = " " + title.lower() + " "
    for label, words in TAG_RULES:
        if any(re.search(r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])", t) for w in words):
            return label
    return None


def explain(t, meta, C, A, etf_adj, spy_adj, earnings_days):
    c = C[t].dropna()
    a = A[t].dropna()
    win = c.iloc[-252:]
    peak_day = win.idxmax()
    now_day = c.index[-1]
    seg = a[(a.index >= peak_day) & (a.index <= now_day)]
    if len(seg) < 5:
        return None
    stock_ret = float(seg.iloc[-1] / seg.iloc[0] - 1) * 100

    def rng(s):
        s = s.dropna()
        s = s[(s.index >= peak_day) & (s.index <= now_day)]
        return float(s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 1 else None
    spy_ret = rng(spy_adj)
    etf = SECTOR_ETF.get(meta.get("sector"))
    sec_ret = rng(etf_adj[etf]) if etf and etf in etf_adj.columns else None
    if sec_ret is not None and sec_ret <= -20 and stock_ret - sec_ret > -15:
        kind, kind_label = "sector", "🏭 ירידת ענף"
    elif spy_ret is not None and spy_ret <= -15 and stock_ret - spy_ret > -15:
        kind, kind_label = "market", "🌍 ירידת שוק"
    else:
        kind, kind_label = "company", "🏢 בעיה של החברה"
    daily = seg.pct_change().dropna()
    worst = daily.nsmallest(3)
    days = []
    for d, r in sorted(worst.items(), key=lambda x: x[1]):
        is_earn = any(abs((d - e).days) <= 1 for e in earnings_days)
        days.append({"date": d.strftime("%Y-%m-%d"), "pct": round(float(r) * 100, 1), "earnings": is_earn})
    headlines = []
    for dd in days[:2]:
        for h in _news_on(t, meta.get("name"), pd.Timestamp(dd["date"])):
            if len(headlines) >= 3 or any(h["title"] == x["title"] for x in headlines):
                continue
            h["tag"] = tag_headline(h["title"])
            h["date"] = dd["date"]
            headlines.append(h)
        time.sleep(1)
    peak_s = peak_day.strftime("%d.%m.%y")
    parts = [f"{t} ירדה {abs(stock_ret):.0f}% מהשיא ({peak_s})."]
    ctx = []
    if spy_ret is not None:
        ctx.append(f"ה-S&P {'עלה' if spy_ret >= 0 else 'ירד'} {abs(spy_ret):.0f}%")
    if sec_ret is not None:
        ctx.append(f"הענף ({etf}) {'עלה' if sec_ret >= 0 else 'ירד'} {abs(sec_ret):.0f}%")
    if ctx:
        parts.append("באותה תקופה " + " ו".join(ctx) + f" - {kind_label.split(' ', 1)[1]}.")
    w = days[0] if days else None
    if w:
        parts.append(f"הירידה החדה ביותר: {w['pct']:.0f}% ב-{pd.Timestamp(w['date']).strftime('%d.%m.%y')}"
                     + (" (יום דוח רבעוני 📊)." if w["earnings"] else "."))
    tags = [h["tag"] for h in headlines if h.get("tag")]
    if tags:
        parts.append("בכותרות: " + ", ".join(dict.fromkeys(tags)) + ".")
    return {"kind": kind, "kind_label": kind_label, "drop_pct": round(stock_ret, 1), "peak_date": peak_day.strftime("%Y-%m-%d"),
            "spy_pct": None if spy_ret is None else round(spy_ret, 1), "sector_etf": etf,
            "sector_pct": None if sec_ret is None else round(sec_ret, 1), "worst_days": days, "headlines": headlines,
            "summary": " ".join(parts)}


def earnings_days_of(t):
    try:
        df = yf.Ticker(t).get_earnings_dates(limit=12)
        if df is None or df.empty:
            return []
        idx = df.index.tz_localize(None) if getattr(df.index, "tz", None) is not None else df.index
        return [pd.Timestamp(d.date()) for d in idx]
    except Exception:
        return []


# ---------------------------------------------------------------- selection
def select(st):
    meta = members_meta()
    tickers = sorted(meta)
    start = (date.today() - timedelta(days=int(365 * 6.3))).isoformat()
    log(f"selection: {len(tickers)} members, prices since {start}")
    O, C, A, splits = download(tickers + ["SPY"] + list(SECTOR_ETF.values()), start, actions=True)
    spy_adj = A.pop("SPY")
    etf_adj = A[[e for e in SECTOR_ETF.values() if e in A.columns]]
    keep = [t for t in tickers if t in C.columns]
    C, A = C[keep], A[keep]
    session = D.sec_session()
    mode, diag = W.probe_sec(session)
    if mode != "api":
        log(f"SEC not available ({diag}) - skipping selection this week")
        st.setdefault("selection_log", []).append({"date": date.today().isoformat(), "error": diag[:300]})
        return []
    funds = D.load_fundamentals(set(keep), session)
    members = [(pd.Timestamp("1990-01-01"), set(keep))]
    today = pd.Timestamp(C.index[-1].date())
    qdates = [d for d in pd.date_range(start, today, freq="QS")] + [today]
    snaps = D.quarterly_snapshots(funds, C, members, qdates, splits)
    snap = snaps.get(today) or {}
    draw = (C / C.rolling(252, min_periods=200).max() - 1).iloc[-1]
    cands = []
    for t in keep:
        s = snap.get(t) or {}
        if not np.isfinite(draw.get(t, np.nan)) or draw[t] > -DRAWDOWN:
            continue
        if not (s.get("q") and s.get("l1") is not None and s["l1"] <= 1.0 and s.get("l2") is not None and s["l2"] <= 1.0):
            continue
        cands.append({"ticker": t, "name": meta[t]["name"], "sector": meta[t]["sector"],
                      "drawdown_pct": round(float(draw[t]) * 100, 1), "ev_ebit": round(s["ev_ebit"], 1),
                      "vs_industry": round(s["l1"], 2), "vs_itself": round(s["l2"], 2), "cheap_score": round(s["l1"] * s["l2"], 3)})
    cands.sort(key=lambda r: r["cheap_score"])
    held = {p["ticker"] for p in st["positions"]} | {p["ticker"] for p in st["pending"]}
    free = MAX_POS - len(held)
    new = [c for c in cands if c["ticker"] not in held][:max(free, 0)]
    for c in cands[:30]:
        c["explanation"] = explain(c["ticker"], meta[c["ticker"]], C, A, etf_adj, spy_adj, earnings_days_of(c["ticker"]))
        c["status"] = "held" if c["ticker"] in held else ("new" if c in new else "waiting")
    sel_day = today.strftime("%Y-%m-%d")
    for c in new:
        st["pending"].append({**c, "selected_on": sel_day})
    st["candidates"] = cands[:30]
    st["candidates_date"] = sel_day
    st["selection_log"].append({"date": sel_day, "qualified": len(cands), "added": [c["ticker"] for c in new],
                                "free_slots": free, "with_fundamentals": len(funds)})
    st["selection_log"] = st["selection_log"][-60:]
    if st["started"] is None and new:
        st["started"] = sel_day
    log(f"selection {sel_day}: {len(cands)} qualify, {len(new)} added")
    return new


# ---------------------------------------------------------------- mark to market
def mark(st):
    tickers = {p["ticker"] for p in st["positions"] + st["pending"]}
    if not tickers and not st["closed"]:
        return [], []
    first = min([p.get("entry_date") or p["selected_on"] for p in st["positions"] + st["pending"]] +
                [c["entry_date"] for c in st["closed"]])
    start = (pd.Timestamp(first) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    meta = members_meta()
    all_t = tickers | {c["ticker"] for c in st["closed"]} | set(meta) | {"SPY"}
    O, C, A, _ = download(all_t, start)
    spy = A["SPY"].dropna()
    idx = spy.index
    ret = A.pct_change()
    mem = [t for t in meta if t in ret.columns]
    ew_daily = ret[mem].clip(-0.9, 3).mean(axis=1).reindex(idx).fillna(0)
    filled, exited = [], []
    # fill pending at the open of the session after selection
    still = []
    for p in st["pending"]:
        t = p["ticker"]
        after = [d for d in idx if d > pd.Timestamp(p["selected_on"])]
        if not after or t not in O.columns or not np.isfinite(O.loc[after[0], t]):
            still.append(p)
            continue
        e = after[0]
        p["entry_date"] = e.strftime("%Y-%m-%d")
        p["exit_planned_after_sessions"] = HOLD
        st["positions"].append(p)
        filled.append(p)
    st["pending"] = still

    def adj_open(t, d):
        return float(O.loc[d, t] * (A.loc[d, t] / C.loc[d, t]))

    def window(t, e, x):
        s = A[t].dropna()
        entry = adj_open(t, e)
        px = float(s[s.index <= x].iloc[-1])
        r = px / entry * (1 - FEE) ** 2 - 1
        sp = float(spy.loc[x] / spy[spy.index < e].iloc[-1] - 1)
        ew = float((1 + ew_daily[(ew_daily.index >= e) & (ew_daily.index <= x)]).prod() - 1)
        return r, sp, ew, px, entry

    keep = []
    last = idx[-1]
    for p in st["positions"]:
        t = p["ticker"]
        e = pd.Timestamp(p["entry_date"])
        if t not in A.columns or e not in idx:
            keep.append(p)
            continue
        ei = idx.get_loc(e)
        xi = ei + HOLD - 1
        s = A[t].dropna()
        data_end = s.index[-1] < last - pd.Timedelta(days=7)      # delisted / merged
        if xi <= len(idx) - 1 or data_end:
            x = idx[min(xi, len(idx) - 1)] if not data_end else s.index[-1]
            r, sp, ew, px, entry = window(t, e, x)
            st["closed"].append({**{k: p[k] for k in ("ticker", "name", "sector", "selected_on", "entry_date")},
                                 "exit_date": x.strftime("%Y-%m-%d"), "ret_pct": round(r * 100, 2),
                                 "spy_pct": round(sp * 100, 2), "ew_pct": round(ew * 100, 2),
                                 "reason": "data_end" if data_end else "one_year",
                                 "kind": (p.get("explanation") or {}).get("kind")})
            exited.append(st["closed"][-1])
            continue
        r, sp, ew, px, entry = window(t, e, last)
        p.update({"entry_price": round(entry, 2), "last_price": round(px, 2), "ret_pct": round(r * 100, 2),
                  "spy_pct": round(sp * 100, 2), "ew_pct": round(ew * 100, 2),
                  "sessions_left": int(xi - (len(idx) - 1)),
                  "exit_estimate": (last + pd.tseries.offsets.BDay(int(xi - (len(idx) - 1)))).strftime("%Y-%m-%d")})
        keep.append(p)
    st["positions"] = keep

    # portfolio series: equal weight across whatever is held each day, in SPY when nothing is
    book = [(c["ticker"], pd.Timestamp(c["entry_date"]), pd.Timestamp(c["exit_date"])) for c in st["closed"]] + \
           [(p["ticker"], pd.Timestamp(p["entry_date"]), None) for p in st["positions"]]
    if book:
        s0 = min(b[1] for b in book)
        series, v, vs, ve = [], 100.0, 100.0, 100.0
        spy_r = spy.pct_change().fillna(0)
        for d in idx[idx >= s0]:
            rs = []
            for t, e, x in book:
                if d < e or (x is not None and d > x) or t not in A.columns:
                    continue
                if d == e:
                    r = A.loc[d, t] / adj_open(t, d) - 1 - FEE
                else:
                    r = ret.loc[d, t]
                if x is not None and d == x:
                    r -= FEE
                if np.isfinite(r):
                    rs.append(float(r))
            v *= 1 + (float(np.mean(rs)) if rs else float(spy_r.loc[d]))
            vs *= 1 + float(spy_r.loc[d])
            ve *= 1 + float(ew_daily.loc[d])
            series.append({"date": d.strftime("%Y-%m-%d"), "port": round(v, 2), "spy": round(vs, 2), "ew": round(ve, 2), "n": len(rs)})
        st["series"] = series
        allp = st["closed"] + st["positions"]
        st["stats"] = {
            "port_pct": round(series[-1]["port"] - 100, 2), "spy_pct": round(series[-1]["spy"] - 100, 2),
            "ew_pct": round(series[-1]["ew"] - 100, 2), "since": series[0]["date"],
            "daily_pct": round((series[-1]["port"] / series[-2]["port"] - 1) * 100, 2) if len(series) > 1 else 0.0,
            "trades": len(allp), "closed": len(st["closed"]),
            "beat_spy": sum(1 for p in allp if p.get("ret_pct") is not None and p["ret_pct"] > p["spy_pct"]),
            "beat_ew": sum(1 for p in allp if p.get("ret_pct") is not None and p["ret_pct"] > p["ew_pct"]),
            "by_kind": {k: {"n": len(g), "avg_vs_ew_pct": round(float(np.mean([p["ret_pct"] - p["ew_pct"] for p in g])), 2)}
                        for k in ("market", "sector", "company")
                        for g in [[p for p in allp if p.get("ret_pct") is not None and
                                   ((p.get("explanation") or {}).get("kind") or p.get("kind")) == k]] if g},
        }
    return filled, exited


def main():
    st = load_state()
    force = os.environ.get("FORCE_SELECT", "").lower() == "true"
    is_friday = datetime.now(timezone.utc).weekday() == 4
    added = []
    if force or is_friday or not st.get("selection_log"):
        try:
            added = select(st)
        except Exception as e:
            log(f"selection failed: {type(e).__name__}: {e}")
            st.setdefault("selection_log", []).append({"date": date.today().isoformat(), "error": f"{type(e).__name__}: {e}"[:300]})
    filled, exited = [], []
    try:
        filled, exited = mark(st)
    except Exception as e:
        log(f"mark-to-market failed: {type(e).__name__}: {e}")
    save_state(st)
    lines = []
    if added:
        lines.append("🆕 נבחרו (קנייה בפתיחה הבאה): " + ", ".join(f"{c['ticker']} ({c['drawdown_pct']:.0f}%)" for c in added))
    if exited:
        lines.append("🔚 נסגרו אחרי שנה: " + ", ".join(f"{c['ticker']} {c['ret_pct']:+.1f}% (S&P {c['spy_pct']:+.1f}%)" for c in exited))
    if lines:
        telegram("🏆 מניות מנצחות 🧪\n" + "\n".join(lines) + "\nניסוי - לא הוכח במחקר, לא ייעוץ השקעות.")
    log("done")


if __name__ == "__main__":
    main()
