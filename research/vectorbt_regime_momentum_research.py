"""
מחקר אופליין - לא נוגע בקוד החי, לא רץ ב-cron הרגיל!
מופעל רק דרך .github/workflows/vectorbt_research.yml (workflow_dispatch,
הפעלה ידנית בלבד), שומר תוצאות כ-artifact להורדה.

v2 (2026-09-23) - שלוש תוספות על הגרסה הראשונה, בעקבות הממצא שהיקום
הראשון (50 ענקיות בלו-צ'יפ, 8 שנים) כמעט ואף פעם לא הפעיל את Dual
Momentum/Low-Vol בפועל - לא כי הרעיונות לא עובדים, אלא כי לא נתנו להם
תנאים אמיתיים להיבדק:
1. LOOKBACK_YEARS הוארך ל-26 שנה - כדי לתפוס בפועל משברי-דובי מתמשכים
   (2000-2002, 2008-2009), לא רק את הירידות הקצרות-יחסית של 2019-2026.
2. RESEARCH_UNIVERSE הורחב והתגוון - לא רק ענקיות יציבות (שבהגדרה כמעט
   אף פעם לא מפתחות מומנטום-מוחלט שלילי) אלא גם שמות תנודתיים/מחזוריים
   יותר (טיסות, נסיעות, שבבים קטנים, ביוטק) שבהם ל-Dual Momentum/Low-Vol
   יש סיכוי אמיתי להראות הבדל.
3. קונפיגורציה שישית - "RS מהיר": בונוס לפי מומנטום-יחסי קצר-טווח
   (20-30 יום, אחוזון חוצה-יקום) בכל תאריך-רה-בלנס, בדיוק כמו
   compute_fast_rs_score שכבר חי בפרודקשן (v5.6.0) - נבדק כאן במקביל
   כדי לראות אם התוצאה החיה (שעוד אין לה מספיק מדגם) עקבית עם באקטסט
   היסטורי ארוך.

שאלת המחקר המקורית עדיין בתוקף: האם הוספת הרכיבים האלה משפרת בפועל את
הביצועים בזמן שוק דובי, בלי "להרוג" יותר מדי הזדמנויות אמיתיות בזמן
שוק שורי?

חשוב - שתי מגבלות מכוונות של המחקר הזה, שלא פוגמות בהשוואה בין
הקונפיגורציות (כולן סובלות מהן באופן שווה), אבל צריך לזכור אותן
כשמסתכלים על המספרים המוחלטים:

1. אין רכיב אנליסטים (analyst_score_0_100) - recommendationMean הוא
   "עכשווי" בלבד ב-yfinance, אין לו היסטוריה זמינה בחינם. המחקר משתמש
   רק בשני הרכיבים הטכניים (עוצמת חיזוי מקורית + יחס סיכוי/סיכון),
   בהתאמה ל-DEFAULT_FORMULA_ALPHA=1.0 (המשקל הנוכחי בפרודקשן) בלי הרכיב
   האנליסטי.
2. אין גיוון סקטורים (select_diversified_top10 דורש נתון "sector" שגם
   הוא לא נשמר היסטורית בלי קריאת רשת יקרה לכל תאריך) - הבחירה כאן היא
   Top10 גולמי לפי הציון המשוקלל, בלי מכסת-סקטור.

כל שאר הלוגיקה (compute_technical_factors, compute_prediction_score,
compute_risk_reward_score, compute_blended_top10_score, Trend Template,
data_suspect) מיובאת ומופעלת ישירות מ-stock_alerts.py - לא משוכפלת -
כדי שהמחקר יבדוק את הנוסחה האמיתית, לא גרסה משוערכת שלה.
"""

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import stock_alerts as sa  # noqa: E402  (reuse the real production formula)

# --------------------------------------------------------------------------
# Config - safe to tweak between runs without touching the logic below
# --------------------------------------------------------------------------
LOOKBACK_YEARS = 26             # v2: long enough to include 2000-02 and 2008-09
TOP_N = 10
RISK_FREE_ANNUAL_PCT = 4.0      # simplified constant risk-free proxy (T-bill-ish)
MOMENTUM_LOOKBACK_DAYS = 252    # ~12 months, per the Dual Momentum literature
FAST_MOMENTUM_LOOKBACK_DAYS = 21  # ~1 trading month, matches compute_fast_rs_score in production
FAST_RS_ADDITIVE_MAX = 2.0      # additive nudge scale - see config_fast_rs's docstring for why additive, not multiplicative
LOW_BETA_THRESHOLD = 0.8
LOW_BETA_BONUS = 1.5            # same order of magnitude as the other +/- nudges in compute_prediction_score
MIN_HISTORY_DAYS = 260          # must clear this before a ticker is eligible for a rebalance date
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# v2: expanded and diversified - not just mega-cap blue chips (which almost
# never develop negative absolute momentum, so Dual Momentum/Low-Vol barely
# ever fired on the original 50-name universe) but a deliberate mix across
# volatility/cyclicality profiles, so those signals get a real chance to
# differentiate. Some names won't have the full 26-year history (recent
# IPOs) - that's fine, they simply join the study from whenever they have
# MIN_HISTORY_DAYS of data (handled per-rebalance-date, not by exclusion).
RESEARCH_UNIVERSE = [
    # mega-cap / stable (original core, kept for continuity with v1 results)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "MA", "PG", "HD", "MRK", "ABBV", "COST", "PEP", "KO",
    "AVGO", "CSCO", "TMO", "MCD", "ADBE", "CRM", "ACN", "LIN", "ABT", "DHR",
    "WMT", "NKE", "TXN", "NEE", "PM", "UPS", "ORCL", "INTC", "QCOM", "AMD",
    "HON", "IBM", "UNP", "LOW", "SBUX", "CAT", "GE", "BA", "GS", "AMGN",
    # cyclical / travel / consumer-discretionary (historically hit hard in
    # both 2008-09 and 2020, good test cases for absolute momentum)
    "DAL", "UAL", "AAL", "CCL", "RCL", "MGM", "LVS", "F", "GM", "MAR",
    # energy (volatile, cyclical, historically low-correlation stretches to
    # the broad market - a natural test bed for the low-vol/defensive tilt)
    "SLB", "HAL", "OXY", "MRO", "DVN", "COP",
    # financials (hit especially hard in 2008-09 specifically)
    "C", "BAC", "WFC", "MS", "AIG",
    # volatile tech / biotech / small-mid-cap (high-beta, meant to actually
    # trigger the low-beta-bonus comparison and show real dispersion)
    "PLTR", "COIN", "MRNA", "MSTR", "SMCI", "CRWD", "NET", "DKNG", "RIVN", "SNAP",
]



def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def download_universe():
    tickers = RESEARCH_UNIVERSE + ["SPY"]
    log(f"Downloading {len(tickers)} tickers, {LOOKBACK_YEARS}y history...")
    data = yf.download(
        tickers=" ".join(tickers), period=f"{LOOKBACK_YEARS}y", group_by="ticker",
        threads=True, progress=False, auto_adjust=True,
    )
    frames = {}
    for t in tickers:
        try:
            df = data[t][["Close", "Volume", "High", "Low"]].dropna(how="all")
            if len(df) >= MIN_HISTORY_DAYS:
                frames[t] = df
        except Exception as e:
            log(f"  skipping {t}: {e}")
    log(f"Usable price history for {len(frames)}/{len(tickers)} tickers.")
    return frames


# --------------------------------------------------------------------------
# Historical (point-in-time) equivalents of production helpers that only
# know how to look at "right now" - get_market_regime() in stock_alerts.py
# always downloads a fresh 1y SPY window, which is correct for daily
# production use but useless for a backtest that must stay strictly
# point-in-time at every rebalance date.
# --------------------------------------------------------------------------
def historical_market_regime(spy_close_upto_t):
    s = spy_close_upto_t.dropna()
    if len(s) < 60:
        return {"bullish": None}
    sma50 = float(s.rolling(50).mean().iloc[-1])
    sma200 = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else float(s.mean())
    return {"bullish": bool(sma50 > sma200)}


def absolute_momentum_negative(closes_upto_t):
    """Dual Momentum's 'absolute momentum' leg: trailing ~12-month total
    return vs. a constant risk-free proxy. True = momentum says get out."""
    s = closes_upto_t.dropna()
    if len(s) < MOMENTUM_LOOKBACK_DAYS + 1:
        return None
    total_return_pct = float((s.iloc[-1] - s.iloc[-MOMENTUM_LOOKBACK_DAYS]) / s.iloc[-MOMENTUM_LOOKBACK_DAYS] * 100)
    return total_return_pct < RISK_FREE_ANNUAL_PCT


def rolling_beta(closes_upto_t, spy_closes_upto_t):
    s = closes_upto_t.pct_change().dropna()
    m = spy_closes_upto_t.pct_change().dropna()
    joined = pd.concat([s, m], axis=1, join="inner").tail(MOMENTUM_LOOKBACK_DAYS)
    if len(joined) < 60:
        return None
    joined.columns = ["stock", "mkt"]
    var = joined["mkt"].var()
    if not var or np.isnan(var):
        return None
    cov = joined["stock"].cov(joined["mkt"])
    return float(cov / var)


def fast_momentum_return(closes_upto_t):
    """v2: trailing ~1-month return, feeds the cross-sectional 'RS מהיר'
    percentile below - same window as production's run_up_30d."""
    s = closes_upto_t.dropna()
    if len(s) < FAST_MOMENTUM_LOOKBACK_DAYS + 1:
        return None
    return float((s.iloc[-1] - s.iloc[-FAST_MOMENTUM_LOOKBACK_DAYS]) / s.iloc[-FAST_MOMENTUM_LOOKBACK_DAYS] * 100)


# --------------------------------------------------------------------------
# One rebalance date: score every ticker with the REAL production formula,
# then build all config variants from the same scored universe.
# --------------------------------------------------------------------------
def score_universe_at(frames, spy_close_full, t_idx, dates):
    asof = dates[t_idx]
    market_regime = historical_market_regime(spy_close_full.loc[:asof])
    entries = {}
    for ticker, df in frames.items():
        if ticker == "SPY":
            continue
        sl = df.loc[:asof]
        if len(sl) < MIN_HISTORY_DAYS:
            continue
        tf = sa.compute_technical_factors(sl["Close"], sl["Volume"], sl["High"], sl["Low"])
        if not tf:
            continue
        score = sa.compute_prediction_score(tf, market_regime)
        multiplier = (tf.get("trend_template") or {}).get("multiplier", 1.0)
        score = round(score * multiplier, 2)
        predicted = "up" if score >= 0 else "down"
        entry = {"ticker": ticker, "score": score, "predicted": predicted, **tf}
        entry["mom_negative"] = absolute_momentum_negative(sl["Close"])
        entry["beta"] = rolling_beta(sl["Close"], spy_close_full.loc[:asof])
        entry["fast_momentum_return"] = fast_momentum_return(sl["Close"])
        entries[ticker] = entry

    # v2: cross-sectional "RS מהיר" percentile at this date, same style as
    # production's fast_rs_rating (see run_predictions in stock_alerts.py)
    fast_pairs = [(t, e["fast_momentum_return"]) for t, e in entries.items() if e["fast_momentum_return"] is not None]
    if fast_pairs:
        ranked = sorted(fast_pairs, key=lambda p: p[1])
        total = len(ranked)
        for rank, (ticker, _) in enumerate(ranked):
            entries[ticker]["fast_rs_rating"] = round(rank / max(total - 1, 1) * 100, 1)

    return market_regime, entries


def select_top_n(entries, score_fn, n=TOP_N):
    breadth = [e for e in entries.values()
               if abs(e["score"]) >= sa.PREDICTION_SCORE_THRESHOLD and not e.get("data_suspect")]
    if not breadth:
        return []
    rank_a = {e["ticker"]: i for i, e in enumerate(sorted(breadth, key=lambda e: abs(e["score"]), reverse=True))}
    rank_b = {e["ticker"]: i for i, e in enumerate(sorted(breadth, key=sa.compute_risk_reward_score, reverse=True))}
    scored = sorted(breadth, key=lambda e: score_fn(e, rank_a, rank_b), reverse=True)
    return [e["ticker"] for e in scored[:n]]


def config_baseline(e, rank_a, rank_b):
    return sa.compute_blended_top10_score(e, 1.0, rank_a, rank_b)


def config_dual_momentum_gate(e, rank_a, rank_b):
    if e["predicted"] == "up" and e.get("mom_negative") is True:
        return -1e9  # excluded, binary gate
    return sa.compute_blended_top10_score(e, 1.0, rank_a, rank_b)


def config_dual_momentum_penalty(e, rank_a, rank_b):
    base = sa.compute_blended_top10_score(e, 1.0, rank_a, rank_b)
    if e["predicted"] == "up" and e.get("mom_negative") is True:
        return base - abs(base) * 0.5  # graduated penalty, not exclusion
    return base


def config_lowvol_tilt(regime):
    def fn(e, rank_a, rank_b):
        base = sa.compute_blended_top10_score(e, 1.0, rank_a, rank_b)
        if regime.get("bullish") is False and e.get("beta") is not None and e["beta"] < LOW_BETA_THRESHOLD:
            base += LOW_BETA_BONUS
        return base
    return fn


def config_combined(regime):
    def fn(e, rank_a, rank_b):
        if e["predicted"] == "up" and e.get("mom_negative") is True:
            return -1e9
        base = sa.compute_blended_top10_score(e, 1.0, rank_a, rank_b)
        if regime.get("bullish") is False and e.get("beta") is not None and e["beta"] < LOW_BETA_THRESHOLD:
            base += LOW_BETA_BONUS
        return base
    return fn


def config_fast_rs(e, rank_a, rank_b):
    """v2: cross-sectional ~1-month RS percentile (fast_rs_rating, see
    score_universe_at), nudging the SAME rank-based blended score every
    other config here uses. Deliberately ADDITIVE, not multiplicative -
    compute_blended_top10_score returns a rank-based value that's usually
    negative (closer to 0 = better rank), so multiplying it by a >1 factor
    would push a good rank the WRONG way. Scaled to the same rough
    magnitude as LOW_BETA_BONUS above, not copied from production's
    compute_fast_rs_score (which multiplies abs(raw score) - a different,
    always-positive quantity that multiplication works correctly on)."""
    base = sa.compute_blended_top10_score(e, 1.0, rank_a, rank_b)
    fast_rs = e.get("fast_rs_rating")
    if fast_rs is not None:
        base += (fast_rs - 50) / 50 * FAST_RS_ADDITIVE_MAX
    return base


# --------------------------------------------------------------------------
# Backtest loop
# --------------------------------------------------------------------------
def forward_basket_return(frames, tickers, t_date, t_next_date):
    if not tickers:
        return None
    rets = []
    for tk in tickers:
        df = frames.get(tk)
        if df is None:
            continue
        try:
            p0 = df["Close"].asof(t_date)
            p1 = df["Close"].asof(t_next_date)
            if p0 and p1 and p0 > 0:
                rets.append((p1 - p0) / p0)
        except Exception:
            continue
    return float(np.mean(rets)) if rets else None


def max_drawdown(period_returns):
    equity = np.cumprod([1 + r for r in period_returns])
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def summarize(rows, config_name):
    df = pd.DataFrame(rows)
    df = df[df[config_name].notna()]
    if df.empty:
        return {"config": config_name, "n_periods": 0}
    rets = df[config_name].tolist()
    bull = df[df["regime_bullish"] == True][config_name].tolist()
    bear = df[df["regime_bullish"] == False][config_name].tolist()
    total_return = float(np.prod([1 + r for r in rets]) - 1)
    n_years = max(len(rets) / 12, 0.01)
    annualized = float((1 + total_return) ** (1 / n_years) - 1)
    return {
        "config": config_name,
        "n_periods": len(rets),
        "total_return_pct": round(total_return * 100, 1),
        "annualized_return_pct": round(annualized * 100, 1),
        "max_drawdown_pct": round(max_drawdown(rets) * 100, 1),
        "win_rate_pct": round(100 * sum(1 for r in rets if r > 0) / len(rets), 1),
        "bull_periods": len(bull),
        "bull_avg_return_pct": round(100 * float(np.mean(bull)), 2) if bull else None,
        "bear_periods": len(bear),
        "bear_avg_return_pct": round(100 * float(np.mean(bear)), 2) if bear else None,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = download_universe()
    if "SPY" not in frames:
        log("FATAL: could not load SPY data, aborting.")
        sys.exit(1)
    spy_close = frames["SPY"]["Close"]

    all_dates = spy_close.index
    month_end_positions = [
        i for i in range(MIN_HISTORY_DAYS, len(all_dates) - 1)
        if all_dates[i].month != all_dates[i + 1].month
    ]
    log(f"{len(month_end_positions)} monthly rebalance points to evaluate.")

    configs = {
        "baseline": config_baseline,
        "dual_momentum_gate": config_dual_momentum_gate,
        "dual_momentum_penalty": config_dual_momentum_penalty,
        "fast_rs": config_fast_rs,
    }

    rows = []
    for n, t_idx in enumerate(month_end_positions[:-1]):
        t_date = all_dates[t_idx]
        t_next_date = all_dates[month_end_positions[n + 1]] if n + 1 < len(month_end_positions) else all_dates[-1]
        regime, entries = score_universe_at(frames, spy_close, t_idx, all_dates)
        row = {"date": t_date.date().isoformat(), "regime_bullish": regime.get("bullish")}

        for name, fn in configs.items():
            picks = select_top_n(entries, fn)
            row[name] = forward_basket_return(frames, picks, t_date, t_next_date)

        picks_lowvol = select_top_n(entries, config_lowvol_tilt(regime))
        row["lowvol_tilt"] = forward_basket_return(frames, picks_lowvol, t_date, t_next_date)

        picks_combined = select_top_n(entries, config_combined(regime))
        row["combined"] = forward_basket_return(frames, picks_combined, t_date, t_next_date)

        rows.append(row)
        if n % 6 == 0:
            log(f"  processed {n}/{len(month_end_positions) - 1} rebalance points ({t_date.date()})...")

    detail_df = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "vectorbt_research_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    log(f"Wrote per-period detail: {detail_path}")

    all_config_names = ["baseline", "dual_momentum_gate", "dual_momentum_penalty", "fast_rs", "lowvol_tilt", "combined"]
    summary = [summarize(rows, name) for name in all_config_names]
    summary_path = OUTPUT_DIR / "vectorbt_research_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": date.today().isoformat(),
            "backend_version_tested_against": sa.BACKEND_VERSION,
            "lookback_years": LOOKBACK_YEARS,
            "universe_size": len(RESEARCH_UNIVERSE),
            "limitations": [
                "no analyst-score component (no historical recommendationMean data)",
                "no sector diversification cap (no point-in-time sector data)",
            ],
            "results": summary,
        }, f, ensure_ascii=False, indent=2)
    log(f"Wrote summary: {summary_path}")

    log("\n=== SUMMARY ===")
    for s in summary:
        log(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
