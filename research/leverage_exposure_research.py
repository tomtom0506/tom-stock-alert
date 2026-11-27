"""
🔬 מחקר: כלל החשיפה עם מינוף בשוק עולה (השערה 4).
אופליין בלבד. מופעל רק דרך .github/workflows/research.yml (הפעלה ידנית).
לא נוגע ב-stock_alerts.py, ב-index.html או בשום קובץ של האפליקציה החיה.

אושר ע"י תומר ב-21.11.2026. הכלל ורף ההצלחה נקבעו מראש
(research/2026-11-21-leveraged-exposure.md) ואסור לשנות אותם אחרי שרואים תוצאות.

Rule: ^GSPC above its 200-day average -> 50% SPY + 50% SSO (2x daily) = 150% exposure;
      below -> 50% SPY + 50% cash (the current rule). Re-balanced the day after every
      regime switch and on the first trading day of each month.
SSO: real from its first trading day (06.2006); before that synthetic =
      2 x SPY total return - T-bill financing on 100% notional - 0.89%/yr fee.
Costs, cash, start capital, test/holdout split: identical to breadth_lowvol_research.

Usage:  python research/leverage_exposure_research.py            # test period only
        python research/leverage_exposure_research.py --holdout  # holdout only - ONE run, after the test results are written
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_lowvol_research as B  # noqa: E402  (costs, simulator, metrics, regime)

OUTPUT_DIR = HERE / "output_leverage_exposure"
LEV = "SSO"
LEV_FEE_ANNUAL = 0.0089
W_ABOVE = {B.ETF: 0.5, LEV: 0.5}
W_BELOW = {B.ETF: B.REDUCED}
BAR_EXCESS_CAGR = 0.02


# ---------------------------------------------------------------- pure helpers (tested)
def synthetic_2x(spy_ret, cash_ret, fee_annual=LEV_FEE_ANNUAL):
    """Daily return of a 2x daily-reset fund: 2x the index, minus financing of the
    borrowed 1x at the T-bill rate, minus the fund fee."""
    return 2 * spy_ret - cash_ret - fee_annual / 252


def splice(real_ret, synth_ret):
    """Real fund returns where it has data, synthetic before its first day."""
    first = real_ret.first_valid_index()
    out = synth_ret.copy()
    if first is not None:
        out[out.index >= first] = real_ret[real_ret.index >= first]
    return out


def leverage_targets(above, days):
    out = {}
    for d in days:
        a = above.get(d)
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        if bool(a):
            t = dict(W_ABOVE)
            t["_m"] = d.strftime("%Y-%m")           # monthly re-balance back to 50/50
            out[d] = t
        else:
            out[d] = dict(W_BELOW)
    return out


def verdict(strat, spy):
    return strat["cagr"] >= spy["cagr"] + BAR_EXCESS_CAGR and abs(strat["max_dd"]) <= abs(spy["max_dd"])


def worst_moves(eq):
    r1 = eq.pct_change().min()
    r5 = (eq / eq.shift(5) - 1).min()
    return float(r1), float(r5)


# ---------------------------------------------------------------- data (network)
def load_data():
    import yfinance as yf
    gspc, spy, irx = B.load_market()
    raw = yf.download(LEV, start=B.PRICE_START, auto_adjust=False, progress=False)
    sso = raw["Adj Close"]
    if isinstance(sso, pd.DataFrame):
        sso = sso.iloc[:, 0]
    sso = sso.dropna()
    if getattr(sso.index, "tz", None) is not None:
        sso.index = sso.index.tz_localize(None)
    days = gspc.index.intersection(spy.index)
    days = days[days >= pd.Timestamp(B.PRICE_START)]
    cash_ret = (irx.reindex(days).ffill() / 100 / 252).shift(1).fillna(0.0)
    spy_ret = spy.reindex(days).pct_change(fill_method=None)
    real = sso.reindex(days).pct_change(fill_method=None)
    real[real.index <= sso.index[0]] = np.nan              # no return on its first day
    synth = synthetic_2x(spy_ret, cash_ret)
    lev_ret = splice(real, synth)
    rets = pd.DataFrame({B.ETF: spy_ret, LEV: lev_ret, B.CASH: cash_ret}, index=days)
    overlap = real.dropna().index
    track = {"first_real_day": str(sso.index[0].date()),
             "synthetic_vs_real_annual_gap": float((synth[overlap].mean() - real[overlap].mean()) * 252),
             "daily_corr": float(np.corrcoef(synth[overlap], real[overlap])[0, 1])}
    return gspc.reindex(days), rets, cash_ret, days, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="run the 5-year holdout ONCE, after the test results are written")
    args = ap.parse_args()
    start, end = B.window(args.holdout)
    tag = "holdout" if args.holdout else "test"
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    gspc, rets, cash_ret, days, track = load_data()
    end = min(end, days[-1])
    above = B.regime_above(gspc)
    targets = {"h4_leverage": leverage_targets(above, days),
               "baseline": B.baseline_targets(above, days),
               "spy_buy_hold": {d: {B.ETF: 1.0} for d in days}}
    results, curves = {}, {}
    for name, tg in targets.items():
        B.log(f"simulate {name}")
        eq, trades, n_reb = B.simulate(B.orders_from_targets(tg, start, end), rets, start, end)
        curves[name] = eq
        m = B.metrics(eq, cash_ret)
        w1, w5 = worst_moves(eq)
        m.update({"trades": len(trades), "rebalances": n_reb, "costs": round(sum(t["cost"] for t in trades), 2),
                  "final_value": round(float(eq.iloc[-1]), 2), "worst_day": w1, "worst_5d": w5,
                  "decades": B.decade_table(eq, cash_ret)})
        results[name] = m
        pd.DataFrame(trades).to_csv(out_dir / f"trades_{name}.csv", index=False)
    passed = bool(verdict(results["h4_leverage"], results["spy_buy_hold"]))
    pd.DataFrame(curves).to_csv(out_dir / "equity_curves.csv")
    summary = {"period": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")], "run": tag,
               "results": results, "pass": passed, "sso_tracking": track}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "SUMMARY.md").write_text(render_md(summary))
    B.log(f"done: H4 {'PASS' if passed else 'FAIL'}")


def render_md(s):
    R = s["results"]
    names = {"h4_leverage": "השערה 4: 150% מעל / 50% מתחת", "baseline": "כלל החשיפה הנוכחי",
             "spy_buy_hold": "S&P 500 (SPY)"}
    L = [f"# תוצאות ({'תקופת בקרה' if s['run'] == 'holdout' else 'תקופת בדיקה'}) {s['period'][0]} עד {s['period'][1]}", "",
         "| אסטרטגיה | תשואה שנתית | ירידה מקסימלית | Sharpe | יום גרוע | 5 ימים גרועים | עסקאות | עלויות $ | שווי סופי $ |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, n in names.items():
        r = R[k]
        L.append(f"| {n} | {B.pct(r['cagr'])} | {B.pct(r['max_dd'])} | {B.num(r['sharpe'])} | {B.pct(r['worst_day'])} | "
                 f"{B.pct(r['worst_5d'])} | {r['trades']} | {r['costs']:,.0f} | {r['final_value']:,.0f} |")
    L += ["", f"**השערה 4:** {'✅ עומדת ברף' if s['pass'] else '❌ לא עומדת ברף'}", "",
          "## לפי עשור", "", "| אסטרטגיה | עשור | תשואה שנתית | ירידה מקסימלית |", "|---|---|---|---|"]
    for k, n in names.items():
        for dec, m in R[k]["decades"].items():
            L.append(f"| {n} | {dec} | {B.pct(m['cagr'])} | {B.pct(m['max_dd'])} |")
    t = s["sso_tracking"]
    L += ["", "## בדיקת SSO מסונתז מול אמיתי (בתקופה המשותפת)", "",
          f"SSO אמיתי מ-{t['first_real_day']}. פער שנתי ממוצע (סינתטי פחות אמיתי): {B.pct(t['synthetic_vs_real_annual_gap'], 2)}. "
          f"מתאם יומי: {t['daily_corr']:.4f}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
