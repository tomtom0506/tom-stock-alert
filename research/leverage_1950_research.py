"""
🔬 מחקר: השערה 4 (מינוף בשוק עולה) - בדיקה עצמאית על 1950-1999.
אופליין בלבד. מופעל רק דרך .github/workflows/research.yml (הפעלה ידנית).
לא נוגע ב-stock_alerts.py, ב-index.html או בשום קובץ של האפליקציה החיה.

אושר ע"י תומר ב-21.11.2026. הכלל והרף זהים להשערה 4 ונקבעו מראש
(research/2026-11-21-leveraged-exposure-1950.md).

Index proxy = ^GSPC price + an assumed 3.5%/yr dividend (yfinance has no index
dividends this far back); 2x fund = 2 x proxy - T-bill financing - 0.89%/yr.
Main result (the one judged against the bar) includes the dividend; a price-only
run is reported alongside. One run only - there is no holdout for this period.

Usage:  python research/leverage_1950_research.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_lowvol_research as B  # noqa: E402
import leverage_exposure_research as L  # noqa: E402

OUTPUT_DIR = HERE / "output_leverage_1950"
DATA_START = "1948-01-01"           # 200-day warm-up
WINDOW_START = pd.Timestamp("1950-01-01")
WINDOW_END = pd.Timestamp("1999-12-31")
DIV_ANNUAL = 0.035
DECADES_1950 = [(f"{y}s", f"{y}-01-01", f"{y + 9}-12-31") for y in range(1950, 2000, 10)]


# ---------------------------------------------------------------- pure helpers (tested)
def window_start(irx):
    first = irx.dropna().index.min()
    if first is None or pd.isna(first):
        raise RuntimeError("no ^IRX data")
    return max(WINDOW_START, first + pd.Timedelta(days=1))


def build_returns(gspc, irx, div_annual):
    days = gspc.index
    cash_ret = (irx.reindex(days).ffill() / 100 / 252).shift(1).fillna(0.0)
    idx_ret = gspc.pct_change(fill_method=None) + div_annual / 252
    lev_ret = L.synthetic_2x(idx_ret, cash_ret)
    return pd.DataFrame({B.ETF: idx_ret, L.LEV: lev_ret, B.CASH: cash_ret}, index=days), cash_ret


def decade_table(eq, cash_ret):
    out = {}
    for name, a, b in DECADES_1950:
        s = eq[(eq.index >= a) & (eq.index <= b)]
        if len(s) > 20:
            out[name] = B.metrics(s, cash_ret)
    return out


def run_variant(gspc, irx, start, end, div_annual):
    rets, cash_ret = build_returns(gspc, irx, div_annual)
    above = B.regime_above(gspc)
    days = rets.index
    targets = {"h4_leverage": L.leverage_targets(above, days),
               "baseline": B.baseline_targets(above, days),
               "index_buy_hold": {d: {B.ETF: 1.0} for d in days}}
    res, curves = {}, {}
    for name, tg in targets.items():
        eq, trades, n_reb = B.simulate(B.orders_from_targets(tg, start, end), rets, start, end)
        m = B.metrics(eq, cash_ret)
        w1, w5 = L.worst_moves(eq)
        m.update({"trades": len(trades), "rebalances": n_reb, "costs": round(sum(t["cost"] for t in trades), 2),
                  "final_value": round(float(eq.iloc[-1]), 2), "worst_day": w1, "worst_5d": w5,
                  "worst_day_date": str(eq.pct_change().idxmin().date()),
                  "decades": decade_table(eq, cash_ret)})
        res[name], curves[name] = m, eq
    return res, curves


# ---------------------------------------------------------------- main (network)
def main():
    import yfinance as yf
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    data = yf.download(f"{B.INDEX} {B.TBILL}", start=DATA_START, end="2000-01-10", auto_adjust=False,
                       group_by="ticker", progress=False)

    def col(t):
        s = data[t]["Close"].dropna()
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        return s
    gspc, irx = col(B.INDEX), col(B.TBILL)
    start = window_start(irx)
    end = min(WINDOW_END, gspc.index[-1])
    B.log(f"window {start.date()} - {end.date()}, ^GSPC from {gspc.index[0].date()}, ^IRX from {irx.index[0].date()}")

    main_res, curves = run_variant(gspc, irx, start, end, DIV_ANNUAL)
    price_res, _ = run_variant(gspc, irx, start, end, 0.0)
    passed = bool(L.verdict(main_res["h4_leverage"], main_res["index_buy_hold"]))
    passed_price = bool(L.verdict(price_res["h4_leverage"], price_res["index_buy_hold"]))
    pd.DataFrame(curves).to_csv(out_dir / "equity_curves.csv")
    summary = {"period": [str(start.date()), str(end.date())], "dividend_assumed": DIV_ANNUAL,
               "results": main_res, "results_price_only": price_res,
               "pass": passed, "pass_price_only": passed_price}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "SUMMARY.md").write_text(render_md(summary))
    B.log(f"done: H4-1950 {'PASS' if passed else 'FAIL'} (price-only {'PASS' if passed_price else 'FAIL'})")


def render_md(s):
    names = {"h4_leverage": "השערה 4: 150% מעל / 50% מתחת", "baseline": "כלל החשיפה הנוכחי",
             "index_buy_hold": "מדד קנה והחזק"}

    def table(R):
        L_ = ["| אסטרטגיה | תשואה שנתית | ירידה מקסימלית | Sharpe | יום גרוע (תאריך) | 5 ימים גרועים | עסקאות | שווי סופי $ |",
              "|---|---|---|---|---|---|---|---|"]
        for k, n in names.items():
            r = R[k]
            L_.append(f"| {n} | {B.pct(r['cagr'])} | {B.pct(r['max_dd'])} | {B.num(r['sharpe'])} | "
                      f"{B.pct(r['worst_day'])} ({r['worst_day_date']}) | {B.pct(r['worst_5d'])} | {r['trades']} | "
                      f"{r['final_value']:,.0f} |")
        return L_
    L_ = [f"# בדיקה עצמאית {s['period'][0]} עד {s['period'][1]}", "",
          f"## עיקרי (עם דיבידנד משוער {s['dividend_assumed'] * 100:.1f}%) — {'✅ עומדת ברף' if s['pass'] else '❌ לא עומדת ברף'}", ""]
    L_ += table(s["results"])
    L_ += ["", f"## בלי דיבידנד (לעיון) — {'✅' if s['pass_price_only'] else '❌'}", ""]
    L_ += table(s["results_price_only"])
    L_ += ["", "## לפי עשור (עיקרי)", "", "| אסטרטגיה | עשור | תשואה שנתית | ירידה מקסימלית |", "|---|---|---|---|"]
    for k, n in names.items():
        for dec, m in s["results"][k]["decades"].items():
            L_.append(f"| {n} | {dec} | {B.pct(m['cagr'])} | {B.pct(m['max_dd'])} |")
    return "\n".join(L_) + "\n"


if __name__ == "__main__":
    main()
