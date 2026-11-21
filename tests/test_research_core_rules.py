"""Unit tests for the 20.11.2026 core-rule research scripts (research/ only, offline,
synthetic data - no network). Run: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research"))
import breadth_lowvol_research as B  # noqa: E402
import strong_stocks_research as S  # noqa: E402
import leverage_exposure_research as L  # noqa: E402


def bdays(n, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


class ThrustEvents(unittest.TestCase):
    def test_fast_thrust_is_one_event(self):
        d = bdays(8)
        b = pd.Series([0.5, 0.15, 0.3, 0.55, 0.6, 0.45, 0.55, 0.6], index=d)
        self.assertEqual(B.thrust_events(b), [d[3]])

    def test_new_low_rearms(self):
        d = bdays(8)
        b = pd.Series([0.15, 0.55, 0.3, 0.1, 0.4, 0.52, 0.6, 0.6], index=d)
        self.assertEqual(B.thrust_events(b), [d[1], d[5]])

    def test_slow_recovery_is_not_a_thrust(self):
        d = bdays(13)
        vals = [0.15] + [0.3] * 10 + [0.55, 0.6]       # low 11 trading days before the high
        self.assertEqual(B.thrust_events(pd.Series(vals, index=d)), [])

    def test_exactly_ten_days_counts(self):
        d = bdays(12)
        vals = [0.15] + [0.3] * 9 + [0.55, 0.6]        # high exactly 10 days after the low
        self.assertEqual(B.thrust_events(pd.Series(vals, index=d)), [d[10]])

    def test_nan_breadth_ignored(self):
        d = bdays(4)
        b = pd.Series([np.nan, 0.1, np.nan, 0.6], index=d)
        self.assertEqual(B.thrust_events(b), [d[3]])


class Breadth(unittest.TestCase):
    def test_breadth_counts_members_only(self):
        d = bdays(60)
        up = pd.Series(np.linspace(10, 20, 60), index=d)
        down = pd.Series(np.linspace(20, 10, 60), index=d)
        adj = pd.DataFrame({"A": up, "B": up, "C": down})
        member = pd.DataFrame(True, index=d, columns=adj.columns)
        member["B"] = False
        b, n = B.breadth_series(adj, member, sma_days=50, min_names=2)
        self.assertAlmostEqual(b.iloc[-1], 0.5)
        self.assertEqual(int(n.iloc[-1]), 2)
        self.assertTrue(np.isnan(b.iloc[10]))           # no 50-day average yet

    def test_min_names(self):
        d = bdays(60)
        adj = pd.DataFrame({"A": np.linspace(10, 20, 60)}, index=d)
        b, _ = B.breadth_series(adj, pd.DataFrame(True, index=d, columns=["A"]), sma_days=50, min_names=2)
        self.assertTrue(b.isna().all())


class Membership(unittest.TestCase):
    def test_point_in_time(self):
        rows = [(pd.Timestamp("2020-01-01"), {"A", "B"}), (pd.Timestamp("2020-01-08"), {"A", "C"})]
        d = bdays(10, "2020-01-01")
        m = B.membership_matrix(rows, d, ["A", "B", "C"])
        self.assertTrue(m.loc["2020-01-07", "B"])
        self.assertFalse(m.loc["2020-01-08", "B"])
        self.assertTrue(m.loc["2020-01-08", "C"])
        self.assertFalse(m.loc["2020-01-07", "C"])


class Costs(unittest.TestCase):
    def test_minimum_commission(self):
        self.assertAlmostEqual(B.trade_cost(1000, B.ETF), 8 + 0.5)

    def test_percentage_commission_and_stock_slippage(self):
        self.assertAlmostEqual(B.trade_cost(-100_000, "AAPL"), 100 + 100)

    def test_small_trade_skipped_but_closing_not(self):
        pos = {B.CASH: 0.0, "A": 100.0, "B": 10_000.0}
        tr = B.rebalance(pos, {"B": 10_100.0})           # +$100 on B is below MIN_TRADE
        self.assertEqual([t["asset"] for t in tr], ["A"])  # A ($100) is still closed
        self.assertNotIn("A", pos)
        self.assertEqual(pos["B"], 10_000.0)


class Simulator(unittest.TestCase):
    def setUp(self):
        self.d = bdays(5)
        self.rets = pd.DataFrame({B.ETF: [0.0, 0.01, 0.01, 0.01, 0.01], B.CASH: 0.0}, index=self.d)

    def test_executes_next_day_at_close(self):
        eq, trades, n = B.simulate({self.d[0]: {B.ETF: 1.0}}, self.rets, self.d[0], self.d[-1], initial=100_000)
        self.assertEqual(trades[0]["date"], self.d[1].strftime("%Y-%m-%d"))
        self.assertAlmostEqual(eq.iloc[1], 100_000 - B.trade_cost(100_000, B.ETF))  # day-1 return missed
        self.assertAlmostEqual(eq.iloc[-1] / eq.iloc[1], 1 + (1.01 ** 3 - 1) * (1 + 150 / (100_000 - 150)), places=4)
        self.assertEqual(n, 1)

    def test_nan_return_held_flat(self):
        r = self.rets.copy()
        r["X"] = [0.0, 0.0, np.nan, 0.1, 0.0]
        eq, _, _ = B.simulate({self.d[0]: {"X": 1.0}}, r, self.d[0], self.d[-1], initial=10_000)
        self.assertTrue(np.isfinite(eq).all())

    def test_underscore_keys_ignored(self):
        eq, trades, _ = B.simulate({self.d[0]: {B.ETF: 1.0, "_m": "2020-01"}}, self.rets, self.d[0], self.d[-1])
        self.assertEqual([t["asset"] for t in trades], [B.ETF])


class Metrics(unittest.TestCase):
    def test_max_drawdown(self):
        d = bdays(4)
        m = B.metrics(pd.Series([100, 120, 90, 130], index=d), pd.Series(0.0, index=d))
        self.assertAlmostEqual(m["max_dd"], -0.25)


class Targets(unittest.TestCase):
    def test_baseline_and_thrust(self):
        d = bdays(6)
        above = pd.Series([True, False, False, False, False, False], index=d, dtype=object)
        base = B.baseline_targets(above, d)
        self.assertEqual(base[d[0]], {B.ETF: 1.0})
        self.assertEqual(base[d[2]], {B.ETF: B.REDUCED})
        th = B.thrust_targets(above, d, [d[2]], hold=2)
        self.assertEqual([th[x][B.ETF] for x in d], [1.0, 0.5, 1.0, 1.0, 0.5, 0.5])

    def test_orders_only_on_change(self):
        d = bdays(5)
        tg = {x: {B.ETF: 1.0} for x in d}
        tg[d[3]] = {B.ETF: 0.5}
        self.assertEqual(sorted(B.orders_from_targets(tg, d[0], d[-1])), [d[0], d[3], d[4]])

    def test_lowvol_pick_members_only(self):
        vol = pd.Series({"A": 0.01, "B": 0.005, "C": 0.02, "D": np.nan})
        mem = pd.Series({"A": True, "B": False, "C": True, "D": True})
        self.assertEqual(B.lowvol_pick(vol, mem, n=2), ["A", "C"])

    def test_lowvol_monthly_reselect(self):
        d = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
        above = pd.Series([True, False, False, False], index=d, dtype=object)
        vol = pd.DataFrame({"A": [0.01, 0.01, 0.03, 0.03], "B": [0.02, 0.02, 0.01, 0.01]}, index=d)
        mem = pd.DataFrame(True, index=d, columns=["A", "B"])
        tg = B.lowvol_targets(above, d, vol, mem, n=1)
        self.assertEqual(tg[d[0]], {B.ETF: 1.0})
        self.assertIn("A", tg[d[1]])
        self.assertIn("B", tg[d[2]])                    # new month -> re-picked
        self.assertEqual(tg[d[2]], tg[d[3]])           # same month -> no new order
        self.assertEqual(len(B.orders_from_targets(tg, d[0], d[-1])), 3)


class Verdicts(unittest.TestCase):
    def test_h1_needs_return_and_no_bigger_dd(self):
        base = {"cagr": 0.08, "max_dd": -0.30}
        self.assertTrue(B.verdict_h1({"cagr": 0.09, "max_dd": -0.30}, base))
        self.assertFalse(B.verdict_h1({"cagr": 0.09, "max_dd": -0.31}, base))
        self.assertFalse(B.verdict_h1({"cagr": 0.08, "max_dd": -0.20}, base))

    def test_h2_allows_two_points_dd(self):
        base = {"cagr": 0.08, "max_dd": -0.30}
        self.assertTrue(B.verdict_h2({"cagr": 0.081, "max_dd": -0.32}, base))
        self.assertFalse(B.verdict_h2({"cagr": 0.081, "max_dd": -0.3201}, base))

    def test_h3_bar(self):
        spy = {"cagr": 0.08, "max_dd": -0.50}
        good = {"cagr": 0.10, "max_dd": -0.55, "trades_per_month": 2.0}
        self.assertTrue(S.verdict(good, spy))
        self.assertFalse(S.verdict({**good, "cagr": 0.0999}, spy))
        self.assertFalse(S.verdict({**good, "max_dd": -0.551}, spy))
        self.assertFalse(S.verdict({**good, "trades_per_month": 2.01}, spy))


class StrongStocks(unittest.TestCase):
    def test_buffer_keeps_until_out_of_top45(self):
        ranked = [f"T{i}" for i in range(60)]
        keep, new = S.select_with_buffer(ranked, {"T40", "T50", "T2"})
        self.assertEqual(keep, ["T2", "T40"])
        self.assertNotIn("T50", keep + new)
        self.assertEqual(len(keep) + len(new), 15)
        self.assertEqual(new[:3], ["T0", "T1", "T3"])

    def test_profitability_only_from_2010(self):
        mom = pd.Series({"A": 0.5, "B": 0.4, "C": 0.3})
        mem = pd.Series(True, index=mom.index)
        prof = lambda t, d: t != "A"  # noqa: E731
        self.assertEqual(S.ranked_eligible_at(pd.Timestamp("2005-01-31"), mom, mem, prof), ["A", "B", "C"])
        self.assertEqual(S.ranked_eligible_at(pd.Timestamp("2015-01-30"), mom, mem, prof), ["B", "C"])

    def test_momentum_skips_last_month(self):
        d = bdays(300)
        px = pd.DataFrame({"A": np.r_[np.full(279, 100.0), np.full(21, 200.0)]}, index=d)
        px.iloc[300 - 1 - 252, 0] = 50.0
        m = S.momentum_12_1(px)
        self.assertAlmostEqual(m["A"].iloc[-1], 100 / 50 - 1)   # last-month jump ignored

    def test_exposure_variant_scales_on_switch_and_trades(self):
        d = pd.bdate_range("2020-01-01", "2020-03-31")
        above = pd.Series(True, index=d, dtype=object)
        above[d >= "2020-02-10"] = False
        tickers = [f"T{i}" for i in range(20)]
        rank_fn = lambda day: tickers  # noqa: E731
        orders, log = S.build_orders(d, d[0], d[-1], rank_fn, above)
        kinds = [getattr(orders[x], "kind") for x in sorted(orders)]
        self.assertEqual(kinds, ["select", "scale", "select", "select"])
        rets = pd.DataFrame(0.0, index=d, columns=tickers + [B.ETF, B.CASH])
        eq, trades, _ = B.simulate(orders, rets, d[0], d[-1], initial=100_000)
        held = [t for t in trades if t["kind"] == "select"]
        self.assertEqual(len(held), 15)                 # bought once, never re-traded (same ranking)
        self.assertEqual(sum(t["kind"] == "scale" for t in trades), 15)
        self.assertGreater(eq.iloc[-1], 0)
        self.assertEqual(log[0]["held"], 15)

    def test_month_ends(self):
        d = pd.bdate_range("2020-01-01", "2020-03-31")
        self.assertEqual(sorted(S.month_ends(d)), [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28"),
                                                   pd.Timestamp("2020-03-31")])


class EndToEndSynthetic(unittest.TestCase):
    """Whole H1/H2 pipeline on random prices: runs, no NaN, report renders."""
    def test_pipeline(self):
        rng = np.random.default_rng(0)
        d = pd.bdate_range("2000-01-03", "2002-12-31")
        tick = [f"S{i}" for i in range(40)]
        adj = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(d), 40)), axis=0)), index=d, columns=tick)
        gspc = adj.mean(axis=1)
        member = pd.DataFrame(True, index=d, columns=tick)
        rets, _ = B.clean_returns(adj)
        rets[B.ETF] = gspc.pct_change()
        rets[B.CASH] = 0.0001
        above = B.regime_above(gspc)
        breadth, _ = B.breadth_series(adj, member, min_names=10)
        ev = B.thrust_events(breadth)
        vol = rets.drop(columns=[B.ETF, B.CASH]).rolling(60, min_periods=60).std()
        start, end = pd.Timestamp("2000-11-20"), d[-1]
        res = {}
        for name, tg in {"baseline": B.baseline_targets(above, d), "h1_breadth_thrust": B.thrust_targets(above, d, ev),
                         "h2_lowvol": B.lowvol_targets(above, d, vol, member),
                         "spy_buy_hold": {x: {B.ETF: 1.0} for x in d}}.items():
            eq, tr, n = B.simulate(B.orders_from_targets(tg, start, end), rets, start, end)
            self.assertTrue(np.isfinite(eq).all(), name)
            m = B.metrics(eq, rets[B.CASH])
            m.update({"trades": len(tr), "rebalances": n, "costs": 0.0, "decades": B.decade_table(eq, rets[B.CASH])})
            res[name] = m
        md = B.render_md({"run": "test", "period": ["a", "b"], "results": res, "h1_pass": False, "h2_pass": False,
                          "h1_events": [], "h1_unconditional_fwd": {63: 0.01, 126: 0.02},
                          "member_price_coverage_by_year": {}, "bad_daily_returns_masked": 0,
                          "tickers_ever": 40, "tickers_with_prices": 40})
        self.assertIn("השערה 1", md)


class LeverageExposure(unittest.TestCase):
    def test_synthetic_2x(self):
        d = bdays(3)
        r = L.synthetic_2x(pd.Series([0.01, -0.02, 0.0], index=d), pd.Series(0.0001, index=d), fee_annual=0.0252)
        np.testing.assert_allclose(r.values, [0.02 - 0.0001 - 0.0001, -0.04 - 0.0002, -0.0002])

    def test_splice_uses_real_from_first_day(self):
        d = bdays(4)
        real = pd.Series([np.nan, np.nan, 0.05, 0.06], index=d)
        synth = pd.Series([0.1, 0.2, 0.3, 0.4], index=d)
        self.assertEqual(L.splice(real, synth).tolist(), [0.1, 0.2, 0.05, 0.06])

    def test_targets_and_monthly_rebalance(self):
        d = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
        above = pd.Series([True, True, True, False], index=d, dtype=object)
        tg = L.leverage_targets(above, d)
        self.assertEqual(tg[d[0]][L.LEV], 0.5)
        self.assertEqual(tg[d[3]], {B.ETF: B.REDUCED})
        self.assertEqual(sorted(B.orders_from_targets(tg, d[0], d[-1])), [d[0], d[2], d[3]])

    def test_sso_gets_etf_slippage(self):
        self.assertAlmostEqual(B.trade_cost(100_000, L.LEV), 100 + 50)

    def test_verdict(self):
        spy = {"cagr": 0.08, "max_dd": -0.55}
        self.assertTrue(L.verdict({"cagr": 0.10, "max_dd": -0.55}, spy))
        self.assertFalse(L.verdict({"cagr": 0.0999, "max_dd": -0.40}, spy))
        self.assertFalse(L.verdict({"cagr": 0.12, "max_dd": -0.551}, spy))

    def test_leverage_simulation(self):
        d = bdays(6)
        rets = pd.DataFrame({B.ETF: 0.01, L.LEV: 0.02, B.CASH: 0.0}, index=d)
        above = pd.Series(True, index=d, dtype=object)
        eq, trades, _ = B.simulate(B.orders_from_targets(L.leverage_targets(above, d), d[0], d[-1]),
                                   rets, d[0], d[-1], initial=100_000)
        self.assertAlmostEqual(eq.iloc[2] / eq.iloc[1], 1.015, places=3)   # 150% exposure
        self.assertEqual(sorted(t["asset"] for t in trades), [B.ETF, L.LEV])


if __name__ == "__main__":
    unittest.main()
