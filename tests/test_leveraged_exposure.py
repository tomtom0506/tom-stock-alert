"""v5.36.0: 🧪 "leveraged exposure" paper test - allocation, rebalancing the session
after a cross and on each new month, costs, and the stop rule.
Run: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stock_alerts as sa  # noqa: E402


def days(n, start_month="2026-11", first_day=2):
    return [f"{start_month}-{first_day + i:02d}" for i in range(n)]


class LeveragedExposureTest(unittest.TestCase):
    def test_above_is_150pct(self):
        d = days(3)
        spy = {d[0]: 100.0, d[1]: 110.0, d[2]: 110.0}
        sso = {d[0]: 100.0, d[1]: 120.0, d[2]: 120.0}   # 2x of +10%
        flags = {x: True for x in d}
        r = sa.simulate_leveraged_exposure(spy, sso, flags, d[0])
        # 50% SPY +10% and 50% SSO +20% -> +15% minus the opening trade costs
        self.assertAlmostEqual(r["total_return_pct"], 15.0, delta=0.3)
        self.assertEqual(r["daily_log"][0]["exposure_pct"], 150)          # allocated at 150%
        self.assertTrue(145 <= r["daily_log"][-1]["exposure_pct"] <= 160)  # drifts until the next rebalance
        self.assertAlmostEqual(r["spy_pct"], 10.0)

    def test_below_is_half_cash(self):
        d = days(2)
        spy = {d[0]: 100.0, d[1]: 90.0}
        sso = {d[0]: 100.0, d[1]: 80.0}
        r = sa.simulate_leveraged_exposure(spy, sso, {x: False for x in d}, d[0])
        self.assertAlmostEqual(r["total_return_pct"], -5.0, delta=0.2)   # half in SPY -10%, half cash
        self.assertEqual(r["daily_log"][0]["exposure_pct"], 50)

    def test_rebalances_the_session_after_a_cross(self):
        d = days(4)
        spy = {x: 100.0 for x in d}
        sso = {x: 100.0 for x in d}
        flags = {d[0]: False, d[1]: True, d[2]: True, d[3]: True}       # crosses up at the close of d1
        r = sa.simulate_leveraged_exposure(spy, sso, flags, d[0])
        exp = [e["exposure_pct"] for e in r["daily_log"]]
        self.assertEqual(exp[1], 50)    # the cross day itself: still the old allocation
        self.assertEqual(exp[2], 150)   # the session after: rebalanced up

    def test_monthly_rebalance_and_costs(self):
        d = ["2026-11-27", "2026-11-30", "2026-12-01"]
        spy = {d[0]: 100.0, d[1]: 100.0, d[2]: 100.0}
        sso = {d[0]: 100.0, d[1]: 150.0, d[2]: 150.0}   # SSO drifts to 60% of the account
        r = sa.simulate_leveraged_exposure(spy, sso, {x: True for x in d}, d[0])
        self.assertGreater(r["costs_usd"], 2 * sa.LEV_MIN_FEE)   # opening trades + the December rebalance

    def test_stop_rule(self):
        d = days(3)
        spy = {d[0]: 100.0, d[1]: 95.0, d[2]: 95.0}     # S&P -5%
        sso = {d[0]: 100.0, d[1]: 60.0, d[2]: 60.0}     # leveraged leg collapses
        r = sa.simulate_leveraged_exposure(spy, sso, {x: True for x in d}, d[0])
        self.assertLess(r["max_dd_pct"] - r["spy_max_dd_pct"], -sa.LEV_STOP_GAP_PCT)
        self.assertTrue(r["stop_hit"])

    def test_no_data(self):
        self.assertIsNone(sa.simulate_leveraged_exposure({}, {}, {}, "2026-11-02"))

    def test_comparison_row_is_experiment(self):
        store = {"leveraged_exposure_sim": {"value": 101000.0, "start_date": "2026-11-23", "total_return_pct": 1.0,
                                            "start_value": 100000}}
        sa.build_strategy_comparison(store)
        row = [r for r in store["strategy_comparison"]["rows"] if r["key"] == "leveraged_exposure"][0]
        self.assertEqual(row["tag"], "experiment")


if __name__ == "__main__":
    unittest.main()
