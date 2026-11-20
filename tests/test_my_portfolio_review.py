"""v5.34.0: my portfolio vs the core, the per-holding facts, the weekly
Telegram digest, and the pinned start date (the 90-row log trim used to move
it while the return still counted from the real start).
Run: python -m unittest discover -s tests"""
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stock_alerts as sa  # noqa: E402


class CoreReturnTest(unittest.TestCase):
    def test_half_exposure_below_sma(self):
        closes = {"2026-01-01": 100.0, "2026-01-02": 110.0, "2026-01-05": 99.0}
        series = [["2026-01-01", True], ["2026-01-02", False]]
        core, spy = sa.core_return_since(closes, series, "2026-01-01")
        # day 1: +10% fully invested; day 2: -10% at half exposure -> 1.1 * 0.95
        self.assertAlmostEqual(core, round((1.1 * 0.95 - 1) * 100, 2))
        self.assertAlmostEqual(spy, -1.0)

    def test_too_short(self):
        self.assertEqual(sa.core_return_since({"2026-01-01": 1.0}, [], "2026-01-01"), (None, None))


class ReviewHoldingTest(unittest.TestCase):
    def test_flags_and_since_start(self):
        idx = pd.bdate_range("2025-09-01", periods=260)
        c = pd.Series(np.linspace(200, 100, 260), index=idx)   # steady fall: below its 200-day average
        r = sa.review_holding(c, idx[200].strftime("%Y-%m-%d"), 23.2)
        self.assertIn("below_sma200", r["flags"])
        self.assertIn("concentrated", r["flags"])
        self.assertLess(r["since_start_pct"], 0)
        self.assertLess(r["from_high_pct"], 0)

    def test_healthy_holding_no_flags(self):
        idx = pd.bdate_range("2025-09-01", periods=260)
        c = pd.Series(np.linspace(100, 200, 260), index=idx)
        r = sa.review_holding(c, idx[0].strftime("%Y-%m-%d"), 10.0)
        self.assertEqual(r["flags"], [])
        self.assertAlmostEqual(r["since_start_pct"], 100.0)


class PinnedStartDateTest(unittest.TestCase):
    def test_start_date_survives_log_trim(self):
        log = [{"date": f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}", "return_pct": 0.0, "value": 100.0} for i in range(95)]
        store = {"my_portfolio_sim": {"value": 100.0, "pending_date": "2026-01-01", "pending_return_pct": 0.0,
                                      "daily_log": log}}
        snap = {"rows": [], "weighted_return_pct": 0.0, "total_value_ils": 0, "usdils_rate": 3.3}
        with mock.patch.object(sa, "compute_my_portfolio_snapshot", return_value=snap), \
             mock.patch.object(sa, "load_json", return_value=[]), \
             mock.patch.object(sa, "build_my_portfolio_review"):
            sa.update_my_portfolio(store)
        self.assertEqual(store["my_portfolio_sim"]["start_date"], log[0]["date"])


class WeeklyDigestTest(unittest.TestCase):
    def test_due_only_sunday_morning_once_a_week(self):
        sun = datetime(2026, 11, 22, 9, 0)
        due, week = sa.weekly_digest_due(sun, None)
        self.assertTrue(due)
        self.assertFalse(sa.weekly_digest_due(sun, week)[0])                       # already sent this week
        self.assertFalse(sa.weekly_digest_due(datetime(2026, 11, 22, 7, 0), None)[0])  # too early
        self.assertFalse(sa.weekly_digest_due(datetime(2026, 11, 23, 9, 0), None)[0])  # Monday

    def test_message_content(self):
        store = {
            "exposure": {"recommended_exposure_pct": 50, "above_sma": False},
            "picks": {"book_actions": {
                "B": {"buys": [{"ticker": "NVDA"}], "sell_due": [],
                      "holdings": [{"ticker": "OKTA", "exit_estimate": "2026-11-24"},
                                   {"ticker": "FICO", "exit_estimate": "2026-12-24"}]},
                "A": {"buys": [{"ticker": "HUM"}], "sell_due": []}}},
            "my_portfolio_review": {"start_date": "2026-08-26", "mine_pct": -20.26, "spy_pct": -6.3, "core_pct": -5.2,
                                    "holdings": [{"ticker": "ZVRA", "weight_pct": 23.2, "flags": ["concentrated"]}]},
        }
        msg = sa.build_weekly_digest(store, date(2026, 11, 22))
        self.assertIn("50% במניות", msg)
        self.assertIn("לקנות: NVDA", msg)
        self.assertIn("OKTA (24.11)", msg)
        self.assertNotIn("FICO", msg)                 # exit is not this week
        self.assertIn("פעולה אחת", msg)
        self.assertIn("-20.3%", msg)
        self.assertIn("ZVRA 23% מהתיק", msg)
        self.assertIn("לא ייעוץ השקעות", msg)

    def test_digest_kind_not_muted(self):
        self.assertNotIn("weekly_digest", sa.TELEGRAM_MUTED_KINDS)



class FollowedListsTest(unittest.TestCase):
    """v5.34.0: Top10 was stopped 20.11.2026 - its picks must not count as 'your lists'
    (6.26.0 still tagged unusual moves with 'Top10' in the mine tab)."""
    def test_top10_not_followed(self):
        today = date.today().isoformat()
        store = {"history": [{"date": today, "ticker": "SLS", "top10": True}],
                 "strategy_book": {"B": {"holdings": [{"ticker": "OKTA"}]}}}
        with mock.patch.object(sa, "load_json", return_value=[]):
            lists = sa._followed_lists(store)
        self.assertNotIn("SLS", lists)
        self.assertIn("OKTA", lists)


if __name__ == "__main__":
    unittest.main()
