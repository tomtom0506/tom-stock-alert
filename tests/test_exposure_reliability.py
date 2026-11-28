"""v5.37.0: exposure-rule reliability - append-only decision log, one Telegram alert per
rule change, the per-session checks, closes only (no intraday bar), and the leveraged
experiment's 3-month review.
Run: python -m unittest discover -s tests"""
import sys
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stock_alerts as sa  # noqa: E402


def store_with(session, pct, close=6900.0, sma=7000.0):
    return {"exposure": {"as_of": session, "recommended_exposure_pct": pct, "index_close": close,
                         "sma200": sma, "pct_vs_sma": round((close / sma - 1) * 100, 2)}}


def at(iso):
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


class DecisionLogTest(unittest.TestCase):
    def test_once_per_session_never_rewritten(self):
        st = store_with("2026-11-27", 50)
        first = sa.log_exposure_decision(st, at("2026-11-27T21:30:00"))
        self.assertIsNotNone(first)
        st["exposure"]["index_close"] = 1.0           # a later run must not rewrite the record
        self.assertIsNone(sa.log_exposure_decision(st, at("2026-11-27T23:00:00")))
        self.assertEqual(len(st["exposure_log"]), 1)
        self.assertEqual(st["exposure_log"][0]["index_close"], 6900.0)

    def test_flip_sends_exactly_one_alert(self):
        st = store_with("2026-11-26", 50)
        sent = []
        with mock.patch.object(sa, "send_telegram_message", side_effect=lambda m, **k: sent.append((m, k))):
            sa.record_exposure_reliability(st)                    # first entry: no flip
            st["exposure"].update(as_of="2026-11-27", recommended_exposure_pct=100, index_close=7100.0)
            sa.record_exposure_reliability(st)                    # flip -> alert
            sa.record_exposure_reliability(st)                    # same session again -> nothing
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1].get("kind"), "exposure_flip")
        self.assertIn("100%", sent[0][0])
        self.assertTrue(st["exposure_log"][-1]["alert_sent"])
        self.assertNotIn("exposure_flip", sa.TELEGRAM_MUTED_KINDS)


class ReliabilityTest(unittest.TestCase):
    def entry(self, session, logged, **kw):
        e = {"session": session, "exposure_pct": 50, "index_close": 6900.0, "sma200": 7000.0,
             "logged_at": logged, "flip": False, "alert_sent": False}
        e.update(kw)
        return e

    def test_clean_streak_and_ready(self):
        days = pd.bdate_range("2026-10-01", periods=20)
        log = [self.entry(d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d") + "T21:30:00+00:00") for d in days]
        r = sa.exposure_reliability(log, today=days[-1].date())
        self.assertEqual(r["clean_streak"], 20)
        self.assertTrue(r["ready"])

    def test_each_failure_breaks_the_streak(self):
        log = [self.entry("2026-11-23", "2026-11-23T21:30:00+00:00"),
               self.entry("2026-11-24", "2026-11-26T09:00:00+00:00"),                 # late
               self.entry("2026-11-25", "2026-11-25T21:30:00+00:00", index_close=None),  # missing price
               self.entry("2026-11-26", "2026-11-26T21:30:00+00:00", flip=True),       # no alert
               self.entry("2026-11-28", "2026-11-28T21:30:00+00:00")]                  # Saturday
        r = sa.exposure_reliability(log, today=date(2026, 11, 28))
        self.assertEqual(r["clean_streak"], 0)
        reasons = {f for x in r["failed"] for f in x["fails"]}
        self.assertTrue({"נרשם באיחור", "מחיר חסר", "לא נשלחה התראה על החלפה", "יום בלי מסחר"} <= reasons)

    def test_stale_log_is_not_ready(self):
        log = [self.entry("2026-11-02", "2026-11-02T21:30:00+00:00")]
        r = sa.exposure_reliability(log, today=date(2026, 11, 20))
        self.assertTrue(r["stale"])
        self.assertFalse(r["ready"])


class CompleteClosesTest(unittest.TestCase):
    def test_drops_the_in_progress_bar(self):
        idx = pd.to_datetime(["2026-11-25", "2026-11-26", "2026-11-27"]).tz_localize("America/New_York")
        closes = pd.Series([1.0, 2.0, 3.0], index=idx)
        out = sa.complete_closes(closes, date(2026, 11, 26))       # 27.11 still trading
        self.assertEqual(list(out.values), [1.0, 2.0])


class LeverageReviewTest(unittest.TestCase):
    def test_first_review_after_3_months(self):
        d = ["2026-11-20", "2026-11-23"]
        sim = sa.simulate_leveraged_exposure({x: 100.0 for x in d}, {x: 100.0 for x in d}, {x: False for x in d}, d[0])
        st = {"leveraged_exposure_sim": {"start_date": "2026-11-20"}, "exposure": {"series": []}}
        idx = pd.to_datetime(d)
        with mock.patch.object(sa.yf, "Ticker") as T, \
             mock.patch.object(sa, "simulate_leveraged_exposure", return_value=dict(sim)):
            T.return_value.history.return_value = {"Close": pd.Series([100.0, 100.0], index=idx)}
            sa.build_leveraged_exposure_sim(st)
        self.assertEqual(st["leveraged_exposure_sim"]["review_on"], "2027-02-20")


if __name__ == "__main__":
    unittest.main()
