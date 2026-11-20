"""Regression for 20.11.2026: Yahoo had no 19.11 close for OKTA/PLTR/PANW,
trades in picks_short_sim / exit_forward_test closed at NaN, predictions.json
was written with a bare NaN and no browser could parse it (app blank).
Run: python -m unittest discover -s tests"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stock_alerts as sa  # noqa: E402


def strict_dumps(o):
    """Exactly what a browser accepts: no NaN / Infinity."""
    return json.dumps(o, allow_nan=False, default=float)


def no_bad_floats(o):
    if isinstance(o, float):
        return math.isfinite(o)
    if isinstance(o, dict):
        return all(no_bad_floats(v) for v in o.values())
    if isinstance(o, list):
        return all(no_bad_floats(v) for v in o)
    return True


def panels(n=8, nan_last_close=True):
    idx = pd.bdate_range("2026-11-09", periods=n)
    cols = ["OKTA", "AAPL"]
    C = pd.DataFrame({"OKTA": np.linspace(100, 107, n), "AAPL": np.linspace(200, 207, n)}, index=idx)
    O = C * 0.99
    if nan_last_close:
        C.iloc[-1, 0] = np.nan          # the 19.11 situation: no close yet for OKTA
    H, L = C * 1.01, C * 0.98
    spy = pd.Series(np.linspace(500, 507, n), index=idx)
    return idx, O[cols], H[cols], L[cols], C[cols], spy


class SimulatePicksShortTest(unittest.TestCase):
    def test_trade_without_exit_close_stays_open(self):
        idx, O, H, L, C, spy = panels()
        hold = len(idx) - 1                  # exit session = last session (whose close is NaN)
        log = {idx[0].strftime("%Y-%m-%d"): [{"ticker": "OKTA", "strategy": "A", "hold": hold},
                                             {"ticker": "AAPL", "strategy": "A", "hold": hold}]}
        out = sa.simulate_picks_short(log, O, C, spy)
        closed = {c["ticker"]: c for c in out["closed"]}
        opened = {c["ticker"]: c for c in out["open"]}
        self.assertIn("AAPL", closed)
        self.assertNotIn("OKTA", closed)
        self.assertIn("OKTA", opened)
        self.assertIsNotNone(opened["OKTA"]["last"])          # last REAL close, not NaN
        self.assertEqual(opened["OKTA"]["sessions_left"], 0)
        self.assertTrue(no_bad_floats(out))
        strict_dumps(out)

    def test_normal_close_unchanged(self):
        idx, O, H, L, C, spy = panels(nan_last_close=False)
        log = {idx[0].strftime("%Y-%m-%d"): [{"ticker": "OKTA", "strategy": "A", "hold": len(idx) - 1}]}
        out = sa.simulate_picks_short(log, O, C, spy)
        self.assertEqual([c["ticker"] for c in out["closed"]], ["OKTA"])
        self.assertTrue(math.isfinite(out["closed"][0]["pnl_pct"]))


class CompareExitsForwardTest(unittest.TestCase):
    def test_nan_exit_close_does_not_poison_stats(self):
        idx, O, H, L, C, spy = panels()
        hold = len(idx) - 2
        log = {idx[1].strftime("%Y-%m-%d"): [
            {"ticker": "OKTA", "strategy": "A", "hold": hold, "levels_frozen": True,
             "tech_target": 1000.0, "tech_stop": 1.0, "tech_target_far": 2000.0},
            {"ticker": "AAPL", "strategy": "A", "hold": hold, "levels_frozen": True,
             "tech_target": 1000.0, "tech_stop": 1.0, "tech_target_far": 2000.0}]}
        out = sa.compare_exits_forward(log, O, H, L, C, spy)
        okta = [t for t in out["trades"] if t["ticker"] == "OKTA"][0]
        self.assertIsNone(okta["time"])       # not closed until there is a real close
        self.assertIsNone(okta["price"])
        self.assertEqual(out["time"]["closed"], 1)
        self.assertTrue(no_bad_floats(out))
        strict_dumps(out)


class JsonSafeWritersTest(unittest.TestCase):
    def test_json_safe(self):
        self.assertEqual(sa.json_safe({"a": [1.5, float("nan"), np.float64("inf"), np.float32("nan"), "NaN", 3]}),
                         {"a": [1.5, None, None, None, "NaN", 3]})

    def test_writers_never_emit_nan(self):
        tmp = Path(tempfile.mkdtemp())
        bad = {"x": float("nan"), "y": [np.float64("-inf")], "ok": 1.0}
        sa.save_json(tmp / "a.json", bad)
        with mock.patch.object(sa, "PREDICTIONS_FILE", tmp / "p.json"), \
             mock.patch.object(sa, "ARCHIVE_DIR", tmp / "archive"), \
             mock.patch.object(sa, "TODAY_FILE", tmp / "t.json"), \
             mock.patch.object(sa, "_archive_files", lambda: []):
            sa.save_prediction_store({"history": [], "picks_short_sim": {"closed": [{"pnl_pct": float("nan")}]}}, archive=False)
            sa.write_today_json({"strategy_comparison": {"rows": [{"vs_sp500_pct": float("nan")}]}})
        for f in ("a.json", "p.json", "t.json"):
            txt = (tmp / f).read_text(encoding="utf-8")
            self.assertNotIn("NaN", txt, f)
            self.assertNotIn("Infinity", txt, f)
            json.loads(txt, parse_constant=lambda c: self.fail(f"{f}: {c}"))


if __name__ == "__main__":
    unittest.main()


class TodayJsonHomeFieldsTest(unittest.TestCase):
    """v5.33.0: the home screen renders from today.json alone - it must carry
    the market badge and list statuses (else the app falls back to the 18MB file)."""
    def test_home_fields_present(self):
        tmp = Path(tempfile.mkdtemp())
        store = {"market_regime": {"regime": "bull"}, "market_breadth": {"pct": 55.0},
                 "list_status": {"book_A": {"status": "warning"}}, "picks": {}}
        with mock.patch.object(sa, "TODAY_FILE", tmp / "t.json"):
            sa.write_today_json(store)
        out = json.loads((tmp / "t.json").read_text(encoding="utf-8"))
        self.assertEqual(out["market_regime"], {"regime": "bull"})
        self.assertEqual(out["market_breadth"], {"pct": 55.0})
        self.assertEqual(out["list_status"]["book_A"]["status"], "warning")
