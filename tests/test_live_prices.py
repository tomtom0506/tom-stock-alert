"""Unit tests for live-price coverage (v5.32.0 / app 6.24.0).
Regression for the backlog bug: tickers picked AFTER run_watchlist_alerts
wrote current_prices.json (AMD/SWKS in strategy_book A pending, 19.11.2026)
had no live price. No network: get_price_and_prev_close is stubbed.
Run: python -m unittest discover -s tests"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stock_alerts as sa  # noqa: E402


class MissingPriceTickersTest(unittest.TestCase):
    def test_returns_only_missing_in_order(self):
        prices = {"AAPL": {"price": 1.0}, "MSFT": {"price": None}}
        self.assertEqual(sa.missing_price_tickers(["AMD", "AAPL", "MSFT", "AMD", "", None, "TEVA.TA"], prices),
                         ["AMD", "MSFT", "TEVA.TA"])

    def test_bad_prices_map(self):
        self.assertEqual(sa.missing_price_tickers(["X"], None), ["X"])


class FillMissingLivePricesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        files = {"WATCHLIST_FILE": "watchlist.json", "STARRED_FILE": "starred.json",
                 "CRYPTO_EXPOSED_FILE": "crypto_exposed.json", "MY_PORTFOLIO_FILE": "my_portfolio.json",
                 "CURRENT_PRICES_FILE": "current_prices.json"}
        self.patches = [mock.patch.object(sa, "BASE_DIR", self.tmp)]
        self.patches += [mock.patch.object(sa, k, self.tmp / v) for k, v in files.items()]
        for p in self.patches:
            p.start()
        (self.tmp / "watchlist.json").write_text(json.dumps([{"ticker": "AAPL"}]))
        self.cp = {"updated_at": "2026-11-19T21:21:40+00:00",
                   "prices": {"AAPL": {"price": 200.0, "prev_close": 199.0, "pct_change": 0.5}},
                   "indices": {"sp500": {"price": 1}}}
        (self.tmp / "current_prices.json").write_text(json.dumps(self.cp))
        # state right after the bug: strategy book picked AMD after prices were written
        self.store = {"history": [], "strategy_book": {"A": {"holdings": [], "pending": [{"ticker": "AMD"}]}}}

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def _fake_price(self, ticker):
        self.fetched.append(ticker)
        return {"AMD": (110.0, 100.0)}.get(ticker, (None, None))

    def test_fills_ticker_picked_after_prices_were_written(self):
        self.fetched = []
        with mock.patch.object(sa, "get_price_and_prev_close", side_effect=self._fake_price):
            added = sa.fill_missing_live_prices(self.store)
        out = json.loads((self.tmp / "current_prices.json").read_text())
        self.assertEqual(added, ["AMD"])
        self.assertAlmostEqual(out["prices"]["AMD"]["price"], 110.0)
        self.assertAlmostEqual(out["prices"]["AMD"]["pct_change"], 10.0)
        # existing prices, timestamp and indices untouched; no re-fetch of AAPL
        self.assertEqual(out["prices"]["AAPL"], self.cp["prices"]["AAPL"])
        self.assertEqual(out["updated_at"], self.cp["updated_at"])
        self.assertEqual(out["indices"], self.cp["indices"])
        self.assertEqual(self.fetched, ["AMD"])

    def test_no_price_available_does_not_write(self):
        self.fetched = []
        self.store["strategy_book"]["A"]["pending"] = [{"ticker": "NOPE.TA"}]
        before = (self.tmp / "current_prices.json").read_text()
        with mock.patch.object(sa, "get_price_and_prev_close", side_effect=self._fake_price):
            self.assertEqual(sa.fill_missing_live_prices(self.store), [])
        self.assertEqual((self.tmp / "current_prices.json").read_text(), before)

    def test_fetch_error_is_skipped(self):
        def boom(t):
            raise RuntimeError("network")
        with mock.patch.object(sa, "get_price_and_prev_close", side_effect=boom):
            self.assertEqual(sa.fill_missing_live_prices(self.store), [])


class MainOrderTest(unittest.TestCase):
    def test_fill_runs_after_every_picking_step_in_main(self):
        import inspect
        src = inspect.getsource(sa.main)
        last_fill = src.rindex("fill_missing_live_prices(")
        for step in ("manage_long_term_picks(", "run_strategy_book(", "build_recommendations(", "run_picks("):
            self.assertLess(src.rindex(step), last_fill, step)
        # off-day path too: fill before its early return
        off = src[:src.index("return\n", src.index("if not is_market_trading_day()"))]
        self.assertIn("fill_missing_live_prices(", off)
        self.assertLess(off.rindex("build_picks_view("), off.rindex("fill_missing_live_prices("))


if __name__ == "__main__":
    unittest.main()
