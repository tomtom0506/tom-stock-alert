"""
🧪 בדיקת אפליקציה מקצה לקצה (v5.30.0, 31.10.2026 - אחרי שבאג TWLO "קנה וגם מכור"
לא נתפס): פותח את האפליקציה בדפדפן אמיתי על הנתונים האמיתיים מה-repo, עובר על
כל הלשוניות, ובודק מה שהמשתמש רואה - לא רק את הנתונים במנוע:
  * אף מניה לא מופיעה גם בקנייה וגם במכירה, ואין רשימת קנייה שנייה מתחרה
  * לכל אחזקה יש מחיר עם שעה או "סגירה אחרונה"
  * אין undefined / NaN / null / Infinity בשום לשונית, אין "טוען..." תקוע
  * אין שגיאות JavaScript
רץ מ-.github/workflows/app_qa.yml כל יום ובהפעלה ידנית; בכישלון - טלגרם.
"""
import os, re, sys, json, base64
if os.path.isdir("/opt/pw-browsers"):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
from playwright.sync_api import sync_playwright
REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
DATA = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else REPO)
def local(name):
    for d in (DATA, REPO):
        p = os.path.join(d, name)
        if os.path.exists(p): return open(p, "rb").read()
fails, notes, puts = [], [], []
SHOTS = os.path.join(REPO, 'qa', 'screens'); os.makedirs(SHOTS, exist_ok=True)
def check(cond, msg):
    (notes if cond else fails).append(("OK  " if cond else "FAIL") + " " + msg)
# v6.24.0 (20.11.2026 - AMD/SWKS in strategy A's buy list had no live price):
# every ticker on a screen must be in current_prices.json, or the app shows
# an old close / the entry price instead of a price from today
def _j(name, default):
    raw = local(name)
    try: return json.loads(raw) if raw else default
    except ValueError: return default
# v6.24.0 (20.11.2026 - a bare NaN blanked the app; Python's json reads it, browsers don't):
# every data file must be STRICT JSON, exactly as a browser parses it
def _strict(raw):
    def bad(c): raise ValueError(c)
    json.loads(raw, parse_constant=bad)
for _f in ("predictions.json", "today.json", "current_prices.json", "winners.json", "insider.json", "watchlist.json", "my_portfolio.json"):
    _raw = local(_f)
    if _raw is None: continue
    try: _strict(_raw); check(True, f"data file is browser-readable JSON: {_f}")
    except ValueError as e: check(False, f"data file is browser-readable JSON: {_f} (bad value: {str(e)[:60]})")
_store = _j("predictions.json", {}) or {}
_prices = (_j("current_prices.json", {}) or {}).get("prices") or {}
_need = {}
def _add(src, items):
    for x in items or []:
        t = x if isinstance(x, str) else (x or {}).get("ticker")
        if t: _need.setdefault(t, src)
_book = _store.get("strategy_book") or {}
for _k in ("A", "B"):
    _add(f"book {_k}", (_book.get(_k) or {}).get("holdings")); _add(f"book {_k} pending", (_book.get(_k) or {}).get("pending"))
_add("top_picks", (_store.get("top_picks") or {}).get("tickers"))
_add("long_term", (_store.get("long_term_picks") or {}).get("picks"))
_add("picks_long", (_store.get("picks_long") or {}).get("holdings"))
_add("monthly", (_store.get("monthly_portfolio") or {}).get("holdings"))
_add("watchlist", _j("watchlist.json", []))
_miss = sorted(f"{t} ({s})" for t, s in _need.items() if (_prices.get(t) or {}).get("price") is None)
check(not _miss, f"live price for every listed ticker ({len(_need)} tickers){' missing: ' + ', '.join(_miss) if _miss else ''}")
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    # v6.24.0: the app catches its render errors and only logs them - catch those too
    cerrs = []; pg.on("console", lambda m: cerrs.append(m.text) if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
    def route(r):
        u = r.request.url
        if "raw.githubusercontent.com" in u:
            body = local(u.split("/main/")[1].split("?")[0]); return r.fulfill(status=200 if body else 404, body=body or b"", content_type="application/json")
        if "api.github.com" in u and "/contents/" in u:
            name = u.split("/contents/")[-1].split("?")[0]
            if r.request.method == "PUT":
                puts.append(name); return r.fulfill(status=200, content_type="application/json", body=json.dumps({"content": {"sha": "y"}}))
            body = local(name)
            if body:
                return r.fulfill(status=200, content_type="application/json", body=json.dumps({"content": base64.b64encode(body).decode(), "sha": "x"}))
        if "api.github.com" in u: return r.fulfill(status=404, body="{}")
        return r.continue_()
    pg.route("**/*", route)
    pg.goto("file://" + REPO + "/index.html")
    pg.evaluate("() => localStorage.setItem('gh-watchlist-config', JSON.stringify({owner:'o', repo:'r', token:'t'}))")
    pg.reload(); pg.wait_for_timeout(6000)
    pg.evaluate("() => { const o = document.getElementById('legalOverlay'); if (o) o.remove(); }")
    # v6.25.0: home renders from today.json; predictions.json only when it can't (old today.json)
    _today = _j("today.json", {}) or {}
    if _today.get("market_regime"):
        check(not pg.evaluate("() => !!window.__predictionsLoaded"), "home: first screen does not load predictions.json")
    pg.evaluate("() => document.querySelectorAll('#homeActionSection details').forEach(d => d.open = true)")
    act = pg.inner_text("#homeActionSection")
    buys = set(re.findall(r"קנה ([A-Z.\-]+)", act)); sells = set(re.findall(r"מכור ([A-Z.\-]+)", act))
    check(not (buys & sells), f"action: no ticker both buy and sell (buy={sorted(buys)}, sell={sorted(sells)})")
    check("בדיקה עצמית מצאה בעיה" not in act, "action: self-check banner absent")
    n_pos = len(re.findall(r"נקנתה", act)); n_time = len(re.findall(r"נכון ל-|סגירה אחרונה", act))
    check(n_time >= n_pos, f"action: every holding shows price time ({n_pos} positions, {n_time} price stamps)")
    pg.evaluate("() => { switchTab('home'); switchSubtab('full'); }"); pg.wait_for_timeout(600)
    full = pg.inner_text("#homePicksSection")
    full_buys = set(re.findall(r"\n([A-Z][A-Z.\-]{0,5}) ", full)) & set(re.findall(r"[A-Z][A-Z.\-]{0,5}", "".join(re.findall(r"לקנות[^\n]*\n(?:[^\n]*\n){0,40}", full))))
    check("לקנות עכשיו - טווח קצר" not in full, "full screen: no second, competing short-term buy list")
    for tab, subs in (("home", ["action", "full"]), ("mine", [None]), ("results", [None]), ("lab", ["top10", "longterm", "winners", "exit", "insider"])):
        for sb in subs:
            pg.evaluate(f"() => {{ switchTab('{tab}'); {'switchSubtab(%r);' % sb if sb else ''} }}"); pg.wait_for_timeout(700)
            try: pg.wait_for_function("() => window.__predictionsLoaded === true", timeout=30000)
            except Exception: check(False, f"{tab}/{sb or '-'}: predictions.json finished loading")
            pg.wait_for_timeout(300)
            txt = pg.evaluate("() => [...document.querySelectorAll('[data-tab]')].filter(e => e.offsetParent !== null).map(e => e.innerText).join('\\n')")
            bad = [w for w in ("undefined", "NaN", "null%", "[object Object]", "Infinity") if w in txt]
            check(not bad, f"{tab}/{sb or '-'}: no broken values {bad if bad else ''}")
            check("טוען..." not in txt, f"{tab}/{sb or '-'}: nothing stuck on 'loading'")
            check(len(txt) > 200, f"{tab}/{sb or '-'}: has content ({len(txt)} chars)")
            # v5.31.0: values must be PLAUSIBLE, not just present - no percentage below -100%
            pcts = [float(x) for x in re.findall(r"(?<![\d.])-(\d{3,}(?:\.\d+)?)%", txt) if float(x) > 100]
            check(not pcts, f"{tab}/{sb or '-'}: no percentage below -100% {pcts[:3] if pcts else ''}")
            pg.screenshot(path=os.path.join(SHOTS, f"{tab}_{sb or 'main'}.png"), full_page=True)
    # v5.31.0: a floating message must sit on its own opaque layer, never on top of page text
    pg.evaluate("() => { switchTab('mine'); document.getElementById('floatingRecomputeStatus').innerHTML = '<div class=\\'status\\'>✓ הופעל! מחכה לסיום</div>'; }")
    pg.wait_for_timeout(300)
    bg = pg.evaluate("() => getComputedStyle(document.getElementById('floatingRecomputeStatus')).backgroundColor")
    check(bg not in ("rgba(0, 0, 0, 0)", "transparent"), f"run-status message has an opaque background ({bg})")
    pg.evaluate("() => { document.getElementById('floatingRecomputeStatus').innerHTML = ''; }")
    # v5.31.0: every user action that exists must still exist and respond
    for tab, ids in (("mine", ["watchlistAddToggleBtn", "myPortfolioAddToggleBtn"]), ("home", ["contextActionBtn"]), ("lab", ["refreshBtn"])):
        pg.evaluate(f"() => switchTab('{tab}')"); pg.wait_for_timeout(300)
        for i in ids:
            vis = pg.evaluate(f"() => {{ const e = document.getElementById('{i}'); return !!(e && (e.offsetParent !== null || getComputedStyle(e).position === 'fixed')); }}")
            check(vis, f"action available: {i} on {tab}")
    pg.evaluate("() => switchTab('check')"); pg.wait_for_timeout(300)
    check(pg.evaluate("() => !!document.getElementById('checkStockInput') && !!document.getElementById('checkStockBtn')"), "action available: check-stock input + button")
    pg.evaluate("() => switchTab('mine')"); pg.wait_for_timeout(300)
    try:
        pg.click("#watchlistAddToggleBtn", timeout=3000); pg.wait_for_timeout(200)
        check(pg.evaluate("() => !document.getElementById('watchlistAddForm').classList.contains('hidden')"), "add-to-watchlist: form opens")
        pg.fill("#wlAddTicker", "qatest", timeout=3000); pg.click("#wlAddSaveBtn", timeout=3000); pg.wait_for_timeout(1500)
        added = pg.evaluate("() => rows.some(r => r.ticker === 'QATEST')")
        check(added and "watchlist.json" in puts, f"add-to-watchlist: saves a watch-only stock (saved={puts})")
    except Exception as e:
        check(False, f"add-to-watchlist: works ({str(e).splitlines()[0][:80]})")
    check(not errs, f"no JS errors {errs[:2]}")
    check(not cerrs, f"no errors logged by the app {[c[:120] for c in cerrs[:2]]}")
    check(pg.evaluate("() => !document.getElementById('dataLoadError')"), "no 'could not read data file' banner")
    b.close()
report = "\n".join(fails + notes) + f"\n\n{len(fails)} failed / {len(fails) + len(notes)} checks"
print(report)
if fails and os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
    import requests
    requests.post(f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage",
                  data={"chat_id": os.environ["TELEGRAM_CHAT_ID"],
                        "text": "⚠️ בדיקת האפליקציה מקצה לקצה נכשלה - לא לפעול לפי המסך עד שזה נבדק:\n" + "\n".join(f[5:] for f in fails)[:3500]},
                  timeout=20)
sys.exit(1 if fails else 0)
