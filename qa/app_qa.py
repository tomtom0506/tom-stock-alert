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
import os, re, sys, json
if os.path.isdir("/opt/pw-browsers"):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
from playwright.sync_api import sync_playwright
REPO = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
DATA = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else REPO)
def local(name):
    for d in (DATA, REPO):
        p = os.path.join(d, name)
        if os.path.exists(p): return open(p, "rb").read()
fails, notes = [], []
def check(cond, msg):
    (notes if cond else fails).append(("OK  " if cond else "FAIL") + " " + msg)
with sync_playwright() as pw:
    b = pw.chromium.launch(); pg = b.new_page(viewport={"width": 390, "height": 844})
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    def route(r):
        u = r.request.url
        if "raw.githubusercontent.com" in u:
            body = local(u.split("/main/")[1].split("?")[0]); return r.fulfill(status=200 if body else 404, body=body or b"", content_type="application/json")
        if "api.github.com" in u: return r.fulfill(status=404, body="{}")
        return r.continue_()
    pg.route("**/*", route)
    pg.goto("file://" + REPO + "/index.html")
    pg.evaluate("() => localStorage.setItem('gh-watchlist-config', JSON.stringify({owner:'o', repo:'r', token:'t'}))")
    pg.reload(); pg.wait_for_timeout(6000)
    pg.evaluate("() => { const o = document.getElementById('legalOverlay'); if (o) o.remove(); }")
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
            txt = pg.evaluate("() => [...document.querySelectorAll('[data-tab]')].filter(e => e.offsetParent !== null).map(e => e.innerText).join('\\n')")
            bad = [w for w in ("undefined", "NaN", "null%", "[object Object]", "Infinity") if w in txt]
            check(not bad, f"{tab}/{sb or '-'}: no broken values {bad if bad else ''}")
            check("טוען..." not in txt, f"{tab}/{sb or '-'}: nothing stuck on 'loading'")
            check(len(txt) > 200, f"{tab}/{sb or '-'}: has content ({len(txt)} chars)")
    check(not errs, f"no JS errors {errs[:2]}")
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
