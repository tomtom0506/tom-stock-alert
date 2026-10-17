"""
🔁 בדיקת התאמה: קניות אנשי פנים - הקוד החי מול מקור המחקר (אופליין).
מופעל מ-.github/workflows/research.yml (script = insider_parity_check).

The live tracker (insider_live.py) reads each Form 4 XML from EDGAR; the
research (dip_insider_research.py) read SEC's quarterly insider data set.
Cluster logic is shared code, so the only thing that can differ is the
PARSING. This takes one finished quarter, parses it the research way, then
fetches a sample of the same filings as XML and parses them the live way:

  1. purchase level - for each sampled filing: same verdict (counts / not),
     same owner, value within 1%
  2. cluster level  - clusters from both parses of the sample must match

Target: 95%+ on both. Writes research/output_insider_parity/report.md.
"""
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import dip_insider_research as R   # noqa: E402
import insider_live as L           # noqa: E402

OUT = HERE / "output_insider_parity"
QUARTER = os.environ.get("PARITY_QUARTER", "2025q4")
SAMPLE = 400
UA = os.environ.get("SEC_USER_AGENT") or L.UA


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    url = f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{QUARTER}_form345.zip"
    print("downloading", url, flush=True)
    r = s.get(url, timeout=300)
    r.raise_for_status()
    import zipfile, io
    z = zipfile.ZipFile(io.BytesIO(r.content))
    sub = R._read_tsv(z, "SUBMISSION.tsv")
    tx = R._read_tsv(z, "NONDERIV_TRANS.tsv")
    p_acc = set(tx[tx["TRANS_CODE"].astype(str).str.strip() == "P"]["ACCESSION_NUMBER"])
    issuers = sorted(set(pd.to_numeric(sub[sub["ACCESSION_NUMBER"].isin(p_acc)]["ISSUERCIK"], errors="coerce").dropna().astype(int)))
    random.seed(7)
    pick = set(random.sample(issuers, min(len(issuers), 120)))
    acc_of = sub[["ACCESSION_NUMBER", "ISSUERCIK", "FILING_DATE"]]
    cand = acc_of[pd.to_numeric(acc_of["ISSUERCIK"], errors="coerce").isin(pick)]
    cand = cand.sample(min(len(cand), SAMPLE), random_state=7)
    keep = set(cand["ACCESSION_NUMBER"])
    sub_r = sub[sub["ACCESSION_NUMBER"].isin(keep)]
    own = R._read_tsv(z, "REPORTINGOWNER.tsv")
    own = own[own["ACCESSION_NUMBER"].isin(keep)]
    txk = R._read_tsv(z, "NONDERIV_TRANS.tsv")
    txk = txk[txk["ACCESSION_NUMBER"].isin(keep)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zz:
        for name, df in (("SUBMISSION.tsv", sub_r), ("REPORTINGOWNER.tsv", own), ("NONDERIV_TRANS.tsv", txk)):
            zz.writestr(name, df.to_csv(sep="\t", index=False, quoting=3, escapechar="\\"))
    part = R.parse_insider_zip(buf.getvalue())
    # compare as sets of (issuer, owner, filing date, value) - parse_insider_zip drops the accession number
    def norm_owner(o):
        return str(o).split(".")[0].lstrip("0")
    R_set = {(int(r_.cik), norm_owner(r_.owner), r_.filing_date.strftime("%Y-%m-%d"), round(float(r_.value))) for r_ in part.itertuples()}
    L_rows = []
    for k, rr in enumerate(cand.itertuples()):
        a, cik = rr.ACCESSION_NUMBER, int(float(rr.ISSUERCIK))
        try:
            fr = s.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{a}.txt", timeout=30)
        except Exception:
            continue
        time.sleep(L.REQ_PAUSE)
        if fr.status_code != 200:
            continue
        x = L.extract_xml(fr.text)
        p = L.parse_form4(x) if x else None
        if p:
            d = pd.to_datetime(rr.FILING_DATE, format="mixed").strftime("%Y-%m-%d")
            L_rows.append({"ticker": str(cik), "owner": norm_owner(p["owner"]), "filing_date": d, "value": p["value"], "title": p["title"]})
        if (k + 1) % 100 == 0:
            print(f"  fetched {k + 1}/{len(cand)}", flush=True)
    L_set = {(int(r_["ticker"]), r_["owner"], r_["filing_date"], round(float(r_["value"]))) for r_ in L_rows}
    # match with the documented 1% value tolerance (the quarterly TSV rounds shares/prices)
    def close(a, b):
        return a[0] == b[0] and a[2] == b[2] and abs(a[3] - b[3]) <= 0.01 * max(a[3], 1)
    matched, other_owner, used = 0, 0, set()
    for a in R_set:
        hit = next((b for b in L_set if b not in used and close(a, b) and a[1] == b[1]), None)
        if hit:
            matched += 1; used.add(hit); continue
        hit = next((b for b in L_set if b not in used and close(a, b)), None)
        if hit:
            other_owner += 1; used.add(hit)
    either = len(R_set) + len(L_set - used)
    pct = round(matched / max(either, 1) * 100, 1)
    diffs = [{"only": "research", "row": list(x)} for x in sorted(R_set)
             if not any(close(x, b) for b in L_set)][:30] + \
            [{"only": "live", "row": list(x)} for x in sorted(L_set - used)][:30]
    # cluster level, same shared function on both parses
    rd = pd.DataFrame([{"ticker": str(c), "owner": o, "filing_date": pd.Timestamp(d), "value": v, "title": ""} for c, o, d, v in R_set])
    ld = pd.DataFrame([{**r_, "filing_date": pd.Timestamp(r_["filing_date"]), "title": ""} for r_ in L_rows])
    cr = R.find_clusters(rd) if len(rd) else pd.DataFrame()
    cl = R.find_clusters(ld) if len(ld) else pd.DataFrame()
    ks = lambda c: set() if c.empty else {(t, d.strftime("%Y-%m-%d")) for t, d in zip(c["ticker"], c["signal_date"])}
    ck_r, ck_l = ks(cr), ks(cl)
    cpct = round(len(ck_r & ck_l) / max(len(ck_r | ck_l), 1) * 100, 1) if (ck_r | ck_l) else 100.0
    checked = len(cand)
    rep = ["# 🔁 התאמת אנשי פנים - קוד חי מול מקור המחקר", "",
           f"רבעון {QUARTER} · {checked} דיווחים במדגם · {len(R_set)} קניות כשרות לפי המחקר · {len(L_set)} לפי הקוד החי", "",
           f"**התאמה ברמת הקנייה: {pct}%** ({'✅ עובר' if pct >= 95 else '❌ לא עובר'}, יעד 95%) · עוד {other_owner} אותה קנייה עם איש פנים אחר (דיווח משותף)",
           f"**התאמה ברמת האשכול: {cpct}%** ({len(ck_r)} אשכולות במחקר, {len(ck_l)} בקוד החי)", "",
           "## דוגמאות לפערים (מנפיק, איש פנים, תאריך, סכום)", *[f"- רק ב{'מחקר' if d['only'] == 'research' else 'קוד החי'}: {d['row']}" for d in diffs[:20]]]
    (OUT / "report.md").write_text("\n".join(rep))
    (OUT / "summary.json").write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "quarter": QUARTER,
                                                  "checked": checked, "agree_pct": pct, "same_purchase_other_owner": other_owner, "cluster_agree_pct": cpct, "diffs": diffs}, ensure_ascii=False, indent=1))
    print("\n".join(rep[:6]))


if __name__ == "__main__":
    main()
