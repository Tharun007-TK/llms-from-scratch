"""
Indian Kanoon API test — run this BEFORE building the full AAR/AAAR fetch
pipeline. This answers the one open question that determines whether the
whole plan works: are GST advance rulings actually indexed here, and under
what doctype?

Usage:
    set INDIANKANOON_TOKEN=your_token_here      (PowerShell: $env:INDIANKANOON_TOKEN="...")
    python test_indiankanoon.py
"""

import json
import os
import requests

TOKEN = os.environ.get("INDIANKANOON_TOKEN")
if not TOKEN:
    raise SystemExit("Set INDIANKANOON_TOKEN environment variable first.")

HEADERS = {"Authorization": f"Token {TOKEN}"}
BASE = "https://api.indiankanoon.org"


def search(query: str, doctypes: str = None, pagenum: int = 0):
    params = {"formInput": query, "pagenum": pagenum}
    if doctypes:
        params["doctypes"] = doctypes
    resp = requests.post(f"{BASE}/search/", headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def summarize(label: str, result: dict):
    found = result.get("found", "?")
    docs = result.get("docs", [])
    print(f"\n--- {label} ---")
    print(f"Reported total found: {found}")
    print(f"Docs on this page: {len(docs)}")
    for d in docs[:5]:
        print(f"  - [{d.get('docsource')}] {d.get('title')}  (docsize={d.get('docsize')})")


if __name__ == "__main__":
    # 1. Broad query, no doctype filter — see what comes back at all
    r1 = search('"advance ruling" GST')
    summarize('formInput="advance ruling" GST, no doctype filter', r1)

    # 2. Same query restricted to tribunals, in case AAR/AAAR is bucketed there
    r2 = search('"advance ruling" GST', doctypes="tribunals")
    summarize('Same query, doctypes=tribunals', r2)

    # 3. Try the AAR-specific phrase directly
    r3 = search('"Authority for Advance Ruling" GST')
    summarize('formInput="Authority for Advance Ruling" GST', r3)

    print("\n\nCheck the 'docsource' values above — that tells you the real "
          "doctype tag to filter on (if any) for the full fetch script. "
          "If 'found' is near zero across all three, AAR/AAAR coverage on "
          "Indian Kanoon may be thinner than assumed — worth knowing now, "
          "not after building the full pipeline.")