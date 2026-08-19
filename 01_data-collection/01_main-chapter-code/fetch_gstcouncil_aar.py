"""
gstcouncil.gov.in AAR ruling scraper — GST/tax regulatory LLM project.

Confirmed against the live site (2026-08-14):
  - URL: https://gstcouncil.gov.in/en/authority-for-advance-ruling
        ?field_states_ut_target_id=All&field_year_target_id=All&page=N
  - Standard Drupal Views pagination, 0-indexed, confirmed through page=307
    (308 pages total, 3,071 records, ~10 rows/page)
  - Table columns (confirmed from live fetch): Sr. No. | Name of the
    Applicant | States/UT | Brief of Order in Appeal (OIA) | Order No. &
    Date | Download | Category
  - Download column contains a direct, already-absolute PDF link
    (gstcouncil.gov.in/sites/default/files/AAR/<name>.pdf) — no URL
    resolution needed, unlike the India Code ViewFileUploaded scheme.

This parser locates columns by HEADER TEXT, not fixed index — more
robust if the site ever reorders columns, and the same code should work
unmodified for gstcouncil.gov.in/appellate-orders (77 records, same site,
very likely same table shape) by pointing --url at that listing instead.

Usage:
    python fetch_gstcouncil_aar.py --out-dir raw --source-type aar_rulings

    Defaults to the full "-Any-/-Any-" AAR listing, all 308 pages. Use
    --max-pages to limit for testing, and --start-page to resume after
    an interruption without re-downloading everything.
"""

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://gstcouncil.gov.in/en/authority-for-advance-ruling"
SITE_ROOT = "https://gstcouncil.gov.in"
HEADERS = {"User-Agent": "Mozilla/5.0 (research corpus collection)"}


def fetch_page(page_num: int) -> str:
    params = {
        "field_states_ut_target_id": "All",
        "field_year_target_id": "All",
        "page": page_num,
    }
    resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_results_table(html: str) -> list[dict]:
    """Find the results table by locating a header row containing both
    'Download' and 'Applicant' — specific enough to avoid matching an
    unrelated table on the page, general enough to survive minor markup
    changes."""
    soup = BeautifulSoup(html, "html.parser")

    target_table = None
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        header_text = " ".join(th.get_text(strip=True) for th in header_cells).lower()
        if "download" in header_text and "applicant" in header_text:
            target_table = table
            break

    if target_table is None:
        return []

    headers = [th.get_text(strip=True) for th in target_table.find_all("th")]
    download_idx = next((i for i, h in enumerate(headers) if "download" in h.lower()), None)

    records = []
    for tr in target_table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds or download_idx is None or download_idx >= len(tds):
            continue

        row = {headers[i]: tds[i].get_text(strip=True) for i in range(min(len(headers), len(tds)))}

        pdf_link = tds[download_idx].find("a")
        row["pdf_url"] = urljoin(SITE_ROOT, pdf_link["href"].strip()) if pdf_link and pdf_link.get("href") else None

        if row.get("pdf_url"):
            records.append(row)

    return records


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80] if slug else "doc"


def record_filename(rec: dict) -> str:
    applicant_key = next((k for k in rec if "applicant" in k.lower()), None)
    order_key = next((k for k in rec if "order no" in k.lower()), None)
    name_part = slugify(rec.get(applicant_key, "") if applicant_key else "")
    order_part = slugify(rec.get(order_key, "") if order_key else "")
    return f"{name_part}__{order_part}.pdf"


def download_pdfs(records: list[dict], out_dir: Path, delay: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded, skipped, failed = 0, 0, 0

    for i, rec in enumerate(records):
        fname = record_filename(rec)
        fpath = out_dir / fname

        if fpath.exists():
            skipped += 1
            continue

        try:
            resp = requests.get(rec["pdf_url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()
            fpath.write_bytes(resp.content)
            downloaded += 1
        except Exception as e:
            print(f"    FAILED: {fname} ({e})")
            failed += 1

        time.sleep(delay)

    return downloaded, skipped, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("raw"))
    ap.add_argument("--source-type", default="aar_rulings")
    ap.add_argument("--start-page", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=308,
                     help="confirmed total is 308 pages (page=0..307)")
    ap.add_argument("--delay", type=float, default=1.5,
                     help="seconds between PDF downloads (page fetches use the same delay)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir / args.source_type
    total_downloaded = total_skipped = total_failed = total_parsed = 0

    for page in range(args.start_page, min(args.start_page + args.max_pages, 308)):
        print(f"Page {page + 1}/308 ...", end=" ")
        try:
            html = fetch_page(page)
        except Exception as e:
            print(f"FETCH FAILED: {e} — stopping here, resume with --start-page {page}")
            break

        records = parse_results_table(html)
        total_parsed += len(records)
        print(f"{len(records)} records")

        if records and not args.dry_run:
            d, s, f = download_pdfs(records, out_dir, args.delay)
            total_downloaded += d
            total_skipped += s
            total_failed += f

        if args.dry_run and page == args.start_page:
            for r in records[:2]:
                print("   ", r)

        time.sleep(args.delay)

    print(f"\nTotal records parsed: {total_parsed}")
    if not args.dry_run:
        print(f"Downloaded: {total_downloaded}  Skipped (already had): {total_skipped}  Failed: {total_failed}")
        print(f"\nIf this run was interrupted before page 308, re-run with "
              f"--start-page N to resume — already-downloaded files are skipped "
              f"automatically, so re-running the full range is also safe, just slower.")


if __name__ == "__main__":
    main()