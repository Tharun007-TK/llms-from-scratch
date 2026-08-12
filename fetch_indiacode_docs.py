"""
India Code Act sub-document scraper — GST/tax regulatory LLM project.

Confirmed against the live site (2026-08-13):
  - A single Act "handle" page (e.g. .../handle/123456789/15689) embeds
    ALL rows for Circulars/Notifications/Rules/Orders statically in the
    HTML. DataTables just hides rows client-side for pagination display —
    there is nothing to paginate against on the server side. One fetch
    per Act gets everything.
  - Each row's cells cycle through class="modaltd1" (date), two cells of
    class="modaltd2" (description, hindi description), and two cells of
    class="modaltd3" (files-eng link, files-hindi link). This was
    confirmed against real row HTML pulled from the site, not guessed.
  - PDF links use a ViewFileUploaded?path=...&file=... scheme, NOT the
    /bitstream/.../<id>/... scheme the Act PDF itself uses. IDs are not
    sequential or predictable across documents — you must parse the
    table, not construct URLs.

WHAT I HAVEN'T BEEN ABLE TO VERIFY:
    The exact modal container markup (id/class of the <div> wrapping
    each section's table) — I only have real HTML for the row level,
    pasted from your browser, not the full page source. This script
    finds the right table by locating a heading/label whose text matches
    the section name ("Circulars", "Notifications", etc.) and taking the
    nearest following <table>. If India Code's actual markup nests things
    differently, this heading-based lookup may need a small adjustment —
    run it, and if it reports 0 rows found, paste the actual modal HTML
    (view source around the section heading) and I'll fix the selector
    against real markup rather than guessing again.

Usage:
    python fetch_indiacode_docs.py \\
        --act-url https://www.indiacode.nic.in/handle/123456789/15689 \\
        --section Circulars \\
        --source-type circulars \\
        --out-dir raw

    Downloads PDFs into raw/circulars/, matching the folder structure
    extract_corpus.py expects.
"""

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.indiacode.nic.in"
HEADERS = {"User-Agent": "Mozilla/5.0 (research corpus collection)"}


def fetch_act_page(act_url: str) -> str:
    resp = requests.get(act_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


# Confirmed against real hrefs pulled from the site: circular links contain
# "circularindividualfile", rules links contain "rulesindividualfile". The
# rest are educated guesses following the same naming convention — pass
# --link-keyword to override if a guess turns out wrong for your section.
LINK_KEYWORD_GUESSES = {
    "circulars": "circularindividualfile",
    "rules": "rulesindividualfile",
    "notifications": "notificationindividualfile",
    "orders": "orderindividualfile",
    "regulations": "regulationindividualfile",
    "ordinance": "ordinanceindividualfile",
    "statutes": "statuteindividualfile",
}


def find_section_table(soup: BeautifulSoup, section_name: str, link_keyword: str = None):
    """Locate the table for a named section by finding links whose href
    contains a section-specific keyword (e.g. 'circularindividualfile'),
    then walking up to the containing <table>. This is more reliable than
    matching heading text — 'Circulars' also appears as a nav-tab label
    elsewhere on the page, and heading-based lookup can grab the wrong
    (possibly empty) table, e.g. an adjacent section with zero rows."""
    keyword = link_keyword or LINK_KEYWORD_GUESSES.get(section_name.lower())
    if not keyword:
        raise SystemExit(
            f"No known link-keyword pattern for section '{section_name}'. "
            f"Pass --link-keyword explicitly (inspect a real PDF link's href "
            f"on the page for the section you want)."
        )

    matching_links = [a for a in soup.find_all("a", href=True) if keyword in a["href"]]
    print(f"  (debug) found {len(matching_links)} links containing '{keyword}'")

    if not matching_links:
        return None, keyword

    table = matching_links[0].find_parent("table")
    return table, keyword


def parse_rows(table) -> list[dict]:
    """Extract (date, description, hindi_description, pdf_url, pdf_hindi_url)
    from a section table using the confirmed modaltd1/modaltd2/modaltd3
    cell pattern."""
    records = []

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue  # header row or malformed row, skip

        date = tds[0].get_text(strip=True)
        description = tds[1].get_text(strip=True)
        hindi_description = tds[2].get_text(strip=True) if len(tds) > 2 else ""

        eng_link = tds[3].find("a") if len(tds) > 3 else None
        hindi_link = tds[4].find("a") if len(tds) > 4 else None

        pdf_url = urljoin(BASE, eng_link["href"].strip()) if eng_link and eng_link.get("href") else None
        pdf_url_hindi = urljoin(BASE, hindi_link["href"].strip()) if hindi_link and hindi_link.get("href") else None

        if not date and not description:
            continue  # likely a stray/empty row

        records.append({
            "date": date,
            "description": description,
            "hindi_description": hindi_description,
            "pdf_url": pdf_url,
            "pdf_url_hindi": pdf_url_hindi,
        })

    return records


def slugify(description: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", description.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80] if slug else "doc"


def download_pdfs(records: list[dict], out_dir: Path, delay: float = 1.5):
    """Download each record's English PDF. Rate-limited — government sites
    tend to be more aggressive about blocking than commercial ones, so
    this errs slow rather than fast. Skips files that already exist so
    a re-run after an interruption doesn't re-download everything."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded, skipped, failed = 0, 0, 0

    for i, rec in enumerate(records):
        if not rec["pdf_url"]:
            print(f"  [{i+1}/{len(records)}] SKIP (no PDF link): {rec['description']}")
            failed += 1
            continue

        fname = f"{slugify(rec['description'])}.pdf"
        fpath = out_dir / fname

        if fpath.exists():
            skipped += 1
            continue

        try:
            resp = requests.get(rec["pdf_url"], headers=HEADERS, timeout=30)
            resp.raise_for_status()
            fpath.write_bytes(resp.content)
            downloaded += 1
            print(f"  [{i+1}/{len(records)}] OK: {fname}")
        except Exception as e:
            print(f"  [{i+1}/{len(records)}] FAILED: {rec['description']} ({e})")
            failed += 1

        time.sleep(delay)

    print(f"\nDownloaded: {downloaded}  Skipped (already had): {skipped}  Failed: {failed}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--act-url", required=True, help="e.g. https://www.indiacode.nic.in/handle/123456789/15689")
    ap.add_argument("--section", required=True, help="Circulars | Notifications | Rules | Orders | Regulations")
    ap.add_argument("--source-type", required=True, help="folder name under --out-dir, e.g. circulars")
    ap.add_argument("--out-dir", type=Path, default=Path("raw"))
    ap.add_argument("--delay", type=float, default=1.5, help="seconds between PDF downloads")
    ap.add_argument("--dry-run", action="store_true", help="parse and print counts only, don't download")
    ap.add_argument("--link-keyword", default=None,
                     help="override the auto-guessed href substring used to find this section's table")
    args = ap.parse_args()

    print(f"Fetching {args.act_url} ...")
    html = fetch_act_page(args.act_url)
    soup = BeautifulSoup(html, "html.parser")

    table, keyword_used = find_section_table(soup, args.section, args.link_keyword)
    if table is None:
        raise SystemExit(
            f"Found 0 links containing '{keyword_used}' on this page — either "
            f"this Act has no documents in the '{args.section}' section, or the "
            f"guessed keyword is wrong. Open a PDF link in that section's table "
            f"in your browser, copy its href, and pass the matching substring "
            f"via --link-keyword."
        )

    records = parse_rows(table)
    print(f"Parsed {len(records)} rows for section '{args.section}'")

    if not records:
        raise SystemExit("Found the table but extracted 0 rows — check parse_rows() "
                          "against this page's actual cell structure.")

    if args.dry_run:
        for r in records[:5]:
            print(" ", r)
        print("  ...")
        return

    out_dir = args.out_dir / args.source_type
    download_pdfs(records, out_dir, delay=args.delay)


if __name__ == "__main__":
    main()