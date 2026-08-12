"""
Corpus extraction pipeline — GST/tax regulatory LLM project.

Pipeline stages, run in order:
    1. extract    — raw PDF/HTML -> plain text (per document)
    2. clean      — strip boilerplate, normalize whitespace
    3. dedup      — exact + near-duplicate removal (minhash)
    4. tokenize   — real token counts per source type (GPT-2 BPE)
    5. split      — stratified train/val split by source_type, whole-document

WHAT THIS SCRIPT DOES NOT DO FOR YOU:
    Fetching/scraping the source sites is deliberately NOT hardcoded here —
    CBIC, GST Council, Indian Kanoon, and indiacode.nic.in each have
    different HTML structures, and hardcoding selectors against pages I
    haven't fetched live would just be guessed code that breaks on first
    run. Drop your downloaded raw files (PDFs and/or saved HTML) into
    raw/<source_type>/ and this pipeline takes it from there. Write the
    per-site fetch loop separately once you've inspected each site's
    actual markup — happy to help with that per-source once you're ready.

Directory layout expected:
    raw/
      aar_rulings/      *.pdf or *.html
      bare_acts/        *.pdf or *.html
      circulars/        *.pdf or *.html
      notifications/    *.pdf or *.html

Output:
    processed/<source_type>/<doc_id>.jsonl   (one JSON object per doc)
    corpus_stats.json                        (token counts per source)
    train.jsonl / val.jsonl                  (final stratified split)
"""

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Stage 1: extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: Path) -> str:
    """Try direct text extraction first; fall back to OCR if the PDF has
    no embedded text layer (i.e. it's a scan). Checking fonts first avoids
    silently returning empty strings for scanned circulars, which would
    quietly shrink your token count without any error."""
    import pdfplumber

    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    text = "\n".join(text_parts).strip()

    # Heuristic: if extraction yielded almost nothing relative to page
    # count, this is very likely a scanned document with no text layer.
    if len(text) < 20 * len(text_parts):
        return _ocr_pdf(path)
    return text


def _ocr_pdf(path: Path) -> str:
    """OCR fallback for scanned PDFs. Slower — only invoked when direct
    extraction looks empty. Flag OCR'd docs downstream (see extract_document)
    since OCR error rate is nonzero and worth tracking separately in your
    corpus stats, not silently blended in with clean-text documents."""
    import pytesseract
    from pdf2image import convert_from_path

    pages = convert_from_path(str(path), dpi=200)
    return "\n".join(pytesseract.image_to_string(p) for p in pages).strip()


def extract_html_text(path: Path) -> str:
    """Strip nav/header/footer/script/style, keep the article body.
    Adjust the CSS selector list per source once you've inspected each
    site's actual markup — this default is a reasonable starting point,
    not a guarantee it matches CBIC/GST Council/Indian Kanoon exactly."""
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Prefer a main-content container if one exists; fall back to body.
    main = soup.select_one("main, article, .content, #content, .entry-content")
    container = main if main else soup.body if soup.body else soup
    text = container.get_text(separator="\n")
    return text.strip()


def extract_document(path: Path) -> tuple[str, bool]:
    """Returns (text, was_ocr)."""
    if path.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            sample = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
        if len(sample) < 20:
            return _ocr_pdf(path), True
        return extract_pdf_text(path), False
    elif path.suffix.lower() in (".html", ".htm"):
        return extract_html_text(path), False
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


# ---------------------------------------------------------------------------
# Stage 2: cleaning
# ---------------------------------------------------------------------------

# High-frequency, low-information boilerplate that shows up across CBIC/
# GST Council documents. Extend this list as you see recurring junk in
# your actual pulled corpus — this is a starting set, not exhaustive.
BOILERPLATE_PATTERNS = [
    r"Government of India\s*\n?\s*Ministry of Finance",
    r"Department of Revenue\s*\n?\s*Central Board of Indirect Taxes and Customs",
    r"F\.?\s*No\.?\s*[\w./-]+",           # file number lines
    r"Page \d+ of \d+",
    r"^\s*\d+\s*$",                        # bare page-number lines
    r"To\s*\n\s*All Principal Chief Commissioners.*?(?=\n\n)",
]

_COMPILED_BOILERPLATE = [re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                          for p in BOILERPLATE_PATTERNS]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    for pattern in _COMPILED_BOILERPLATE:
        text = pattern.sub("", text)

    # Collapse excessive whitespace left behind by stripped boilerplate.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Metadata + document record
# ---------------------------------------------------------------------------

@dataclass
class DocRecord:
    doc_id: str
    source_type: str        # aar_rulings | bare_acts | circulars | notifications
    source_path: str
    text: str
    char_count: int
    was_ocr: bool
    content_hash: str


def make_doc_id(source_type: str, path: Path) -> str:
    return f"{source_type}__{path.stem}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage 3: deduplication
# ---------------------------------------------------------------------------

def exact_dedup(records: list[DocRecord]) -> list[DocRecord]:
    seen = set()
    out = []
    for r in records:
        if r.content_hash in seen:
            continue
        seen.add(r.content_hash)
        out.append(r)
    return out


def near_dedup(records: list[DocRecord], threshold: float = 0.85) -> list[DocRecord]:
    """MinHash-based near-duplicate removal. Legal corpora reuse whole
    paragraphs verbatim across documents (amended circulars re-quoting the
    original, rulings cross-citing prior rulings at length) — exact hash
    dedup alone won't catch this."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    out = []

    for r in records:
        mh = MinHash(num_perm=128)
        for shingle in _shingles(r.text, k=5):
            mh.update(shingle.encode("utf-8"))

        if lsh.query(mh):
            continue  # near-duplicate of something already kept
        lsh.insert(r.doc_id, mh)
        minhashes[r.doc_id] = mh
        out.append(r)

    return out


def _shingles(text: str, k: int = 5):
    words = text.split()
    for i in range(max(len(words) - k + 1, 1)):
        yield " ".join(words[i:i + k])


# ---------------------------------------------------------------------------
# Stage 4: tokenization stats
# ---------------------------------------------------------------------------

def token_stats(records: list[DocRecord]) -> dict:
    """Real token counts per source type using the GPT-2 BPE tokenizer —
    replace the earlier word-count estimate with this before finalizing
    the data-prep post."""
    stats: dict[str, dict] = {}

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        count_fn = lambda t: len(enc.encode(t))
    except Exception as e:
        # tiktoken needs a one-time download of the GPT-2 merge/vocab files;
        # if that's blocked in your environment, fall back to a rough
        # chars/4 estimate so the pipeline still completes. Re-run with
        # network access before trusting these numbers for the data-prep
        # post — this fallback is for pipeline development only.
        print(f"  WARNING: tiktoken unavailable ({e}); using chars/4 "
              f"estimate instead of real token counts")
        count_fn = lambda t: len(t) // 4

    for r in records:
        n_tokens = count_fn(r.text)
        s = stats.setdefault(r.source_type, {"docs": 0, "tokens": 0})
        s["docs"] += 1
        s["tokens"] += n_tokens

    stats["_total"] = {
        "docs": sum(v["docs"] for k, v in stats.items() if k != "_total"),
        "tokens": sum(v["tokens"] for k, v in stats.items() if k != "_total"),
    }
    return stats


# ---------------------------------------------------------------------------
# Stage 5: stratified split (whole-document, per source type)
# ---------------------------------------------------------------------------

def stratified_split(records: list[DocRecord], val_frac: float = 0.10, seed: int = 42):
    import random

    rng = random.Random(seed)
    by_source: dict[str, list[DocRecord]] = {}
    for r in records:
        by_source.setdefault(r.source_type, []).append(r)

    train, val = [], []
    for source_type, docs in by_source.items():
        docs = docs[:]
        rng.shuffle(docs)
        n_val = max(1, int(len(docs) * val_frac)) if len(docs) >= 10 else 0
        val.extend(docs[:n_val])
        train.extend(docs[n_val:])

    return train, val


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(raw_dir: Path, out_dir: Path, val_frac: float = 0.10):
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[DocRecord] = []

    source_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not source_dirs:
        raise SystemExit(
            f"No source_type subdirectories found under {raw_dir}. "
            f"Expected e.g. {raw_dir}/aar_rulings/, {raw_dir}/circulars/, etc."
        )

    for source_dir in source_dirs:
        source_type = source_dir.name
        files = [f for f in source_dir.iterdir()
                 if f.suffix.lower() in (".pdf", ".html", ".htm")]
        print(f"[{source_type}] {len(files)} raw files found")

        for f in files:
            try:
                raw_text, was_ocr = extract_document(f)
            except Exception as e:
                print(f"  SKIP {f.name}: extraction failed ({e})")
                continue

            cleaned = clean_text(raw_text)
            if len(cleaned) < 100:
                print(f"  SKIP {f.name}: near-empty after cleaning "
                      f"({len(cleaned)} chars) — check if this is a scan "
                      f"OCR mishandled")
                continue

            rec = DocRecord(
                doc_id=make_doc_id(source_type, f),
                source_type=source_type,
                source_path=str(f),
                text=cleaned,
                char_count=len(cleaned),
                was_ocr=was_ocr,
                content_hash=content_hash(cleaned),
            )
            records.append(rec)

    print(f"\nExtracted {len(records)} documents before dedup")
    records = exact_dedup(records)
    print(f"{len(records)} after exact dedup")
    records = near_dedup(records)
    print(f"{len(records)} after near-duplicate dedup")

    stats = token_stats(records)
    (out_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print("\nToken counts by source:")
    for source_type, s in stats.items():
        print(f"  {source_type:20s} docs={s['docs']:5d}  tokens={s['tokens']:,}")

    train, val = stratified_split(records, val_frac=val_frac)
    print(f"\nSplit: {len(train)} train docs / {len(val)} val docs")

    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "val.jsonl", val)

    ocr_count = sum(1 for r in records if r.was_ocr)
    if ocr_count:
        print(f"\nNOTE: {ocr_count} documents went through OCR — "
              f"spot-check a sample of these for garbled text before "
              f"trusting the token counts above.")


def _write_jsonl(path: Path, records: list[DocRecord]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("processed"))
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    run_pipeline(args.raw_dir, args.out_dir, args.val_frac)