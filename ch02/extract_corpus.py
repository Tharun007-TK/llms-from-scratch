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

# If Tesseract is installed but pytesseract can't auto-locate it (common on
# Windows even when `tesseract --version` works fine in your terminal), set
# the exe path explicitly here or via --tesseract-cmd on the command line.
_TESSERACT_CMD_OVERRIDE = None

# Scanning-app watermarks that sit ON TOP of an otherwise-image-only PDF page.
# pdfplumber sees this text as a real text layer, which fools the "is this a
# scan?" length check into thinking the page has substantive content when it
# doesn't. Confirmed against a real file: "Scanned by CamScanner" (22 chars)
# slipped past a 20-char threshold, causing OCR to be skipped entirely on a
# document that was actually a pure image. Strip these before judging length.
_SCANNER_WATERMARK_PATTERNS = [
    re.compile(r"scanned\s*(?:by|with)?\s*camscanner", re.IGNORECASE),
    re.compile(r"scanned\s*(?:by|with)?\s*adobe\s*scan", re.IGNORECASE),
    re.compile(r"scanned\s*(?:by|with)?\s*genius\s*scan", re.IGNORECASE),
    re.compile(r"created\s*with\s*scanner\s*app", re.IGNORECASE),
]


def _strip_scanner_watermarks(text: str) -> str:
    for pattern in _SCANNER_WATERMARK_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Stage 1: extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: Path, engine: str = "paddleocr") -> str:
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
    # Strip scanner-app watermarks first — they can inflate apparent text
    # length enough to wrongly skip OCR (confirmed: "Scanned by CamScanner"
    # alone was enough to fool a naive length check).
    meaningful_text = _strip_scanner_watermarks(text)
    if len(meaningful_text) < 20 * len(text_parts):
        return _ocr_pdf(path, engine=engine)
    return text


def _ocr_pdf_tesseract(path: Path) -> str:
    """OCR fallback using pytesseract. Slower — only invoked when direct
    extraction looks empty. Flag OCR'd docs downstream (see extract_document)
    since OCR error rate is nonzero and worth tracking separately in your
    corpus stats, not silently blended in with clean-text documents."""
    import pytesseract
    from pdf2image import convert_from_path

    if _TESSERACT_CMD_OVERRIDE:
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD_OVERRIDE

    pages = convert_from_path(str(path), dpi=200)
    return "\n".join(pytesseract.image_to_string(p) for p in pages).strip()


# PaddleOCR engine is expensive to initialize (loads model weights) — reuse
# one instance per worker process instead of recreating it per file, or
# per-file overhead alone could dominate runtime on a large batch.
_PADDLE_OCR_INSTANCE = None


def _get_paddle_ocr():
    global _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE is None:
        from paddleocr import PaddleOCR
        # lang='en' — rulings are English-language legal text. Text-line
        # orientation classification helps with slightly rotated/skewed
        # phone scans, common in this corpus given the CamScanner watermarks
        # we found. enable_mkldnn=False works around a confirmed open bug in
        # PaddlePaddle 3.3.x's oneDNN CPU backend (NotImplementedError:
        # ConvertPirAttribute2RuntimeAttribute) — see PaddlePaddle/Paddle
        # issue #77340 and PaddleOCR issue #17955. Costs some CPU speed
        # versus oneDNN-accelerated inference, but oneDNN is currently
        # broken outright on affected CPUs, so this isn't optional.
        _PADDLE_OCR_INSTANCE = PaddleOCR(use_textline_orientation=True, lang="en",
                                          enable_mkldnn=False,
                                          use_doc_orientation_classify=False,
                                          use_doc_unwarping=False)
    return _PADDLE_OCR_INSTANCE


def _ocr_pdf_paddle(path: Path) -> str:
    """OCR fallback using PaddleOCR — generally faster and more accurate
    than Tesseract on stamped/signed scanned legal documents, still CPU-only
    (no local GPU assumed). Converts each page to an image via pdf2image
    (still needs Poppler installed) then runs PaddleOCR per page."""
    from pdf2image import convert_from_path
    import numpy as np

    ocr = _get_paddle_ocr()
    pages = convert_from_path(str(path), dpi=200)

    all_text = []
    for page_img in pages:
        result = ocr.predict(np.array(page_img))
        for res in result:
            # PaddleOCR 3.x returns dict-like result objects with 'rec_texts'
            # holding the recognized text lines directly — confirmed against
            # PaddleOCR's current documentation, not the older nested
            # box/tuple format from PaddleOCR 2.x.
            texts = res.get("rec_texts", [])
            if texts:
                all_text.append("\n".join(texts))

    return "\n\n".join(all_text).strip()


def _ocr_pdf(path: Path, engine: str = "paddleocr") -> str:
    if engine == "tesseract":
        return _ocr_pdf_tesseract(path)
    elif engine == "paddleocr":
        return _ocr_pdf_paddle(path)
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")


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


def extract_document(path: Path, engine: str = "paddleocr") -> tuple[str, bool]:
    """Returns (text, was_ocr)."""
    if path.suffix.lower() == ".pdf":
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            sample = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
        sample_meaningful = _strip_scanner_watermarks(sample)
        if len(sample_meaningful) < 20:
            return _ocr_pdf(path, engine=engine), True
        return extract_pdf_text(path, engine=engine), False
    elif path.suffix.lower() in (".html", ".htm"):
        return extract_html_text(path), False
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def extract_document_with_timeout(path: Path, timeout: int = 90, engine: str = "paddleocr") -> tuple[str, bool]:
    """Wraps extract_document with a hard timeout so one pathological file
    (e.g. a huge scanned multi-page PDF, or a hung Poppler subprocess) can't
    stall the entire run. On timeout, skips this file and moves on.

    Caveat: Python can't force-kill a stuck thread, so a timed-out
    extraction's underlying work may keep running in the background even
    after we give up waiting on it. That's an acceptable tradeoff to
    unblock the overall run — but if you hit many timeouts, your machine
    may be doing more background work than the progress log suggests."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract_document, path, engine)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"extraction exceeded {timeout}s — likely a large "
                                f"scanned PDF or a hung OCR/Poppler call")


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

def _process_one_file(args: tuple) -> dict:
    """Top-level, picklable worker function for ProcessPoolExecutor — extracts,
    OCR-fallbacks, and cleans a single file, returning a plain dict (not a
    DocRecord dataclass, which round-trips through multiprocessing fine but
    dict is simpler to reason about across the process boundary)."""
    import os
    # Tesseract uses OpenMP to multithread internally by default. Running N
    # worker PROCESSES that each also spawn a multithreaded Tesseract call
    # oversubscribes the CPU (N processes x M internal threads each, fighting
    # over the same cores) — often slower than sequential, not faster. Pin
    # each Tesseract call to one thread so our outer-level process pool is
    # the only source of parallelism.
    os.environ["OMP_THREAD_LIMIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    source_type, path_str, timeout, tesseract_cmd, engine = args
    path = Path(path_str)

    global _TESSERACT_CMD_OVERRIDE
    if tesseract_cmd:
        _TESSERACT_CMD_OVERRIDE = tesseract_cmd

    try:
        raw_text, was_ocr = extract_document_with_timeout(path, timeout=timeout, engine=engine)
    except TimeoutError as e:
        return {"status": "skip", "file": path.name, "reason": str(e)}
    except Exception as e:
        return {"status": "skip", "file": path.name, "reason": f"extraction failed ({e})"}

    cleaned = clean_text(raw_text)
    if len(cleaned) < 100:
        return {"status": "skip", "file": path.name,
                "reason": f"near-empty after cleaning ({len(cleaned)} chars)"}

    return {
        "status": "ok",
        "file": path.name,
        "was_ocr": was_ocr,
        "doc_id": make_doc_id(source_type, path),
        "source_type": source_type,
        "source_path": path_str,
        "text": cleaned,
        "char_count": len(cleaned),
        "content_hash": content_hash(cleaned),
    }


def run_pipeline(raw_dir: Path, out_dir: Path, val_frac: float = 0.10, timeout: int = 90,
                  workers: int = 1, tesseract_cmd: str = None, ocr_engine: str = "paddleocr"):
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[DocRecord] = []

    source_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not source_dirs:
        raise SystemExit(
            f"No source_type subdirectories found under {raw_dir}. "
            f"Expected e.g. {raw_dir}/aar_rulings/, {raw_dir}/circulars/, etc."
        )

    all_jobs = []
    for source_dir in source_dirs:
        source_type = source_dir.name
        files = [f for f in source_dir.iterdir()
                 if f.suffix.lower() in (".pdf", ".html", ".htm")]
        print(f"[{source_type}] {len(files)} raw files found")
        for f in files:
            all_jobs.append((source_type, str(f), timeout, tesseract_cmd, ocr_engine))

    total = len(all_jobs)
    print(f"\nProcessing {total} files with {workers} worker(s)...\n")

    if workers <= 1:
        results = []
        for i, job in enumerate(all_jobs):
            if i % 100 == 0 and i > 0:
                print(f"  ... {i}/{total} processed")
            r = _process_one_file(job)
            if r["status"] == "skip":
                print(f"  SKIP {r['file']}: {r['reason']}")
            elif r["was_ocr"]:
                print(f"  [OCR] {r['file']}")
            results.append(r)
    else:
        import concurrent.futures
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_one_file, job) for job in all_jobs]
            import time as _time
            start_time = _time.time()
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                if i % 20 == 0 and i > 0:
                    elapsed = _time.time() - start_time
                    rate = i / elapsed * 60  # files per minute
                    remaining_min = (total - i) / (rate + 1e-6)
                    print(f"  ... {i}/{total} processed "
                          f"({rate:.1f}/min, ~{remaining_min:.0f} min remaining)")
                r = future.result()
                # Print per-file status live as each result actually completes,
                # not batched at the end — fixes a real gap where the parallel
                # path silently deferred every SKIP/[OCR] line until the whole
                # 2,976-file run finished, making it look stuck when it wasn't.
                if r["status"] == "skip":
                    print(f"  SKIP {r['file']}: {r['reason']}")
                elif r["was_ocr"]:
                    print(f"  [OCR] {r['file']}")
                results.append(r)

    for r in results:
        if r["status"] == "skip":
            continue
        rec = DocRecord(
            doc_id=r["doc_id"], source_type=r["source_type"], source_path=r["source_path"],
            text=r["text"], char_count=r["char_count"], was_ocr=r["was_ocr"],
            content_hash=r["content_hash"],
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
    ap.add_argument("--tesseract-cmd", type=str, default=None,
                     help=r"Full path to tesseract.exe if pytesseract can't auto-detect it, "
                          r"e.g. --tesseract-cmd \"C:\Program Files\Tesseract-OCR\tesseract.exe\"")
    ap.add_argument("--timeout", type=int, default=90,
                     help="max seconds per document before skipping it (guards against hangs)")
    ap.add_argument("--workers", type=int, default=1,
                     help="parallel worker processes for extraction (try 4 or your CPU core count "
                          "for a large batch like AAR rulings — this is the main lever if extraction "
                          "is projecting many hours)")
    ap.add_argument("--ocr-engine", choices=["paddleocr", "tesseract"], default="paddleocr",
                     help="OCR engine for scanned PDFs — paddleocr is generally faster/more "
                          "accurate on stamped legal scans and needs no separate system install "
                          "beyond pip; tesseract remains available via --ocr-engine tesseract")
    args = ap.parse_args()

    if args.tesseract_cmd:
        _TESSERACT_CMD_OVERRIDE = args.tesseract_cmd

    run_pipeline(args.raw_dir, args.out_dir, args.val_frac, timeout=args.timeout,
                 workers=args.workers, tesseract_cmd=args.tesseract_cmd, ocr_engine=args.ocr_engine)
