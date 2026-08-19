"""
Single-file timing test. Run this on ONE real file from your raw/aar_rulings/
folder to get a genuine, measured per-document time — this ends the guessing
and gives us a real number to multiply against 2,828 files.

Usage:
    python time_single_file.py "raw\\aar_rulings\\<pick any real filename>.pdf" --tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
"""

import argparse
import time
from pathlib import Path

import extract_corpus  # reuse the real extraction logic, not a reimplementation

ap = argparse.ArgumentParser()
ap.add_argument("file", type=Path)
ap.add_argument("--tesseract-cmd", type=str, default=None)
ap.add_argument("--timeout", type=int, default=120)
ap.add_argument("--ocr-engine", choices=["paddleocr", "tesseract"], default="tesseract",
                 help="which engine to time — defaults to tesseract here since that's "
                      "the proven path; pass paddleocr explicitly if you want to test that instead")
args = ap.parse_args()

if args.tesseract_cmd:
    extract_corpus._TESSERACT_CMD_OVERRIDE = args.tesseract_cmd

print(f"Timing extraction of: {args.file} (engine={args.ocr_engine})")
start = time.time()
try:
    text, was_ocr = extract_corpus.extract_document_with_timeout(
        args.file, timeout=args.timeout, engine=args.ocr_engine)
    elapsed = time.time() - start
    print(f"\nDONE in {elapsed:.1f} seconds")
    print(f"was_ocr: {was_ocr}")
    print(f"extracted chars: {len(text)}")
    print(f"\n--- first 300 chars ---\n{text[:300]}")
    print(f"\n\nAt this rate, 2828 files sequentially would take "
          f"~{elapsed * 2828 / 3600:.1f} hours. "
          f"With N working workers in parallel, roughly divide that by N "
          f"(optimistically — real speedup is usually less than perfect).")
except TimeoutError as e:
    elapsed = time.time() - start
    print(f"\nTIMED OUT after {elapsed:.1f}s: {e}")
    print("This single file alone would blow the per-file budget at scale — "
          "worth knowing whether this is a one-off bad file or typical.")
