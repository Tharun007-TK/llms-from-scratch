"""
Build a Hugging Face `datasets` DatasetDict from the extracted corpus —
GST/tax regulatory LLM project.

Takes processed/train.jsonl and processed/val.jsonl (from extract_corpus.py)
and produces a per-document (not pre-packed) HF dataset — the reusable,
inspectable artifact for the series, distinct from the packed train.bin/
val.bin used only by the training loop itself.

Usage:
    # Build + save locally, don't push anywhere yet:
    python build_hf_dataset.py --processed-dir processed --out-dir hf_dataset

    # Build AND push to your HF Hub account (requires `huggingface-cli login`
    # first, or an HF_TOKEN env var):
    python build_hf_dataset.py --processed-dir processed --out-dir hf_dataset \\
        --push-to-hub your-username/gst-rulings-corpus

IMPORTANT — LICENSE: this script writes "TODO" into the dataset card's
license field on purpose. Indian government legal text (circulars, AAR
rulings) isn't automatically public domain — unlike US federal documents,
there's no blanket public-domain rule under the Indian Copyright Act.
Check the actual copyright status (e.g. Section 52 exceptions, or explicit
government reuse policies) before publishing this publicly, and fill in
the real license yourself. Don't ship this with a guessed license.
"""

import argparse
import json
from pathlib import Path


def load_jsonl_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Keep fields useful to someone browsing/using the dataset;
            # drop content_hash (internal dedup artifact, no external value)
            # and source_path (a local file path, meaningless off your machine).
            records.append({
                "doc_id": rec["doc_id"],
                "source_type": rec["source_type"],
                "text": rec["text"],
                "char_count": rec["char_count"],
                "was_ocr": rec["was_ocr"],
            })
    return records


DATASET_CARD_TEMPLATE = """\
---
language:
- en
license: TODO-CHECK-COPYRIGHT-STATUS-BEFORE-PUBLISHING
task_categories:
- text-generation
pretty_name: GST/Tax Regulatory Text Corpus (India)
size_categories:
- 1K<n<10K
---

# GST/Tax Regulatory Text Corpus

A narrow-domain corpus of Indian GST (Goods and Services Tax) regulatory
text, assembled for pretraining a small (~130M parameter) language model
from scratch, following Sebastian Raschka's *Build a Large Language Model
From Scratch*.

## Contents

- **{n_train} training documents / {n_val} validation documents**
- **~{total_tokens:,} tokens** (GPT-2 BPE)
- Two source types:
  - `circulars` — CGST circulars from India Code (indiacode.nic.in)
  - `aar_rulings` — Authority for Advance Ruling (AAR) orders from
    gstcouncil.gov.in, covering all Indian states/UTs

## Known limitations — read before use

- **Small for pretraining a language model from scratch.** ~{total_tokens:,}
  tokens is far below the Chinchilla-optimal token budget for a 130M
  parameter model (~2.6B tokens). A model trained on this corpus is
  necessarily undertrained relative to compute-optimal scaling, and/or
  relies on repeated epochs over this data — this is a deliberate,
  disclosed tradeoff of the project, not an oversight.
- **OCR-derived text.** A meaningful fraction of `aar_rulings` documents
  were extracted via OCR (Tesseract) from scanned PDFs, not a clean text
  layer. OCR error rate is nonzero — check the `was_ocr` field per
  document if working with a subset where accuracy matters.
- **Narrow domain.** This is not general-purpose text. It's useful for
  domain-specific language modeling (GST/tax regulatory language) and not
  intended as, or suitable for, general knowledge or reasoning tasks.
- **Not legal advice, not verified against primary sources by a legal
  professional.** Sourced from official government publications, but
  provided as-is for ML research/education, not as an authoritative legal
  reference.

## License

TODO — verify the copyright status of Indian government circulars and
AAR/AAAR rulings before treating this as freely licensed. Fill in before
publishing.

## Source

Built as part of a public series on building an LLM from scratch:
[link to your blog/LinkedIn series here]
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", type=Path, default=Path("processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("hf_dataset"))
    ap.add_argument("--push-to-hub", type=str, default=None,
                     help="HF Hub repo id to push to, e.g. your-username/gst-rulings-corpus. "
                          "Omit to just build and save locally.")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict

    train_records = load_jsonl_records(args.processed_dir / "train.jsonl")
    val_records = load_jsonl_records(args.processed_dir / "val.jsonl")

    print(f"Train: {len(train_records)} docs")
    print(f"Val:   {len(val_records)} docs")

    ds = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(val_records),
    })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out_dir))
    print(f"\nSaved locally to {args.out_dir}")

    total_tokens = sum(r["char_count"] for r in train_records + val_records) // 4  # rough fallback estimate
    card = DATASET_CARD_TEMPLATE.format(
        n_train=len(train_records), n_val=len(val_records), total_tokens=total_tokens,
    )
    (args.out_dir / "README.md").write_text(card, encoding="utf-8")
    print(f"Wrote dataset card to {args.out_dir / 'README.md'} — "
          f"EDIT THE LICENSE FIELD before publishing, see the warning at the top of this script.")

    if args.push_to_hub:
        print(f"\nPushing to HF Hub: {args.push_to_hub} ...")
        print("(requires prior `huggingface-cli login` or HF_TOKEN env var)")
        ds.push_to_hub(args.push_to_hub)
        print("Pushed. Note: push_to_hub does NOT upload the README.md dataset "
              "card automatically — upload it separately via the Hub web UI or "
              "huggingface_hub's upload_file, after you've filled in the license.")


if __name__ == "__main__":
    main()
