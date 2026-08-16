"""
Tokenize + pack corpus — GST/tax regulatory LLM project.

Takes processed/train.jsonl and processed/val.jsonl (from extract_corpus.py)
and produces train.bin / val.bin — flat uint16 arrays of GPT-2 BPE token IDs,
memory-mappable during training so you're not loading the whole corpus into
RAM at once. This is the same packing approach the book (and nanoGPT) uses:

  1. Tokenize each document with GPT-2 BPE.
  2. Join all documents in a split with an end-of-text token between them
     (so the model learns document boundaries, not that unrelated rulings
     are one continuous text).
  3. Concatenate into one long token stream, then slice into fixed-length
     context_len chunks — no padding, no wasted compute. The remainder
     that doesn't fill a full chunk is dropped.

Usage:
    python tokenize_and_pack.py --processed-dir processed --context-len 1024
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_docs(jsonl_path: Path) -> list[str]:
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(rec["text"])
    return docs


def tokenize_and_concat(docs: list[str], enc, eot_token: int) -> np.ndarray:
    """Tokenize every doc, join with an EOT token between them. Returns one
    flat array of token IDs for the whole split."""
    all_ids = []
    for doc in docs:
        ids = enc.encode(doc)
        all_ids.extend(ids)
        all_ids.append(eot_token)
    return np.array(all_ids, dtype=np.uint16)


def pack_into_sequences(token_ids: np.ndarray, context_len: int) -> np.ndarray:
    """Slice a flat token stream into (N, context_len) chunks. Drops the
    trailing remainder that doesn't fill a full sequence — for a corpus
    this size, that's a handful of tokens lost, not worth padding for."""
    n_sequences = len(token_ids) // context_len
    usable = token_ids[: n_sequences * context_len]
    return usable.reshape(n_sequences, context_len)


def process_split(jsonl_path: Path, out_bin_path: Path, context_len: int) -> dict:
    if not jsonl_path.exists():
        print(f"  {jsonl_path} not found — skipping")
        return {"docs": 0, "tokens": 0, "sequences": 0}

    docs = load_docs(jsonl_path)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        eot_token = enc.eot_token  # 50256
    except Exception as e:
        raise SystemExit(
            f"tiktoken unavailable ({e}). Real token IDs are required here — "
            f"unlike extract_corpus.py's rough char/4 estimate, packing needs "
            f"actual GPT-2 BPE token IDs to produce a usable training file. "
            f"Ensure you have network access for tiktoken's one-time vocab download."
        )

    print(f"  Tokenizing {len(docs)} documents from {jsonl_path.name}...")
    token_ids = tokenize_and_concat(docs, enc, eot_token)
    sequences = pack_into_sequences(token_ids, context_len)

    sequences.tofile(out_bin_path)

    return {
        "docs": len(docs),
        "tokens": int(len(token_ids)),
        "sequences": int(sequences.shape[0]),
        "context_len": context_len,
        "dropped_remainder_tokens": int(len(token_ids) % context_len),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", type=Path, default=Path("processed"))
    ap.add_argument("--context-len", type=int, default=1024)
    args = ap.parse_args()

    stats = {}
    for split in ["train", "val"]:
        jsonl_path = args.processed_dir / f"{split}.jsonl"
        bin_path = args.processed_dir / f"{split}.bin"
        print(f"\n[{split}]")
        stats[split] = process_split(jsonl_path, bin_path, args.context_len)
        s = stats[split]
        if s["docs"] > 0:
            print(f"  {s['docs']} docs -> {s['tokens']:,} tokens -> "
                  f"{s['sequences']:,} sequences of {args.context_len} tokens "
                  f"({s['dropped_remainder_tokens']} trailing tokens dropped)")

    (args.processed_dir / "pack_stats.json").write_text(json.dumps(stats, indent=2))

    total_seq = stats.get("train", {}).get("sequences", 0)
    if total_seq:
        print(f"\nDone. {total_seq:,} training sequences of {args.context_len} tokens "
              f"written to {args.processed_dir / 'train.bin'} (uint16, memory-mappable).")
        print(f"Load during training with:\n"
              f"  np.memmap('{args.processed_dir / 'train.bin'}', dtype=np.uint16, mode='r')"
              f".reshape(-1, {args.context_len})")


if __name__ == "__main__":
    main()