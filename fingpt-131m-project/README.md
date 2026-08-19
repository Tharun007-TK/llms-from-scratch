# FinGPT-131M: Training a GST/Tax LLM from Scratch

A decoder-only transformer (131M parameters) trained from scratch on Indian
GST regulatory text — Advance Authority Rulings, CGST circulars, and India Code acts.
Built to understand the full pipeline from raw web scraping through HF Hub publication.

**Trained model:** [Tharun007/FinGPT-131M](https://huggingface.co/Tharun007/FinGPT-131M)  
**Training corpus:** [Tharun007/gst-rulings-corpus](https://huggingface.co/datasets/Tharun007/gst-rulings-corpus)

---

## Project Structure

| Stage | Folder | What It Does |
|---|---|---|
| 1 | [`01_data-collection/`](01_data-collection/) | Scrapers for gstcouncil.gov.in Advance Authority Rulings (PDFs, 3,071 records across 308 pages) and India Code acts/circulars/notifications. Each scraper locates table columns by header text, not fixed index, and handles resumable pagination. |
| 2 | [`02_corpus-extraction/`](02_corpus-extraction/) | Five-stage pipeline (extract → clean → dedup → tokenize → split) that turns raw PDFs/HTML into a stratified train/val JSONL corpus. Deduplication uses minhash; split is whole-document to avoid token leakage. |
| 3 | [`03_dataset-publishing/`](03_dataset-publishing/) | Packs the JSONL corpus into fixed-length token sequences (context_length=1024) using GPT-2 BPE and publishes a per-document HuggingFace `DatasetDict` to the Hub for reuse and inspection. |
| 4 | [`04_model-architecture/`](04_model-architecture/) | Decoder-only GPT-2-style transformer: 13 layers, 768 hidden, 12 heads, ~131.3M parameters. Uses `scaled_dot_product_attention` (causal), GELU activation, weight-tied embeddings, and pre-norm LayerNorm. One layer deeper than GPT-2 small to hit the 130M target. |
| 5 | [`05_training/`](05_training/) | FP16 mixed-precision training loop with gradient accumulation (effective batch = 8), activation checkpointing, AdamW (lr=3e-4, wd=0.1), early stopping (patience=10), and AMP-aware checkpoint saving/resuming. Best result: val_loss 2.7641 at step 11,250 on Kaggle T4. |
| 6 | [`06_export-to-huggingface/`](06_export-to-huggingface/) | Converts the raw PyTorch checkpoint to a `PreTrainedModel` subclass (`FinGPTForCausalLM`) with a custom config class (`FinGPTConfig`), exports to `safetensors`, tests generation locally, and uploads to the HF Hub. See the bonus debugging postmortem for the tied-weight export bug. |
| 7 | [`07_evaluation/`](07_evaluation/) | Qualitative generation test across 10 prompts (in-domain GST/tax text at multiple temperatures + two off-domain controls) to check coherence, repetition, and domain specificity. |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate text with the trained model
from transformers import AutoTokenizer
from fingpt-131m-project.06_export-to-huggingface.01_main-chapter-code.configuration_fingpt import FinGPTConfig
from fingpt-131m-project.06_export-to-huggingface.01_main-chapter-code.modeling_fingpt import FinGPTForCausalLM
import torch

model = FinGPTForCausalLM.from_pretrained("Tharun007/FinGPT-131M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

inputs = tokenizer("Input Tax Credit can be claimed when", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.8, top_k=50)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## Data

Raw PDFs and processed JSONL files are **not tracked** in this repo (see `.gitignore`) —
they are large and reproducible from the pipeline. See [`data/README.md`](data/README.md)
for instructions to regenerate locally, or use the published HF dataset directly.

---

## Model Card Summary

| Property | Value |
|---|---|
| Parameters | ~131.3M |
| Architecture | Decoder-only transformer (GPT-2 style) |
| Layers / Heads / Hidden | 13 / 12 / 768 |
| Context length | 1,024 tokens |
| Tokenizer | GPT-2 BPE (50,257 vocab) |
| Training data | GST Advance Rulings + India Code acts/circulars |
| Best val loss | 2.7641 (step 11,250) |
| Training hardware | Kaggle T4 GPU (FP16) |
