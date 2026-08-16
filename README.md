# LLMs-from-scratch (GST/Tax Regulatory LLM Project)

This project contains a data extraction and preprocessing pipeline designed to build a corpus of GST/tax regulatory documents for LLM training.

## Features

- **Document Fetching**: Scripts for fetching documents from Indian Code, GST Council, and Indian Kanoon.
- **Corpus Extraction**: A multi-stage pipeline (`extract_corpus.py`) that extracts text from raw PDFs/HTML, cleans, deduplicates, and splits the dataset into training and validation sets.
- **Tokenization**: Uses GPT-2 BPE tokenizer to accurately count tokens.
- **Dataset Building**: Scripts to format the processed corpus into a Hugging Face dataset format (`build_hf_dataset.py`, `tokenize_and_pack.py`).

## Directory Structure

Expected layout:
- `raw/`: Raw downloaded documents (PDFs/HTML) organized by source type (e.g., `aar_rulings/`, `bare_acts/`, `circulars/`, `notifications/`).
- `processed/`: Output directory containing JSONL files with processed text and corpus statistics.
- `hf_dataset/`: Hugging Face dataset format output.

## Workflow

1. **Fetch**: Download raw files using the provided fetch scripts.
2. **Extract & Clean**: Run `extract_corpus.py` to process the raw documents.
3. **Tokenize & Pack**: Prepare the dataset for training using `tokenize_and_pack.py` or `build_hf_dataset.py`.

## Requirements

Dependencies can be installed from `requirements.txt`.
