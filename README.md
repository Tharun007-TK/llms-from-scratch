# LLMs-from-scratch (GST/Tax Regulatory LLM Project)

This project contains a data extraction and preprocessing pipeline designed to build a corpus of GST/tax regulatory documents for LLM training.

## Features

- **Document Fetching**: Scripts for fetching documents from Indian Code, GST Council, and Indian Kanoon.
- **Corpus Extraction**: A multi-stage pipeline that extracts text from raw PDFs/HTML, cleans, deduplicates, and splits the dataset into training and validation sets.
- **Tokenization**: Uses GPT-2 BPE tokenizer to accurately count tokens.
- **Dataset Building**: Scripts to format the processed corpus into a Hugging Face dataset format.

## Table of Contents

| Chapter Title                                              | Main Code                                                                                                    |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Ch 1: Fetching Raw Documents                  | - [fetch_gstcouncil_aar.py](ch01/fetch_gstcouncil_aar.py)<br/>- [fetch_indiacode_docs.py](ch01/fetch_indiacode_docs.py)<br/>- [test_indiankanoon.py](ch01/test_indiankanoon.py) |
| Ch 2: Corpus Extraction & Cleaning                               | - [extract_corpus.py](ch02/extract_corpus.py)<br/>- [time_single_file.py](ch02/time_single_file.py)               |
| Ch 3: Tokenization & Dataset Building                          | - [tokenize_and_pack.py](ch03/tokenize_and_pack.py)<br/>- [build_hf_dataset.py](ch03/build_hf_dataset.py) |

## Directory Structure

Expected layout for data:
- `raw/`: Raw downloaded documents (PDFs/HTML) organized by source type (e.g., `aar_rulings/`, `bare_acts/`, `circulars/`, `notifications/`).
- `processed/`: Output directory containing JSONL files with processed text and corpus statistics.
- `ch03/hf_dataset/`: Hugging Face dataset format output.

## Workflow

1. **Fetch**: Download raw files using the scripts in `ch01/`.
2. **Extract & Clean**: Run the extraction pipeline in `ch02/` to process the raw documents.
3. **Tokenize & Pack**: Prepare the dataset for training using the scripts in `ch03/`.

## Requirements

Dependencies can be installed from `requirements.txt`.
