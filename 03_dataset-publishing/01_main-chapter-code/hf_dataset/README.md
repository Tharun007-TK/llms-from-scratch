---
language:
- en
license: other
license_name: indian-govt-public-text
license_link: https://www.indiacode.nic.in/
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

- **2150 training documents / 238 validation documents**
- **~10,621,967 tokens** (GPT-2 BPE)
- Two source types:
  - `circulars` — CGST circulars from India Code (indiacode.nic.in)
  - `aar_rulings` — Authority for Advance Ruling (AAR) orders from
    gstcouncil.gov.in, covering all Indian states/UTs

## Known limitations — read before use

- **Small for pretraining a language model from scratch.** ~10,621,967
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

This dataset consists of text sourced from official Government of India
publications: CGST circulars (India Code, indiacode.nic.in) and Authority
for Advance Ruling orders (GST Council, gstcouncil.gov.in).

**Legal basis (not a formal legal opinion):** Section 52(1)(q) of the
Indian Copyright Act, 1957 provides an exception for reproduction of
certain government and judicial/quasi-judicial materials. The Supreme
Court's ruling in *Eastern Book Company v. D.B. Modak* (2007) interpreted
this section as placing such material effectively in the public domain.
AAR/AAAR rulings are quasi-judicial orders; circulars are official
government notifications — both plausibly fall within this exception.

**What this is NOT:** a confirmed license grant from indiacode.nic.in or
gstcouncil.gov.in themselves (unlike data.gov.in, which explicitly
publishes under the Government Open Data License – India / GODL, I could
not confirm these two specific portals declare an equivalent license), and
not a lawyer's opinion. If you plan to redistribute or build a commercial
product on this dataset, verify the copyright position independently
before relying on this summary.

Source portals: indiacode.nic.in, gstcouncil.gov.in — attribution to the
Government of India / GST Council as the original source is included here
in that spirit, independent of the legal question above.

## Source

Built as part of a public series on building an LLM from scratch:
[link to your blog/LinkedIn series here]