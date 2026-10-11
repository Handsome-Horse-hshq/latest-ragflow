# AVeriTeC — source and license

`train.json` and `dev.json` in this directory are the unmodified official files of
the AVeriTeC dataset.

- **Dataset**: AVeriTeC: A Dataset for Real-world Claim Verification with Evidence
  from the Web
- **Authors**: Michael Schlichtkrull, Zhijiang Guo, Andreas Vlachos
- **Paper**: NeurIPS 2023 Datasets and Benchmarks, <https://arxiv.org/abs/2305.13117>
- **Source**: <https://huggingface.co/chenxwh/AVeriTeC>, repository revision
  `2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031`
- **License**: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0), <https://creativecommons.org/licenses/by-nc/4.0/>

| File | SHA-256 |
| --- | --- |
| `train.json` | `ae5eda7c42ddf169…` (full digest in `data/processed/averitec_v1/manifest.json`) |
| `dev.json` | `499793726b4a5406…` (full digest in `data/processed/averitec_v1/manifest.json`) |

The same license and attribution apply to everything derived from these files in
this repository: `data/processed/averitec_v1/` (a balanced four-class subset, each
question-answer pair turned into one evidence document) and
`outputs/ragchecker_averitec/` (NLI inputs and outputs that contain the claim and
evidence text). These files may be used for non-commercial purposes only, with
credit to the AVeriTeC authors.
