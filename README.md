# Radiology Mini-VLM + Fairness Audit

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20ready-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Educational demo of **CLIP-style vision-language retrieval for radiology images**,
plus a **subgroup fairness audit** of a classifier's performance across patient
subgroups. Built as a portfolio project exploring vision-language models (VLMs)
in medical imaging.

> ⚠️ **Educational demo only — not a clinical tool.** Nothing here is validated
> for medical use, and no real patient data is included or required.

## What this is / is not

**This is:**
- A minimal, readable example of text↔image retrieval with a pretrained CLIP model
  (via `open_clip`) applied to radiology-style images.
- A self-contained fairness-audit utility: given model scores and subgroup labels,
  it reports per-subgroup AUROC with bootstrap confidence intervals and the
  performance gap — the kind of check expected before any clinical evaluation.

**This is not:**
- A diagnostic model, a foundation model, or a clinically validated system.
- Trained or evaluated on real patient data in this repo (all demo data is synthetic).

## Quickstart

```bash
pip install -r requirements.txt

# 1) Fairness audit on synthetic demo data (no downloads, CPU only)
python examples/demo.py
# → prints per-subgroup AUROC table, saves examples/outputs/subgroup_auroc.png

# 2) CLIP text-to-image retrieval (downloads ViT-B-32 weights on first run)
python - <<'EOF'
from PIL import Image
from src.retrieval import text_to_image_search

images = [Image.open("examples/sample_01.png"), Image.open("examples/sample_02.png")]
idx, scores = text_to_image_search("bilateral lung opacity", images, top_k=2)
print(list(zip(idx, scores)))
EOF
```

## Fairness audit

`src/fairness_audit.py` implements the audit:

1. **Input:** a CSV with columns `y_true` (0/1), `y_score` (model score),
   `subgroup` (e.g. sex, age band, scanner site).
2. **Metric:** per-subgroup AUROC with a 95% confidence interval from
   nonparametric bootstrap resampling (default 1000 replicates).
3. **Output:** a table of AUROC per subgroup plus `gap_vs_best` — the absolute
   AUROC gap to the best-performing subgroup — and a bar plot with CI whiskers.

Why this matters: aggregate metrics can hide that a model works well on average
but underperforms for a specific subgroup. Reporting subgroup metrics and gaps
is standard practice in responsible medical-AI evaluation (e.g. the reporting
guidelines discussed in the literature on algorithmic fairness in clinical ML).

```python
from src.fairness_audit import make_demo_csv, audit_csv, print_gap_table

make_demo_csv("audit.csv", n=2000, seed=0)   # synthetic demo data
results = audit_csv("audit.csv", plot_path="gaps.png")
print_gap_table(results)
```

## Datasets

This repo ships **no medical images**. To extend the retrieval demo to real data,
use these public datasets (access subject to their data-use agreements; the code
does not assume they are downloaded):

- **NIH ChestX-ray14** — 112,120 frontal chest X-rays with 14 pathology labels.
  https://nihcc.app.box.com/v/ChestXray-NIHCC
- **CheXpert** — 224,316 chest radiographs from Stanford, with uncertainty labels.
  https://stanfordmlgroup.github.io/competitions/chexpert/

Both pair images with labels/text that a VLM-style approach can exploit; the
`src/retrieval.py` functions accept any list of PIL images.

## Project structure

```
radiology-mini-vlm/
├── src/
│   ├── retrieval.py       # CLIP embeddings + text-to-image search
│   └── fairness_audit.py  # subgroup AUROC audit with bootstrap CIs
├── examples/
│   └── demo.py            # end-to-end audit demo (synthetic data, no downloads)
├── tests/
│   └── test_smoke.py      # CPU-only tests, no weights downloaded
├── requirements.txt
└── README.md
```

## References

- Radford, A. et al. *Learning Transferable Visual Models From Natural Language
  Supervision (CLIP).* ICML 2021.
- Wang, Z. et al. *MedCLIP: Contrastive Learning from Unpaired Medical Images
  and Text.* EMNLP 2022.
- Cherti, M. et al. *Reproducible scaling laws for contrastive language-image
  learning (OpenCLIP).* CVPR 2023.

## License

MIT — see [LICENSE](LICENSE).
