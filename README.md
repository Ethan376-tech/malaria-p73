# Stain-Invariant Semi-Supervised Learning for Robust Malaria Diagnosis at Low Parasitaemia

Code for an MSc dissertation (UCL Computer Science, project **P73**), extending the
weakly-supervised malaria pipeline **MILCA** — which learns parasite detection on
thick-blood-film microscopy from *sample-level* diagnostic labels rather than manual
boxes — toward **stain-invariance** and **label-efficiency**, and using it to
**diagnose where the weak-supervision paradigm stops** at low parasitaemia and across
sites.

> ⚠️ **This repository contains code only.** The datasets are patient microscopy images
> and are **not** included or redistributable (see [Data availability](#data-availability)).
> Trained model weights are also not included.

## Overview

The organising finding is an **asymmetry in what transfers across a stain shift**: a
self-supervised tile encoder can be made stain-invariant (it separates parasites from
background on an unseen site almost as well as in-domain), yet the *sample-level
decision* built on that representation does **not** transfer, and the gap is sharpest at
low parasitaemia — a *signal-aggregation* problem, not a representation or
domain-adaptation one. The pipeline is carried end-to-end (detection, counting with a
per-µL limit of detection, label efficiency), and the residual is bounded with a
distribution-free selective-prediction (conformal) guarantee.

### Selected results

| Result | Value |
|---|---|
| Cross-site parasite-vs-background **tile AUC** (ThickDINO encoder) | **0.945** (≈75% smaller cross-site drop than an ImageNet baseline) |
| Cross-site **low-parasitaemia** sample classification | ≈ chance (AUC **0.47**) — a characterised boundary, not a met target |
| **MicroYOLO** detector, AP@0.5 | **0.31** (≈2× the RetinaNet baseline 0.15); refined AP@0.3 0.62 |
| **Faintness-protected filter** | retains **90%** of faint true positives; pseudo-box precision 0.070 → 0.095 |
| **Space-signal filter** (exploits intra-field parasite aggregation) | cuts the per-µL limit of detection by **≈ two-thirds** |
| **Label efficiency** | weakly-pretrained detector beats from-scratch supervision at the smallest annotation budget |

## Named methods

- **ThickDINO** — a self-supervised, stain-invariant thick-film tile encoder (ViT-S, DINOv2 recipe).
- **ViCAM** — a single-pass ViT adaptation of MILCA's iterative-CAM pseudo-box generator.
- **Faintness-protected filter** — an uncertainty-aware retain rule for pseudo-boxes, in
  **single-signal**, **multi-signal** (adds TTA-variance and feature-typicality) and
  **space-signal** (contextual, co-detection density) forms.
- **FilterDistil** — the filter embedded as the retain rule of a teacher–student
  pseudo-box refinement loop.
- **MicroYOLO** — a resolution-aware YOLOv8n specialisation for tiny parasites.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── MANIFEST.md          # per-script index, grouped G01–G19 (what each script produces)
├── code/                # all experiment / analysis scripts (see MANIFEST.md)
├── results/             # (optional) metric JSONs behind the reported tables
└── figures/             # (optional) final result figures
```

`MANIFEST.md` is the authoritative map from each script to the report section / table /
figure it produces.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12
# install the torch build matching your CUDA from https://pytorch.org, then:
pip install -r requirements.txt
```

## Data availability

The datasets used are **patient thick-blood-film microscopy** and are governed by their
original data-use agreements; **they are not included here and cannot be redistributed.**

- **Ibadan / FASTMAL** (Nigeria, lab camera) — domain A; the primary target site.
- **Chittagong-1 / -2** (Bangladesh, smartphone) — domain B.
- **Tanzania** (4K micro-camera) — domain C (image-level labels only).
- **Ghana / Lacuna** (smartphone) — domain D (external validation).

Access must be sought from the original data providers / the MILCA authors under the
relevant agreements. This repository redistributes **no** imaging data, annotations, or
patient information.

## Reproducibility notes

- Paths are centralised in `code/common/paths.py`. Scripts expect data, checkpoints and
  outputs under a data root (`RAW`, `CHECKPOINTS`, `OUTPUTS`); none of these are shipped.
- Scripts run with `XFORMERS_DISABLED=1` and are seeded; multi-seed experiments report
  mean ± std over 3 random field splits (following MILCA's protocol).
- Because the data are restricted, the scripts are provided for **transparency and
  method reproducibility**, not as a turnkey run on public data.

## Third-party code

The encoder builds on **DINOv2** (Meta AI) and the blood-cell adaptation **RedDino**;
detection uses **Ultralytics YOLOv8**; the pipeline extends **MILCA**. These are used
from their own repositories under their own licences and are **not vendored** here — see
each project's licence. Cite them alongside this work.

## Dissertation

The methods, experiments and analysis are described in the accompanying MSc dissertation
(UCL, 2026). *(Link / DOI to be added after submission and grading.)*

## Use of generative AI

A generative-AI coding assistant was used under the author's direction (code, figures,
analysis and drafting/editing of text), with all outputs manually checked. This is
declared in full in the **Acknowledgements** section of the dissertation.

## Citation

```bibtex
@mastersthesis{P73_2026,
  title  = {Stain-Invariant Semi-Supervised Learning for Robust Malaria Diagnosis at Low Parasitaemia},
  author = {Ma, Jianhong},
  school = {University College London},
  year   = {2026},
  note   = {MSc dissertation, Department of Computer Science}
}
```

## License

Code released under the MIT License (see [LICENSE](LICENSE)). The licence covers the code
only — not the datasets, third-party code, or trained weights. Confirm compatibility with
UCL's IP policy before publishing.
