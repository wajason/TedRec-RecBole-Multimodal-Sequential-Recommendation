# TedRec-Multimodal-Sequential-Recommendation
[![GitHub Stars](https://img.shields.io/github/stars/wajason/TedRec-Multimodal-Sequential-Recommendation?style=for-the-badge&logo=github&color=4C8EDA)](https://github.com/wajason/TedRec-Multimodal-Sequential-Recommendation/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/wajason/TedRec-Multimodal-Sequential-Recommendation?style=for-the-badge&logo=github&color=4C8EDA)](https://github.com/wajason/TedRec-Multimodal-Sequential-Recommendation/network/members)
[![Issues](https://img.shields.io/github/issues/wajason/TedRec-Multimodal-Sequential-Recommendation?style=for-the-badge&color=4C8EDA)](https://github.com/wajason/TedRec-Multimodal-Sequential-Recommendation/issues)
[![License](https://img.shields.io/badge/License-MIT-4C8EDA?style=for-the-badge)](./LICENSE)
[![Run in Colab](https://img.shields.io/badge/Open%20in-Colab-4C8EDA?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/wajason/TedRec-Multimodal-Sequential-Recommendation/blob/main/TedRec.ipynb)


Short description:
A production-ready, reproducible implementation of the TedRec multimodal sequential recommender built on RecBole. Includes MoE-based text compression, Colab-ready notebook, and MovieLens-1M benchmark results.

================================================================================
OVERVIEW
--------------------------------------------------------------------------------
TedRec implements a multimodal sequential recommendation model on top of RecBole.
Core ideas:
 - fuse standard item-ID embeddings with text semantic vectors
 - compress the text vectors via an MoE adapter to keep representation compact
 - train & evaluate on MovieLens-1M as a reproducible benchmark

This repo is designed to be:
 - research-friendly (clear configs, reproducible)
 - engineer-friendly (Colab-ready notebook, easy entrypoint)
 - shareable (citation snippet included)

================================================================================
QUICKSTART — Run in Google Colab (recommended)
--------------------------------------------------------------------------------
1) Open the notebook in Colab:
   Click the "Open in Colab" badge above, or open:
   https://colab.research.google.com/github/wajason/TedRec-Multimodal-Sequential-Recommendation/blob/main/TedRec.ipynb

2) Runtime settings (Colab menu):
   Runtime -> Change runtime type -> Hardware accelerator -> GPU (recommended: T4)

3) Install minimal dependencies (run first notebook cell or run these commands):
   !pip install recbole torch

4) Execute notebook cells sequentially:
   - dataset download & preprocessing (RecBole auto)
   - train TedRec (default 3 epochs for quick benchmark)
   - eval & save checkpoint

Notes:
 - The notebook contains the exact commands used to reproduce the logged runs and final metrics.

================================================================================
LOCAL / SCRIPT-BASED QUICKSTART
--------------------------------------------------------------------------------
(Useful for local machine or server)

1) Create env (conda recommended)
   conda create -n tedrec python=3.10 -y
   conda activate tedrec

2) Install packages
   pip install recbole torch

3) Train (example)
   python run_tedrec.py --config config/config.yaml

4) Checkpoints saved to:
   saved/TedRec-<DATE>.pth

================================================================================
PROJECT STRUCTURE (recommended)
--------------------------------------------------------------------------------
TedRec-Multimodal-Sequential-Recommendation/
│
├── TedRec.ipynb                # Colab-ready notebook (highest priority for demos)
├── run_tedrec.py               # script: entrypoint for training & evaluation
├── models/
│   └── tedrec.py               # TedRec model implementation (RecBole style)
├── config/
│   └── config.yaml             # default hyperparameters + dataset config
├── data/                       # (auto) MovieLens-1M (downloaded by RecBole)
├── saved/                      # checkpoints (gitignore)
├── assets/                     # visuals: banner, demo images (optional)
├── CITATION.bib                # citation metadata
└── README.md                   # this file

================================================================================
CONFIGURATION HIGHLIGHTS
--------------------------------------------------------------------------------
- Default dataset: MovieLens-1M (standard RecBole split)
- Default epochs: 3 (fast reproducible baseline)
- Feature fusion: item_id_embedding + moe_compressed_text_vector
- Checkpoints & logs: saved/* and logs/*

================================================================================
EXPERIMENT LOG (exact run from Colab / local notebook)
--------------------------------------------------------------------------------
Epoch 0:
  time (train): 496.38s, train loss: 3029.5624
  validating: valid_score: 0.093800
  valid result:
    recall@10 : 0.1902
    recall@20 : 0.3060
    ndcg@10   : 0.0938
    ndcg@20   : 0.1229
  checkpoint saved: saved/TedRec-Nov-07-2025_11-07-08.pth

Epoch 1:
  time (train): 497.14s, train loss: 2730.1905
  validating: valid_score: 0.117800
  valid result:
    recall@10 : 0.2258
    recall@20 : 0.3435
    ndcg@10   : 0.1178
    ndcg@20   : 0.1475
  checkpoint saved: saved/TedRec-Nov-07-2025_11-07-08.pth

Epoch 2:
  time (train): 497.08s, train loss: 2669.6013
  validating: valid_score: 0.124100
  valid result:
    recall@10 : 0.2359
    recall@20 : 0.3540
    ndcg@10   : 0.1241
    ndcg@20   : 0.1540
  checkpoint saved: saved/TedRec-Nov-07-2025_11-07-08.pth

Best valid (selected):
  recall@10: 0.2359
  recall@20: 0.3540
  ndcg@10:   0.1241
  ndcg@20:   0.1540

Test result (after loading best checkpoint):
  recall@10 : 0.2306
  recall@20 : 0.3389
  ndcg@10   : 0.1254
  ndcg@20   : 0.1527

(Short summary: model converged in 3 epochs; NDCG@10=0.1254 on test set.)

================================================================================
TOP-K DEMO (example presentation for README)
--------------------------------------------------------------------------------
You can generate a CSV or table with Top-K results and embed into README or assets.
Example format (paste into README or render as HTML table):

UserID, Top1, Top2, Top3, Top4, Top5
127, 608, 50, 174, 356, 12
204, 21, 305, 411, 87, 90

(Notebook includes a cell that prints Top-K for a few sample users; export as PNG for README visuals.)

================================================================================
CITATION (CITATION.bib)
--------------------------------------------------------------------------------
@misc{TedRec2025,
  title = {TedRec: Multimodal Sequential Recommendation with MoE compression},
  author = {Jason and Contributors},
  year = {2025},
  howpublished = {GitHub repository},
  note = {https://github.com/wajason/TedRec-Multimodal-Sequential-Recommendation}
}

================================================================================
GOOD PRACTICES & TIPS
--------------------------------------------------------------------------------
- If you need larger-scale or longer training, increase epochs in config/config.yaml.
- For reproducibility: set random seed in config and save the full config alongside checkpoints.
- To make Colab runs even faster: use GPU runtime and pin compatible torch build for Colab.

