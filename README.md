# Histopathological Cell Classification System
### Colorectal Cancer Detection & Cell-Type Phenotyping — GLS University Capstone 2026-27

**Students:** Barot Pratham Vipulbhai (202201619010002), Trivedi Kathan Vrajeshbhai (202201619010346)

## What this project does

Upload a cell nucleus image → the system predicts:
1. **isCancerous** — is this a cancerous (neoplastic) cell? (binary)
2. **Cell type** — epithelial / inflammatory / connective(fibroblast) / dead-other (multi-class)

...and **explains its reasoning**:
- A **Grad-CAM heatmap** overlaid on the image showing which region drove the prediction.
- A **plain-language morphological explanation** (nucleus size, shape irregularity, staining/chromatin density, texture) generated from measurable features, not just a black-box heatmap.

## Dataset

**Primary dataset: PanNuke** (Gamper et al., 2019/2020, Warwick TIA Centre) — chosen over the originally
proposed CRCHistoPhenotypes (Sirinukunwattana et al., 2016) because:

| | CRCHistoPhenotypes | PanNuke (used) |
|---|---|---|
| Access | Gated (email request required, no guaranteed/timely response) | Open download, no login |
| Labeled nuclei | ~22,444 | ~190,000+ |
| Annotation type | Centroid point only | Full instance segmentation mask per nucleus |
| Cancer label | Inferred indirectly | Explicit "Neoplastic" class |

This substitution is documented for transparency. CRCHistoPhenotypes access was formally requested from the
Warwick TIA Centre (email sent 13 Aug 2026); the TIA Centre confirmed on 14 Aug 2026 that the dataset is now
gated behind a partner hospital's (UHCW) forthcoming Data Access Committee, with access not expected before
early autumn 2026 — outside this project's timeline. This exchange is kept on record as evidence of due
diligence and is cited in the final report's dataset section. PanNuke is therefore used as the sole dataset
for this project.

**Citations required in the final report:**
- Gamper, J., Koohbanani, N.A., Benet, K., Khuram, A. and Rajpoot, N., 2019. PanNuke: an open pan-cancer
  histology dataset for nuclei instance segmentation and classification. ECDP 2019.
- Gamper, J. et al., 2020. PanNuke Dataset Extension, Insights and Baselines. arXiv:2003.10778.

(Sirinukunwattana et al., 2016 is referenced above only to explain the original dataset choice and why it
was substituted — since CRCHistoPhenotypes is not used in this project, it is not cited as a dataset source
in the final report.)

## Repo structure

```
crc-cell-classifier/
├── ml-pipeline/
│   ├── data/
│   │   ├── raw/            # downloaded PanNuke folds (not committed)
│   │   └── processed/      # extracted patches + labels + patient-level splits
│   ├── notebooks/          # EDA, experiment notebooks
│   ├── src/                # module scripts (01_acquire_data.py, 02_prepare_data.py, ...)
│   ├── models/             # saved trained models (.pt / .pkl)
│   └── reports/figures/    # EDA plots, confusion matrices, Grad-CAM samples
├── backend/                 # Django app (Module 9)
├── frontend/                 # React app
└── docs/                     # capstone report, module writeups
```

## Modules mapped to this repo

| Capstone Module | Location |
|---|---|
| 1. Data Acquisition & Patch Extraction | `ml-pipeline/src/01_acquire_data.py`, `02_extract_patches.py` |
| 2. Data Preparation (patient-level split) | `ml-pipeline/src/03_prepare_splits.py` |
| 3. EDA | `ml-pipeline/notebooks/03_eda.ipynb` |
| 4. isCancerous Classification | `ml-pipeline/src/04_train_cancer_models.py` |
| 5. Cell-Type Classification | `ml-pipeline/src/05_train_celltype_models.py` |
| 6. Hyperparameter Tuning | `ml-pipeline/src/06_tune.py` |
| 7. Advanced Technique | `ml-pipeline/src/07_advanced.py` |
| 8. Evaluation & Analysis | `ml-pipeline/notebooks/08_evaluation.ipynb` |
| 9. Web Interface | `backend/`, `frontend/` |
| 10. DevOps & Deployment | `docker-compose.yml`, `.github/workflows/` |

## Setup

```bash
cd ml-pipeline
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python src/01_acquire_data.py
```
