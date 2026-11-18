# PATHOAI — Histopathological Cell Classification

### Colorectal Cancer Cell Detection and Cell-Type Phenotyping

**Academic Research Project | GLS University | 2026–2027**

PATHOAI is a deep learning–based histopathological image analysis project developed to investigate automated classification of cell nuclei in colorectal tissue images.

The project explores convolutional neural networks (CNNs) and traditional machine learning methods for binary cancer-associated cell classification and multi-class cell-type identification.

The system combines a Python-based machine learning pipeline with a Django REST backend and a React frontend for image-based predictions and model evaluation.

## 1. Project Overview

Histopathological examination plays an important role in cancer research and diagnosis. However, manually analyzing large numbers of cell nuclei can be time-consuming and requires specialized expertise.

PATHOAI investigates whether computational methods can identify morphological patterns in histopathological images and distinguish between different cell categories.

The project focuses on two classification tasks:

- **Binary classification:** Predict whether a cell nucleus belongs to the neoplastic or non-neoplastic category.
- **Multi-class classification:** Predict one of five nucleus categories: Neoplastic, Inflammatory, Connective, Dead, or Non-Neoplastic Epithelial.

The project is intended for academic research and experimentation, not clinical diagnosis.

## 2. Dataset

**Dataset:** PanNuke — Pan-Cancer Nuclei Dataset

PanNuke provides annotated histopathological images with nucleus instance segmentation masks and cell-type labels.

For this project, the dataset is processed with a focus on colorectal tissue images.

### Data preparation

1. Acquire and extract the PanNuke dataset.
2. Select colorectal tissue images.
3. Extract 64 × 64 pixel nucleus-centered patches.
4. Generate binary and multi-class labels.
5. Prepare training, validation, and testing datasets.
6. Apply an image-level split to reduce leakage between dataset partitions.

**Dataset split:**

| Partition | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The split is performed at the source-image level. Since patient identifiers are not available in the dataset schema used by this project, this is not a patient-level split.

## 3. Models and Methodology

### Binary CNN Classification

A convolutional neural network is trained to distinguish neoplastic from non-neoplastic nuclei using image patches.

### Multi-Class CNN Classification

A separate CNN is trained to classify nuclei into five histopathological cell categories.

### Random Forest Classification

A traditional machine learning model uses extracted morphological and image-derived features:

- Area
- Eccentricity
- Solidity
- Perimeter
- Mean intensity
- Texture contrast

This provides a feature-based approach for comparison with deep learning models.

## 4. Experimental Results

The following results summarize the project's recorded test-set evaluations.

| Model | Task | Accuracy | Macro F1-score |
|---|---|---:|---:|
| Binary CNN | Neoplastic vs. non-neoplastic | 91.20% | 0.890 |
| Multi-Class CNN | Five cell categories | ~76% | 0.635 |
| Random Forest | Five cell categories | ~74% | 0.707 |

Performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

These results are specific to the project's experimental dataset and evaluation procedure. They do not establish clinical reliability or generalization to independent medical institutions.

## 5. Technology Stack

| Component | Technologies |
|---|---|
| Programming | Python, JavaScript |
| Deep Learning | PyTorch, CNN |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | NumPy, Pandas, OpenCV |
| Backend | Django |
| Frontend | React, Vite |
| Visualization | Matplotlib |
| Version Control | Git, GitHub |

## 6. Repository Structure

```text
crc-cell-classifier/
├── backend/
│   ├── classifier/
│   ├── config/
│   └── manage.py
├── frontend/
│   ├── public/
│   ├── src/
│   └── package.json
├── ml-pipeline/
│   ├── src/
│   ├── reports/
│   │   └── figures/
│   └── requirements.txt
├── docs/
│   └── module8_evaluation_and_analysis.md
├── .gitignore
└── README.md
```

The repository contains the application source code, model training scripts, and selected evaluation materials.

**Note:** Raw datasets, processed data, trained model weights, and local environments are excluded from version control.

## 7. Getting Started

### Prerequisites

- Python
- Node.js and npm
- Git
- PanNuke dataset for reproducing the training pipeline

### Clone the repository

```bash
git clone https://github.com/pratham195/PATHOAI-Histopathological-Cell-Classification.git
cd PATHOAI-Histopathological-Cell-Classification
```

### Set up the machine learning environment

```bash
cd ml-pipeline
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Acquire and prepare data

```bash
python src/01_acquire_data.py
python src/02_extract_patches.py
python src/03_prepare_splits.py
```

Dataset acquisition may require additional setup depending on the data source and local configuration.

### Train the models

Relevant training scripts include:

```text
04_train_cancer_model.py
06_train_cnn.py
07_train_cnn_celltype.py
09_train_celltype_classical.py
10_hyperparameter_tuning.py
```

Review each script's configuration and input requirements before running it.

### Start the frontend

From the repository root:

```bash
cd frontend
npm install
npm run dev
```

### Start the backend

From the repository root:

```bash
cd backend
python manage.py runserver
```

Install the backend's required Python dependencies and configure model paths before running inference. Trained model files are not distributed with this repository.

## 8. Research Scope and Limitations

- The project focuses on colorectal histopathological nucleus classification.
- Experiments use PanNuke-derived data rather than a clinically validated deployment dataset.
- Image-level separation does not guarantee patient-level independence.
- Model predictions may be affected by staining differences, image quality, and dataset distribution.
- Performance on independent clinical datasets has not been established.
- The application is a research prototype and should not be used for medical decision-making.

## 9. Research Documentation

The project was developed as an academic capstone at GLS University.

A research manuscript based on the experimental methodology and evaluation is currently in preparation under faculty guidance.

**Project contributors:**
- Pratham Barot
- Kathan Trivedi

**Faculty supervisor:** Dr. Ankita Kanojiya

## 10. References

1. Gamper, J., Koohbanani, N. A., Benet, K., Khuram, A., and Rajpoot, N. (2019). *PanNuke: An Open Pan-Cancer Histology Dataset for Nuclei Instance Segmentation and Classification.*
2. Gamper, J., et al. (2020). *PanNuke Dataset Extension, Insights and Baselines.* arXiv:2003.10778.

---

**Disclaimer:** PATHOAI is an academic research prototype. It is not a certified medical device and is not intended for clinical diagnosis or treatment decisions.
