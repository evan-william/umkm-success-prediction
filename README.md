<div align="center">

<img src="assets/magic-cloud-logo.svg" alt="Magic Cloud Logo" width="420"/>

# UMKM Success Prediction

**Machine learning analysis for predicting MSME/UMKM business success in IN-FEST 2026 Data Science Competition**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Team Magic Cloud**  
Universitas Widya Mandala Surabaya

</div>

---

## Overview

This repository contains the preliminary-round submission package for the **Data Science Competition - Informatics Festival 2026**. The project predicts MSME/UMKM business success from owner and business profile variables, then converts model findings into practical recommendations for business assistance programs.

The solution follows the competition rule: the analysis only uses the official dataset provided by the committee, `umkm_success.csv`.

---

## Team

| Role | Name |
|---|---|
| Member | Evan William |
| Member | Daniel |
| Member | Bambang Herlambang |

Institution: **Universitas Widya Mandala Surabaya**

---

## Repository Info

**GitHub repo title:** `umkm-success-prediction`  
**GitHub repo description:** `Machine learning analysis for predicting MSME/UMKM business success using official IN-FEST 2026 competition data.`  
**Suggested tags:** `data-science machine-learning kaggle-notebook classification umkm msme scikit-learn python business-analytics infest-2026`

---

## Key Highlights

- Binary classification for UMKM success prediction.
- Complete EDA, data quality audit, and target distribution analysis.
- Handles imbalanced target data using appropriate validation metrics.
- Compares baseline and machine learning models.
- Uses stratified split, repeated stratified cross-validation, and model tuning.
- Includes model interpretation through feature importance.
- Provides practical recommendations for UMKM assistance programs.

---

## Dataset

The dataset contains **250 rows** and **13 columns**.

Target column:

| Column | Meaning |
|---|---|
| `Success` | `1` = successful UMKM, `0` = unsuccessful UMKM |

Target distribution:

| Class | Count | Percentage |
|---|---:|---:|
| `0` - Not successful | 188 | 75.2% |
| `1` - Successful | 62 | 24.8% |

Because the target is imbalanced, accuracy alone is not enough. The notebook also reports balanced accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve, and feature importance.

---

## Project Structure

```text
dsc_umkm_success_submission/
├── assets/
│   └── magic-cloud-logo.svg
├── data/
│   ├── .gitkeep
│   └── umkm_success.csv
├── notebooks/
│   ├── dsc_umkm_success_kaggle.ipynb
│   └── dsc_umkm_success_kaggle.py
├── paper/
│   ├── MAKALAH_DSC_Magic_Cloud.pdf
│   └── makalah_source.md
├── src/
│   └── validate_dataset.py
├── submission/
│   ├── GUIDELINE_AUDIT.md
│   └── UPLOAD_CHECKLIST.md
├── tests/
│   └── test_dataset_contract.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Quick Start on Kaggle

Kaggle is the recommended environment for the competition because it already includes common data science libraries such as pandas, NumPy, matplotlib, seaborn, and scikit-learn.

1. Open Kaggle and create a new notebook.
2. Import `notebooks/dsc_umkm_success_kaggle.ipynb`.
3. Upload `data/umkm_success.csv` as a Kaggle dataset or notebook input.
4. Click **Run All**.
5. Check that every cell runs without error.
6. Click **Save Version / Commit**.
7. Set notebook visibility to **Public**.
8. Submit the public Kaggle Notebook URL and the paper PDF to the committee form.

---

## Local Jupyter Setup

If Python is available in PATH:

```powershell
cd D:\CodeSpace\Competition\dsc_umkm_success_submission
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Windows fallback when Python is installed but not available in PATH:

```powershell
cd D:\CodeSpace\Competition\dsc_umkm_success_submission
& "C:\Users\Evan\AppData\Local\Programs\Python\Python311\python.exe" -m jupyter lab --notebook-dir "D:\CodeSpace\Competition\dsc_umkm_success_submission"
```

Then open:

```text
notebooks/dsc_umkm_success_kaggle.ipynb
```

If a dataset error appears, confirm this file exists:

```text
D:\CodeSpace\Competition\dsc_umkm_success_submission\data\umkm_success.csv
```

---

## Validation Test

Run this after installing dependencies:

```powershell
pytest
```

The test checks that the official dataset exists, has the expected schema, has no missing values, and uses a binary `Success` target.

---

## Model Evaluation

The competition guideline does not define a minimum required accuracy. The scoring emphasizes model validity and analysis depth, so the notebook reports several metrics:

| Metric | Purpose |
|---|---|
| Accuracy | General correct prediction ratio |
| Balanced Accuracy | Fairer score for imbalanced classes |
| Precision | How reliable positive predictions are |
| Recall | How many successful UMKM are detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ranking quality across thresholds |
| Confusion Matrix | Counts of true/false predictions |

The majority baseline is around **75.2% accuracy** because most records are class `0`. A useful model should beat this baseline and show stronger balanced accuracy, F1-score, and ROC-AUC.

---

## Main Findings

The strongest actionable factors in the dataset are:

- Business plan availability.
- Sufficient initial capital.
- Good financial record keeping.
- Internet usage.
- Industry experience.
- Professional advice frequency.

These findings lead to practical recommendations: business plan mentoring, bookkeeping training, digital onboarding, targeted micro-capital support, professional consultation clinics, and industry-based mentoring.

---

## Competition Submission

Submit only these assets to the committee:

| Asset | Status |
|---|---|
| Public Kaggle Notebook URL | Required |
| Paper PDF | Required |

Submission form:

```text
https://s.id/DSC_INFEST26_Penyisihan
```

Deadline:

```text
Saturday, 13 June 2026, 23:59 WIB
```

After submitting the Kaggle URL, do not modify the notebook again.

---

## Security and Reproducibility Notes

- No external dataset is used.
- No credentials, API keys, or private tokens are required.
- The notebook uses a fixed random seed for reproducibility.
- The data file is small and loaded locally from the Kaggle input directory.
- The project does not store user data, send network requests, or depend on hidden services.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Magic Cloud**  
Universitas Widya Mandala Surabaya

Members:

- Evan William
- Daniel
- Bambang Herlambang
