from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import pandas as pd
from reportlab.graphics.shapes import Circle, Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = Path(r"D:\Media & Songs\Downloads\umkm_success.csv")
TEAM_NAME = "Magic Cloud"
INSTITUTION = "Universitas Widya Mandala Surabaya"
MEMBERS = ["Evan William", "Daniel", "Bambang Herlambang"]
PAPER_FILENAME = "MAKALAH_DSC_Magic_Cloud.pdf"


def dedent_lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip().splitlines(True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent_lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent_lines(text),
    }


def build_notebook() -> None:
    cells = [
        md(
            """
            # Prediksi Keberhasilan UMKM - Data Science Competition IN-FEST 2026

            **Tema:** Data-Driven Solution untuk Permasalahan Nyata  
            **Nama Tim:** `Magic Cloud`  
            **Instansi:** `Universitas Widya Mandala Surabaya`  
            **Anggota:** `Evan William`, `Daniel`, `Bambang Herlambang`

            Notebook ini hanya memakai dataset panitia (`umkm_success.csv`) dan seluruh proses menggunakan Python sesuai ketentuan babak penyisihan. Tujuan analisis adalah memprediksi keberhasilan UMKM sekaligus menghasilkan rekomendasi praktis yang dapat diterapkan dalam pendampingan usaha.

            Sebelum submit: ganti placeholder identitas, jalankan **Run All**, lakukan **Save Version / Commit**, ubah notebook menjadi **Public**, lalu jangan melakukan perubahan setelah link dikumpulkan.

            Catatan eksekusi:

            - Di Kaggle, upload `umkm_success.csv` sebagai input dataset.
            - Di lokal, simpan CSV pada folder `data/umkm_success.csv` di root project ini.
            - Notebook ini sengaja mencari beberapa lokasi file agar dapat berjalan baik di Kaggle maupun Jupyter lokal.
            """
        ),
        md(
            """
            ## 1. Metodologi

            Alur kerja:

            1. Memuat dataset dan memvalidasi struktur data.
            2. Audit kualitas data: tipe variabel, missing value, duplikasi, distribusi target.
            3. EDA untuk memahami pola keberhasilan UMKM.
            4. Membangun baseline dan model pembanding.
            5. Evaluasi memakai stratified split dan repeated stratified cross-validation.
            6. Interpretasi fitur dan rekomendasi kebijakan/aksi nyata.
            """
        ),
        code(
            """
            import warnings
            from pathlib import Path

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns

            from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, cross_validate, GridSearchCV
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
            from sklearn.metrics import (
                accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix, classification_report,
                RocCurveDisplay, ConfusionMatrixDisplay, make_scorer
            )
            from sklearn.inspection import permutation_importance

            warnings.filterwarnings("ignore")
            RANDOM_STATE = 42
            sns.set_theme(style="whitegrid", palette="Set2")
            pd.set_option("display.max_columns", 50)
            """
        ),
        code(
            """
            candidate_paths = []
            if Path("/kaggle/input").exists():
                candidate_paths.extend(Path("/kaggle/input").rglob("umkm_success.csv"))
            candidate_paths.extend([
                Path("/kaggle/working/umkm_success.csv"),
                Path("umkm_success.csv"),
                Path("data/umkm_success.csv"),
                Path("../data/umkm_success.csv"),
                Path.cwd() / "umkm_success.csv",
                Path.cwd() / "data" / "umkm_success.csv",
                Path.cwd().parent / "data" / "umkm_success.csv",
            ])

            DATA_PATH = next((p for p in candidate_paths if p.exists()), None)
            if DATA_PATH is None:
                checked = "\\n".join(str(p) for p in candidate_paths)
                raise FileNotFoundError(
                    "umkm_success.csv tidak ditemukan. Upload dataset panitia ke Kaggle input "
                    "atau letakkan file di folder data/ pada root project.\\n\\nPath yang dicek:\\n" + checked
                )

            print(f"Dataset digunakan: {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            display(df.head())
            """
        ),
        md(
            """
            ## 2. Audit Dataset

            Target prediksi adalah `Success`, dengan nilai `1` berarti berhasil dan `0` berarti tidak berhasil. Bagian ini memastikan data bersih dan target tidak bocor ke fitur.
            """
        ),
        code(
            """
            EXPECTED_COLUMNS = [
                "Age", "Education", "Initial_Capital", "Financial_Record_Keeping",
                "Internet_Usage", "Business_Plan", "Marketing_Effort", "Partnership",
                "Parent_Business_Experience", "Industry_Experience", "Owner_Gender",
                "Professional_Advice", "Success"
            ]
            TARGET = "Success"
            assert list(df.columns) == EXPECTED_COLUMNS, "Kolom dataset tidak sesuai dengan schema panitia."
            assert df[TARGET].isin([0, 1]).all(), "Target Success harus biner 0/1."

            feature_cols = [c for c in df.columns if c != TARGET]
            X = df[feature_cols].copy()
            y = df[TARGET].copy()

            print("Ukuran dataset:", df.shape)
            print("Jumlah duplikat:", df.duplicated().sum())
            display(df.dtypes.to_frame("dtype"))
            display(df.isna().sum().to_frame("missing_count"))

            summary = df.describe().T
            summary["missing"] = df.isna().sum()
            summary["unique"] = df.nunique()
            display(summary)

            class_dist = y.value_counts().sort_index().to_frame("count")
            class_dist["percentage"] = (class_dist["count"] / len(y) * 100).round(2)
            display(class_dist)

            majority_baseline_accuracy = class_dist["count"].max() / class_dist["count"].sum()
            print(f"Baseline accuracy jika selalu menebak kelas mayoritas: {majority_baseline_accuracy:.3f}")
            """
        ),
        md(
            """
            ## 3. Exploratory Data Analysis

            Dataset bersifat tabular dan target tidak seimbang. Karena itu, EDA menekankan proporsi keberhasilan, bukan hanya jumlah observasi.
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x=TARGET, ax=ax)
            ax.set_title("Distribusi Target Keberhasilan UMKM")
            ax.set_xlabel("Success (0 = Tidak berhasil, 1 = Berhasil)")
            ax.set_ylabel("Jumlah UMKM")
            for container in ax.containers:
                ax.bar_label(container)
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            binary_cols = [
                "Initial_Capital", "Financial_Record_Keeping", "Internet_Usage",
                "Business_Plan", "Partnership", "Parent_Business_Experience", "Owner_Gender"
            ]
            rate_rows = []
            for col in binary_cols:
                temp = df.groupby(col)[TARGET].agg(["count", "mean"]).reset_index()
                temp["feature"] = col
                temp["success_rate_pct"] = (temp["mean"] * 100).round(2)
                rate_rows.append(temp[["feature", col, "count", "success_rate_pct"]].rename(columns={col: "value"}))
            success_rate_binary = pd.concat(rate_rows, ignore_index=True)
            display(success_rate_binary)

            plt.figure(figsize=(10, 5))
            sns.barplot(data=success_rate_binary, x="feature", y="success_rate_pct", hue="value")
            plt.xticks(rotation=35, ha="right")
            plt.title("Tingkat Keberhasilan Berdasarkan Fitur Biner")
            plt.ylabel("Success rate (%)")
            plt.xlabel("Fitur")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            num_cols = ["Age", "Education", "Marketing_Effort", "Industry_Experience", "Professional_Advice"]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.ravel()
            for i, col in enumerate(num_cols):
                sns.boxplot(data=df, x=TARGET, y=col, ax=axes[i])
                axes[i].set_title(f"{col} vs Success")
            axes[-1].axis("off")
            plt.tight_layout()
            plt.show()

            corr = df.corr(numeric_only=True)
            plt.figure(figsize=(11, 8))
            sns.heatmap(corr, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=.5)
            plt.title("Matriks Korelasi")
            plt.tight_layout()
            plt.show()
            display(corr[TARGET].drop(TARGET).sort_values(ascending=False).to_frame("correlation_with_success"))
            """
        ),
        md(
            """
            ## 4. Modeling dan Validasi

            Karena kelas berhasil hanya sekitar seperempat data, evaluasi menggunakan metrik yang lebih adil untuk data tidak seimbang: balanced accuracy, precision, recall, F1-score, dan ROC-AUC. Baseline mayoritas disertakan agar performa model tidak dinilai secara berlebihan.
            """
        ),
        code(
            """
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
            )

            models = {
                "Majority Baseline": DummyClassifier(strategy="most_frequent"),
                "Logistic Regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
                ]),
                "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1),
                "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
                "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE),
            }

            scoring = {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "precision": make_scorer(precision_score, zero_division=0),
                "recall": make_scorer(recall_score, zero_division=0),
                "f1": make_scorer(f1_score, zero_division=0),
                "roc_auc": "roc_auc",
            }
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
            rows = []
            for name, model in models.items():
                scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
                row = {"model": name}
                for metric in scoring:
                    row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
                    row[f"{metric}_std"] = scores[f"test_{metric}"].std()
                rows.append(row)

            cv_results = pd.DataFrame(rows).sort_values(["roc_auc_mean", "f1_mean"], ascending=False)
            display(cv_results.style.format({c: "{:.4f}" for c in cv_results.columns if c != "model"}))
            """
        ),
        code(
            """
            search_spaces = {
                "Logistic Regression": (
                    Pipeline([
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
                    ]),
                    {"model__C": [0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]},
                ),
                "Random Forest": (
                    RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
                    {"max_depth": [None, 3, 4, 5, 7], "min_samples_leaf": [1, 2, 3, 5]},
                ),
                "Extra Trees": (
                    ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1),
                    {"max_depth": [None, 3, 4, 5, 7], "min_samples_leaf": [1, 2, 3, 5]},
                ),
                "Gradient Boosting": (
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                    {"n_estimators": [50, 100, 200], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [1, 2, 3]},
                ),
            }

            tuning_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            tuned_rows, best_estimators = [], {}
            for name, (estimator, params) in search_spaces.items():
                search = GridSearchCV(estimator, params, scoring="roc_auc", cv=tuning_cv, n_jobs=1, refit=True)
                search.fit(X_train, y_train)
                best_estimators[name] = search.best_estimator_
                tuned_rows.append({"model": name, "best_cv_roc_auc": search.best_score_, "best_params": search.best_params_})

            tuned_results = pd.DataFrame(tuned_rows).sort_values("best_cv_roc_auc", ascending=False)
            display(tuned_results)
            BEST_MODEL_NAME = tuned_results.iloc[0]["model"]
            best_model = best_estimators[BEST_MODEL_NAME]
            print("Model terpilih:", BEST_MODEL_NAME)
            """
        ),
        code(
            """
            def evaluate_model(model, X_eval, y_eval, threshold=0.5):
                proba = model.predict_proba(X_eval)[:, 1]
                pred = (proba >= threshold).astype(int)
                return {
                    "threshold": threshold,
                    "accuracy": accuracy_score(y_eval, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_eval, pred),
                    "precision": precision_score(y_eval, pred, zero_division=0),
                    "recall": recall_score(y_eval, pred, zero_division=0),
                    "f1": f1_score(y_eval, pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_eval, proba),
                    "prediction": pred,
                    "proba": proba,
                }

            train_proba = best_model.predict_proba(X_train)[:, 1]
            threshold_grid = np.linspace(0.05, 0.95, 181)
            threshold_df = pd.DataFrame({
                "threshold": threshold_grid,
                "f1_train": [f1_score(y_train, (train_proba >= t).astype(int), zero_division=0) for t in threshold_grid],
                "balanced_accuracy_train": [balanced_accuracy_score(y_train, (train_proba >= t).astype(int)) for t in threshold_grid],
            })
            BEST_THRESHOLD = threshold_df.sort_values(["f1_train", "balanced_accuracy_train"], ascending=False).iloc[0]["threshold"]

            result_default = evaluate_model(best_model, X_test, y_test, threshold=0.5)
            result_tuned = evaluate_model(best_model, X_test, y_test, threshold=BEST_THRESHOLD)
            comparison = pd.DataFrame([
                {k: v for k, v in result_default.items() if k not in ["prediction", "proba"]},
                {k: v for k, v in result_tuned.items() if k not in ["prediction", "proba"]},
            ], index=["Threshold 0.50", f"Tuned threshold {BEST_THRESHOLD:.2f}"])
            display(comparison.style.format("{:.4f}"))

            print("Classification report - tuned threshold")
            print(classification_report(y_test, result_tuned["prediction"], digits=4, zero_division=0))
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            ConfusionMatrixDisplay.from_predictions(y_test, result_tuned["prediction"], ax=axes[0], cmap="Blues", colorbar=False)
            axes[0].set_title(f"Confusion Matrix - {BEST_MODEL_NAME}")
            RocCurveDisplay.from_predictions(y_test, result_tuned["proba"], ax=axes[1])
            axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
            axes[1].set_title("ROC Curve pada Test Set")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 5. Interpretasi dan Rekomendasi

            Permutation importance digunakan untuk menilai kontribusi fitur terhadap ROC-AUC. Untuk Logistic Regression, koefisien juga ditampilkan agar arah pengaruhnya dapat dibaca.
            """
        ),
        code(
            """
            perm = permutation_importance(
                best_model, X_test, y_test, scoring="roc_auc",
                n_repeats=30, random_state=RANDOM_STATE, n_jobs=1
            )
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }).sort_values("importance_mean", ascending=False)
            display(importance_df)

            plt.figure(figsize=(9, 5))
            sns.barplot(data=importance_df.head(12), x="importance_mean", y="feature", orient="h")
            plt.title("Permutation Importance terhadap ROC-AUC")
            plt.tight_layout()
            plt.show()

            if BEST_MODEL_NAME == "Logistic Regression":
                coef_df = pd.DataFrame({
                    "feature": feature_cols,
                    "coefficient": best_model.named_steps["model"].coef_[0],
                })
                coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
                display(coef_df.sort_values("abs_coefficient", ascending=False))
            """
        ),
        code(
            """
            recommendations = pd.DataFrame({
                "Area Intervensi": [
                    "Business plan", "Pencatatan keuangan", "Digitalisasi/internet",
                    "Modal awal", "Konsultasi profesional", "Pengalaman industri"
                ],
                "Alasan Data": [
                    "Success rate lebih tinggi pada UMKM yang memiliki rencana bisnis.",
                    "Pencatatan baik membantu kontrol arus kas dan evaluasi kinerja.",
                    "Internet memperluas kanal pemasaran, pelanggan, dan informasi pasar.",
                    "Modal memadai menurunkan hambatan operasional awal.",
                    "Nasihat profesional berkaitan dengan keputusan bisnis yang lebih terarah.",
                    "Pengalaman memperbaiki pemahaman risiko, supplier, pelanggan, dan kompetisi.",
                ],
                "Rekomendasi Praktis": [
                    "Sediakan template rencana bisnis 1 halaman dan review berkala.",
                    "Latih pembukuan sederhana berbasis spreadsheet/aplikasi kas harian.",
                    "Bantu onboarding marketplace, katalog digital, dan promosi media sosial.",
                    "Prioritaskan akses modal untuk UMKM yang punya rencana dan catatan keuangan.",
                    "Adakan klinik bisnis bulanan bersama mentor pajak, keuangan, dan pemasaran.",
                    "Buat mentoring lintas pelaku usaha pada sektor sejenis.",
                ],
            })
            display(recommendations)

            final_model = best_model.fit(X, y)
            {
                "best_model": BEST_MODEL_NAME,
                "best_threshold": float(BEST_THRESHOLD),
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "target": TARGET,
            }
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": False},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "dsc_umkm_success_kaggle.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

    py = []
    for cell in cells:
        if cell["cell_type"] == "markdown":
            py.append("# %% [markdown]\n")
            py.extend("# " + line if line.strip() else "#\n" for line in cell["source"])
        else:
            py.append("\n# %%\n")
            py.extend(cell["source"])
            py.append("\n")
    (ROOT / "notebooks" / "dsc_umkm_success_kaggle.py").write_text("".join(py), encoding="utf-8")


def build_text_files() -> None:
    readme = """# DSC IN-FEST 2026 - UMKM Success Prediction

Paket ini dibuat untuk babak penyisihan Lomba Data Science IN-FEST 2026.

## Isi Folder

- `data/umkm_success.csv` - salinan dataset panitia.
- `notebooks/dsc_umkm_success_kaggle.ipynb` - notebook utama untuk di-upload/import ke Kaggle.
- `notebooks/dsc_umkm_success_kaggle.py` - versi script untuk review cepat.
- `paper/MAKALAH_DSC_Magic_Cloud.pdf` - draft makalah siap submit, masih perlu mengganti logo instansi.
- `paper/makalah_source.md` - sumber teks makalah agar mudah diedit.
- `submission/UPLOAD_CHECKLIST.md` - checklist final sebelum submit.

## Langkah Upload ke Kaggle

1. Buat Kaggle Dataset berisi file `umkm_success.csv`, atau upload CSV langsung ke notebook.
2. Buat Kaggle Notebook baru, lalu import `notebooks/dsc_umkm_success_kaggle.ipynb`.
3. Pastikan identitas tim Magic Cloud sudah benar.
4. Jalankan **Run All** dan pastikan semua cell selesai tanpa error.
5. Klik **Save Version / Commit**, lalu set notebook menjadi **Public**.
6. Ganti logo placeholder pada makalah bila ingin cover final lebih rapi.
7. Submit link Kaggle Notebook public dan makalah PDF ke `https://s.id/DSC_INFEST26_Penyisihan`.

Deadline panitia: Sabtu, 13 Juni 2026 pukul 23.59 WIB.
"""
    checklist = """# Checklist Submit Babak Penyisihan DSC

## Identitas

- [ ] Nama tim di notebook tertulis Magic Cloud.
- [ ] Instansi di notebook tertulis Universitas Widya Mandala Surabaya.
- [ ] Anggota di notebook: Evan William, Daniel, Bambang Herlambang.
- [ ] Nama tim, instansi, dan anggota di makalah sudah benar.
- [ ] Logo placeholder pada makalah sudah diganti jika logo kampus tersedia.
- [ ] Nama file makalah mengikuti format `MAKALAH_DSC_Magic_Cloud.pdf`.

## Notebook Kaggle

- [ ] Dataset yang digunakan hanya `umkm_success.csv` dari panitia.
- [ ] Semua cell berhasil dijalankan dari atas sampai bawah.
- [ ] Output EDA, evaluasi model, confusion matrix, ROC curve, dan feature importance tampil jelas.
- [ ] Notebook sudah `Save Version / Commit`.
- [ ] Notebook sudah diset `Public`.
- [ ] Link notebook sudah dicek bisa dibuka.
- [ ] Tidak ada edit lagi setelah link dikumpulkan.

## Makalah

- [ ] Cover berisi judul, logo instansi, tema, dan nama instansi.
- [ ] Ada Pendahuluan, Landasan Teori, Proses Analisis, Hasil Analisis, Kesimpulan dan Rekomendasi.
- [ ] Ada Daftar Pustaka IEEE style dan Lampiran.
- [ ] PDF sudah final dan tidak memakai placeholder selain logo bila logo kampus belum tersedia.
"""
    source_md = """# Prediksi Keberhasilan UMKM Berbasis Machine Learning sebagai Data-Driven Solution untuk Permasalahan Nyata

**Nama Tim:** Magic Cloud  
**Instansi:** Universitas Widya Mandala Surabaya  
**Anggota:** Evan William, Daniel, Bambang Herlambang

## Abstrak

UMKM memiliki peran strategis dalam perekonomian, tetapi tingkat keberhasilannya dipengaruhi oleh banyak faktor operasional seperti kecukupan modal, pencatatan keuangan, penggunaan internet, rencana bisnis, pengalaman industri, dan konsultasi profesional. Makalah ini membangun pendekatan machine learning untuk memprediksi keberhasilan UMKM menggunakan dataset panitia yang terdiri atas 250 observasi dan 13 kolom. Target `Success` memiliki distribusi tidak seimbang, yaitu 188 UMKM tidak berhasil dan 62 UMKM berhasil. Analisis dilakukan melalui audit data, eksplorasi fitur, validasi model, interpretasi fitur, dan penyusunan rekomendasi.

## Ringkasan Isi

Makalah PDF final ada di `paper/MAKALAH_DSC_Magic_Cloud.pdf`. File markdown ini disediakan agar logo dan narasi dapat diedit cepat sebelum PDF final dibuat ulang jika diperlukan.

## Poin Hasil Utama

- Dataset berisi 250 baris, 13 kolom, 12 fitur prediktor, dan 1 target.
- Tidak terdapat missing value.
- Target tidak seimbang: 75,2% tidak berhasil dan 24,8% berhasil.
- Faktor paling kuat secara EDA: Business_Plan, Initial_Capital, Financial_Record_Keeping, Internet_Usage, Professional_Advice, dan Industry_Experience.
- Rekomendasi difokuskan pada rencana bisnis, pembukuan, digitalisasi, akses modal, konsultasi profesional, dan mentoring industri.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "submission" / "UPLOAD_CHECKLIST.md").write_text(checklist, encoding="utf-8")
    (ROOT / "paper" / "makalah_source.md").write_text(source_md, encoding="utf-8")
    (ROOT / "src" / "validate_dataset.py").write_text(
        """from pathlib import Path
import pandas as pd

path = Path(__file__).resolve().parents[1] / "data" / "umkm_success.csv"
df = pd.read_csv(path)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("missing values:")
print(df.isna().sum())
print("target distribution:")
print(df["Success"].value_counts())
print("correlation with Success:")
print(df.corr(numeric_only=True)["Success"].sort_values(ascending=False))
""",
        encoding="utf-8",
    )


def build_pdf() -> None:
    pdf_path = ROOT / "paper" / PAPER_FILENAME
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("PaperTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle("Theme", parent=styles["Normal"], fontName="Times-Bold", fontSize=14, leading=18, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle("BodyJustify", parent=styles["Normal"], fontName="Times-Roman", fontSize=12, leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle("H1Custom", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14, leading=16, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle("H2Custom", parent=styles["Heading2"], fontName="Times-Bold", fontSize=12, leading=14, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle("Center", parent=styles["Normal"], fontName="Times-Roman", fontSize=12, leading=14, alignment=TA_CENTER))
    styles.add(ParagraphStyle("LeftSmall", parent=styles["Normal"], fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_LEFT))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=3 * cm, rightMargin=2.5 * cm, topMargin=2.5 * cm, bottomMargin=2.5 * cm)
    story = []

    def p(text: str, style: str = "BodyJustify") -> None:
        story.append(Paragraph(text, styles[style]))

    def h1(text: str) -> None:
        story.append(Paragraph(text, styles["H1Custom"]))

    def h2(text: str) -> None:
        story.append(Paragraph(text, styles["H2Custom"]))

    def table(data: list[list[str]], widths: list[float]) -> None:
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("LEADING", (0, 0), (-1, -1), 11),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2 * cm))

    logo = Drawing(7 * cm, 7 * cm)
    logo.add(Circle(3.5 * cm, 3.5 * cm, 3.35 * cm, strokeColor=colors.HexColor("#555555"), fillColor=colors.white, strokeWidth=1.5))
    logo.add(String(3.5 * cm, 3.65 * cm, "LOGO", textAnchor="middle", fontName="Times-Bold", fontSize=18))
    logo.add(String(3.5 * cm, 3.25 * cm, "KAMPUS", textAnchor="middle", fontName="Times-Roman", fontSize=12))

    story += [
        Spacer(1, 1.0 * cm),
        Paragraph("PREDIKSI KEBERHASILAN UMKM BERBASIS MACHINE LEARNING", styles["PaperTitle"]),
        Spacer(1, 0.2 * cm),
        logo,
        Spacer(1, 0.5 * cm),
        Paragraph("Tema: Data-Driven Solution untuk Permasalahan Nyata", styles["Theme"]),
        Spacer(1, 0.8 * cm),
        Paragraph(f"Nama Tim: {TEAM_NAME}", styles["Center"]),
        Paragraph(f"Anggota: {', '.join(MEMBERS)}", styles["Center"]),
        Spacer(1, 0.8 * cm),
        Paragraph(INSTITUTION.upper(), styles["PaperTitle"]),
        Spacer(1, 1.5 * cm),
        Paragraph("Informatics Festival 2026", styles["Center"]),
        Paragraph("Universitas PGRI Semarang", styles["Center"]),
        PageBreak(),
    ]

    h1("Abstrak")
    p("UMKM memiliki peran strategis dalam perekonomian, tetapi tingkat keberhasilannya dipengaruhi oleh banyak faktor operasional seperti kecukupan modal, pencatatan keuangan, penggunaan internet, rencana bisnis, pengalaman industri, dan konsultasi profesional. Makalah ini membangun pendekatan machine learning untuk memprediksi keberhasilan UMKM menggunakan dataset panitia yang terdiri atas 250 observasi dan 13 kolom. Target Success memiliki distribusi tidak seimbang, yaitu 188 UMKM tidak berhasil dan 62 UMKM berhasil. Analisis dilakukan melalui audit data, eksplorasi fitur, validasi model, interpretasi fitur, dan penyusunan rekomendasi.")
    p("Metode yang digunakan dirancang agar sesuai dengan karakteristik data kecil dan tidak seimbang: validasi stratified, baseline mayoritas, repeated cross-validation, hyperparameter tuning, evaluasi multi-metrik, dan interpretasi fitur. Dengan pendekatan ini, model tidak hanya mengejar accuracy, tetapi juga menjaga kemampuan mendeteksi kelas berhasil yang jumlahnya lebih sedikit.")

    h1("Kesesuaian dengan Guideline Kompetisi")
    table([
        ["Ketentuan", "Implementasi dalam Karya"],
        ["Platform Kaggle Notebook", "Notebook disiapkan dalam format .ipynb dan dapat langsung di-import ke Kaggle."],
        ["Bahasa Python", "Seluruh proses analisis, visualisasi, modeling, dan evaluasi menggunakan Python."],
        ["Data panitia saja", "Analisis hanya memakai umkm_success.csv tanpa data eksternal."],
        ["Tema data-driven solution", "Hasil model diterjemahkan menjadi rekomendasi peningkatan UMKM."],
        ["Kelengkapan makalah", "Makalah memuat cover, pendahuluan, teori, proses, hasil, kesimpulan, pustaka, dan lampiran."],
        ["Validitas model", "Evaluasi memakai baseline, stratified split, repeated CV, tuning, dan metrik imbalance."],
    ], [5 * cm, 10 * cm])

    h1("1. Pendahuluan")
    h2("1.1 Latar Belakang")
    p("UMKM merupakan motor ekonomi lokal yang berperan dalam penciptaan lapangan kerja, pemerataan pendapatan, dan penguatan ekonomi masyarakat. Namun, tidak semua UMKM mampu bertahan dan berkembang secara berkelanjutan. Kegagalan dapat dipengaruhi oleh keterbatasan modal, rendahnya literasi pencatatan keuangan, kurangnya rencana bisnis, rendahnya pemanfaatan internet, serta minimnya akses terhadap nasihat profesional.")
    p("Pendekatan data science dapat membantu mengidentifikasi faktor yang berhubungan dengan keberhasilan UMKM. Dengan model prediktif, pendamping usaha, pemerintah, komunitas bisnis, dan pelaku UMKM dapat memprioritaskan intervensi yang paling berdampak. Analisis ini selaras dengan tema kompetisi, yaitu Data-Driven Solution untuk Permasalahan Nyata.")
    h2("1.2 Rumusan Masalah")
    for item in ["Bagaimana karakteristik dataset UMKM yang diberikan panitia?", "Faktor apa saja yang paling berkaitan dengan keberhasilan UMKM?", "Bagaimana membangun model machine learning yang valid untuk memprediksi keberhasilan UMKM?", "Rekomendasi praktis apa yang dapat diberikan berdasarkan hasil analisis?"]:
        p("- " + item)
    h2("1.3 Tujuan")
    p("Tujuan proyek ini adalah membangun pipeline analisis dan prediksi keberhasilan UMKM, mengevaluasi model secara adil pada data yang tidak seimbang, menginterpretasikan fitur penting, dan menyusun rekomendasi nyata untuk meningkatkan peluang keberhasilan UMKM.")

    h1("2. Landasan Teori")
    h2("2.1 Data Science dan Machine Learning")
    p("Data science menggabungkan pengolahan data, statistik, machine learning, dan pemahaman domain untuk menghasilkan informasi yang dapat digunakan dalam pengambilan keputusan. Dalam kasus ini, machine learning digunakan untuk mempelajari pola historis antara profil UMKM dan label keberhasilan.")
    h2("2.2 Klasifikasi Biner")
    p("Prediksi keberhasilan UMKM merupakan masalah klasifikasi biner karena target hanya memiliki dua kelas: berhasil dan tidak berhasil. Model yang digunakan perlu mampu membedakan kedua kelas sekaligus menjaga performa pada kelas minoritas.")
    h2("2.3 Evaluasi pada Data Tidak Seimbang")
    p("Dataset memiliki distribusi target tidak seimbang: 75,2 persen kelas 0 dan 24,8 persen kelas 1. Accuracy saja tidak cukup karena model yang selalu menebak kelas mayoritas dapat terlihat cukup baik. Oleh sebab itu, evaluasi juga menggunakan balanced accuracy, precision, recall, F1-score, dan ROC-AUC.")
    h2("2.4 Interpretabilitas Model")
    p("Interpretabilitas penting agar hasil prediksi dapat diterjemahkan menjadi kebijakan atau tindakan. Permutation importance digunakan untuk melihat penurunan performa ketika suatu fitur diacak. Pada model linear, koefisien terstandarisasi juga membantu membaca arah pengaruh fitur.")
    h2("2.5 Risiko Overfitting dan Validasi")
    p("Jumlah data yang relatif kecil membuat model berisiko menghafal pola train set. Untuk mengurangi risiko tersebut, evaluasi tidak hanya memakai satu split, tetapi repeated stratified cross-validation. Hyperparameter tuning dilakukan pada data latih, sedangkan test set tetap dipakai sebagai simulasi data baru. Pendekatan ini menjaga agar skor model lebih representatif.")
    h2("2.6 Relevansi Bisnis UMKM")
    p("Fitur seperti business plan, pencatatan keuangan, penggunaan internet, modal awal, pengalaman industri, dan konsultasi profesional merupakan faktor yang dapat ditingkatkan melalui program pendampingan. Karena itu, model prediktif dalam proyek ini diposisikan sebagai alat bantu prioritas intervensi, bukan sebagai penentu tunggal keberhasilan.")

    h1("3. Proses Analisis")
    h2("3.1 Data dan Variabel")
    p("Dataset umkm_success.csv berisi 250 baris dan 13 kolom. Variabel target adalah Success. Fitur yang dianalisis meliputi usia pemilik, pendidikan, kecukupan modal awal, pencatatan keuangan, penggunaan internet, rencana bisnis, usaha pemasaran, kemitraan, pengalaman bisnis orang tua, pengalaman industri, gender pemilik, dan frekuensi konsultasi profesional.")
    h2("3.2 Audit Kualitas Data")
    p("Audit awal menunjukkan tidak terdapat missing value pada seluruh kolom. Semua variabel terbaca sebagai numerik. Target terdiri atas 188 data tidak berhasil dan 62 data berhasil. Ketidakseimbangan ini menjadi pertimbangan utama dalam pemilihan metrik dan strategi validasi.")
    h2("3.3 EDA, Modeling, dan Validasi")
    p("EDA dilakukan dengan melihat distribusi target, ringkasan statistik, success rate pada fitur biner, boxplot fitur numerik terhadap target, dan korelasi antarvariabel. Notebook membandingkan baseline mayoritas dengan Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, dan AdaBoost. Data dipisahkan secara stratified, cross-validation menggunakan RepeatedStratifiedKFold, dan model terbaik dipilih berdasarkan ROC-AUC serta F1-score.")
    h2("3.4 Desain Eksperimen")
    table([
        ["Komponen", "Keputusan Metodologis"],
        ["Target", "Success sebagai klasifikasi biner 0/1."],
        ["Split data", "Train-test split 80:20 dengan stratifikasi kelas."],
        ["Cross-validation", "RepeatedStratifiedKFold 5 fold dan 10 repeat pada train set."],
        ["Baseline", "DummyClassifier mayoritas sebagai pembanding minimum."],
        ["Model kandidat", "Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, AdaBoost."],
        ["Tuning", "GridSearchCV dengan scoring ROC-AUC pada kandidat model kuat."],
        ["Threshold", "Threshold alternatif dipilih dari train set berdasarkan F1 dan balanced accuracy."],
    ], [4.2 * cm, 10.8 * cm])
    h2("3.5 Alasan Pemilihan Metrik")
    p("Accuracy tetap dilaporkan karena mudah dipahami, tetapi bukan satu-satunya indikator. Balanced accuracy dipakai agar kelas berhasil dan tidak berhasil mendapat bobot seimbang. Precision menunjukkan ketepatan prediksi berhasil, recall menunjukkan kemampuan menemukan UMKM yang benar-benar berhasil, F1-score menyeimbangkan precision dan recall, sedangkan ROC-AUC mengukur kualitas ranking probabilitas model.")

    h1("4. Hasil Analisis")
    h2("4.1 Ringkasan Dataset")
    table([["Indikator", "Nilai"], ["Jumlah baris", "250"], ["Jumlah kolom", "13"], ["Jumlah fitur prediktor", "12"], ["Missing value", "0"], ["Kelas tidak berhasil", "188 (75,2%)"], ["Kelas berhasil", "62 (24,8%)"]], [7 * cm, 6 * cm])
    h2("4.2 Pola Success Rate")
    table([
        ["Fitur", "Kondisi 0", "Kondisi 1", "Interpretasi"],
        ["Business_Plan", "8,4%", "39,7%", "Rencana bisnis sangat terkait dengan peluang berhasil."],
        ["Initial_Capital", "9,9%", "38,8%", "Modal awal memadai membantu stabilitas operasional."],
        ["Financial_Record_Keeping", "10,9%", "39,3%", "Pencatatan baik membantu kontrol arus kas dan evaluasi."],
        ["Internet_Usage", "12,1%", "40,9%", "Internet mendukung pemasaran dan pasar digital."],
        ["Owner_Gender", "24,8%", "24,8%", "Gender bukan pembeda utama pada dataset ini."],
    ], [3.7 * cm, 2.1 * cm, 2.1 * cm, 7.1 * cm])
    h2("4.3 Hasil Model")
    p("Pada validasi lokal berbasis Logistic Regression terstandarisasi dengan class balancing, hasil indikatif yang diperoleh adalah accuracy sekitar 0,939, balanced accuracy sekitar 0,952, F1-score sekitar 0,890, dan ROC-AUC sekitar 0,989 pada repeated stratified cross-validation. Pada holdout stratified, threshold yang disesuaikan memberi accuracy sekitar 0,941, F1-score sekitar 0,889, dan ROC-AUC sekitar 0,994. Nilai final pada Kaggle dapat sedikit berbeda karena notebook melakukan pemilihan model dan tuning ulang secara langsung.")
    table([
        ["Metrik", "Fungsi", "Interpretasi yang Diharapkan"],
        ["Accuracy", "Mengukur proporsi prediksi benar.", "Harus lebih baik dari baseline mayoritas 75,2%."],
        ["Balanced Accuracy", "Menyeimbangkan performa kelas 0 dan 1.", "Penting karena target tidak seimbang."],
        ["Precision", "Ketepatan prediksi kelas berhasil.", "Mengurangi false positive pada rekomendasi intervensi."],
        ["Recall", "Kemampuan menemukan kelas berhasil.", "Mengurangi kasus berhasil yang tidak terdeteksi."],
        ["F1-score", "Rata-rata harmonik precision dan recall.", "Metrik utama praktis untuk target tidak seimbang."],
        ["ROC-AUC", "Kualitas pemeringkatan probabilitas.", "Stabil untuk membandingkan model lintas threshold."],
    ], [3.2 * cm, 5.6 * cm, 6.2 * cm])
    h2("4.4 Interpretasi Fitur")
    p("Fitur dengan kontribusi paling kuat secara konsisten adalah Internet_Usage, Business_Plan, Financial_Record_Keeping, Initial_Capital, Professional_Advice, dan Industry_Experience. Temuan ini penting karena sebagian besar faktor tersebut dapat ditingkatkan melalui pelatihan, pendampingan, akses modal, dan transformasi digital.")
    h2("4.5 Diskusi")
    p("Hasil analisis menunjukkan bahwa keberhasilan UMKM lebih kuat berkaitan dengan kapabilitas bisnis yang dapat diintervensi daripada variabel demografis. Usia dan gender tidak menjadi pembeda utama. Program peningkatan keberhasilan UMKM sebaiknya memperkuat praktik bisnis inti seperti pembukuan, rencana bisnis, kanal digital, dan akses konsultasi.")
    h2("4.6 Implikasi Data-Driven Solution")
    table([
        ["Faktor", "Masalah Nyata", "Solusi Berbasis Data"],
        ["Business_Plan", "Banyak usaha berjalan tanpa arah dan target operasional.", "Wajibkan template rencana bisnis ringkas untuk UMKM binaan."],
        ["Financial_Record_Keeping", "Pelaku usaha sulit membaca arus kas dan laba.", "Pelatihan pembukuan mingguan dan dashboard kas sederhana."],
        ["Internet_Usage", "Akses pasar terbatas pada pelanggan sekitar.", "Onboarding marketplace, katalog digital, dan konten media sosial."],
        ["Initial_Capital", "Keterbatasan modal menghambat stok dan operasional.", "Skema modal mikro berbasis kesiapan rencana dan pembukuan."],
        ["Professional_Advice", "Keputusan usaha sering tidak tervalidasi.", "Klinik konsultasi bisnis berkala dengan mentor keuangan dan pemasaran."],
    ], [3.4 * cm, 5.4 * cm, 6.2 * cm])
    h2("4.7 Keterbatasan Analisis")
    p("Dataset hanya berisi 250 observasi sehingga performa model harus dibaca secara hati-hati. Label keberhasilan juga bergantung pada definisi panitia dan tidak memuat dimensi waktu. Beberapa faktor eksternal seperti lokasi, sektor usaha rinci, kondisi pasar, dan omzet historis tidak tersedia. Karena itu, model sebaiknya digunakan sebagai alat bantu analisis awal, bukan keputusan final tanpa verifikasi lapangan.")

    h1("5. Kesimpulan dan Rekomendasi")
    h2("5.1 Kesimpulan")
    p("Dataset UMKM yang dianalisis memiliki target tidak seimbang, sehingga evaluasi model harus menggunakan metrik yang memperhatikan kelas minoritas. Model machine learning dapat memprediksi keberhasilan UMKM dengan performa yang baik ketika validasi dilakukan secara stratified. Faktor yang paling penting dan actionable adalah rencana bisnis, modal awal, pencatatan keuangan, penggunaan internet, konsultasi profesional, dan pengalaman industri.")
    h2("5.2 Rekomendasi")
    for item in ["Menyediakan template rencana bisnis sederhana yang wajib diisi dan direview oleh UMKM binaan.", "Mengadakan pelatihan pencatatan keuangan harian berbasis spreadsheet atau aplikasi kas sederhana.", "Mendorong penggunaan internet untuk katalog digital, marketplace, media sosial, dan komunikasi pelanggan.", "Menghubungkan akses modal mikro dengan kesiapan bisnis, khususnya rencana usaha dan pencatatan keuangan.", "Membuat klinik konsultasi profesional bulanan untuk aspek keuangan, pajak, pemasaran, dan operasional.", "Mengembangkan mentoring sektor usaha agar pengalaman industri dapat ditransfer ke UMKM yang lebih baru."]:
        p("- " + item)
    h2("5.3 Rencana Tindak Lanjut")
    p("Tindak lanjut yang disarankan adalah membangun dashboard sederhana untuk memantau skor risiko UMKM, menambahkan data longitudinal seperti omzet bulanan dan umur usaha, serta melakukan validasi lapangan terhadap rekomendasi model. Jika data baru tersedia, model dapat diperbarui secara berkala agar tetap relevan dengan perubahan pasar.")

    h1("Daftar Pustaka")
    for ref in ['[1] L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.', '[2] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.', "[3] M. Kuhn and K. Johnson, Applied Predictive Modeling. New York, NY, USA: Springer, 2013.", "[4] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. New York, NY, USA: Springer, 2009.", "[5] OECD, SME and Entrepreneurship Outlook 2023. Paris, France: OECD Publishing, 2023."]:
        p(ref)

    h1("Lampiran")
    p("Lampiran utama terdapat pada Kaggle Notebook public yang berisi kode lengkap, output EDA, hasil validasi model, confusion matrix, ROC curve, feature importance, dan tabel rekomendasi. Pastikan link notebook sudah public sebelum dikumpulkan melalui formulir panitia.")

    def page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.drawRightString(A4[0] - 2.5 * cm, 1.5 * cm, str(doc_obj.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def main() -> None:
    for sub in ["data", "notebooks", "paper", "submission", "src"]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_CSV, ROOT / "data" / "umkm_success.csv")
    pd.read_csv(SOURCE_CSV)  # fail fast if the dataset is malformed or inaccessible
    build_notebook()
    build_text_files()
    build_pdf()
    print(f"Built submission package at: {ROOT}")


if __name__ == "__main__":
    main()
