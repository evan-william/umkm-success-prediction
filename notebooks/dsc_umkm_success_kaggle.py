# %% [markdown]
# # Prediksi Keberhasilan UMKM - Data Science Competition IN-FEST 2026
#
# **Tema:** Data-Driven Solution untuk Permasalahan Nyata  
# **Nama Tim:** `Magic Cloud`  
# **Instansi:** `Universitas Widya Mandala Surabaya`  
# **Anggota:** `Evan William`, `Daniel`, `Bambang Herlambang`
#
# Notebook ini hanya memakai dataset panitia (`umkm_success.csv`) dan seluruh proses menggunakan Python sesuai ketentuan babak penyisihan. Tujuan analisis adalah memprediksi keberhasilan UMKM sekaligus menghasilkan rekomendasi praktis yang dapat diterapkan dalam pendampingan usaha.
#
# Sebelum submit: ganti placeholder identitas, jalankan **Run All**, lakukan **Save Version / Commit**, ubah notebook menjadi **Public**, lalu jangan melakukan perubahan setelah link dikumpulkan.
#
# Catatan eksekusi:
#
# - Di Kaggle, upload `umkm_success.csv` sebagai input dataset.
# - Di lokal, simpan CSV pada folder `data/umkm_success.csv` di root project ini.
# - Notebook ini sengaja mencari beberapa lokasi file agar dapat berjalan baik di Kaggle maupun Jupyter lokal.# %% [markdown]
# ## 1. Metodologi
#
# Alur kerja:
#
# 1. Memuat dataset dan memvalidasi struktur data.
# 2. Audit kualitas data: tipe variabel, missing value, duplikasi, distribusi target.
# 3. EDA untuk memahami pola keberhasilan UMKM.
# 4. Membangun baseline dan model pembanding.
# 5. Evaluasi memakai stratified split dan repeated stratified cross-validation.
# 6. Interpretasi fitur dan rekomendasi kebijakan/aksi nyata.
# %%
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

# %%
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
    checked = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "umkm_success.csv tidak ditemukan. Upload dataset panitia ke Kaggle input "
        "atau letakkan file di folder data/ pada root project.\n\nPath yang dicek:\n" + checked
    )

print(f"Dataset digunakan: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
display(df.head())
# %% [markdown]
# ## 2. Audit Dataset
#
# Target prediksi adalah `Success`, dengan nilai `1` berarti berhasil dan `0` berarti tidak berhasil. Bagian ini memastikan data bersih dan target tidak bocor ke fitur.
# %%
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
# %% [markdown]
# ## 3. Exploratory Data Analysis
#
# Dataset bersifat tabular dan target tidak seimbang. Karena itu, EDA menekankan proporsi keberhasilan, bukan hanya jumlah observasi.
# %%
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x=TARGET, ax=ax)
ax.set_title("Distribusi Target Keberhasilan UMKM")
ax.set_xlabel("Success (0 = Tidak berhasil, 1 = Berhasil)")
ax.set_ylabel("Jumlah UMKM")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.show()

# %%
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

# %%
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
# %% [markdown]
# ## 4. Modeling dan Validasi
#
# Karena kelas berhasil hanya sekitar seperempat data, evaluasi menggunakan metrik yang lebih adil untuk data tidak seimbang: balanced accuracy, precision, recall, F1-score, dan ROC-AUC. Baseline mayoritas disertakan agar performa model tidak dinilai secara berlebihan.
# %%
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

# %%
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

# %%
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

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, result_tuned["prediction"], ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title(f"Confusion Matrix - {BEST_MODEL_NAME}")
RocCurveDisplay.from_predictions(y_test, result_tuned["proba"], ax=axes[1])
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[1].set_title("ROC Curve pada Test Set")
plt.tight_layout()
plt.show()
# %% [markdown]
# ## 5. Interpretasi dan Rekomendasi
#
# Permutation importance digunakan untuk menilai kontribusi fitur terhadap ROC-AUC. Untuk Logistic Regression, koefisien juga ditampilkan agar arah pengaruhnya dapat dibaca.
# %%
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

# %%
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
