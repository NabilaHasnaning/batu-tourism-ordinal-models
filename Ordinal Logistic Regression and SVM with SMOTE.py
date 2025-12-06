
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from statsmodels.miscmodels.ordinal_model import OrderedModel

# --- Load data
file_path = r' '
df = pd.read_excel(file_path, sheet_name=' ')
print(df.head())
df.info()

# --- Jadikan Y sebagai ordinal (langsung di kolom Y)
df['Y'] = pd.Categorical(df['Y'], ordered=True)

# --- Pisah X dan Y (nama tetap)
Y = df['Y']
X = df[['X1', 'X2', 'X3', 'X4','X5']]
print("Distribusi Y (full data):")
print(df['Y'].value_counts().sort_index())

# --- Seed & split
import random
random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
print(len(X_train))
print(y_train.head())
print("Jumlah sampel per kelas di y_train:")
print(y_train.value_counts().sort_index())
print("Jumlah sampel per kelas di y_test:")
print(y_test.value_counts().sort_index())

# --- Visualisasi sebaran kelas (full, train, test)

# bikin palet otomatis sesuai jumlah kelas
palette_train = sns.color_palette("Set2", n_colors=y_train.nunique())
palette_test = sns.color_palette("Set2", n_colors=y_test.nunique())

# Plot training
import matplotlib.pyplot as plt
import seaborn as sns
# Atur font global ke Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# Plot training
plt.figure(figsize=(8,4))
ax1 = sns.countplot(x=y_train, hue=y_train, palette=palette_train, legend=False)
# Judul dan label dengan font Times New Roman
plt.xlabel("Kelas", fontsize=12)
plt.ylabel("Banyaknya Wisatawan", fontsize=12)
# Tambahkan angka di atas batang
for container in ax1.containers:
    ax1.bar_label(container, fontsize=10, color='black', padding=3)
# Tambah ruang di atas batang
ymax = y_train.value_counts().max()
ax1.set_ylim(0, ymax + 5)
plt.show()
# Set font global ke Times New Roman
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12
})
# Plot testing
plt.figure(figsize=(8,4))
ax2 = sns.countplot(x=y_test, hue=y_test, palette=palette_test, legend=False)
plt.xlabel("Kelas", fontname='Times New Roman', fontsize=12)
plt.ylabel("Banyaknya Wisatawan", fontname='Times New Roman', fontsize=12)

# Kasih angka di atas batang
for container in ax2.containers:
    ax2.bar_label(container, fontsize=10, color='black', padding=3, fontname='Times New Roman')
# Tambah ruang di atas batang
ymax = y_test.value_counts().max()
ax2.set_ylim(0, ymax + 2)
plt.show()

# --- Fungsi OATN Forward SVM
class OMSVM_OATN_Forward:
    """
    Ordinal Multi-class SVM dengan strategi One-Against-The-Next (OATN, forward).
    Melatih k-1 SVC untuk pasangan kelas berurutan: (c1 vs c2), (c2 vs c3), ...
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        # probability=False karena kita tidak pakai predict_proba; ini mempercepat training
        self.svm_params = {'kernel': kernel, 'C': C, 'gamma': gamma, 'probability': False}
        self.classifiers_ = []
        self.classes_ = None

    def _get_ordered_classes(self, y: pd.Series):
        """Ambil urutan kelas ordinal yang benar."""
        if isinstance(y.dtype, pd.CategoricalDtype) and y.cat.ordered:
            return list(y.cat.categories)  # pakai urutan kategori
        # kalau bukan categorical ordered, fallback ke sort unik biasa
        return list(np.sort(y.unique()))

    def fit(self, X, y):
        # Pastikan tipe data
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        y_ser = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y))

        # Tentukan urutan kelas ordinal
        self.classes_ = self._get_ordered_classes(y_ser)
        if len(self.classes_) < 2:
            raise ValueError("Butuh ≥2 kelas untuk melatih OATN.")

        self.classifiers_ = []

        # Latih k-1 classifier: (c_i vs c_{i+1})
        for i in range(len(self.classes_) - 1):
            c1, c2 = self.classes_[i], self.classes_[i+1]
            mask = y_ser.isin([c1, c2])
            X_sub = X_df.loc[mask].values
            y_sub = y_ser.loc[mask].values

            clf = SVC(**self.svm_params)
            clf.fit(X_sub, y_sub)
            self.classifiers_.append(clf)

            print(f"Model untuk {c1} vs {c2} berhasil dilatih.")
        return self

    def predict(self, X):
        if not self.classifiers_:
            raise RuntimeError("Model belum dilatih! Jalankan .fit() dulu.")

        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        preds = []

        for row in X_np:
            found = False
            r = row.reshape(1, -1)
            for i, clf in enumerate(self.classifiers_):
                lower_class = self.classes_[i]
                pred = clf.predict(r)[0]
                if pred == lower_class:
                    preds.append(lower_class)
                    found = True
                    break
            if not found:
                preds.append(self.classes_[-1])  # jatuh ke kelas tertinggi
        return np.array(preds, dtype=object)

# --- Metrik Evaluasi
from itertools import combinations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_classifier(
    nama_set, y_true, y_pred, labels,
    cmap='rocket_r',                 # <- default ganti ke palet kontras
    title_prefix='HASIL EVALUASI',
    show_cm=True, show_pairwise=True, return_dict=True,
    cm_linewidth=1.2,                # <- tebal garis antar sel
    cm_linecolor='grey',            # <- warna garis
):
    # --- Accuracy & CM
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{'='*35}\n {title_prefix} PADA DATA {nama_set}\n{'='*35}")
    print(f"Akurasi (Overall): {acc:.2%}")
    print(f"Misclassification Rate (Overall): {(1-acc):.2%}\n")

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=[str(l) for l in labels],
        zero_division=0,
        output_dict=True
    )

    # --- Specificity & BA per class
    specificities, balanced_acc = {}, {}
    for i, _ in enumerate(labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec_i = TN / (TN + FP) if (TN + FP) else 0.0
        rec_i  = report[str(labels[i])]['recall']
        ba_i   = (rec_i + spec_i) / 2.0
        specificities[str(labels[i])] = spec_i
        balanced_acc[str(labels[i])]  = ba_i

    macro_recall = np.mean([report[str(l)]['recall'] for l in labels])
    macro_spec   = np.mean(list(specificities.values()))
    macro_ba     = np.mean(list(balanced_acc.values()))

    header = f"{'Kelas':<10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'specificity':>12} {'BA':>8} {'support':>10}"
    print("Classification Report (+Specificity & Balanced Accuracy):")
    print(header)
    print('-' * len(header))
    for lbl in labels:
        s = str(lbl)
        print(f"{s:<10} "
              f"{report[s]['precision']:>10.2f} "
              f"{report[s]['recall']:>10.2f} "
              f"{report[s]['f1-score']:>10.2f} "
              f"{specificities[s]:>12.2f} "
              f"{balanced_acc[s]:>8.2f} "
              f"{int(report[s]['support']):>10}")
    print('-' * len(header))
    print(f"{'macro avg':<10} "
          f"{report['macro avg']['precision']:>10.2f} "
          f"{macro_recall:>10.2f} "
          f"{report['macro avg']['f1-score']:>10.2f} "
          f"{macro_spec:>12.2f} "
          f"{macro_ba:>8.2f} "
          f"{int(report['macro avg']['support']):>10}")
    print(f"{'weighted avg':<10} "
          f"{report['weighted avg']['precision']:>10.2f} "
          f"{report['weighted avg']['recall']:>10.2f} "
          f"{report['weighted avg']['f1-score']:>10.2f} "
          f"{'N/A':>12} "
          f"{'N/A':>8} "
          f"{int(report['weighted avg']['support']):>10}")

    # --- Pairwise (opsional, sama seperti punyamu) ---
    if show_pairwise and len(labels) >= 2:
        print("\nPairwise Accuracy & MR (i↔j):")
        print(f"{'Pair':<9} {'Acc_i↔j':>10} {'MR_i↔j':>10} {'n_sub':>8}")
        for (ia, ib) in combinations(range(len(labels)), 2):
            i_lab, j_lab = labels[ia], labels[ib]
            nii = cm[ia, ia]; nij = cm[ia, ib]
            nji = cm[ib, ia]; njj = cm[ib, ib]
            denom = nii + nij + nji + njj
            if denom == 0:
                acc_ij = mr_ij = np.nan
            else:
                acc_ij = (nii + njj) / denom
                mr_ij  = (nij + nji) / denom
            pair = f"{i_lab}↔{j_lab}"
            print(f"{pair:<9} {acc_ij:>10.2%} {mr_ij:>10.2%} {int(denom):>8}")

    # --- Confusion Matrix Plot (revisi tampilannya) ---
    if show_cm:
        plt.figure(figsize=(6.4, 4.6))
        ax = sns.heatmap(
            cm, annot=True, fmt='d', cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            linewidths=cm_linewidth, linecolor=cm_linecolor,  # garis antar sel
            square=True, cbar=True,
            cbar_kws={'shrink': 0.85, 'label': 'Count'},
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        ax.set_title(f'{title_prefix} - {nama_set}')
        plt.xticks(rotation=0); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()



# --- Reglog tanpa SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
# 1) Model Regresi Logistik Ordinal
reglog = OrderedModel(
    endog=y_train,
    exog=X_train,
    distr="logit"   # cumulative logit link
)
result_reglog = reglog.fit(method="newton", disp=False)
print(result_reglog.summary())
# 2) Ambil parameter cutpoint (cek namanya sesuai hasil summary)
p = result_reglog.params
# contoh kalau parameternya bernama '1/2' dan '2/3'
if '1/2' in p.index and '2/3' in p.index:
    cut1 = p['1/2']
    delta2 = np.exp(p['2/3'])
    cut2 = cut1 + delta2
    print("ζ1:", cut1)
    print("ζ2:", cut2)
else:
    print("Nama cutpoint berbeda, cek result_reglog.params.index")
# 3) Hitung VIF
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Standarisasi X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Hitung VIF pakai data yang sudah distandarisasi
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_scaled, i)
                   for i in range(X_train_scaled.shape[1])]
print("\nUji Asumsi Multikolinearitas (VIF):")
print(vif_data)
# 4) Uji signifikansi
# Parsial (Wald test)
wald_pvalues = result_reglog.pvalues
print("\nUji Wald (Parsial) - P-values:")
print(wald_pvalues)
# Simultan (Likelihood Ratio test)
lr_test = result_reglog.llr
lr_test_pvalue = result_reglog.llr_pvalue
print(f"\nLikelihood Ratio Test (Serentak) - statistics: {lr_test}")
print(f"p-value: {lr_test_pvalue}")
# 5) Prediksi probabilitas di data test
# Prediksi probabilitas
proba_reglog = result_reglog.model.predict(
    result_reglog.params,
    exog=X_test,
    which="prob"
)
# Urutan label (pakai urutan ordinal kalau ada)
unique_labels = list(y_train.cat.categories) if hasattr(y_train, "cat") else sorted(Y.unique())
# Konversi proba → label
idx = np.argmax(proba_reglog, axis=1)
y_pred_reglog_lbl = np.array([unique_labels[i] for i in idx])
print(y_pred_reglog_lbl)
# 7) Evaluasi
evaluate_classifier(
    nama_set="Data Uji",
    y_true=y_test,
    y_pred=y_pred_reglog_lbl,
    labels=unique_labels,
    title_prefix="REGLOG tanpa SMOTE",
    cmap="Blues"
)
# 7) Prediksi probabilitas di data train
# Prediksi probabilitas
proba_reglog = result_reglog.model.predict(
    result_reglog.params,
    exog=X_train,
    which="prob"
)
# Urutan label (pakai urutan ordinal kalau ada)
unique_labels = list(y_train.cat.categories) if hasattr(y_train, "cat") else sorted(Y.unique())
# Konversi proba → label
idx = np.argmax(proba_reglog, axis=1)
y_pred_reglog_lbl = np.array([unique_labels[i] for i in idx])
print(y_pred_reglog_lbl)
# 7) Evaluasi
evaluate_classifier(
    nama_set="Data Latih",
    y_true=y_train,
    y_pred=y_pred_reglog_lbl,
    labels=unique_labels,
    title_prefix="REGLOG Tanpa SMOTE",
    cmap="Blues"
)
# --- Grid Search manual untuk tuning hyperparameter ---
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from itertools import product
import numpy as np
Cs     = [0.1, 1, 10]
gammas = ['scale', 0.01, 0.1]
best_score, best_params = -1, None
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for C, gamma in product(Cs, gammas):
    scores = []
    for tr, val in cv.split(X_train, y_train):
        model = OMSVM_OATN_Forward(kernel='rbf', C=C, gamma=gamma)
        model.fit(X_train.iloc[tr], y_train.iloc[tr])
        y_val_pred = pd.Series(model.predict(X_train.iloc[val])).astype(int).values
        y_val_true = pd.Series(y_train.iloc[val]).astype(int).values
        scores.append(balanced_accuracy_score(y_val_true, y_val_pred))
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score, best_params = mean_score, {'C': C, 'gamma': gamma}
print("Best params:", best_params, "| CV Balanced Acc:", best_score)

# --- SVM tanpa smote
print("\n--- Melatih Model OATN-Forward (SVM) ---")
svm_oatn_nosmote = OMSVM_OATN_Forward(kernel='rbf', C=1, gamma=0.1)
svm_oatn_nosmote.fit(X_train, y_train)
y_pred_svm_nosmote = svm_oatn_nosmote.predict(X_test)
y_true_svm  = pd.Series(y_test).astype(int).values
y_pred_svm  = pd.Series(y_pred_svm_nosmote).astype(int).values
labels_svm  = list(map(int, svm_oatn_nosmote.classes_))
print(y_pred_svm)
evaluate_classifier(
    nama_set="Data Uji",
    y_true=y_true_svm,
    y_pred=y_pred_svm,
    labels=labels_svm,
    title_prefix="SVM OATN tanpa SMOTE",
    cmap="Blues"
)
y_pred_svm_nosmote = svm_oatn_nosmote.predict(X_train)
y_true_svm  = pd.Series(y_train).astype(int).values
y_pred_svm  = pd.Series(y_pred_svm_nosmote).astype(int).values
labels_svm  = list(map(int, svm_oatn_nosmote.classes_))
print(y_pred_svm)
evaluate_classifier(
    nama_set="Data Latih",
    y_true=y_true_svm,
    y_pred=y_pred_svm,
    labels=labels_svm,
    title_prefix="SVM OATN tanpa SMOTE",
    cmap="Blues"
)

# --- SMOTE
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# --- 0) Urutan label ordinal (buat konsistensi nantinya)
unique_labels = list(y_train.cat.categories)  # [1,2,3]
# --- 1) Cek distribusi awal (train only)
print("Sebelum SMOTE (train):")
print(y_train.value_counts().sort_index())
# --- 2) Tentukan k_neighbors aman berdasar kelas terkecil
min_count = y_train.value_counts().min()      # = 4 (kelas 1)
k_neighbors = max(1, min(min_count - 1, 5))   # = 3
print(f"k_neighbors yang digunakan: {k_neighbors}")
# --- 3) Terapkan SMOTE di TRAIN SAJA
sm = SMOTE(random_state=42, k_neighbors=k_neighbors)  # sampling_strategy default: not_majority
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
# --- 4) Kembalikan tipe kategori terurut (supaya nyambung ke model ordinal & evaluasi)
y_train_bal = pd.Categorical(y_train_bal,
                             categories=unique_labels,
                             ordered=True)
# --- 5) Cek distribusi sesudah SMOTE
print("\nSesudah SMOTE (train):")
print(pd.Series(y_train_bal).value_counts().sort_index())
# --- 6) (Opsional) Visualisasi sebelum–sesudah SMOTE
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
sns.countplot(x=y_train, ax=axes[0], hue=y_train, legend=False,
              palette=sns.color_palette("Set2", n_colors=len(unique_labels)))
axes[0].set_title("Train (Sebelum SMOTE)")
axes[0].set_xlabel("Kelas"); axes[0].set_ylabel("Jumlah")
sns.countplot(x=pd.Series(y_train_bal), ax=axes[1], hue=pd.Series(y_train_bal),
              legend=False, palette=sns.color_palette("Set2", n_colors=len(unique_labels)))
axes[1].set_title("Train (Sesudah SMOTE)")
axes[1].set_xlabel("Kelas"); axes[1].set_ylabel("")
plt.tight_layout(); plt.show()

# --- Reglog SMOTE
# 1) Model Regresi Logistik Ordinal
reglog_smote = OrderedModel(
    endog=y_train_bal,
    exog=X_train_bal,
    distr="logit"   # cumulative logit link
)
result_reglog_smote = reglog_smote.fit(method="newton", disp=False)
print(result_reglog_smote.summary())
# 2) Ambil parameter cutpoint (cek namanya sesuai hasil summary)
p = result_reglog_smote.params
# contoh kalau parameternya bernama '1/2' dan '2/3'
if '1/2' in p.index and '2/3' in p.index:
    cut1 = p['1/2']
    delta2 = np.exp(p['2/3'])
    cut2 = cut1 + delta2
    print("ζ1:", cut1)
    print("ζ2:", cut2)
else:
    print("Nama cutpoint berbeda, cek result_reglog.params.index")
# 3) Hitung VIF
# Standarisasi X_train
scaler = StandardScaler()
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
# Hitung VIF pakai data yang sudah distandarisasi
vif_data_smote = pd.DataFrame()
vif_data_smote["feature"] = X_train_bal.columns
vif_data_smote["VIF"] = [variance_inflation_factor(X_train_bal_scaled, i)
                   for i in range(X_train_bal_scaled.shape[1])]
print("\nUji Asumsi Multikolinearitas (VIF):")
print(vif_data_smote)
# 4) Uji signifikansi
# Parsial (Wald test)
wald_pvalues_smote = result_reglog_smote.pvalues
print("\nUji Wald (Parsial) - P-values:")
print(wald_pvalues_smote)
# Simultan (Likelihood Ratio test)
lr_test_smote = result_reglog_smote.llr
lr_test_pvalue_smote = result_reglog_smote.llr_pvalue
print(f"\nLikelihood Ratio Test (Serentak) - statistics: {lr_test}")
print(f"p-value: {lr_test_pvalue}")
# 5) Odds Ratio
odds_ratios_smote = np.exp(result_reglog_smote.params)
print("\nOdds Ratio (eksponeansial dari koefisien):")
print(odds_ratios_smote)
# 5) Prediksi probabilitas di data test
# Prediksi probabilitas
proba_reglog_smote = result_reglog_smote.model.predict(
    result_reglog_smote.params,
    exog=X_test,
    which="prob"
)
# Urutan label (pakai urutan ordinal kalau ada)
unique_labels = list(y_train_bal.cat.categories) if hasattr(y_train_bal, "cat") else sorted(Y.unique())
# Konversi proba → label
idx = np.argmax(proba_reglog_smote, axis=1)
y_pred_reglog_lbl_smote = np.array([unique_labels[i] for i in idx])
print(y_pred_reglog_lbl_smote)
# 5) Evaluasi
evaluate_classifier(
    nama_set="Data Uji",
    y_true=y_test,
    y_pred=y_pred_reglog_lbl_smote,
    labels=unique_labels,
    title_prefix="REGLOG dengan SMOTE",
    cmap="Blues"
)
# 6) Prediksi probabilitas di data test
# Prediksi probabilitas
proba_reglog_smote = result_reglog_smote.model.predict(
    result_reglog_smote.params,
    exog=X_train_bal,
    which="prob"
)
# Urutan label (pakai urutan ordinal kalau ada)
unique_labels = list(y_train_bal.cat.categories) if hasattr(y_train_bal, "cat") else sorted(Y.unique())
# Konversi proba → label
idx = np.argmax(proba_reglog_smote, axis=1)
y_pred_reglog_lbl_smote = np.array([unique_labels[i] for i in idx])
print(y_pred_reglog_lbl_smote)
# 6) Evaluasi
evaluate_classifier(
    nama_set="Data Latih",
    y_true=y_train_bal,
    y_pred=y_pred_reglog_lbl_smote,
    labels=unique_labels,
    title_prefix="REGLOG dengan SMOTE",
    cmap="Blues"
)

# --- SVM SMOTE
print("\n--- Melatih Model OATN-Forward (SVM SMOTE) ---")
svm_oatn_smote = OMSVM_OATN_Forward(kernel='rbf', C=1, gamma=0.1)
svm_oatn_smote.fit(X_train_bal, y_train_bal)
y_pred_svm_smote = svm_oatn_smote.predict(X_test)
y_true_svm_smote  = pd.Series(y_test).astype(int).values
y_pred_svm_smote  = pd.Series(y_pred_svm_smote).astype(int).values
labels_svm_smote  = list(map(int, svm_oatn_smote.classes_))
print(y_pred_svm_smote)
evaluate_classifier(
    nama_set="Data Uji",
    y_true=y_true_svm_smote,
    y_pred=y_pred_svm_smote,
    labels=labels_svm,
    title_prefix="SVM OATN dengan SMOTE",
    cmap="Blues"
)
print("\n--- Melatih Model OATN-Forward (SVM SMOTE) ---")
svm_oatn_smote = OMSVM_OATN_Forward(kernel='rbf', C=1, gamma=0.1)
svm_oatn_smote.fit(X_train_bal, y_train_bal)
y_pred_svm_smote = svm_oatn_smote.predict(X_train_bal)
y_true_svm_smote  = pd.Series(y_train_bal).astype(int).values
y_pred_svm_smote  = pd.Series(y_pred_svm_smote).astype(int).values
labels_svm_smote  = list(map(int, svm_oatn_smote.classes_))
print(y_pred_svm_smote)
evaluate_classifier(
    nama_set="Data Latih",
    y_true=y_true_svm_smote,
    y_pred=y_pred_svm_smote,
    labels=labels_svm,
    title_prefix="SVM OATN dengan SMOTE",
    cmap="Blues"
)

# --- SHAP
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
def _oatn_predict_proba_like(model, X):
    """
    Build K-class 'probabilities' from the (K-1) OATN pairwise SVM margins via a sequential-logit scheme:
      P(c1) = σ(s1)
      P(c2) = (1 - σ(s1)) * σ(s2)
      ...
      P(cK) = Π_{i=1..K-1} (1 - σ(si))
    where si = decision_function margin of classifier i.

    Returns: np.ndarray of shape (n_samples, K), column order aligned with model.classes_
    """
    if not getattr(model, "classifiers_", None):
        raise RuntimeError("Model not fitted. Call .fit() first.")

    # Ensure ndarray input
    X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    # Collect decision margins from each adjacent classifier -> (n, K-1)
    margins = []
    for clf in model.classifiers_:
        s = clf.decision_function(X_np)  # shape (n,)
        if s.ndim == 1:
            s = s.reshape(-1, 1)
        margins.append(s)
    S = np.hstack(margins) if margins else np.zeros((X_np.shape[0], 0))  # (n, K-1)

    # Sigmoid transform
    sig = 1.0 / (1.0 + np.exp(-S))  # (n, K-1)
    n, k_minus_1 = sig.shape
    K = k_minus_1 + 1

    # Compose sequential probabilities
    P = np.zeros((n, K), dtype=float)
    if K == 1:
        P[:, 0] = 1.0
    else:
        P[:, 0] = sig[:, 0]  # first class
        for j in range(1, K - 1):
            P[:, j] = np.prod(1.0 - sig[:, :j], axis=1) * sig[:, j]
        P[:, K - 1] = np.prod(1.0 - sig, axis=1)  # last class

    # Normalize for numerical safety
    row_sums = P.sum(axis=1, keepdims=True)
    P = np.divide(P, row_sums, out=np.full_like(P, 1.0 / K), where=row_sums != 0)
    return P  # columns correspond to model.classes_

print("\n====================\n SHAP INTERPRETATION (SVM-OATN, no retrain) \n====================")

# 1) Background data via KMeans on SMOTE-balanced train (reproducible)
k_bg = 10  # you can adjust (5–20 typical)
km = KMeans(n_clusters=k_bg, random_state=42, n_init=10)
km.fit(X_train_bal.values if hasattr(X_train_bal, "values") else X_train_bal)
X_background = km.cluster_centers_

# 2) Build KernelExplainer using your trained model
explainer = shap.KernelExplainer(
    model=lambda X: _oatn_predict_proba_like(svm_oatn_smote, X),
    data=X_background
)
print(f"SHAP KernelExplainer ready (background = {k_bg} KMeans centroids).")
# 3) Compute SHAP values on TEST set (subsample if slow)
X_test_for_shap = X_test  # e.g., X_test.sample(n=80, random_state=42) to speed up
print("\nCalculating SHAP values on the test set...")
shap_values = explainer.shap_values(X_test_for_shap)
print("SHAP values computed.")

# --- pastikan X_test_for_shap adalah DataFrame yg sama persis dg yg dipakai saat hitung shap_values
Xf = X_test_for_shap.values if hasattr(X_test_for_shap, "values") else np.asarray(X_test_for_shap)
feat_names = list(X_test_for_shap.columns) if hasattr(X_test_for_shap, "columns") \
             else [f"x{i}" for i in range(Xf.shape[1])]
n_feat = Xf.shape[1]

# --- Pisahkan per kelas -> list panjang n_classes
shap_arr = np.stack(shap_values, axis=0)   # (n_samples, n_features, n_classes)
shap_per_class = [shap_arr[:, :, j] for j in range(shap_arr.shape[2])]  # list of (n_samples, n_features)

# --- Nama kelas yang ingin ditampilkan (1,2,3)
classes_display = [1, 2, 3]  # pastikan panjangnya = jumlah kelas
legend_names = [f"Kategori {c}" for c in classes_display]

# Beeswarm per kelas 
for k, sv in enumerate(shap_per_class):
    shap.summary_plot(
        sv,
        features=Xf,
        feature_names=feat_names,
        max_display=len(feat_names),
        show=False  # supaya bisa diberi judul sebelum show
    )
    plt.title(f"SHAP Beeswarm — Kategori {classes_display[k]}")
    plt.tight_layout()
    plt.show()

# Bar plot multiclass 
shap.summary_plot(
    shap_per_class,
    feature_names=feat_names,
    plot_type="bar",
    max_display=len(feat_names),
    class_names=legend_names,  # kunci agar legend bukan 0/1/2
    show=False
)
plt.title("Global Feature Importance (stacked per kategori)")
plt.tight_layout()
plt.show()

