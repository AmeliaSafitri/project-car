import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from load_data import load_arff_data, TRAIN_FILE_NAME

# -------------------------------
# MAPPING LABEL
# -------------------------------
# Mengubah angka 1-4 menjadi nama mobil
label_map = {
    "1": "Sedan",
    "2": "Pickup",
    "3": "Minivan",
    "4": "SUV"
}

# -------------------------------
# LOAD DATA TRAINING
# -------------------------------
# Memuat dataset training dari file ARFF
df_train, target_col, feature_cols, _ = load_arff_data(TRAIN_FILE_NAME)

# -------------------------------
# KONVERSI LABEL KE STRING
# -------------------------------
# Ubah kolom target menjadi string sesuai label_map
df_train[target_col] = df_train[target_col].astype(str).map(label_map)

# -------------------------------
# PERSIAPAN FITUR & TARGET
# -------------------------------
# Ambil nilai fitur (X) dan target (y)
X = df_train[feature_cols].astype(float).values  # Fitur numerik
y = df_train[target_col].values                  # Target berupa string

# -------------------------------
# SCALING / PREPROCESSING FITUR
# -------------------------------
# Normalisasi fitur agar memiliki mean=0 dan std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# TRAIN MODEL SVC
# -------------------------------
# Membuat model Support Vector Classifier dengan kernel RBF
svc = SVC(kernel='rbf', probability=True)
svc.fit(X_scaled, y)  # Training model dengan data yang sudah diskalakan

# -------------------------------
# SIMPAN MODEL + PREPROCESSING
# -------------------------------
# Semua komponen penting disimpan dalam satu dictionary
model_dict = {
    "model": svc,           # Model SVC terlatih
    "scaler": scaler,       # Scaler untuk preprocessing fitur
    "features": feature_cols,  # Nama-nama fitur
    "target": target_col,      # Nama kolom target
    "df_test": df_train        # Test dataset (sama dengan train di sini)
}

# Simpan dictionary ke file pickle
with open("model_svc.pkl", "wb") as f:
    pickle.dump(model_dict, f)

print("✅ DONE! Model siap digunakan")
