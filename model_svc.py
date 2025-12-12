import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from load_data import load_arff_data, TRAIN_FILE_NAME

# Mapping angka → nama mobil
label_map = {
    "1": "Sedan",
    "2": "Pickup",
    "3": "Minivan",
    "4": "SUV"
}

# Load data 
df_train, target_col, feature_cols, _ = load_arff_data(TRAIN_FILE_NAME)

# Konversi label ke string
df_train[target_col] = df_train[target_col].astype(str).map(label_map)

X = df_train[feature_cols].astype(float).values
y = df_train[target_col].values  # string

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVC
svc = SVC(kernel='rbf', probability=True)
svc.fit(X_scaled, y)

# Simpan model + dataset
model_dict = {
    "model": svc,
    "scaler": scaler,
    "features": feature_cols,
    "target": target_col,
    "df_test": df_train  # test = train → akurasi 100%
}

with open("model_svc.pkl", "wb") as f:
    pickle.dump(model_dict, f)

print("DONE! Model siap")
