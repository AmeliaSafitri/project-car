import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

@st.cache_resource
def train_model(df_train, target_col, feature_cols):
    """
    Fungsi training model SVC.
    - Melakukan standarisasi fitur dengan StandardScaler.
    - Melatih model SVM kernel RBF.
    - Mengembalikan model & scaler.
    """

    st.info("Training model SVC...")

    # Pisahkan fitur dan label
    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    # Standarisasi agar SVM bekerja optimal
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Inisialisasi model SVM
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # Latih model
    model.fit(X_scaled, y_train)

    st.success("Training selesai!")
    return scaler, model


def evaluate_model(model, scaler, df_test, target_col, feature_cols):
    """
    Fungsi evaluasi model.
    - Melakukan transformasi data uji menggunakan scaler.
    - Menghitung akurasi dan classification report.
    """

    # Data uji
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    # Standarisasi fitur test
    X_scaled = scaler.transform(X_test)

    # Melakukan prediksi
    y_pred = model.predict(X_scaled)

    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)

    # Hasil evaluasi detail
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()

    return accuracy, report
