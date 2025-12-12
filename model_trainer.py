# model_trainer.py

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------
# FUNGSI: Melatih model SVC (SUNGGUHAN)
# -------------------------------------------------------------
@st.cache_resource
def train_model_svc(df_train, target_col, feature_cols):
    """Training SVC + standarisasi fitur (scaler)."""
    st.info("Melakukan pelatihan model SVC...")

    try:
        X_train = df_train[feature_cols].values
        y_train = df_train[target_col].values
        
        # Inisialisasi dan latih Scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Inisialisasi dan latih Model SVC
        svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svc_model.fit(X_train_scaled, y_train)
        
        st.success("Pelatihan selesai!")
        return scaler, svc_model
        
    except Exception as e:
        st.error(f"Gagal melatih model: {e}")
        return None, None

# -------------------------------------------------------------
# FUNGSI: Evaluasi Model (Optional: bisa dipindahkan ke app.py)
# -------------------------------------------------------------
def evaluate_model(model, scaler, df_test, target_col, feature_cols):
    """Evaluasi model pada data uji."""
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    # Transformasi data uji menggunakan scaler yang telah dilatih
    X_test_scaled = scaler.transform(X_test)
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)

    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    return accuracy, df_report