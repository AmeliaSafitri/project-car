import streamlit as st
import pandas as pd
import altair as alt
from scipy.io import arff
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------
# KONSTANTA NAMA FILE DATASET
# -------------------------------------------------------------
TRAIN_FILE_NAME = "Car_TRAIN.arff"
TEST_FILE_NAME = "Car_TEST.arff"


# -------------------------------------------------------------
# FUNGSI: Load file ARFF ➜ DataFrame
# -------------------------------------------------------------
@st.cache_data
def load_arff_data(file_path):
    """Memuat file ARFF dan mengubahnya menjadi DataFrame + mapping label."""
    try:
        data_arff, meta_arff = arff.loadarff(file_path)
        df = pd.DataFrame(data_arff)

        # Decode byte → string
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.decode('utf-8')

        # Set nama kolom dari metadata ARFF
        df.columns = meta_arff.names()

        # Kolom label asli
        target_col_raw = df.columns[-1]
        
        # Mapping label numerik → nama kelas
        class_mapping = {
            '1': 'Sedan',
            '2': 'Pickup',
            '3': 'Minivan',
            '4': 'SUV'
        }
        
        df['Class_Label'] = df[target_col_raw].astype(str).map(class_mapping)

        feature_cols = [col for col in df.columns if col.startswith('att')]

        return df, 'Class_Label', feature_cols, target_col_raw

    except Exception as e:
        st.error(f"Gagal memuat file ARFF: {e}")
        return pd.DataFrame(), None, [], None


# -------------------------------------------------------------
# FUNGSI: Melatih model SVC (SUNGGUHAN)
# -------------------------------------------------------------
@st.cache_resource
def train_model_svc(df_train, target_col, feature_cols):
    """Training SVC + standarisasi fitur (scaler)."""
    st.info("Melakukan pelatihan model SVC...")

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svc_model.fit(X_train_scaled, y_train)
    
    st.success("Pelatihan selesai!")
    return scaler, svc_model


# -------------------------------------------------------------
# SETTING APP STREAMLIT
# -------------------------------------------------------------
st.set_page_config(page_title="Prediksi Mobil Time Series", layout="wide")
st.title("🚗 Prediksi Tipe Mobil dan Analisis Deret Waktu")


# -------------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------------
df_train, target_col, feature_cols, _ = load_arff_data(TRAIN_FILE_NAME)
df_test, _, _, _ = load_arff_data(TEST_FILE_NAME)

if df_train.empty or df_test.empty:
    st.error("File ARFF tidak ditemukan. Letakkan file pada direktori yang sama.")
    st.stop()

MAX_FEATURES = len(feature_cols)


# -------------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------------
scaler, svc_model = train_model_svc(df_train, target_col, feature_cols)
st.session_state.model = svc_model
st.session_state.scaler = scaler

st.markdown("---")


# -------------------------------------------------------------
# A. Evaluasi Model (ASLI)
# -------------------------------------------------------------
st.header("A. Proses Pelatihan & Evaluasi")

# --- Perhitungan SVC asli di sini ---
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

X_test_scaled = scaler.transform(X_test)
y_pred = svc_model.predict(X_test_scaled)

# Akurasi asli
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Hasil Evaluasi Model SVC")
st.metric(label="Akurasi pada Data Uji (ASLI)", value=f"{accuracy * 100:.2f} %")

# Classification report asli
report_dict = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
st.dataframe(df_report)

st.markdown("---")


# -------------------------------------------------------------
# 1. PREDIKSI SAMPEL
# -------------------------------------------------------------
st.header("1. Pemilihan Sampel dan Prediksi")

col1, col2 = st.columns(2)
sample_options = df_test.index.to_list()
selected_sample_index = col1.selectbox("Pilih ID Sampel Mobil", options=sample_options)

predicted_class = "Model Belum Dilatih"

if st.session_state.model:
    selected_car = df_test.loc[selected_sample_index]
    X_sample = selected_car[feature_cols].values.reshape(1, -1)

    X_scaled = st.session_state.scaler.transform(X_sample)
    prediction = st.session_state.model.predict(X_scaled)[0]
    predicted_class = prediction

    with col2:
        st.success(f"Prediksi: {predicted_class}")
        st.markdown(f"Label aktual: **{selected_car[target_col]}**")

st.markdown("---")


# -------------------------------------------------------------
# 2. VISUALISASI TIME SERIES
# -------------------------------------------------------------
st.header("2. Diagram Deret Waktu Fitur")

col3, col4 = st.columns(2)
start_step = col3.number_input("Mulai att:", min_value=1, max_value=MAX_FEATURES, value=1)
end_step = col4.number_input("Akhir att:", min_value=1, max_value=MAX_FEATURES, value=MAX_FEATURES)

if start_step <= end_step:
    selected_car = df_test.loc[selected_sample_index]
    plot_features = feature_cols[start_step-1:end_step]
    
    df_plot = pd.DataFrame({
        'Waktu_Langkah': range(start_step, end_step+1),
        'Nilai_Fitur': selected_car[plot_features].values
    })

    chart = alt.Chart(df_plot).mark_line().encode(
        x='Waktu_Langkah',
        y='Nilai_Fitur',
        tooltip=['Waktu_Langkah', 'Nilai_Fitur']
    ).properties(
        title=f'Deret Waktu Mobil {predicted_class}'
    ).interactive()

    st.altair_chart(chart, use_container_width=True)


# -------------------------------------------------------------
# 3. DETAIL DATA
# -------------------------------------------------------------
st.subheader("3. Detail Sampel")
st.dataframe(pd.DataFrame(selected_car).T)

st.markdown("---")

st.subheader("Sekilas Data Uji")
st.dataframe(df_test.head())



