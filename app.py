# app.py

import streamlit as st
import pandas as pd
import altair as alt

# Import fungsi dari file lain
from data_loader import load_arff_data
from model_trainer import train_model_svc, evaluate_model 

# -------------------------------------------------------------
# KONSTANTA NAMA FILE DATASET
# -------------------------------------------------------------
TRAIN_FILE_NAME = "Car_TRAIN.arff"
TEST_FILE_NAME = "Car_TEST.arff"


# -------------------------------------------------------------
# SETTING APP STREAMLIT
# -------------------------------------------------------------
st.set_page_config(page_title="Prediksi Mobil Time Series", layout="wide")
st.title("🚗 Prediksi Tipe Mobil dan Analisis Deret Waktu")


# -------------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------------
# st.spinner digunakan untuk memberikan umpan balik visual saat data dimuat
with st.spinner("Memuat data pelatihan dan uji..."):
    df_train, target_col, feature_cols, _ = load_arff_data(TRAIN_FILE_NAME)
    df_test, _, _, _ = load_arff_data(TEST_FILE_NAME)

if df_train.empty or df_test.empty:
    st.error("Gagal memuat satu atau kedua file ARFF. Pastikan file berada di direktori yang sama.")
    st.stop()

MAX_FEATURES = len(feature_cols)


# -------------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------------
# Model hanya perlu dilatih sekali (dengan st.cache_resource)
scaler, svc_model = train_model_svc(df_train, target_col, feature_cols)

# Simpan model dan scaler di session_state untuk akses mudah
if svc_model and scaler:
    st.session_state.model = svc_model
    st.session_state.scaler = scaler
else:
    st.error("Model gagal dilatih. Hentikan aplikasi.")
    st.stop()


st.markdown("---")


# -------------------------------------------------------------
# A. Evaluasi Model
# -------------------------------------------------------------
st.header("A. Proses Pelatihan & Evaluasi")

# --- Perhitungan SVC menggunakan fungsi evaluasi ---
accuracy, df_report = evaluate_model(svc_model, scaler, df_test, target_col, feature_cols)

st.subheader("Hasil Evaluasi Model SVC")
st.metric(label="Akurasi pada Data Uji", value=f"{accuracy * 100:.2f} %")

st.markdown("**Classification Report**")
st.dataframe(df_report)

st.markdown("---")


# -------------------------------------------------------------
# 1. PREDIKSI SAMPEL
# -------------------------------------------------------------
st.header("1. Pemilihan Sampel dan Prediksi")

col1, col2 = st.columns(2)
sample_options = df_test.index.to_list()
selected_sample_index = col1.selectbox("Pilih ID Sampel Mobil", options=sample_options)

predicted_class = "N/A"
selected_car = df_test.loc[selected_sample_index]

# Lakukan prediksi
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
start_step = col3.number_input("Mulai att:", min_value=1, max_value=MAX_FEATURES, value=1, key="start_step")
end_step = col4.number_input("Akhir att:", min_value=1, max_value=MAX_FEATURES, value=MAX_FEATURES, key="end_step")

if start_step > end_step:
    st.warning("Nilai 'Mulai att' harus lebih kecil atau sama dengan 'Akhir att'.")
else:
    # --- VISUALISASI ---
    plot_features = feature_cols[start_step-1:end_step]
    
    df_plot = pd.DataFrame({
        'Waktu_Langkah': range(start_step, end_step+1),
        'Nilai_Fitur': selected_car[plot_features].values
    })

    # Membuat Chart Altair
    chart = alt.Chart(df_plot).mark_line(point=True).encode(
        x=alt.X('Waktu_Langkah', title='Langkah Waktu (Fitur att)'),
        y=alt.Y('Nilai_Fitur', title='Nilai Fitur'),
        tooltip=['Waktu_Langkah', 'Nilai_Fitur']
    ).properties(
        title=f'Deret Waktu Sampel Mobil ({predicted_class})'
    ).interactive() # Aktifkan zoom dan pan

    st.altair_chart(chart, use_container_width=True)
    

[Image of Time Series Chart Example]



st.markdown("---")

# -------------------------------------------------------------
# 3. DETAIL DATA
# -------------------------------------------------------------
st.subheader("3. Detail Sampel")
st.dataframe(pd.DataFrame(selected_car).T)

st.subheader("Sekilas Data Uji")
st.dataframe(df_test.head())