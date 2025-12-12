import streamlit as st
import pandas as pd
import altair as alt

# Import fungsi dari file load_data dan training model
from load_data import load_arff_data, TRAIN_FILE_NAME, TEST_FILE_NAME
from model_svc import train_model, evaluate_model

# -------------------------------------------------------
# Pengaturan halaman Streamlit
# -------------------------------------------------------
st.set_page_config(page_title="Prediksi Mobil Time Series", layout="wide")
st.title("🚗 Prediksi Tipe Mobil & Analisis Deret Waktu")

# -------------------------------------------------------
# LOAD DATA TRAIN & TEST
# -------------------------------------------------------
df_train, target_col, feature_cols, _ = load_arff_data(TRAIN_FILE_NAME)
df_test, _, _, _ = load_arff_data(TEST_FILE_NAME)

# Jika dataset tidak ditemukan
if df_train.empty or df_test.empty:
    st.error("File ARFF tidak ditemukan! Pastikan file berada di folder yang sama.")
    st.stop()

# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------
scaler, model = train_model(df_train, target_col, feature_cols)

# Simpan model pada session_state
st.session_state.model = model
st.session_state.scaler = scaler

# -------------------------------------------------------
# EVALUASI MODEL
# -------------------------------------------------------
st.header("A. Evaluasi Model SVC (Data Uji Tidak Dihilangkan)")

accuracy, report = evaluate_model(model, scaler, df_test, target_col, feature_cols)

# Tampilkan akurasi
st.metric(label="Akurasi pada Data Uji", value=f"{accuracy*100:.2f}%")

# Tampilkan classification report
st.dataframe(report)

st.markdown("---")

# -------------------------------------------------------
# TAMPILKAN DATA UJI (sesuai permintaan)
# -------------------------------------------------------
st.subheader("📄 Data Uji (TEST) — Tidak Dihilangkan")
st.dataframe(df_test.head())

st.markdown("---")

# -------------------------------------------------------
# PREDIKSI SAMPEL
# -------------------------------------------------------
st.header("1. Prediksi Sampel Mobil")

col1, col2 = st.columns(2)

# Pilih sampel dari Data Uji
idx = col1.selectbox("Pilih ID Sampel Mobil dari Data Uji", df_test.index.to_list())

selected = df_test.loc[idx]

# Siapkan data untuk prediksi
X_sample = selected[feature_cols].values.reshape(1, -1)
pred = model.predict(scaler.transform(X_sample))[0]

with col2:
    st.success(f"Prediksi Model: **{pred}**")
    st.info(f"Label Asli (Ground Truth): **{selected[target_col]}**")

# -------------------------------------------------------
# VISUALISASI DERET WAKTU — Bahasa Indonesia
# -------------------------------------------------------
st.header("2. Grafik Deret Waktu Fitur Mobil")

col3, col4 = st.columns(2)

# Mengatur batas fitur yang ditampilkan
start = col3.number_input("Mulai dari fitur att ke-", min_value=1, max_value=len(feature_cols), value=1)
end = col4.number_input("Sampai fitur att ke-", min_value=1, max_value=len(feature_cols), value=len(feature_cols))

if start <= end:
    features = feature_cols[start-1:end]

    df_plot = pd.DataFrame({
        "Langkah Waktu": range(start, end+1),
        "Nilai Fitur": selected[features].values
    })

    # Grafik dengan label Bahasa Indonesia
    chart = alt.Chart(df_plot).mark_line().encode(
        x="Langkah Waktu",
        y="Nilai Fitur",
        tooltip=["Langkah Waktu", "Nilai Fitur"]
    ).properties(
        title="📈 Grafik Deret Waktu Fitur Mobil (Bahasa Indonesia)"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# -------------------------------------------------------
# DETAIL SAMPEL
# -------------------------------------------------------
st.header("3. Detail Data Sampel Mobil")
st.dataframe(pd.DataFrame(selected).T)
