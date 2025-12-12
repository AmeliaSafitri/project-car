import streamlit as st
import pandas as pd
import altair as alt
import pickle
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# LOAD MODEL
# -------------------------------
MODEL_PATH = "model_svc.pkl"

try:
    # Load model, scaler, fitur, target, dan dataset test
    model_data = pickle.load(open(MODEL_PATH, "rb"))
    model = model_data["model"]          # Model SVC
    scaler = model_data["scaler"]        # Scaler untuk normalisasi
    saved_features = model_data["features"]  # Nama-nama fitur
    saved_target = model_data["target"]      # Nama kolom target
    df_test = model_data["df_test"]      # Data test
    st.success("🟢 model_svc.pkl berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# -------------------------------
# APP TITLE
# -------------------------------
st.title("🚗 Prediksi Tipe Mobil")

# -------------------------------
# EVALUASI MODEL
# -------------------------------
st.header("📊 Evaluasi Model")

# Ambil fitur dan lakukan scaling
X_test_scaled = scaler.transform(df_test[saved_features])
y_test = df_test[saved_target].values

# Prediksi seluruh data test
y_pred = model.predict(X_test_scaled)

# Hitung akurasi
accuracy = (y_test == y_pred).mean() * 100
st.metric("Akurasi Data Uji", f"{accuracy:.2f}%")

# Confusion report / classification report dalam bentuk dataframe
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report)

# -------------------------------
# DATA TEST HEAD
# -------------------------------
st.header("📄 Data Uji (head)")
st.dataframe(df_test.head())  # Menampilkan 5 baris pertama dataset test

# -------------------------------
# PREDIKSI PER SAMPEL
# -------------------------------
st.header("1️⃣ Prediksi Sampel Mobil")

# Membagi layout: kolom kiri untuk select box, kolom kanan untuk hasil prediksi
col1, col2 = st.columns([1,2])

# Pilih sampel berdasarkan index
idx = col1.selectbox("Pilih ID Sampel", df_test.index.to_list())
selected = df_test.loc[idx]

# Ambil fitur sampel dan ubah ke array 2D
X_sample = selected[saved_features].astype(float).values.reshape(1, -1)

# Prediksi model untuk sampel ini
pred_label = model.predict(scaler.transform(X_sample))[0]

# Label asli dari dataset
true_label = selected[saved_target]

# Tampilkan hasil prediksi dan label asli
with col2:
    st.markdown(f"### Prediksi Model: 🚘 **{pred_label}**")
    st.markdown(f"### Label Asli: 🏷️ **{true_label}**")
    
# -------------------------------
# GRAFIK TIME SERIES
# -------------------------------
st.header("2️⃣ Grafik Deret Waktu Fitur Mobil")

# Input untuk memilih range fitur yang ingin ditampilkan
col3, col4 = st.columns(2)
start = col3.number_input("Mulai fitur ke-", 1, len(saved_features), 1)
end = col4.number_input("Sampai fitur ke-", 1, len(saved_features), len(saved_features))

# Jika range valid, buat chart
if start <= end:
    feats = saved_features[start-1:end]  # Ambil nama fitur sesuai range
    df_plot = pd.DataFrame({
        "Langkah Waktu": range(start, end + 1),
        "Nilai Fitur": selected[feats].astype(float).values
    })
    chart = alt.Chart(df_plot).mark_line(point=True).encode(
        x="Langkah Waktu",
        y="Nilai Fitur",
        tooltip=["Langkah Waktu", "Nilai Fitur"]
    ).properties(
        title="📈 Grafik Deret Waktu Fitur Mobil"
    ).interactive()
    
    st.altair_chart(chart, width='stretch')

# -------------------------------
# DETAIL DATA SAMPEL
# -------------------------------
st.header("3️⃣ Detail Data Sampel")

# Expander untuk melihat semua detail kolom sampel
with st.expander("Klik untuk melihat detail"):
    st.dataframe(pd.DataFrame(selected).T)  # Transpose agar kolom menjadi row
