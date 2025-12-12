import streamlit as st
import pandas as pd
import altair as alt
import pickle
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = "model_svc.pkl"

try:
    model_data = pickle.load(open(MODEL_PATH, "rb"))
    model = model_data["model"]
    scaler = model_data["scaler"]
    saved_features = model_data["features"]
    saved_target = model_data["target"]
    df_test = model_data["df_test"]  # test = train
    st.success("🟢 model_svc.pkl berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

st.title("🚗 Prediksi Tipe Mobil")

# -------------------------------
# Evaluasi
# -------------------------------
st.header("📊 Evaluasi Model")
X_test_scaled = scaler.transform(df_test[saved_features])
y_test = df_test[saved_target].values
y_pred = model.predict(X_test_scaled)

accuracy = (y_test == y_pred).mean() * 100
st.metric("Akurasi Data Uji", f"{accuracy:.2f}%")

# Confusion report
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
st.dataframe(report.style.background_gradient(cmap="Blues"))

# -------------------------------
# Data Test Head
# -------------------------------
st.header("📄 Data Uji (head)")
st.dataframe(df_test.head())

# -------------------------------
# Prediksi per sampel
# -------------------------------
st.header("1️⃣ Prediksi Sampel Mobil")
col1, col2 = st.columns([1,2])

idx = col1.selectbox("Pilih ID Sampel", df_test.index.to_list())
selected = df_test.loc[idx]

X_sample = selected[saved_features].astype(float).values.reshape(1, -1)
pred_label = model.predict(scaler.transform(X_sample))[0]
true_label = selected[saved_target]

with col2:
    st.markdown(f"### Prediksi Model: 🚘 **{pred_label}**")
    st.markdown(f"### Label Asli: 🏷️ **{true_label}**")
    

# -------------------------------
# Grafik Time Series
# -------------------------------
st.header("2️⃣ Grafik Deret Waktu Fitur Mobil")
col3, col4 = st.columns(2)
start = col3.number_input("Mulai fitur ke-", 1, len(saved_features), 1)
end = col4.number_input("Sampai fitur ke-", 1, len(saved_features), len(saved_features))

if start <= end:
    feats = saved_features[start-1:end]
    df_plot = pd.DataFrame({
        "Langkah Waktu": range(start, end + 1),
        "Nilai Fitur": selected[feats].astype(float).values
    })
    chart = alt.Chart(df_plot).mark_line(point=True).encode(
        x="Langkah Waktu",
        y="Nilai Fitur",
        tooltip=["Langkah Waktu", "Nilai Fitur"]
    ).properties(title="📈 Grafik Deret Waktu Fitur Mobil").interactive()
    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# Detail Sample
# -------------------------------
st.header("3️⃣ Detail Data Sampel")
with st.expander("Klik untuk melihat detail"):
    st.dataframe(pd.DataFrame(selected).T.style.background_gradient(cmap="viridis"))
