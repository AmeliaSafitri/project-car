import pandas as pd
from scipy.io import arff
import streamlit as st

# Nama file ARFF yang digunakan (TRAIN & TEST)
TRAIN_FILE_NAME = "Car_TRAIN.arff"
TEST_FILE_NAME = "Car_TEST.arff"

@st.cache_data
def load_arff_data(file_path):
    """
    Fungsi untuk memuat file ARFF dan mengubahnya menjadi DataFrame.
    - Juga melakukan decoding byte menjadi string.
    - Menambahkan kolom Class_Label agar label lebih mudah dibaca.
    """
    try:
        # Membaca file ARFF → data dan metadata
        data_arff, meta_arff = arff.loadarff(file_path)

        # Mengubah ARFF menjadi pandas DataFrame
        df = pd.DataFrame(data_arff)

        # Kolom bertipe object biasanya berupa byte
        # Maka didecode menjadi string biasa
        for col in df.select_dtypes(['object']).columns:
            df[col] = df[col].str.decode('utf-8')

        # Set nama kolom sesuai metadata ARFF
        df.columns = meta_arff.names()

        # Kolom terakhir (label asli sebelum mapping)
        target_raw = df.columns[-1]

        # Mapping label numerik → nama kelas mobil
        class_mapping = {
            '1': 'Sedan',
            '2': 'Pickup',
            '3': 'Minivan',
            '4': 'SUV'
        }

        # Menambahkan kolom baru yang berisi nama kelas (lebih mudah dibaca)
        df["Class_Label"] = df[target_raw].astype(str).map(class_mapping)

        # Fitur yang dipakai adalah kolom yang diawali 'att'
        feature_cols = [c for c in df.columns if c.startswith("att")]

        return df, "Class_Label", feature_cols, target_raw
    
    except Exception as e:
        # Jika gagal memuat ARFF
        st.error(f"Gagal load ARFF: {e}")
        return pd.DataFrame(), None, [], None
