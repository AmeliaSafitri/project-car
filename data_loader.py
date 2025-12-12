# data_loader.py

import streamlit as st
import pandas as pd
from scipy.io import arff
import os

# -------------------------------------------------------------
# FUNGSI: Load file ARFF ➜ DataFrame
# -------------------------------------------------------------
@st.cache_data
def load_arff_data(file_path):
    """Memuat file ARFF dan mengubahnya menjadi DataFrame + mapping label."""
    if not os.path.exists(file_path):
        st.error(f"File ARFF tidak ditemukan: {file_path}")
        return pd.DataFrame(), None, [], None

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
        
        # Mapping label numerik → nama kelas (perlu disesuaikan jika label tidak berupa angka)
        # Asumsi kolom label adalah yang terakhir dan isinya adalah '1', '2', '3', '4' (string)
        class_mapping = {
            '1': 'Sedan',
            '2': 'Pickup',
            '3': 'Minivan',
            '4': 'SUV'
        }
        
        # Membuat kolom label yang sudah di-mapping
        df['Class_Label'] = df[target_col_raw].astype(str).map(class_mapping)

        feature_cols = [col for col in df.columns if col.startswith('att')]

        # Mengembalikan DataFrame, nama kolom label hasil mapping, nama kolom fitur, dan nama kolom label asli
        return df, 'Class_Label', feature_cols, target_col_raw

    except Exception as e:
        st.error(f"Gagal memuat file ARFF ({file_path}): {e}")
        return pd.DataFrame(), None, [], None