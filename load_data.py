# load_data.py
import pandas as pd
from scipy.io import arff

# Nama file train & test
TRAIN_FILE_NAME = "Car_TRAIN.arff"
TEST_FILE_NAME = "Car_TEST.arff"

def load_arff_data(path):
    """
    Load file ARFF dan konversi ke DataFrame pandas.
    Mengembalikan: df, nama_label, fitur, raw
    """
    try:
        data, meta = arff.loadarff(path)
    except Exception as e:
        print(f"Error membaca ARFF {path}: {e}")
        return pd.DataFrame(), None, None, None

    df = pd.DataFrame(data)

    # Konversi byte-string -> str
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    target = df.columns[-1]               # kolom terakhir = label
    features = df.columns[:-1].tolist()   # semua kecuali label

    return df, target, features, meta
