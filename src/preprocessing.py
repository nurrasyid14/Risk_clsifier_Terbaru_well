import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.balancer import smote, tomek_links, smote_tomek, smote_enn

class DataPreprocessor:
    """
    Kelas Modular untuk membersihkan dan mentransformasi data mentah
    menjadi format yang bisa dibaca oleh algoritma Machine Learning.
    Kini dilengkapi dengan fitur pembersihan anomali data otomatis saat training.
    """
    def __init__(self, balance_method: str = "smote_tomek"):
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []
        self.balance_method = balance_method

    def fit_transform(self, df):
        """
        Mempelajari pola data (fit) sekaligus mengubahnya (transform).
        Dipanggil HANYA SAAT TRAINING.
        
        Melakukan pembersihan data (data cleaning) terhadap anomali logis:
        1. Menghapus baris jika usia nasabah > 100 tahun.
        2. Menghapus baris jika pengalaman kerja > 60 tahun.
        3. Menghapus baris jika pengalaman kerja >= usia nasabah itu sendiri.
        """
        # --- TAHAP 1: PEMBERSIHAN DATA (AUTOMATIC CLEANING) ---
        df_clean = df.copy()
        
        # Deteksi otomatis kolom pengalaman kerja (bisa berupa 'person_emp_exp' atau 'person_emp_length')
        emp_col = 'person_emp_exp' if 'person_emp_exp' in df_clean.columns else 'person_emp_length'
        
        # 1. Bersihkan usia tidak rasional
        if 'person_age' in df_clean.columns:
            df_clean = df_clean[df_clean['person_age'] <= 100]
            
        # 2. Bersihkan masa kerja tidak rasional
        if emp_col in df_clean.columns:
            df_clean = df_clean[df_clean[emp_col] <= 60]
            
        # 3. Bersihkan jika masa kerja melebihi umur nasabah
        if 'person_age' in df_clean.columns and emp_col in df_clean.columns:
            df_clean = df_clean[df_clean[emp_col] < df_clean['person_age']]

        # --- TAHAP 2: PEMISAHAN FITUR & TARGET ---
        # Pisahkan fitur (X) dan target (y) menggunakan data yang sudah disterilkan
        X = df_clean.drop('loan_status', axis=1)
        y = df_clean['loan_status']

        self.numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Pipeline untuk angka: Isi data kosong dengan median, lalu standarisasi (Scaling)
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline untuk teks: Isi data kosong dengan modus, lalu ubah jadi angka (OneHot)
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Gabungkan kedua pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numeric_features),
                ('cat', cat_transformer, self.categorical_features)
            ])

        # Eksekusi pemrosesan data X
        X_processed = self.preprocessor.fit_transform(X)

        # Pastikan output bukan sparse sebelum balancing
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()

        # --- TAHAP 3: BALANCING DATA ---
        if self.balance_method == "smote":
            X_processed, y = smote(X_processed, y)
        elif self.balance_method == "tomek_links":
            X_processed, y = tomek_links(X_processed, y)
        elif self.balance_method == "smote_tomek":
            X_processed, y = smote_tomek(X_processed, y)
        elif self.balance_method == "smote_enn":
            X_processed, y = smote_enn(X_processed, y)

        return X_processed, y

    def transform(self, df_input):
        """
        Mengubah data baru berdasarkan pola yang sudah dipelajari.
        Dipanggil SAAT PREDIKSI NASABAH BARU (Inference).
        Pada tahap ini kita TIDAK membuang baris data agar aplikasi tidak crash.
        """
        if self.preprocessor is None:
            raise Exception("Preprocessor belum di-fit dengan data training!")
        return self.preprocessor.transform(df_input)