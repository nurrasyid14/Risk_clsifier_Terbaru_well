import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

class CreditRiskModel:
    """
    Kelas Modular khusus untuk membangun, melatih, mengevaluasi,
    dan menyimpan algoritma Random Forest Classifier yang dioptimalkan
    untuk penanganan data risiko kredit yang tidak seimbang (imbalanced).
    """
    def __init__(self, n_estimators=100, max_depth=8, random_state=42):
        # class_weight='balanced' ditambahkan untuk menangani class imbalance
        # agar model lebih sensitif terhadap nasabah berisiko Gagal Bayar (Recall tinggi)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            class_weight='balanced', 
            random_state=random_state
        )
        self.default_class_index = 1

    def train(self, X_processed, y):
        """Melatih model menggunakan data yang sudah di-preprocess."""
        self.model.fit(X_processed, y)
        
        # Deteksi otomatis kelas mana yang merupakan 'Gagal Bayar' (Biasanya bernilai 1)
        classes = list(self.model.classes_)
        if 1 in classes:
            self.default_class_index = classes.index(1)
        else:
            self.default_class_index = 0
            
        # Tampilkan laporan performa sederhana di terminal sebagai feedback latihan model
        y_pred = self.model.predict(X_processed)
        y_prob = self.model.predict_proba(X_processed)[:, self.default_class_index]
        
        print("\n" + "="*50)
        print(" LAPORAN PERFORMA MODEL SAAT LATIHAN (TRAINING PERFORMANCE)")
        print("="*50)
        print(classification_report(y, y_pred, target_names=["Lancar (0)", "Gagal Bayar (1)"]))
        print(f"Skor ROC-AUC Latihan: {roc_auc_score(y, y_prob):.4f}")
        print("="*50 + "\n")

    def predict_default_prob(self, X_input_processed):
        """Mengeluarkan probabilitas (persentase) risiko gagal bayar (PD)."""
        return self.model.predict_proba(X_input_processed)[0][self.default_class_index]

    def save_model(self, file_path="models/credit_risk_rf_model.joblib"):
        """Menyimpan model terlatih ke disk agar bisa dimuat secara instan tanpa training ulang."""
        folder_path = os.path.dirname(file_path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        joblib.dump({
            'model': self.model,
            'default_class_index': self.default_class_index
        }, file_path)
        print(f"Model berhasil diekspor dan disimpan di: {file_path}")

    def load_model(self, file_path="models/credit_risk_rf_model.joblib"):
        """Memuat model terlatih dari disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File model tidak ditemukan di: {file_path}")
            
        saved_data = joblib.load(file_path)
        self.model = saved_data['model']
        self.default_class_index = saved_data['default_class_index']
        print(f"Model berhasil dimuat secara instan dari: {file_path}")