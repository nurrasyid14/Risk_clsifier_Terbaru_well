# Credit Risk Classifier

Proyek klasifikasi risiko kredit menggunakan machine learning untuk memprediksi apakah aplikasi kredit akan disetujui atau ditolak berdasarkan data historis.

## Dataset

Dataset yang digunakan: **Statlog (German Credit Data)** dari UCI Machine Learning Repository
- 1000 sampel
- 20 fitur (mix categorical dan numerical)
- Target: Good (700 sampel) / Bad (300 sampel)

## Model Performance

**Model Terbaik: LightGBM Classifier**
- Metrik evaluasi: Accuracy, Precision, Recall, F1-Score, AUC
- Menggunakan PyCaret untuk AutoML dan hyperparameter tuning

## Setup untuk Teman

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Risk-Classifier.git
cd Credit-Risk-Classifier
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Notebook

```bash
jupyter notebook
```

Buka file `main.ipynb` dan jalankan cell-by-cell.

## Struktur Project

```
Credit-Risk-Classifier/
├── main.ipynb              # Notebook utama (EDA, preprocessing, modeling)
├── requirements.txt        # Daftar library yang dibutuhkan
├── README.md              # Dokumentasi (file ini)
└── saved_models/          # Folder untuk menyimpan model (akan dibuat otomatis)
```

## Cara Pakai

1. Buka `main.ipynb`
2. Jalankan semua cell secara berurutan
3. Model akan otomatis:
   - Download dataset dari UCI
   - Melakukan preprocessing
   - Training dengan PyCaret
   - Evaluasi dan tuning model
   - Menyimpan model terbaik

## Requirements

- Python 3.8+
- RAM minimal 4GB (untuk PyCaret)
- Koneksi internet (saat pertama kali download dataset)

## Troubleshooting

### Error: `ModuleNotFoundError`
```bash
pip install -r requirements.txt
```

### Error: PyCaret installation
Jika ada masalah dengan PyCaret, install secara manual:
```bash
pip install pycaret[full]
```

### Jupyter Kernel tidak ditemukan
```bash
python -m ipykernel install --user --name=venv
```

## Author

Tugas kuliah - Credit Risk Analysis
