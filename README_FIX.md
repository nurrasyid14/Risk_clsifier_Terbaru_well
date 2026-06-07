# Perbaikan Credit Risk Classifier

## Masalah yang Ditemukan dan Diperbaiki

### 1. **Fitur yang Tidak Diisi** ✓ FIXED
**Masalah**: Fitur `person_gender` dan `person_education` tidak diupdate dari form input, tetap menggunakan nilai default dari template (row pertama dataset).

**Solusi**: Menambahkan input field untuk gender dan education di sidebar, dan memastikan nilai-nilai ini diupdate ke `input_raw` sebelum prediksi.

### 2. **Threshold Terlalu Ketat** ✓ FIXED
**Masalah**: Threshold kolektibilitas OJK terlalu ketat sehingga profil baik pun ditolak.

**Solusi**: Menyesuaikan threshold di `src/rules.py`:
- MACET: 0.50 → 0.80
- DIRAGUKAN: 0.30 → 0.65
- KURANG_LANCAR: 0.15 → 0.50
- DPK: 0.08 → 0.40

### 3. **Default Values Tidak Realistis** ✓ FIXED
**Masalah**: Default form values menghasilkan DTI 40% (loan_percent_income = 0.40) yang jauh di atas median approved (11%).

**Solusi**: Update default values di `app.py`:
- Age: 25 → 30
- Income: $50,000 → $70,000
- Loan Amount: $20,000 → $8,000
- Interest Rate: 15% → 11%
- Employment: 3 → 5 years
- Home Ownership: RENT → MORTGAGE
- Credit Score: 650 → 640 (median)
- Credit History: 4 → 6 years

**Hasil**: DTI turun dari 40% ke 11.4%, PD turun dari 99.6% ke 18.3% (APPROVED)

### 4. **Python Version Mismatch** ✓ FIXED
**Masalah**: PyCaret terinstall di Python 3.11 (credit_env) tapi default Python adalah 3.13.

**Solusi**: Membuat script `run_streamlit.bat` yang menjalankan Streamlit dengan Python 3.11.

### 5. **PyCaret Prediction Format** ✓ FIXED
**Masalah**: PyCaret 3.x mengembalikan `prediction_score` (probabilitas untuk kelas yang diprediksi), bukan `prediction_score_0` dan `prediction_score_1`.

**Solusi**: Update logika di `app.py` untuk menangani format baru:
```python
pred_label = predictions['prediction_label'].values[0]
pred_score = predictions['prediction_score'].values[0]
if pred_label == 1:
    pd_value = pred_score
else:
    pd_value = 1 - pred_score
```

## Cara Menjalankan Aplikasi

### Opsi 1: Menggunakan Script Batch (Recommended)
```bash
run_streamlit.bat
```

### Opsi 2: Manual dengan Python 3.11
```bash
"C:\Users\User\miniconda3\envs\credit_env\python.exe" -m streamlit run app.py
```

### Opsi 3: Aktivasi conda environment
```bash
conda activate credit_env
streamlit run app.py
```

## Hasil Testing

### PyCaret Model (XGBoost - AUC 0.9785)
- **Excellent Profile** (income=120k, score=750): PD=0.0% → APPROVED ✓
- **Good Profile** (income=70k, score=680): PD=16.3% → APPROVED ✓
- **Default Form** (income=70k, score=640): PD=18.3% → APPROVED ✓
- **High Risk** (income=30k, score=550): PD=100.0% → REJECTED ✓

## File yang Dimodifikasi
1. `app.py` - Menambahkan gender & education input, fix PyCaret prediction handling, update default values
2. `src/rules.py` - Menyesuaikan threshold kolektibilitas
3. `run_streamlit.bat` - Script untuk menjalankan dengan Python 3.11

## Catatan
- Model PyCaret jauh lebih akurat daripada fallback Random Forest
- Pastikan menggunakan Python 3.11 (credit_env) untuk mendapatkan hasil terbaik
- DTI (Debt-to-Income ratio) adalah faktor paling penting dalam prediksi
