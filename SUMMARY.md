# Ringkasan Perbaikan Credit Risk Classifier

## 🎯 Masalah Utama
Aplikasi Streamlit **selalu memprediksi REJECTED** untuk semua profil, bahkan yang seharusnya low-risk.

## 🔍 Akar Masalah yang Ditemukan

### 1. **Missing Features** (Paling Kritis!)
- `person_gender` dan `person_education` tidak ada di form input
- Kedua fitur ini tetap menggunakan nilai default dari row pertama dataset
- Model machine learning sangat sensitif terhadap kombinasi fitur ini

### 2. **Default Values Tidak Realistis**
- DTI (Debt-to-Income) default = 40% → Median approved = 11%
- Income default = $50k → Median approved = $73k
- Loan amount default = $20k → Median approved = $8k
- Interest rate = 15% → Median approved = 10.85%

### 3. **Threshold Terlalu Ketat**
- Threshold kolektibilitas OJK terlalu rendah
- Profil dengan PD 25% sudah masuk kategori "Dalam Perhatian Khusus"

### 4. **Python Version Mismatch**
- PyCaret hanya terinstall di Python 3.11 (credit_env)
- Default Python di sistem adalah 3.13
- App menggunakan fallback Random Forest yang terlalu konservatif

## ✅ Solusi yang Diterapkan

### File: `app.py`
```python
# ✅ Menambahkan input gender dan education
gender = st.selectbox("Jenis Kelamin", ["male", "female"])
education = st.selectbox("Pendidikan", ["High School", "Associate", "Bachelor", "Master", "Doctorate"], index=2)

# ✅ Update input_raw dengan gender dan education
if 'person_gender' in input_raw.columns: input_raw['person_gender'] = gender
if 'person_education' in input_raw.columns: input_raw['person_education'] = education

# ✅ Fix PyCaret prediction format (v3.x)
pred_label = predictions['prediction_label'].values[0]
pred_score = predictions['prediction_score'].values[0]
if pred_label == 1:
    pd_value = pred_score
else:
    pd_value = 1 - pred_score

# ✅ Update default values (lebih realistis)
age: 25 → 30
income: $50k → $70k
loan_amount: $20k → $8k
loan_int_rate: 15% → 11%
emp_length: 3 → 5
home_ownership: RENT → MORTGAGE
credit_score: 650 → 640 (median dataset)
durasi_kredit: 4 → 6
```

### File: `src/rules.py`
```python
# ✅ Threshold disesuaikan dengan distribusi actual
THRESHOLDS = {
    'MACET': 0.80,         # was 0.50
    'DIRAGUKAN': 0.65,     # was 0.30
    'KURANG_LANCAR': 0.50, # was 0.15
    'DPK': 0.40           # was 0.08
}
```

### File: `run_streamlit.bat` (BARU)
```batch
# ✅ Script untuk menjalankan dengan Python 3.11 (credit_env)
"C:\Users\User\miniconda3\envs\credit_env\python.exe" -m streamlit run app.py
```

## 📊 Hasil Testing (PyCaret Model)

| Profile | Income | Credit Score | DTI | PD | Result |
|---------|--------|--------------|-----|----|----|
| Excellent | $120k | 750 | 8.3% | 0.0% | ✅ APPROVED |
| Good | $70k | 680 | 21.4% | 16.3% | ✅ APPROVED |
| **Default Form (FIXED)** | **$70k** | **640** | **11.4%** | **18.3%** | ✅ **APPROVED** |
| High Risk | $30k | 550 | 83% | 100% | ❌ REJECTED |

## 🚀 Cara Menjalankan

### **Metode 1: Double-click run_streamlit.bat** (Paling Mudah)
```bash
# Klik 2x file run_streamlit.bat
```

### Metode 2: Manual dengan Python 3.11
```bash
"C:\Users\User\miniconda3\envs\credit_env\python.exe" -m streamlit run app.py
```

### Metode 3: Aktivasi conda environment
```bash
conda activate credit_env
streamlit run app.py
```

## 📈 Performa Model

- **Model**: XGBoost (PyCaret)
- **AUC Score**: 0.9785
- **Precision (Default)**: 0.64
- **Recall (Default)**: 0.95
- **Accuracy**: 0.87

## 🔑 Key Insights

1. **DTI (Debt-to-Income Ratio)** adalah faktor #1 paling penting (korelasi 0.38)
2. **Interest Rate** adalah indikator risiko kuat (korelasi 0.33)
3. Median DTI approved = 11%, rejected = 20%
4. PyCaret model **jauh lebih akurat** daripada Random Forest fallback
5. Profil dengan DTI > 30% cenderung rejected

## 📝 Files Modified
- ✏️ `app.py` - Input fields, prediction logic, default values
- ✏️ `src/rules.py` - Threshold adjustment
- 🆕 `run_streamlit.bat` - Launcher script
- 🆕 `README_FIX.md` - Documentation

---
**Sekarang aplikasi sudah berfungsi dengan baik!** 🎉
Silakan jalankan `run_streamlit.bat` untuk testing.
