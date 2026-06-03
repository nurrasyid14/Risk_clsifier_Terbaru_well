"""
NexBank Credit Decision Engine
Menggunakan Model XGBoost dari PyCaret (AUC: 0.9785)
"""

import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go

# Memanggil modul-modul dari folder src/
from src.rules import hitung_kolektibilitas_ojk

# Import PyCaret untuk load model
try:
    from pycaret.classification import load_model, predict_model
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    # Fallback ke model lama jika PyCaret tidak tersedia
    from src.preprocessing import DataPreprocessor
    from src.modeling import CreditRiskModel

st.set_page_config(page_title="Credit Risk Analysis System", page_icon="🏦", layout="wide")

# --- 1. TAHAP LOAD MODEL ---
@st.cache_resource
def load_system():
    """Load model PyCaret atau fallback ke Random Forest manual"""

    # Coba load model PyCaret terlebih dahulu
    if PYCARET_AVAILABLE and os.path.exists("models/best_pycaret_model.pkl"):
        try:
            model = load_model('models/best_pycaret_model')

            # Load dataset untuk template
            file_path = "loan_data.csv"
            if not os.path.exists(file_path) and os.path.exists("data/loan_data.csv"):
                file_path = "data/loan_data.csv"

            df = pd.read_csv(file_path)
            df = df.dropna(subset=['loan_status'])
            X = df.drop('loan_status', axis=1)
            template_df = X.iloc[[0]].copy()

            return model, template_df, "PyCaret", "Sistem Siap! (Model: XGBoost - AUC 0.9785)"
        except Exception as e:
            st.warning(f"Gagal load model PyCaret: {e}. Menggunakan Random Forest manual...")

    # Fallback: Train Random Forest manual
    file_path = "loan_data.csv"
    if not os.path.exists(file_path) and os.path.exists("data/loan_data.csv"):
        file_path = "data/loan_data.csv"

    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['loan_status'])
    except FileNotFoundError:
        return None, None, None, "File loan_data.csv tidak ditemukan!"

    X = df.drop('loan_status', axis=1)
    template_df = X.iloc[[0]].copy()

    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(df)

    model = CreditRiskModel()
    model.train(X_processed, y)

    return (preprocessor, model), template_df, "RandomForest", "Sistem Siap! (Model: Random Forest Manual)"

model_data, template_df, model_type, status_msg = load_system()

# --- 2. TAHAP UI & INPUT DASHBOARD ---
st.title("🏦 Credit Risk Analysis System")
st.markdown("Sistem Penilaian Risiko Pinjaman Berbasis Machine Learning")
st.markdown(f"**{status_msg}**")
st.divider()

if model_data is None:
    st.error(status_msg)
    st.stop()

# Sidebar untuk Input
with st.sidebar:
    st.header("📝 Form Input Data")
    app_name = st.text_input("Nama Aplikan", value="Andi")
    age = st.number_input("Umur (Tahun)", min_value=18, max_value=100, value=25)
    income = st.number_input("Pendapatan Tahunan ($)", min_value=1000, value=50000, step=1000)
    loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=1000, value=20000, step=1000)
    loan_int_rate = st.number_input("Suku Bunga (%)", min_value=1.0, value=15.0, step=0.1)

    st.markdown("---")
    st.subheader("Data Tambahan")
    emp_length = st.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=50, value=3)
    home_ownership = st.selectbox("Status Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE"])
    credit_score = st.number_input("Skor Kredit", min_value=300, max_value=850, value=650)
    hari_tunggakan = st.number_input("Riwayat Tunggakan (Hari)", min_value=0, value=0)
    durasi_kredit = st.number_input("Durasi Histori Kredit (Tahun)", min_value=0, value=4)

    analyze_btn = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

# --- 3. TAHAP DEPLOYMENT / PREDIKSI ---
if analyze_btn:
    with st.spinner("Memproses Analisis Risiko..."):
        time.sleep(0.8)

        dti_ratio = loan_amount / income if income > 0 else 1.0

        # --- DYNAMIC TEMPLATE MATCHING ---
        input_raw = template_df.copy()

        # Timpa nilainya satu per satu HANYA JIKA kolomnya ada
        if 'person_age' in input_raw.columns: input_raw['person_age'] = age
        if 'person_income' in input_raw.columns: input_raw['person_income'] = income

        # Mengatasi nama kolom pengalaman kerja yang berbeda
        if 'person_emp_exp' in input_raw.columns: input_raw['person_emp_exp'] = emp_length
        elif 'person_emp_length' in input_raw.columns: input_raw['person_emp_length'] = emp_length

        if 'person_home_ownership' in input_raw.columns: input_raw['person_home_ownership'] = home_ownership
        if 'loan_amnt' in input_raw.columns: input_raw['loan_amnt'] = loan_amount
        if 'loan_intent' in input_raw.columns: input_raw['loan_intent'] = loan_intent
        if 'loan_int_rate' in input_raw.columns: input_raw['loan_int_rate'] = loan_int_rate
        if 'loan_percent_income' in input_raw.columns: input_raw['loan_percent_income'] = dti_ratio
        if 'cb_person_cred_hist_length' in input_raw.columns: input_raw['cb_person_cred_hist_length'] = durasi_kredit
        if 'credit_score' in input_raw.columns: input_raw['credit_score'] = credit_score

        # Mengatasi kolom riwayat gagal bayar
        if 'cb_person_default_on_file' in input_raw.columns:
            input_raw['cb_person_default_on_file'] = 'Y' if hari_tunggakan > 0 else 'N'
        elif 'previous_loan_defaults_on_file' in input_raw.columns:
            input_raw['previous_loan_defaults_on_file'] = 'Yes' if hari_tunggakan > 0 else 'No'

        # PREDIKSI berdasarkan tipe model
        if model_type == "PyCaret":
            # PyCaret model: gunakan predict_model
            predictions = predict_model(model_data, data=input_raw)

            # Ambil probabilitas untuk kelas 1 (Gagal Bayar)
            if 'prediction_score_1' in predictions.columns:
                pd_value = predictions['prediction_score_1'].values[0]
            elif 'prediction_score' in predictions.columns:
                pd_value = predictions['prediction_score'].values[0]
            else:
                # Fallback: gunakan prediction label
                pd_value = 0.5 if predictions['prediction_label'].values[0] == 1 else 0.1

        else:
            # Random Forest manual
            preprocessor, ml_model = model_data
            input_processed = preprocessor.transform(input_raw)
            pd_value = ml_model.predict_default_prob(input_processed)

        pd_value = max(0.01, min(pd_value, 0.99))
        pd_percent = pd_value * 100

        # Menghitung Expected Loss
        lgd_rate = 0.45
        expected_loss = loan_amount * pd_value * lgd_rate

        # Logika Keputusan berdasarkan Ambang Batas 15%
        if pd_percent < 15.0:
            decision = "APPROVED"
            decision_color = "success"
        elif pd_percent < 30.0:
            decision = "CONDITIONAL APPROVAL"
            decision_color = "warning"
        else:
            decision = "REJECTED"
            decision_color = "error"

        # --- 4. TATA LETAK HASIL ---
        st.markdown("### Detail Aplikan")
        st.markdown(f"""
        <div style='background-color: #2c3e50; padding: 20px; border-radius: 8px; color: white; margin-bottom: 25px;'>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                <div><strong>Nama:</strong> {app_name}</div>
                <div><strong>Tujuan Pinjaman:</strong> {loan_intent}</div>
                <div><strong>Umur:</strong> {age}</div>
                <div><strong>Jumlah Pinjaman:</strong> ${loan_amount:,.2f}</div>
                <div><strong>Pendapatan:</strong> ${income:,.2f}</div>
                <div><strong>Suku Bunga Pinjaman:</strong> {loan_int_rate}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if decision_color == "success":
            st.success(f"**Rekomendasi: {decision}**")
        elif decision_color == "warning":
            st.warning(f"**Rekomendasi: {decision}**")
        else:
            st.error(f"**Rekomendasi: {decision}**")

        col_text, col_chart = st.columns([1, 1])

        with col_text:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #c0392b;'>Potensi Kerugian (Expected Loss): ${expected_loss:,.2f}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Probabilitas Gagal Bayar (PD): {pd_percent:.2f}%**")

        with col_chart:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pd_percent,
                number = {'suffix': "%", 'font': {'size': 40, 'color': '#2c3e50'}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "rgba(41, 128, 185, 0.8)"},
                    'steps': [
                        {'range': [0, 15], 'color': "rgba(46, 204, 113, 0.4)"},
                        {'range': [15, 30], 'color': "rgba(241, 196, 15, 0.4)"},
                        {'range': [30, 100], 'color': "rgba(231, 76, 60, 0.4)"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': pd_percent
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        ambang_batas = 15.0
        status_ambang = "di bawah" if pd_percent < ambang_batas else "di atas"
        keputusan_akhir = "Disetujui (Approved)" if pd_percent < ambang_batas else "Ditolak / Syarat Khusus"

        st.info(f"Aplikan bernama **{app_name}** memiliki probabilitas gagal bayar sebesar **{pd_percent:.2f}%**, yang berada {status_ambang} ambang batas {ambang_batas}%. Dengan demikian, pinjaman **{keputusan_akhir}**.")

else:
    st.write("Silakan isi form di *Sidebar* sebelah kiri dan klik **Jalankan Analisis** untuk melihat detail profil risiko nasabah.")
