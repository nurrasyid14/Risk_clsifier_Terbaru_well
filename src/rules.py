def hitung_kolektibilitas_ojk(pd_value, hari_tunggakan):
    """
    Fungsi Diskrit untuk mengelompokkan risiko ke dalam 5 tingkatan Kolektibilitas (Kol).
    Menggabungkan historis (fakta tunggakan) dan prediktif (Machine Learning).
    """
    # 1. Tentukan Skor Historis (Aturan Baku BI/OJK)
    if hari_tunggakan > 180:
        kol_hist = 5  # Macet
    elif hari_tunggakan > 120:
        kol_hist = 4  # Diragukan
    elif hari_tunggakan > 90:
        kol_hist = 3  # Kurang Lancar
    elif hari_tunggakan > 0:
        kol_hist = 2  # Dalam Perhatian Khusus (DPK)
    else:
        kol_hist = 1  # Lancar

    # 2. Tentukan Skor Prediksi AI (Mencegah nasabah yang kelihatannya lancar tapi berisiko tinggi)
    if pd_value >= 0.50:
        kol_ai = 5
    elif pd_value >= 0.30:
        kol_ai = 4
    elif pd_value >= 0.15:
        kol_ai = 3
    elif pd_value >= 0.08:
        kol_ai = 2
    else:
        kol_ai = 1

    # 3. KEPUTUSAN FINAL: Bank mengambil skor terburuk demi mitigasi risiko (Conservative Approach)
    final_kol = max(kol_hist, kol_ai)

    # 4. Kamus Pemetaan Output untuk Dashboard (UI)
    mapping = {
        1: ("Kol 1 (Lancar)", "🟢 APPROVED", "success", "Tidak ada tunggakan berjalan, profil risiko AI sangat rendah."),
        2: ("Kol 2 (Dalam Perhatian Khusus)", "🟡 CONDITIONAL APPROVAL", "warning", "Terdapat riwayat tunggakan 1-90 hari atau peringatan risiko menengah dari AI."),
        3: ("Kol 3 (Kurang Lancar)", "🟠 REJECTED", "error", "Tunggakan 91-120 hari. Nasabah memerlukan penanganan khusus / restrukturisasi."),
        4: ("Kol 4 (Diragukan)", "🔴 REJECTED", "error", "Tunggakan 121-180 hari. Sangat berisiko, peluang gagal bayar tinggi."),
        5: ("Kol 5 (Macet)", "⛔ REJECTED", "error", "Tunggakan > 180 hari. Kredit macet total, masuk dalam blacklist (NPL).")
    }

    kol_label, decision, color, desc = mapping[final_kol]
    
    return final_kol, kol_label, decision, color, desc