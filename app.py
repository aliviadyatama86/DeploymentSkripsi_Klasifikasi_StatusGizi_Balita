# app.py
# ============================================================
# Aplikasi Streamlit:
# Klasifikasi Status Gizi Balita (Random Forest)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import BytesIO
from datetime import datetime

# -----------------------------
# Config page
# -----------------------------
st.set_page_config(page_title="Prediksi Status Gizi Balita", page_icon="🧒", layout="wide")
st.title("🧒 Prediksi & Pemantauan Status Gizi Balita")

st.markdown("""
**Fitur aplikasi**
- Upload file Excel berisi pengukuran balita.
- Analisis perkembangan per tahun atau seluruh waktu.
- Prediksi status gizi manual menggunakan model Random Forest.
- Simpan & lihat histori hasil prediksi.
""")

# -----------------------------
# Load model (jika ada)
# -----------------------------
MODEL_FILE = "final_model_status_gizi.sav"
model = None
model_features = None
if os.path.exists(MODEL_FILE):
    try:
        bundle = joblib.load(MODEL_FILE)
        model = bundle.get("model", None)
        model_features = bundle.get("features", None)
        st.success("✅ Model ditemukan dan berhasil dimuat.")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model: {e}")
else:
    st.info("⚠️ Model tidak ditemukan. Prediksi otomatis menggunakan model akan dinonaktifkan sampai file model diunggah.")

# -----------------------------
# Utility functions
# -----------------------------
def clean_and_prepare_df(df):
    """Standarisasi nama kolom & tipe untuk dataframe yang diupload."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Tanggal Pengukuran" in df.columns:
        df["Tanggal Pengukuran"] = pd.to_datetime(df["Tanggal Pengukuran"], errors="coerce", dayfirst=True)
    for col in ["Z-Score BB/U", "Z-Score TB/U", "Z-Score BB/TB", "BB", "TB"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".")
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Nama Anak" in df.columns:
        df["Nama Anak"] = df["Nama Anak"].astype(str).str.strip()
    return df

def last_measurement_per_month(df):
    """Ambil pengukuran terakhir tiap bulan per anak."""
    df = df.copy()
    if "Tanggal Pengukuran" not in df.columns:
        return pd.DataFrame()
    df["Periode"] = df["Tanggal Pengukuran"].dt.to_period("M")
    df = df.sort_values(["Nama Anak", "Tanggal Pengukuran"])
    df_last = df.groupby(["Nama Anak", "Periode"], as_index=False).last()
    df_last["Periode_MonthStart"] = df_last["Periode"].dt.to_timestamp()
    return df_last

def interpret_trend(z_series):
    """Interpretasi sederhana trend perkembangan Z-score."""
    if z_series.empty:
        return "Tidak ada data untuk analisis."
    def cat(z):
        if pd.isna(z): return np.nan
        if z < -3: return "Sangat Buruk"
        if -3 <= z < -2: return "Gizi Buruk"
        if -2 <= z < -1: return "Gizi Kurang"
        if -1 <= z < 2: return "Gizi Baik"
        if 2 <= z < 3: return "Risiko Gizi Lebih"
        if z >= 3: return "Gizi Lebih / Obesitas"
    cats = z_series.map(cat).dropna().tolist()
    if not cats: return "Data tidak mencukupi."
    if any(c in ["Sangat Buruk", "Gizi Buruk"] for c in cats):
        return "⚠️ Ada periode dengan status Gizi Buruk — perlu perhatian!"
    changes = sum(1 for i in range(1, len(cats)) if cats[i] != cats[i-1])
    if cats.count("Gizi Baik") / len(cats) >= 0.75 and changes <= 1:
        return "✅ Gizi baik dan stabil sepanjang waktu."
    elif changes >= 3:
        return "🔁 Fluktuatif — perlu pemantauan berkala."
    else:
        return "Cenderung berubah — disarankan pemantauan rutin."

# -----------------------------
# TABs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "1. Perkembangan Anak",
    "2. Prediksi Manual",
    "3. Histori & Reset"
])

# ============================================================
# TAB 1 — Analisis Perkembangan Anak
# ============================================================
with tab1:
    st.header("📊 Analisis Perkembangan Balita")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            # === Baca dan siapkan data ===
            df = pd.read_excel(uploaded_file)
            df = clean_and_prepare_df(df)

            # Pastikan mapping Posyandu diterapkan
            def map_posyandu(rt, rw):
                rt = str(rt).zfill(2)
                rw = str(rw).zfill(2)
                if rw == '06' and rt in ['01', '02', '03']:
                    return 'Larasati 1'
                elif rw in ['04', '05'] and rt in ['01', '02']:
                    return 'Larasati 2'
                elif rw in ['02', '03'] and rt in ['01', '02', '03']:
                    return 'Larasati 3'
                elif rw == '01' and rt in ['01', '02', '03']:
                    return 'Larasati 4'
                elif rw == '07' and rt in ['01', '02', '03']:
                    return 'Larasati 5'
                else:
                    return 'Lainnya'

            if 'RT' in df.columns and 'RW' in df.columns:
                df['Posyandu'] = df.apply(lambda row: map_posyandu(row['RT'], row['RW']), axis=1)
            else:
                df['Posyandu'] = "Tidak diketahui"

            st.success("File berhasil dibaca.")
            st.dataframe(df.head())

            # === Pilih mode analisis utama ===
            mode = st.radio(
                "Pilih Mode Analisis:",
                ["Analisis Per Anak", "Analisis Seluruh Balita"],
                horizontal=True
            )

            # === MODE 1: Analisis per Anak ===
            if mode == "Analisis Per Anak":
                if "Nama Anak" not in df.columns:
                    st.error("Kolom 'Nama Anak' tidak ditemukan.")
                else:
                    nama_pilih = st.selectbox("Pilih Nama Anak", df["Nama Anak"].dropna().unique())
                    jenis_analisis = st.radio("Pilih jenis analisis:", ["Per Tahun", "Seluruh Tanggal"])

                    # Ambil data anak
                    df_anak = df[df["Nama Anak"] == nama_pilih].sort_values("Tanggal Pengukuran")

                    # Info tambahan
                    nama_ibu = df_anak["Nama Ibu"].iloc[0] if "Nama Ibu" in df_anak.columns else "-"
                    posyandu = df_anak["Posyandu"].iloc[0] if "Posyandu" in df_anak.columns else "-"
                    st.markdown(f"**Nama Ibu Kandung:** {nama_ibu}  \n**Tempat Posyandu:** {posyandu}")

                    # === Analisis Per Tahun ===
                    if jenis_analisis == "Per Tahun":
                        tahun_pilih = st.number_input("Masukkan Tahun", 2000, 2100, datetime.now().year)
                        df_last = last_measurement_per_month(df)
                        df_tahun = df_last[
                            (df_last["Nama Anak"] == nama_pilih) &
                            (df_last["Periode_MonthStart"].dt.year == tahun_pilih)
                        ]

                        if df_tahun.empty:
                            st.info("Tidak ada pengukuran di tahun tersebut.")
                        else:
                            st.subheader(f"📋 Pengukuran Terakhir Tiap Bulan ({tahun_pilih})")
                            st.dataframe(df_tahun[["Periode_MonthStart", "Tanggal Pengukuran", "BB", "TB", "Z-Score BB/TB", "Status BB/TB"]])

                            # Grafik
                            st.subheader("📈 Grafik Perkembangan Z-Score BB/TB per Bulan")
                            series = df_tahun.set_index("Periode_MonthStart")["Z-Score BB/TB"]
                            st.line_chart(series)
                            st.info(interpret_trend(series))

                    # === Analisis Seluruh Tanggal ===
                    else:
                        st.subheader(f"📋 Semua Data Pengukuran — {nama_pilih}")
                        display_cols = [c for c in ["Tanggal Pengukuran", "BB", "TB", "Z-Score BB/TB", "Status BB/TB"] if c in df_anak.columns]
                        st.dataframe(df_anak[display_cols].reset_index(drop=True))

                        st.subheader("📈 Grafik Perkembangan Z-Score BB/TB (Seluruh Waktu)")
                        st.line_chart(df_anak.set_index("Tanggal Pengukuran")["Z-Score BB/TB"])
                        st.info(interpret_trend(df_anak["Z-Score BB/TB"]))

            # === MODE 2: Analisis Seluruh Balita ===
            elif mode == "Analisis Seluruh Balita":
                st.subheader("📊 Ringkasan Pengukuran Seluruh Balita")

                # Ambil data terakhir tiap anak
                df_last = last_measurement_per_month(df)
                df_terbaru = df_last.sort_values("Tanggal Pengukuran").groupby("Nama Anak").tail(1)

                if df_terbaru.empty:
                    st.info("Tidak ada data pengukuran yang dapat ditampilkan.")
                else:
                    # Tampilkan ringkasan umum
                    st.dataframe(df_terbaru[[
                        "Nama Anak", "Nama Ibu", "Posyandu", "Tanggal Pengukuran",
                        "BB", "TB", "Z-Score BB/TB", "Status BB/TB"
                    ]].reset_index(drop=True))

                    # Grafik distribusi status gizi
                    st.subheader("📈 Distribusi Status Gizi Terakhir")
                    status_counts = df_terbaru["Status BB/TB"].value_counts()
                    st.bar_chart(status_counts)

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    else:
        st.info("Silakan unggah file Excel terlebih dahulu.")
# ============================================================
# TAB 2 — Prediksi Manual
# ============================================================
with tab2:
    st.header("🧾 Prediksi Manual — Masukkan Pengukuran")
    with st.form("form_prediksi"):
        nama_input = st.text_input("Nama Balita")
        tanggal_input = st.date_input("Tanggal Pengukuran", value=datetime.today())
        z_bbtb = st.text_input("Z-Score BB/TB", "0")
        z_bbu = st.text_input("Z-Score BB/U", "0")
        z_tbu = st.text_input("Z-Score TB/U", "0")
        status_bbu = st.selectbox("Status BB/U", ["Sangat Kurang", "Kurang", "Normal", "Risiko Gizi Lebih"])
        status_tbu = st.selectbox("Status TB/U", ["Sangat Pendek", "Pendek", "Normal", "Tinggi"])
        submit_pred = st.form_submit_button("🔍 Prediksi")

    if submit_pred:
        try:
            z_bbtb, z_bbu, z_tbu = map(lambda x: float(str(x).replace(",", ".")), [z_bbtb, z_bbu, z_tbu])
        except:
            st.error("Pastikan Z-score diisi dengan angka.")
            st.stop()

        map_bbu = {'Sangat Kurang': 0, 'Kurang': 1, 'Normal': 2, 'Risiko Gizi Lebih': 3}
        map_tbu = {'Sangat Pendek': 0, 'Pendek': 1, 'Normal': 2, 'Tinggi': 3}

        X_input = pd.DataFrame([{
            'Z-Score BB/TB': z_bbtb,
            'Z-Score BB/U': z_bbu,
            'Z-Score TB/U': z_tbu,
            'Status BB/U (Encoded)': map_bbu[status_bbu],
            'Status TB/U (Encoded)': map_tbu[status_tbu],
        }])

        if model is not None:
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0]
            label_map = {0: "Gizi Buruk", 1: "Gizi Kurang", 2: "Gizi Baik", 3: "Risiko Gizi Lebih", 4: "Gizi Lebih", 5: "Obesitas"}
            st.success(f"Prediksi: **{label_map.get(pred, 'Tidak diketahui')}** (Probabilitas {prob[pred]:.2%})")
        else:
            st.warning("Model belum dimuat — prediksi otomatis tidak tersedia.")

# ============================================================
# TAB 3 — Histori
# ============================================================
with tab3:
    st.header("📚 Riwayat Prediksi")
    if os.path.exists("riwayat_prediksi.csv"):
        df_hist = pd.read_csv("riwayat_prediksi.csv")
        st.dataframe(df_hist)
        if st.button("🔄 Hapus Riwayat"):
            os.remove("riwayat_prediksi.csv")
            st.success("Riwayat dihapus.")
    else:
        st.info("Belum ada riwayat prediksi.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Catatan: Aplikasi ini adalah alat bantu. Interpretasi medis tetap perlu dikonsultasikan dengan tenaga kesehatan.")

