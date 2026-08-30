"""Generate Laporan Keuangan dari Kwitansi (OCR lokal)

Ekstraksi omset/pembelian/profit otomatis dari foto kwitansi, pakai model
OCR lokal LightOnOCR-2-1B lewat utils/kwitansi_extractor.py - modul yang
sama dipakai fitur "Upload Kwitansi (ZIP)" di
pages_v2/03_Pengajuan_Credit_Baru_Premium.py. Model di-load in-process
(@st.cache_resource), BUKAN lewat endpoint remote/ngrok - diasumsikan app
ini jalan di server/VM dgn GPU/RAM cukup, bukan Streamlit Community Cloud.

Beda dgn halaman 03 (upload 1 file .zip utk 1 nasabah yang SEDANG DIAJUKAN,
auto-fill 2 field Profil Finansial di form pengajuan), halaman ini upload
beberapa foto kwitansi individual (bukan zip) utk 1 nasabah yang SUDAH ADA
di sistem, tujuannya generate laporan omset/pembelian/profit per tahun utk
direview/didownload - bukan mengisi form pengajuan.
"""

import streamlit as st

from utils.data_loader import load_master_data
from utils.kwitansi_extractor import build_raw_text_export, extract_uploaded_files
from utils.ui_components import apply_logo
from utils.ui_premium import inject_css, hero_banner, section_header, kpi_card, rupiah_short

st.set_page_config(page_title="Generate Laporan Keuangan", page_icon="🧾", layout="wide")
apply_logo()
inject_css()

hero_banner(
    "Generate Laporan Keuangan",
    "Ekstraksi omset & profit otomatis dari kwitansi."
)

# ======================================================
# UPLOAD KWITANSI
# ======================================================

section_header("📤", "Upload Kwitansi")

df = load_master_data()
df["label"] = df["application_id"] + " — " + df["company_name"]

with st.container(border=True):
    u1, u2 = st.columns(2)
    with u1:
        selected_nasabah = st.selectbox("Nasabah", df["label"])
    with u2:
        tahun_laporan = st.selectbox("Tahun Laporan", [2025, 2024, 2023])

    uploaded_files = st.file_uploader(
        "Kwitansi (jpg/png) — bisa lebih dari 1, campur pembelian & penjualan",
        type=["jpg", "jpeg", "png"], accept_multiple_files=True,
    )

    run_extraction = st.button(
        f"🤖 Jalankan Ekstraksi ({len(uploaded_files) if uploaded_files else 0} file)",
        type="primary", disabled=not uploaded_files,
    )

# ======================================================
# JALANKAN EKSTRAKSI
# ======================================================

if run_extraction:
    with st.spinner(
        "Menjalankan OCR lokal (LightOnOCR-2-1B) — pemuatan model pertama kali bisa "
        "memakan waktu beberapa menit, proses berikutnya lebih cepat..."
    ):
        try:
            extracted_df = extract_uploaded_files(uploaded_files)
        except Exception as e:
            st.session_state.pop("_kwitansi_extracted_df", None)
            st.error(f"Gagal menjalankan ekstraksi: {type(e).__name__}: {e}")
        else:
            st.session_state["_kwitansi_extracted_df"] = extracted_df
            st.session_state["_kwitansi_nasabah"] = selected_nasabah
            st.session_state["_kwitansi_tahun"] = tahun_laporan

# ======================================================
# REVIEW & EDIT HASIL EKSTRAKSI
# ======================================================

if "_kwitansi_extracted_df" in st.session_state:

    section_header("🤖", "Review & Edit Hasil Ekstraksi")

    extracted_df = st.session_state["_kwitansi_extracted_df"]
    n_gagal = int(extracted_df["total"].isna().sum())
    if n_gagal:
        st.warning(
            f"⚠️ {n_gagal} dari {len(extracted_df)} kwitansi gagal terbaca lengkap (kolom "
            "'Nominal' kosong) — koreksi manual di tabel di bawah, atau baris itu tidak "
            "akan dihitung ke ringkasan."
        )

    st.download_button(
        "⬇️ Download Teks OCR Mentah (.txt)",
        data=build_raw_text_export(extracted_df),
        file_name=f"raw_ocr_kwitansi_{st.session_state.get('_kwitansi_tahun', '')}.txt",
        mime="text/plain",
        help="Transkripsi OCR mentah per file, sebelum di-parse regex — dipakai untuk "
             "cek/koreksi kalau ada field yang salah baca atau kosong.",
    )

    with st.container(border=True):
        display_df = extracted_df.drop(columns=["raw_text"]).rename(columns={
            "source_file": "File", "no_kwitansi": "No. Kwitansi", "jenis_kwitansi": "Jenis",
            "tanggal": "Tanggal", "pihak_terkait": "Pihak Terkait", "nama_usaha": "Nama Usaha",
            "nik_pemilik": "NIK Pemilik", "total": "Nominal", "catatan": "Catatan / Teks OCR",
        })
        edited = st.data_editor(
            display_df,
            use_container_width=True, hide_index=True, key="_kwitansi_editor_06",
            disabled=["File", "Catatan / Teks OCR"],
            column_config={
                "Jenis": st.column_config.SelectboxColumn(options=["penjualan", "pembelian"]),
                "Nominal": st.column_config.NumberColumn(min_value=0, step=1000),
                "Catatan / Teks OCR": st.column_config.TextColumn(width="large"),
            },
        )

    # ======================================================
    # RINGKASAN LAPORAN
    # ======================================================

    section_header("📌", "Ringkasan Laporan")

    valid = edited[edited["Nominal"].notna()]
    total_penjualan = valid.loc[valid["Jenis"] == "penjualan", "Nominal"].sum()
    total_pembelian = valid.loc[valid["Jenis"] == "pembelian", "Nominal"].sum()
    estimasi_profit = total_penjualan - total_pembelian

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Total Penjualan", rupiah_short(total_penjualan), "🟢")
    with c2:
        kpi_card("Total Pembelian", rupiah_short(total_pembelian), "🔴")
    with c3:
        kpi_card("Estimasi Profit", rupiah_short(estimasi_profit), "💰")

    st.caption(
        f"Nasabah: **{st.session_state.get('_kwitansi_nasabah', '-')}** • "
        f"Tahun: **{st.session_state.get('_kwitansi_tahun', '-')}**"
    )

    st.download_button(
        "⬇️ Download Hasil (CSV)",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name=f"laporan_keuangan_kwitansi_{st.session_state.get('_kwitansi_tahun','')}.csv",
        mime="text/csv",
    )

    st.caption(
        "Catatan: tombol download di atas ekspor hasil review ke CSV. Kalau nanti mau "
        "langsung nulis ke `laporan_keuangan.csv` / database, kasih tau format & lokasi "
        "penyimpanannya biar aku sambungin — belum aku asumsiin sendiri di sini."
    )
