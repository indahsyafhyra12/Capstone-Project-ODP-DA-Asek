"""Generate Laporan Keuangan dari Kwitansi (VLM Extraction)

Ekstraksi omset/profit otomatis dari foto kwitansi pembelian & penjualan.
Model VLM (LightOnOCR-2-1B / dsb.) dijalankan TERPISAH di Colab (GPU),
di-expose lewat ngrok jadi endpoint publik — halaman ini cuma manggil
endpoint itu lewat HTTP, TIDAK load model apapun secara lokal. Ini
penting karena app di-deploy ke Streamlit Community Cloud yang RAM/CPU-nya
ga cukup buat inference VLM langsung.

Kontrak API (disepakati, implementasi endpoint dikerjakan terpisah):

  POST {endpoint_url}/extract
  Request: multipart/form-data, field "file" = gambar kwitansi (jpg/png)
  Response (JSON):
    {
      "success": true,
      "jenis": "penjualan" | "pembelian",
      "tanggal": "2025-03-12",       # ISO YYYY-MM-DD
      "nominal": 2400000,             # integer rupiah
      "deskripsi": "...",             # ringkasan singkat, opsional
      "confidence": 0.92,             # 0-1
      "error": null                   # null kalau sukses
    }

  1 request = 1 gambar (bukan batch dalam satu request) — Streamlit yang
  loop kalau user upload banyak file sekaligus.
"""

import requests
import pandas as pd
import streamlit as st

from utils.data_loader import load_master_data
from utils.ui_components import apply_logo
from utils.ui_premium import inject_css, hero_banner, section_header, chart_title, kpi_card, rupiah_short

st.set_page_config(page_title="Generate Laporan Keuangan", page_icon="🧾", layout="wide")
apply_logo()
inject_css()

hero_banner(
    "Generate Laporan Keuangan",
    "Ekstraksi omset & profit otomatis dari kwitansi menggunakan VLM."
)

CONFIDENCE_THRESHOLD = 0.75  # di bawah ini, baris di-flag "perlu dicek manual"

# ======================================================
# KONFIGURASI ENDPOINT
# ======================================================

section_header("🔌", "Konfigurasi Endpoint")

with st.container(border=True):
    e1, e2, e3 = st.columns([3, 1, 1.2])

    with e1:
        endpoint_url = st.text_input(
            "Ngrok URL (dari Colab)",
            value=st.session_state.get("_vlm_endpoint", ""),
            placeholder="https://xxxx.ngrok-free.app",
        ).rstrip("/")
        st.session_state["_vlm_endpoint"] = endpoint_url

    with e2:
        st.write("")
        st.write("")
        test_clicked = st.button("Test Koneksi", use_container_width=True)

    with e3:
        st.write("")
        st.write("")
        if test_clicked and endpoint_url:
            try:
                r = requests.get(f"{endpoint_url}/health", timeout=5)
                if r.ok:
                    st.session_state["_vlm_connected"] = True
                    st.success("● Terhubung")
                else:
                    st.session_state["_vlm_connected"] = False
                    st.error(f"● Gagal ({r.status_code})")
            except Exception as e:
                st.session_state["_vlm_connected"] = False
                st.error(f"● {type(e).__name__}")
        elif st.session_state.get("_vlm_connected"):
            st.success("● Terhubung")
        else:
            st.warning("● Belum dites")

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
    if not endpoint_url:
        st.error("Isi Ngrok URL dulu di atas sebelum jalanin ekstraksi.")
    else:
        results = []
        progress = st.progress(0.0, text="Memproses...")
        for i, f in enumerate(uploaded_files):
            try:
                resp = requests.post(
                    f"{endpoint_url}/extract",
                    files={"file": (f.name, f.getvalue(), f.type)},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("success"):
                    raise ValueError(data.get("error") or "Ekstraksi gagal tanpa detail error.")
                results.append({
                    "file": f.name,
                    "jenis": data.get("jenis", "-"),
                    "tanggal": data.get("tanggal", "-"),
                    "nominal": data.get("nominal", 0),
                    "deskripsi": data.get("deskripsi", ""),
                    "confidence": data.get("confidence", 0.0),
                    "error": None,
                })
            except Exception as e:
                results.append({
                    "file": f.name, "jenis": "-", "tanggal": "-", "nominal": 0,
                    "deskripsi": "", "confidence": 0.0, "error": str(e),
                })
            progress.progress((i + 1) / len(uploaded_files), text=f"Memproses {i + 1}/{len(uploaded_files)}...")
        progress.empty()
        st.session_state["_kwitansi_results"] = results
        st.session_state["_kwitansi_nasabah"] = selected_nasabah
        st.session_state["_kwitansi_tahun"] = tahun_laporan

# ======================================================
# REVIEW & EDIT HASIL EKSTRAKSI
# ======================================================

if "_kwitansi_results" in st.session_state:

    section_header("🤖", "Review & Edit Hasil Ekstraksi")

    results = st.session_state["_kwitansi_results"]
    n_error = sum(1 for r in results if r["error"])
    n_low_conf = sum(1 for r in results if not r["error"] and r["confidence"] < CONFIDENCE_THRESHOLD)

    if n_error:
        st.error(f"{n_error} file gagal diekstrak — cek kolom Error di tabel, upload ulang kalau perlu.")
    if n_low_conf:
        st.warning(f"{n_low_conf} baris confidence-nya di bawah {CONFIDENCE_THRESHOLD:.0%} — cek & koreksi manual sebelum disimpan.")

    with st.container(border=True):
        edit_df = pd.DataFrame(results)
        edited = st.data_editor(
            edit_df.rename(columns={
                "file": "File", "jenis": "Jenis", "tanggal": "Tanggal", "nominal": "Nominal",
                "deskripsi": "Deskripsi", "confidence": "Confidence", "error": "Error",
            }),
            use_container_width=True, hide_index=True,
            column_config={
                "Jenis": st.column_config.SelectboxColumn(options=["penjualan", "pembelian", "-"]),
                "Nominal": st.column_config.NumberColumn(min_value=0, step=1000),
                "Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.0f%%"),
            },
            disabled=["File", "Confidence", "Error"],
        )

    # ======================================================
    # RINGKASAN LAPORAN
    # ======================================================

    section_header("📌", "Ringkasan Laporan")

    valid = edited[edited["Error"].isna()] if "Error" in edited.columns else edited
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