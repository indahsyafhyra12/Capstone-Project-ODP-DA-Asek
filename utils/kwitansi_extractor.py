"""Kwitansi Extractor — OCR ekstraksi kwitansi penjualan/pembelian dari file
.zip pakai model VLM lokal LightOnOCR-2-1B, dipakai fitur "Upload Kwitansi
(ZIP)" di pages_v2/03_Pengajuan_Credit_Baru_Premium.py untuk auto-fill
section Profil Finansial.

Dipindah dari Resources_Pendukung/extract_kwitansi.py (dulu script CLI
mandiri) jadi modul reusable - EXTRACTION_PROMPT dan logika parsing JSON per
kwitansi TIDAK diubah, cuma loading model dipisah jadi _load_model() (lazy
import torch/transformers + @st.cache_resource, mengikuti pola
utils/report_agent.py::_load_model()) supaya Streamlit tidak reload model
~2GB tiap interaksi, dan modul ini tetap bisa di-import di environment tanpa
GPU/torch/transformers terpasang. CLI (`python -m utils.kwitansi_extractor
--zip kwitansi.zip`) tetap tersedia dengan output yang sama seperti sebelumnya.

Catatan skema (lihat utils/feature_builder.py::build_features_from_raw): 4
field Profil Finansial di form hanya 2 yang murni raw ML feature yang
sumbernya form itu sendiri - monthly_turnover_est & transaction_frequency_
monthly. loan_requested murni keputusan pengaju (tidak diautofill).
estimated_dsr butuh info cicilan existing yang tidak ada di kwitansi (tidak
diautofill). revenue_growth_pct/profit_margin_2025/revenue_*/net_profit_*
BUKAN field form sama sekali - itu di-lookup otomatis dari
data/raw/laporan_keuangan.csv by NIK di build_features_from_raw(), di luar
jangkauan halaman ini; kwitansi juga tidak bisa mengisi total_asset/
total_liability/operating_cashflow tanpa data palsu, jadi sengaja TIDAK
disuntikkan ke pipeline ML (lih. diskusi scope sebelum implementasi).
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_ID = "lightonai/LightOnOCR-2-1B"

EXTRACTION_PROMPT = """Baca kwitansi di gambar ini dan keluarkan HANYA JSON valid
(tanpa teks lain, tanpa markdown code fence) dengan schema persis:

{
  "nama_usaha": string,
  "nik_pemilik": string,
  "jenis_kwitansi": "penjualan" | "pembelian",
  "no_kwitansi": string,
  "tanggal": "YYYY-MM-DD",
  "pihak_terkait": string,
  "total": number
}

Aturan:
- Nilai "total" HARUS angka murni (integer), buang "Rp" dan pemisah ribuan.
- jenis_kwitansi: "KWITANSI PENJUALAN" -> "penjualan", "KWITANSI PEMBELIAN" -> "pembelian".
- Jika field tidak terbaca, isi null.
- Output HARUS JSON valid, tanpa teks pembuka/penutup, tanpa ```.
"""

DETAIL_COLUMNS = [
    "source_file", "no_kwitansi", "jenis_kwitansi", "tanggal",
    "pihak_terkait", "nama_usaha", "nik_pemilik", "total", "catatan",
]


@st.cache_resource(show_spinner=False)
def _load_model():
    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(device)
    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
    return model, processor, device, dtype


def extract_one(image_path: Path, model, processor, device, dtype) -> dict:
    """Ekstrak 1 foto kwitansi jadi dict field mentah (schema EXTRACTION_PROMPT)
    + "source_file". Kalau model gagal keluarkan JSON valid, dict berisi
    "error"/"raw_output" alih-alih field kwitansi (bukan exception - dipakai
    supaya kwitansi lain di zip yang sama tetap diproses)."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device, dtype=dtype)
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

    output_text = output_text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    json_str = match.group(0) if match else output_text

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = {"error": "gagal parse JSON", "raw_output": output_text}

    parsed["source_file"] = image_path.name
    return parsed


def _to_detail_row(result: dict) -> dict:
    if "error" in result:
        return {
            "source_file": result.get("source_file"), "no_kwitansi": None,
            "jenis_kwitansi": None, "tanggal": None, "pihak_terkait": None,
            "nama_usaha": None, "nik_pemilik": None, "total": None,
            "catatan": f"{result['error']}: {str(result.get('raw_output', ''))[:200]}",
        }
    return {
        "source_file": result.get("source_file"),
        "no_kwitansi": result.get("no_kwitansi"),
        "jenis_kwitansi": result.get("jenis_kwitansi"),
        "tanggal": result.get("tanggal"),
        "pihak_terkait": result.get("pihak_terkait"),
        "nama_usaha": result.get("nama_usaha"),
        "nik_pemilik": result.get("nik_pemilik"),
        "total": result.get("total"),
        "catatan": "" if result.get("total") is not None else "Field tidak terbaca oleh OCR",
    }


def extract_zip_bytes(zip_bytes: bytes) -> pd.DataFrame:
    """Dipakai halaman Streamlit: terima isi file .zip (dari
    st.file_uploader.getvalue()), jalankan OCR ke semua foto di dalamnya,
    kembalikan 1 baris per kwitansi (kolom DETAIL_COLUMNS) untuk preview/
    koreksi manual SEBELUM dipakai isi form.

    Model di-load sekali lewat @st.cache_resource - baris pertama yang berat
    ada di sini, sebaiknya dipanggil di dalam st.spinner() oleh caller.

    Raises RuntimeError kalau model gagal dimuat, ValueError kalau zip tidak
    berisi foto sama sekali - caller diharapkan menampilkan pesan error dan
    TIDAK mengubah field form yang sudah ada (lihat docstring modul).
    """
    try:
        model, processor, device, dtype = _load_model()
    except Exception as e:
        raise RuntimeError(f"Model OCR gagal dimuat ({type(e).__name__}): {e}") from e

    with tempfile.TemporaryDirectory(prefix="kwitansi_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        zip_path = tmp_dir / "upload.zip"
        zip_path.write_bytes(zip_bytes)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
        except zipfile.BadZipFile as e:
            raise ValueError(f"File bukan ZIP yang valid: {e}") from e

        image_files = sorted(
            p for p in tmp_dir.rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"] and not p.name.startswith(".")
        )
        if not image_files:
            raise ValueError("Tidak ada file gambar (.jpg/.jpeg/.png) di dalam ZIP.")

        rows = [
            _to_detail_row(extract_one(path, model, processor, device, dtype))
            for path in image_files
        ]

    return pd.DataFrame(rows, columns=DETAIL_COLUMNS)


def compute_monthly_estimates(detail_df: pd.DataFrame) -> dict:
    """Dari DataFrame hasil ekstraksi (kolom DETAIL_COLUMNS, sudah dikoreksi
    manual user di preview kalau perlu), hitung 2 field Profil Finansial yang
    murni raw ML feature dari form itu sendiri:

      - monthly_turnover_est = total kwitansi "penjualan" di tahun terakhir
        yang datanya ada, dibagi jumlah BULAN yang datanya tersedia di tahun
        itu (bukan selalu 12).
      - transaction_frequency_monthly = jumlah kwitansi "penjualan" di tahun
        itu, dibagi jumlah bulan yang sama.

    Field lain (loan_requested, estimated_dsr, revenue_growth_pct/profit_
    margin/dst.) sengaja TIDAK dihitung di sini - lihat docstring modul.

    Return dict {"monthly_turnover_est", "transaction_frequency_monthly",
    "n_months", "year_used"} - 2 field pertama None (BUKAN 0) kalau tidak
    ada kwitansi penjualan valid yang bisa dihitung, supaya caller tahu utk
    tidak mengisi form dengan angka yang menyesatkan.
    """
    empty_result = {
        "monthly_turnover_est": None, "transaction_frequency_monthly": None,
        "n_months": 0, "year_used": None,
    }
    if detail_df is None or detail_df.empty:
        return empty_result

    valid = detail_df[
        (detail_df["jenis_kwitansi"] == "penjualan")
        & detail_df["total"].notna()
        & detail_df["tanggal"].notna()
    ].copy()
    if valid.empty:
        return empty_result

    valid["total"] = pd.to_numeric(valid["total"], errors="coerce")
    valid["tanggal_dt"] = pd.to_datetime(valid["tanggal"], errors="coerce")
    valid = valid[valid["tanggal_dt"].notna() & valid["total"].notna()]
    if valid.empty:
        return empty_result

    valid["tahun"] = valid["tanggal_dt"].dt.year
    year_used = int(valid["tahun"].max())
    year_df = valid[valid["tahun"] == year_used].copy()
    year_df["bulan"] = year_df["tanggal_dt"].dt.to_period("M")
    n_months = int(year_df["bulan"].nunique())
    if n_months == 0:
        return empty_result

    return {
        "monthly_turnover_est": round(float(year_df["total"].sum()) / n_months),
        "transaction_frequency_monthly": round(len(year_df) / n_months),
        "n_months": n_months,
        "year_used": year_used,
    }


# ---------------------------------------------------------------------------
# CLI - identik dengan Resources_Pendukung/extract_kwitansi.py sebelumnya:
# hitung omset & profit per nasabah (nik_pemilik) per tahun dari 1 zip
# kwitansi campuran, simpan ke 2 CSV (detail + ringkasan per nasabah).
# ---------------------------------------------------------------------------

def _summarize_omset_profit_by_year(df: pd.DataFrame) -> pd.DataFrame:
    df_valid = df[df["total"].notna() & df["tanggal"].notna()].copy() if not df.empty else df
    if df_valid.empty:
        return pd.DataFrame()

    df_valid["total"] = pd.to_numeric(df_valid["total"], errors="coerce")
    df_valid["tahun"] = pd.to_datetime(df_valid["tanggal"], errors="coerce").dt.year
    df_valid = df_valid[df_valid["tahun"].notna()]
    df_valid["tahun"] = df_valid["tahun"].astype(int)

    grouped = (
        df_valid.groupby(["nik_pemilik", "tahun", "jenis_kwitansi"])["total"]
        .sum()
        .unstack(fill_value=0)
    )
    for col in ["penjualan", "pembelian"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["omset"] = grouped["penjualan"]
    grouped["profit"] = grouped["penjualan"] - grouped["pembelian"]
    grouped = grouped.reset_index()

    nama_usaha_map = df_valid.groupby("nik_pemilik")["nama_usaha"].agg(
        lambda s: s.dropna().iloc[0] if s.notna().any() else None
    )

    summary = grouped.pivot(index="nik_pemilik", columns="tahun", values=["omset", "profit"])
    summary.columns = [f"{metric}_{int(year)}" for metric, year in summary.columns]
    summary = summary.fillna(0).reset_index()
    summary.insert(1, "nama_usaha", summary["nik_pemilik"].map(nama_usaha_map))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="File .zip berisi foto kwitansi (.jpg/.png)")
    ap.add_argument("--output_csv", default="hasil_kwitansi.csv")
    ap.add_argument("--summary_csv", default="ringkasan_omset_profit.csv")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"File zip tidak ditemukan: {zip_path}")
        return

    df = extract_zip_bytes(zip_path.read_bytes())
    df.to_csv(args.output_csv, index=False)
    print(f"\nHasil ekstraksi disimpan ke: {args.output_csv}")

    summary = _summarize_omset_profit_by_year(df)
    if summary.empty:
        print("Tidak ada kwitansi yang berhasil diekstrak dengan lengkap.")
        return
    summary.to_csv(args.summary_csv, index=False)

    print(f"\nHasil ringkasan per nasabah per tahun disimpan ke: {args.summary_csv}")
    print("\n===== OMSET & PROFIT PER NASABAH PER TAHUN =====")
    for _, row in summary.iterrows():
        print(f"NIK {row['nik_pemilik']} ({row['nama_usaha']}):")
        for col in summary.columns:
            if col.startswith("omset_") or col.startswith("profit_"):
                print(f"  {col:<12}: Rp {row[col]:,.0f}")

    n_errors = int(df["catatan"].astype(bool).sum())
    if n_errors:
        print(f"\nPeringatan: {n_errors} kwitansi gagal di-parse, cek kolom 'catatan' di {args.output_csv}.")


if __name__ == "__main__":
    main()
