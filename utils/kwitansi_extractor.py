"""Kwitansi Extractor — OCR ekstraksi kwitansi penjualan/pembelian dari file
.zip pakai model VLM lokal LightOnOCR-2-1B, dipakai fitur "Upload Kwitansi
(ZIP)" di pages_v2/03_Pengajuan_Credit_Baru_Premium.py untuk auto-fill
section Profil Finansial.

Dipindah dari Resources_Pendukung/extract_kwitansi.py (dulu script CLI
mandiri), loading model dipisah jadi _load_model() (lazy import torch/
transformers + @st.cache_resource, mengikuti pola
utils/report_agent.py::_load_model()) supaya Streamlit tidak reload model
~2GB tiap interaksi, dan modul ini tetap bisa di-import di environment tanpa
GPU/torch/transformers terpasang. CLI (`python -m utils.kwitansi_extractor
--zip kwitansi.zip`) tetap tersedia dengan output yang sama seperti sebelumnya.

Ekstraksi 2 tahap (BUKAN one-shot custom JSON seperti versi awal - diverifikasi
gagal 100% di data uji nyata, lihat di bawah):
  1. Model diberi prompt OCR generik "Extract all the text from this image." -
     SATU-SATUNYA gaya prompt yang didemonstrasikan berhasil di
     Resources_Pendukung/vlm-ocr-image.ipynb (notebook referensi resmi utk
     model ini). LightOnOCR-2-1B adalah model OCR/transkripsi murni, BUKAN
     VLM instruction-following - versi awal script ini memintanya keluarkan
     JSON custom schema (translate label, format tanggal ISO, dst.), yang
     ternyata diabaikan model itu: pada uji dgn 20 foto kwitansi asli, SEMUA
     gagal (regex pencarian blok JSON tidak pernah ketemu di output karena
     model cuma transkripsi teks polos, bukan JSON) - bukan gagal sebagian
     yang wajar dari noise OCR, tapi gagal total yang menandakan prompt
     salah total.
  2. _parse_receipt_text() mem-parse teks transkripsi itu dgn regex sesuai
     format kwitansi yang dipakai (lihat contoh kwitansi yg dishare user) -
     "KWITANSI PENJUALAN/PEMBELIAN", "No. Kwitansi :", "Tanggal :",
     "Kepada :"/"Dibeli dari :", "NIK Pemilik:", baris pertama = nama usaha,
     "TOTAL ... Rp X" (word-boundary \bTOTAL\b supaya tidak ketarik ke baris
     "Subtotal" per item). Field yang regex-nya tidak ketemu -> None (bukan
     dikira-kira), teks transkripsi mentah selalu disimpan di kolom "catatan"
     supaya user bisa cross-check/koreksi manual di preview kalau regexnya
     meleset (mis. field custom yang tidak ada di 4 contoh kwitansi ini).

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
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_ID = "lightonai/LightOnOCR-2-1B"

_LOG_PATH = Path(__file__).resolve().parents[1] / "log" / "log_kwitansi.txt"


def debug_print(message: str) -> None:
    """Tampilkan pesan debugging dan simpan salinannya ke file log."""
    print(message)
    # _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # with _LOG_PATH.open("a", encoding="utf-8") as log_file:
        # log_file.write(f"{message}\n")


def log_variable(name: str, value) -> None:
    """Tampilkan nama dan nilai variabel untuk debugging."""
    debug_print(f"variable {name} = {value!r}")

# Prompt OCR generik - lihat docstring modul utk alasan kenapa BUKAN prompt
# custom JSON schema (model ini mengabaikannya, gagal 100% di uji nyata).
EXTRACTION_PROMPT = "Extract all the text from this image."

DETAIL_COLUMNS = [
    "source_file", "no_kwitansi", "jenis_kwitansi", "tanggal",
    "pihak_terkait", "nama_usaha", "nik_pemilik", "total", "catatan", "raw_text",
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
    debug_print(f"OCR model loaded: model_id={MODEL_ID} device={device} dtype={dtype}")
    return model, processor, device, dtype


def extract_one(image_path: Path, model, processor, device, dtype) -> dict:
    """Ekstrak 1 foto kwitansi: jalankan OCR generik (EXTRACTION_PROMPT) lalu
    strukturkan hasil transkripsinya jadi field kwitansi lewat
    _parse_receipt_text(). Selalu berhasil return dict (tidak exception per
    file) - field yang gagal di-regex jadi None, teks transkripsi mentah
    tetap dibawa di key "raw_text" utk audit/koreksi manual di preview."""
    from PIL import Image

    debug_print(f"extract_one started: file={image_path}")
    image = Image.open(image_path).convert("RGB")
    log_variable("image.size", image.size)

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
    log_variable("prompt", prompt)
    log_variable("input_shapes", {key: tuple(value.shape) for key, value in inputs.items() if hasattr(value, "shape")})

    output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
    print("\n===== RAW_TEXT OCR START =====")
    print(raw_text)
    print("===== RAW_TEXT OCR END =====\n")
    log_variable("raw_text", raw_text)

    parsed = _parse_receipt_text(raw_text)
    parsed["source_file"] = image_path.name
    parsed["raw_text"] = raw_text
    log_variable("parsed_result", parsed)
    debug_print(f"extract_one finished: file={image_path.name}")
    return parsed


_JENIS_RE = re.compile(r"\b(PENJUALAN|PEMBELIAN)\b", re.IGNORECASE)
_NO_KWITANSI_RE = re.compile(r"No\.?\s*Kwitansi\s*[:=]?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE)
_TANGGAL_RE = re.compile(r"Tanggal\s*[:=]?\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_TANGGAL_FALLBACK_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_PIHAK_RE = re.compile(r"(?:Kepada|Dibeli\s*dari)\s*[:=]?\s*(.+)", re.IGNORECASE)
_NIK_RE = re.compile(r"NIK\s*Pemilik\s*[:=]?\s*(\d+)", re.IGNORECASE)
_TOTAL_RE = re.compile(r"\bTOTAL\b\s*[:=\-]?\s*Rp\.?\s*([\d.,]+)", re.IGNORECASE)
_SIGNATURE_NAME_RE = re.compile(r"\(\s*([^)\n]+?)\s*\)\s*$")


def _parse_receipt_text(raw_text: str) -> dict:
    """Strukturkan hasil transkripsi OCR polos (EXTRACTION_PROMPT) jadi field
    kwitansi, mengandalkan format kwitansi yang konsisten (lihat docstring
    modul) - bukan LLM/model kedua, murni regex supaya deterministik & tidak
    perlu model tambahan. Field yang tidak ketemu -> None."""
    text = raw_text or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    log_variable("parse.lines", lines)

    jenis_match = _JENIS_RE.search(text)
    jenis_kwitansi = jenis_match.group(1).lower() if jenis_match else None

    tanggal_match = _TANGGAL_RE.search(text) or _TANGGAL_FALLBACK_RE.search(text)
    tanggal = tanggal_match.group(1) if tanggal_match else None

    pihak_match = _PIHAK_RE.search(text)
    pihak_terkait = pihak_match.group(1).strip() if pihak_match else None

    nik_match = _NIK_RE.search(text)
    nik_pemilik = nik_match.group(1) if nik_match else None

    no_kwitansi_match = _NO_KWITANSI_RE.search(text)
    no_kwitansi = no_kwitansi_match.group(1) if no_kwitansi_match else None

    total = None
    total_match = _TOTAL_RE.search(text)
    if total_match:
        digits = re.sub(r"[^\d]", "", total_match.group(1))
        if digits:
            total = int(digits)

    nama_usaha = lines[0] if lines else None
    if nama_usaha and _JENIS_RE.search(nama_usaha):
        # Baris pertama ternyata bukan nama usaha (mis. OCR lewatkan header) -
        # coba ambil dari tanda tangan "( Nama Usaha )" di baris terakhir.
        sig_match = _SIGNATURE_NAME_RE.search(lines[-1]) if lines else None
        nama_usaha = sig_match.group(1) if sig_match else None

    parsed = {
        "nama_usaha": nama_usaha, "nik_pemilik": nik_pemilik,
        "jenis_kwitansi": jenis_kwitansi, "no_kwitansi": no_kwitansi,
        "tanggal": tanggal, "pihak_terkait": pihak_terkait, "total": total,
    }
    log_variable("parse.result", parsed)
    return parsed


def _to_detail_row(result: dict) -> dict:
    raw_text = result.get("raw_text", "") or ""
    missing = [
        label for key, label in [
            ("total", "total"), ("tanggal", "tanggal"), ("jenis_kwitansi", "jenis"),
        ] if not result.get(key)
    ]
    catatan = f"Tidak terbaca: {', '.join(missing)}. " if missing else ""
    catatan += f"Teks OCR: {raw_text[:200]}"
    row = {
        "source_file": result.get("source_file"),
        "no_kwitansi": result.get("no_kwitansi"),
        "jenis_kwitansi": result.get("jenis_kwitansi"),
        "tanggal": result.get("tanggal"),
        "pihak_terkait": result.get("pihak_terkait"),
        "nama_usaha": result.get("nama_usaha"),
        "nik_pemilik": result.get("nik_pemilik"),
        "total": result.get("total"),
        "catatan": catatan,
        "raw_text": raw_text,
    }
    log_variable("detail_row", row)
    return row


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
    debug_print(f"extract_zip_bytes started: zip_bytes={len(zip_bytes)} bytes")
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
        log_variable("image_files", [str(path) for path in image_files])
        if not image_files:
            raise ValueError("Tidak ada file gambar (.jpg/.jpeg/.png) di dalam ZIP.")

        rows = [
            _to_detail_row(extract_one(path, model, processor, device, dtype))
            for path in image_files
        ]

    result_df = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    debug_print(
        f"extract_zip_bytes finished: rows={len(result_df)} columns={list(result_df.columns)}"
    )
    log_variable("result_df", result_df.to_dict(orient="records"))
    return result_df


def build_raw_text_export(detail_df: pd.DataFrame) -> str:
    """Gabungkan kolom "raw_text" (transkripsi OCR mentah, sebelum di-regex)
    semua baris jadi 1 file .txt, dipakai tombol download di halaman
    Streamlit supaya user bisa cek/kirim persis apa yang dikeluarkan model -
    dibutuhkan tiap kali regex _parse_receipt_text() perlu disetel ulang
    krn format kwitansi/transkripsi ternyata beda dari asumsi."""
    if detail_df is None or detail_df.empty:
        return ""
    blocks = []
    for _, row in detail_df.iterrows():
        blocks.append(
            f"===== {row.get('source_file')} =====\n"
            f"{row.get('raw_text') or ''}\n"
        )
    return "\n".join(blocks)


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
    log_variable(
        "compute_monthly_estimates.input",
        None if detail_df is None else detail_df.to_dict(orient="records"),
    )
    if detail_df is None or detail_df.empty:
        log_variable("compute_monthly_estimates.result", empty_result)
        return empty_result

    valid = detail_df[
        (detail_df["jenis_kwitansi"] == "penjualan")
        & detail_df["total"].notna()
        & detail_df["tanggal"].notna()
    ].copy()
    if valid.empty:
        log_variable("compute_monthly_estimates.result", empty_result)
        return empty_result

    valid["total"] = pd.to_numeric(valid["total"], errors="coerce")
    valid["tanggal_dt"] = pd.to_datetime(valid["tanggal"], errors="coerce")
    valid = valid[valid["tanggal_dt"].notna() & valid["total"].notna()]
    if valid.empty:
        log_variable("compute_monthly_estimates.result", empty_result)
        return empty_result

    valid["tahun"] = valid["tanggal_dt"].dt.year
    year_used = int(valid["tahun"].max())
    year_df = valid[valid["tahun"] == year_used].copy()
    year_df["bulan"] = year_df["tanggal_dt"].dt.to_period("M")
    n_months = int(year_df["bulan"].nunique())
    if n_months == 0:
        log_variable("compute_monthly_estimates.result", empty_result)
        return empty_result

    estimates = {
        "monthly_turnover_est": round(float(year_df["total"].sum()) / n_months),
        "transaction_frequency_monthly": round(len(year_df) / n_months),
        "n_months": n_months,
        "year_used": year_used,
    }
    log_variable("compute_monthly_estimates.valid", valid.to_dict(orient="records"))
    log_variable("compute_monthly_estimates.result", estimates)
    return estimates


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

    n_errors = int(df["total"].isna().sum())
    if n_errors:
        print(f"\nPeringatan: {n_errors} kwitansi gagal di-parse (total tidak terbaca), cek kolom 'catatan' di {args.output_csv}.")


if __name__ == "__main__":
    main()
