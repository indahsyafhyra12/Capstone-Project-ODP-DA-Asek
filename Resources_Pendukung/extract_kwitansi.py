"""
extract_kwitansi.py
Ekstraksi data kwitansi (pembelian & penjualan) dari foto (di dalam file .zip)
menggunakan model VLM lokal LightOnOCR-2-1B (sama seperti di notebook contoh),
lalu hitung Omset dan Profit per nasabah, per tahun.

Cara pakai (butuh GPU untuk kecepatan wajar, tapi bisa jalan di CPU juga):
    1. pip install -q git+https://github.com/huggingface/transformers pillow torch pandas
    2. Siapkan satu file .zip berisi foto kwitansi (.jpg/.png), boleh campur
       kwitansi dari beberapa nasabah/usaha sekaligus.
    3. python extract_kwitansi.py --zip kwitansi.zip

Output:
    - hasil_kwitansi.csv          -> detail per kwitansi (untuk audit/debug)
    - ringkasan_omset_profit.csv  -> satu baris per nasabah (nik_pemilik), dengan
                                      kolom omset_<tahun> dan profit_<tahun>
    - ringkasan juga dicetak ke terminal
"""

import os
import re
import json
import argparse
import zipfile
import tempfile
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

# ---------------------------------------------------------------------------
# LOAD MODEL (sama seperti di notebook contoh)
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

print(f"Loading LightOnOCR-2-1B di device={DEVICE} ...")
model = LightOnOcrForConditionalGeneration.from_pretrained(
    "lightonai/LightOnOCR-2-1B", torch_dtype=DTYPE
).to(DEVICE)
processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")


# ---------------------------------------------------------------------------
# PROMPT EKSTRAKSI
# Sama fungsinya seperti prompt "Extract all the text from this image." di
# notebook contoh, tapi diarahkan supaya modelnya keluarkan JSON terstruktur
# sesuai kebutuhan kita (bukan cuma teks OCR mentah).
# ---------------------------------------------------------------------------
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


def extract_one(image_path: Path) -> dict:
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

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE, dtype=DTYPE)
    inputs = {
        k: v.to(device=DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(DEVICE)
        for k, v in inputs.items()
    }

    output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

    # Bersihkan kalau model tetap membungkus dengan ```json ... ```
    output_text = output_text.replace("```json", "").replace("```", "").strip()

    # Kalau ada teks tambahan di luar JSON, coba ambil blok {...} pertama
    match = re.search(r"\{.*\}", output_text, re.DOTALL)
    json_str = match.group(0) if match else output_text

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = {"error": "gagal parse JSON", "raw_output": output_text}

    parsed["source_file"] = image_path.name
    return parsed


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

    tmp_dir = Path(tempfile.mkdtemp(prefix="kwitansi_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    image_files = sorted(
        p for p in tmp_dir.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"] and not p.name.startswith(".")
    )

    if not image_files:
        print(f"Tidak ada file gambar di dalam {zip_path}")
        return

    print(f"Ditemukan {len(image_files)} kwitansi di dalam {zip_path.name}")

    results = []
    for i, path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Memproses {path.name} ...")
        results.append(extract_one(path))

    # ---- Simpan hasil detail per kwitansi ----
    rows = []
    for r in results:
        if "error" in r:
            rows.append(r)
            continue
        rows.append(
            {
                "source_file": r.get("source_file"),
                "no_kwitansi": r.get("no_kwitansi"),
                "jenis_kwitansi": r.get("jenis_kwitansi"),
                "tanggal": r.get("tanggal"),
                "pihak_terkait": r.get("pihak_terkait"),
                "nama_usaha": r.get("nama_usaha"),
                "nik_pemilik": r.get("nik_pemilik"),
                "total": r.get("total"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nHasil ekstraksi disimpan ke: {args.output_csv}")

    # ---- Hitung Omset & Profit per nasabah, dipecah per tahun ----
    df_valid = df[df["total"].notna() & df["tanggal"].notna()].copy() if not df.empty else df
    if df_valid.empty:
        print("Tidak ada kwitansi yang berhasil diekstrak dengan lengkap.")
        return

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
    summary.to_csv(args.summary_csv, index=False)

    print(f"\nHasil ringkasan per nasabah per tahun disimpan ke: {args.summary_csv}")
    print("\n===== OMSET & PROFIT PER NASABAH PER TAHUN =====")
    for _, row in summary.iterrows():
        print(f"NIK {row['nik_pemilik']} ({row['nama_usaha']}):")
        for col in summary.columns:
            if col.startswith("omset_") or col.startswith("profit_"):
                print(f"  {col:<12}: Rp {row[col]:,.0f}")

    n_errors = sum(1 for r in results if "error" in r)
    if n_errors:
        print(f"\nPeringatan: {n_errors} kwitansi gagal di-parse, cek kolom 'error' / 'raw_output'.")

if __name__ == "__main__":
    main()