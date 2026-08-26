"""
extract_kwitansi.py
Ekstraksi data kwitansi (pembelian & penjualan) dari foto (di dalam file .zip)
menggunakan Claude Vision, lalu hitung Omset dan Profit per nasabah.

Cara pakai di Claude Code:
    1. pip install anthropic
    2. export ANTHROPIC_API_KEY=sk-ant-...
    3. Siapkan satu file .zip berisi foto-foto kwitansi (.jpg/.png), boleh campur
       kwitansi dari beberapa nasabah/usaha sekaligus.
    4. python extract_kwitansi.py --zip kwitansi.zip --output_csv hasil_kwitansi.csv

Output:
    - hasil_kwitansi.csv          -> detail per kwitansi (untuk audit/debug)
    - ringkasan_omset_profit.csv  -> satu baris per nasabah (nik_pemilik), dengan
                                      kolom omset_<tahun> dan profit_<tahun> untuk
                                      tiap tahun yang muncul di data (mis. omset_2024,
                                      profit_2024, omset_2025, profit_2025, ...)
    - ringkasan juga dicetak ke terminal
"""

import os
import json
import base64
import argparse
import zipfile
import tempfile
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

client = Anthropic()

MODEL = "claude-sonnet-4-6"  # bisa diganti ke model lain sesuai kebutuhan

# ---------------------------------------------------------------------------
# PROMPT EKSTRAKSI
# Ini bagian paling penting: prompt yang memaksa Claude mengembalikan JSON
# terstruktur dan konsisten, apapun isi kwitansinya (pembelian/penjualan).
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """Kamu adalah sistem OCR + ekstraksi data untuk kwitansi bisnis Indonesia.

Baca gambar kwitansi berikut dan ekstrak informasinya SEBAGAI JSON VALID SAJA,
tanpa teks lain, tanpa markdown code fence, mengikuti schema persis di bawah ini:

{
  "nama_usaha": string,
  "nik_pemilik": string,
  "industri": string,
  "sub_industri": string,
  "jenis_kwitansi": "penjualan" | "pembelian",
  "no_kwitansi": string,
  "tanggal": "YYYY-MM-DD",
  "pihak_terkait": string,           // "Kepada" jika penjualan, "Dibeli dari" jika pembelian
  "items": [
    {
      "deskripsi": string,
      "qty": number,
      "harga_satuan": number,        // angka murni, tanpa "Rp" atau titik ribuan
      "subtotal": number
    }
  ],
  "total": number,                    // angka murni dari baris TOTAL
  "status_lunas": boolean             // true jika ada stempel/label "LUNAS"
}

Aturan:
- Semua nilai uang HARUS angka murni (integer), buang "Rp" dan pemisah ribuan.
- Jika field tidak terbaca/tidak ada di gambar, isi dengan null.
- Jika "total" di gambar tidak sama persis dengan jumlah subtotal items (karena
  pembulatan atau OCR), tetap pakai angka yang tertulis di baris TOTAL.
- jenis_kwitansi ditentukan dari judul dokumen: "KWITANSI PENJUALAN" -> "penjualan",
  "KWITANSI PEMBELIAN" -> "pembelian".
- Output HARUS JSON valid dan bisa langsung di-parse json.loads(), tidak ada teks
  pembuka/penutup, tidak ada ```.
"""


def encode_image(path: Path) -> tuple[str, str]:
    media_type = "image/jpeg" if path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def extract_one(path: Path) -> dict:
    data, media_type = encode_image(path)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )
    raw_text = response.content[0].text.strip()
    # Jaga-jaga kalau model tetap membungkus dengan ```json ... ```
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {"error": "gagal parse JSON", "raw_output": raw_text, "file": path.name}
    parsed["source_file"] = path.name
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

    # Cari semua gambar, termasuk yang ada di dalam subfolder (mis. hasil unzip macOS)
    image_files = sorted(
        [
            p
            for p in tmp_dir.rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"] and not p.name.startswith(".")
        ]
    )

    if not image_files:
        print(f"Tidak ada file gambar di dalam {zip_path}")
        return

    print(f"Ditemukan {len(image_files)} kwitansi di dalam {zip_path.name}")

    results = []
    for i, path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Memproses {path.name} ...")
        parsed = extract_one(path)
        results.append(parsed)

    # ---- Simpan hasil detail (per kwitansi, items dipipihkan jadi JSON string) ----
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
                "industri": r.get("industri"),
                "sub_industri": r.get("sub_industri"),
                "total": r.get("total"),
                "status_lunas": r.get("status_lunas"),
                "items_json": json.dumps(r.get("items", []), ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nHasil ekstraksi disimpan ke: {args.output_csv}")

    # ---- Hitung Omset & Profit per nasabah, dipecah per tahun ----
    df_valid = df[df["total"].notna() & df["tanggal"].notna()].copy()
    df_valid["total"] = pd.to_numeric(df_valid["total"], errors="coerce")
    df_valid["tahun"] = pd.to_datetime(df_valid["tanggal"], errors="coerce").dt.year

    df_valid = df_valid[df_valid["tahun"].notna()]
    df_valid["tahun"] = df_valid["tahun"].astype(int)

    # Total per (nik, tahun, jenis)
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

    # Pivot jadi kolom omset_2024, profit_2024, omset_2025, profit_2025, dst.
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
