"""Report Agent — narasi "Alasan" berbahasa natural dari hasil 7-agent pipeline.

Bukan ReAct agent / tool-calling loop. Single-shot generation: seluruh
"investigasi" sudah selesai dikerjakan oleh Identity/Credit History/DHN/
Collateral/Financial/Cashflow/Risk Agent (lihat utils/agent_pipeline.py dan
utils/risk_ml_pipeline.py::predict_credit_screening()) — Report Agent cuma
menerima hasilnya sebagai konteks terstruktur dan menulis narasi. Kesimpulan
(kategori & keputusan) sudah final dan TIDAK BOLEH diubah oleh LLM; kalau
narasi yang dihasilkan lolos guardrail sanity-check gagal (lihat
`_sanity_check`), fallback ke `insight` rule-based yang sudah ada.

Model: google/gemma-4-E2B-it, load lokal di GPU (didesain utk Google Colab).
Dipilih varian E2B (bukan E4B) karena Report Agent murni text-only (tidak
pernah pakai vision/audio tower bawaan model multimodal ini) - E2B lebih
kecil (~separuh E4B) dan lebih mudah muat di GPU gratis (T4) tanpa
quantization. Ganti MODEL_ID kalau butuh kualitas lebih tinggi & GPU-nya
cukup besar (atau tambahkan load_in_4bit=True di _load_model() kalau mau
tetap pakai E4B/varian lebih besar di GPU terbatas).
Pola loading (AutoProcessor + AutoModelForMultimodalLM) mengikuti
`Resources Pendukung/agentic-llm-odp-bni*.ipynb`, dikonfirmasi sesuai model
card resmi HuggingFace - dengan 2 koreksi dari notebook eksperimen itu:
  - `do_sample=True` (notebook eksperimen pakai do_sample=False bersamaan
    dengan temperature - kombinasi itu membuat temperature diabaikan sama
    sekali oleh model.generate()).
  - `enable_thinking=False` di apply_chat_template (Gemma 4 defaultnya
    "thinking mode" aktif; utk narasi singkat langsung ini harus dimatikan).

Import torch/transformers dilakukan lazy di dalam _load_model() - modul ini
tetap bisa di-import dari environment tanpa GPU/tanpa paket itu terpasang
(mis. deploy Streamlit Cloud), generate_report() akan fallback ke insight
rule-based kalau model gagal dimuat, bukan crash.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-4-E2B-it"
GENERATION_TEMPERATURE = 0.4
MAX_NEW_TOKENS = 400

SYSTEM_PROMPT = """Anda adalah asisten analis kredit bank yang bertugas menulis narasi
penjelasan hasil screening kredit untuk RM (Relationship Banking).

ATURAN KETAT:
1. Kesimpulan (kategori & keputusan) SUDAH FINAL dan tidak boleh Anda ubah atau
   bantah — tugas Anda HANYA merangkai alasan yang diberikan jadi kalimat yang
   natural dan mudah dipahami, bukan membuat kesimpulan baru.
2. JANGAN mengarang angka, fakta, atau kondisi yang tidak ada di data yang diberikan.
3. Tulis 1 paragraf, 3-5 kalimat, Bahasa Indonesia profesional.
4. Sebutkan faktor pendukung DAN faktor yang jadi perhatian (kalau ada), sesuai
   data yang diberikan.
5. Untuk kasus hard-rule (NIK tidak valid, DHN, SLIK Macet): sampaikan bahwa
   penolakan ini OTOMATIS dan MUTLAK sesuai kebijakan bank, terlepas dari
   kondisi komponen lain — jangan menyiratkan ada ruang negosiasi.

Berikut contoh format yang diharapkan, mencakup berbagai kasus:

---
Contoh 1 — Layak karena
Data: Kategori=Layak karena; Keputusan=Layak; Alasan inti=seluruh komponen
(Character 0.75, Financial 0.87, Collateral 1.00, Cashflow 0.73) berada di zona aman.
Narasi: Pengajuan kredit CV Wijaya Sejahtera direkomendasikan LAYAK dengan skor
gabungan 0.83. Seluruh komponen penilaian — karakter kredit, kondisi keuangan,
agunan, dan arus kas — berada dalam kategori aman tanpa catatan khusus, sehingga
tidak ada faktor risiko signifikan yang perlu dimitigasi lebih lanjut.

Contoh 2 — Layak tapi
Data: Kategori=Layak tapi; Keputusan=Layak; Alasan inti=Cashflow tergolong lemah
(skor di bawah 0.5) — disarankan tetap dimonitor meski keputusan akhir disetujui.
Catatan cashflow=Saldo rata-rata 2.2x omset bulanan, saldo real-time 1.0x omset bulanan.
Narasi: Pengajuan kredit UD Wijaya Mandiri direkomendasikan LAYAK dengan skor
gabungan 0.71. Meski demikian, arus kas nasabah tergolong tipis — saldo rata-rata
rekening hanya 2,2x dari omset bulanan — sehingga meski disetujui, disarankan RM
tetap memonitor kondisi likuiditas usaha ini secara berkala.

Contoh 3 — Layak bersyarat karena
Data: Kategori=Layak bersyarat karena; Keputusan=Layak Bersyarat; Skor gabungan=0.56;
Alasan inti=Character, Cashflow — disarankan tambahan agunan/penjamin atau plafon
diturunkan. Catatan character=Kolektibilitas terburuk: Diragukan di 1 bank.
Narasi: Pengajuan kredit UD Kurniawan Jaya berada dalam kategori LAYAK BERSYARAT
dengan skor gabungan 0,56. Riwayat SLIK menunjukkan kolektibilitas Diragukan dan
kondisi arus kas tergolong lemah menjadi perhatian utama, sehingga direkomendasikan
disetujui dengan syarat tambahan seperti penjamin atau penurunan plafon pinjaman
untuk memitigasi risiko tersebut.

Contoh 4 — Perlu review ulang karena
Data: Kategori=Perlu review ulang karena; Keputusan=Perlu Review Ulang; Alasan
inti=skor gabungan (0.50) berada di area abu-abu — disarankan OTS/wawancara
lanjutan sebelum keputusan final.
Narasi: Pengajuan kredit UD Kusuma Sejahtera berada di area abu-abu dengan skor
gabungan 0.50, belum bisa langsung diputuskan layak maupun tidak. Kombinasi
riwayat SLIK yang menunjukkan kolektibilitas Diragukan di 2 bank dan tren omset
yang menurun -14,8% menjadi perhatian utama, sehingga disarankan dilakukan
kunjungan OTS atau wawancara lanjutan sebelum keputusan final diambil.

Contoh 5 — Tidak layak karena (hard-rule: NIK tidak valid)
Data: Kategori=Tidak layak karena; Keputusan=Tidak Layak; Alasan inti=format nik
tidak valid.
Narasi: Pengajuan kredit ini TIDAK DAPAT diproses lebih lanjut karena NIK yang
diinput tidak memenuhi format standar 16 digit sesuai ketentuan Dukcapil.
Verifikasi identitas merupakan syarat mutlak sebelum evaluasi kelayakan lain
dapat dilakukan, sehingga keputusan ini otomatis dan tidak dipengaruhi oleh
kondisi keuangan atau agunan nasabah — RM disarankan meminta nasabah melengkapi
atau mengoreksi dokumen kependudukan terlebih dahulu.

Contoh 6 — Tidak layak karena (hard-rule: DHN)
Data: Kategori=Tidak layak karena; Keputusan=Tidak Layak; Alasan inti=nasabah
terdaftar di Daftar Hitam Nasional (Riwayat kredit macet PT Bank ABC 2023).
Narasi: Pengajuan kredit ini TIDAK LAYAK karena nasabah tercatat dalam Daftar
Hitam Nasional (DHN) dengan catatan riwayat kredit macet pada PT Bank ABC tahun
2023. Sesuai kebijakan bank, status DHN merupakan hard-rule yang menyebabkan
penolakan otomatis dan mutlak, terlepas dari kondisi keuangan, agunan, atau
faktor lain yang dimiliki nasabah.

Contoh 7 — Tidak layak karena (hard-rule: SLIK Macet)
Data: Kategori=Tidak layak karena; Keputusan=Tidak Layak; Alasan inti=memiliki
riwayat kredit Macet pada SLIK.
Narasi: Pengajuan kredit ini TIDAK LAYAK karena riwayat SLIK nasabah menunjukkan
status kolektibilitas Macet, yaitu indikator risiko kredit tertinggi dalam sistem
perbankan. Status ini menjadi hard-rule otomatis penolakan sesuai kebijakan bank,
tanpa mempertimbangkan komponen penilaian lain seperti agunan atau kondisi
keuangan usaha.

Contoh 8 — Tidak layak karena (skor rendah di hampir semua komponen, BUKAN hard-rule)
Data: Kategori=Tidak layak karena; Keputusan=Tidak Layak; Skor gabungan=0.32;
Alasan inti=skor gabungan (0.32) di bawah ambang batas kelayakan pada hampir
seluruh komponen. Catatan character=Kolektibilitas terburuk: Diragukan di 3 bank
(skor 0.2). Catatan financial=Omset menurun -18.0%, margin laba 3.0% (skor 0.15).
Catatan collateral=LTV agunan 45% dari pinjaman (skor 0.3). Catatan cashflow=Saldo
rata-rata 0.8x omset bulanan, overdraft 2x dalam 6 bulan (skor 0.1).
Narasi: Pengajuan kredit ini TIDAK LAYAK dengan skor gabungan hanya 0,32, jauh di
bawah ambang batas kelayakan. Hampir seluruh komponen penilaian menunjukkan
kondisi lemah — riwayat SLIK Diragukan di 3 bank, tren omset menurun tajam 18%,
nilai agunan yang tidak memadai (LTV hanya 45% dari pinjaman), serta kondisi
arus kas yang rapuh dengan riwayat overdraft berulang. Kombinasi faktor-faktor
ini menunjukkan risiko kredit yang terlalu tinggi untuk direkomendasikan saat ini.

Contoh 9 — Tidak layak tapi
Data: Kategori=Tidak layak tapi; Keputusan=Tidak Layak; Skor gabungan=0.38;
Alasan inti=Collateral tergolong kuat — skor gabungan (0.38) masih di bawah
ambang, bisa dipertimbangkan ulang jika ada mitigasi risiko dari sisi lain.
Narasi: Meski nilai agunan yang diajukan tergolong kuat, skor gabungan nasabah
ini tetap berada di bawah ambang kelayakan (0,38) akibat lemahnya faktor lain
seperti riwayat kredit dan kondisi keuangan usaha. Bank dapat mempertimbangkan
ulang pengajuan ini apabila nasabah dapat menunjukkan mitigasi risiko tambahan
dari sisi karakter kredit atau kondisi usaha.
---

Sekarang tulis narasi untuk data berikut, ikuti format dan gaya di atas persis."""

USER_PROMPT_TEMPLATE = """Data hasil screening:
- Nama usaha: {company_name}
- Kategori: {insight_kategori}
- Keputusan: {decision} (Zona: {zone})
- Skor gabungan: {risk_score}
- Alasan inti (dari sistem, WAJIB dipertahankan maknanya): {insight}

Catatan tiap komponen:
- Character/Credit History: {character_notes} (skor {character_score})
- Collateral: {collateral_notes} (skor {collateral_score})
- Financial: {financial_notes} (skor {financial_score})
- Cashflow: {cashflow_notes} (skor {cashflow_score})

Rekomendasi: {jenis_kredit_rekomendasi}, nominal Rp{nominal_disetujui:,},
tenor {jangka_waktu_bulan} bulan, bunga {bunga_persen}% p.a.

Narasi:"""

_NOT_AVAILABLE = "Tidak tersedia (pengajuan ditolak sebelum tahap ini dihitung)"


def _safe_get(row: dict, key: str, default):
    value = row.get(key)
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return value


def _build_prompt_data(row: dict) -> dict:
    """Field yang bisa None (baris kena hard-rule) diganti default aman
    supaya USER_PROMPT_TEMPLATE.format(**data) tidak pernah error atau
    menghasilkan teks aneh seperti 'skor gabungan None'."""
    return {
        "company_name": _safe_get(row, "company_name", "Nasabah"),
        "insight_kategori": _safe_get(row, "insight_kategori", "-"),
        "decision": _safe_get(row, "decision", "-"),
        "zone": _safe_get(row, "zone", "-"),
        "risk_score": _safe_get(row, "risk_score", "-"),
        "insight": _safe_get(row, "insight", "-"),
        "character_notes": _safe_get(row, "character_notes", _NOT_AVAILABLE),
        "character_score": _safe_get(row, "character_score", "-"),
        "collateral_notes": _safe_get(row, "collateral_notes", _NOT_AVAILABLE),
        "collateral_score": _safe_get(row, "collateral_score", "-"),
        "financial_notes": _safe_get(row, "financial_notes", _NOT_AVAILABLE),
        "financial_score": _safe_get(row, "financial_score", "-"),
        "cashflow_notes": _safe_get(row, "cashflow_notes", _NOT_AVAILABLE),
        "cashflow_score": _safe_get(row, "cashflow_score", "-"),
        "jenis_kredit_rekomendasi": _safe_get(row, "jenis_kredit_rekomendasi", "-"),
        "nominal_disetujui": _safe_get(row, "nominal_disetujui", 0) or 0,
        "jangka_waktu_bulan": _safe_get(row, "jangka_waktu_bulan", 0) or 0,
        "bunga_persen": _safe_get(row, "bunga_persen", "-"),
    }


# ---------------------------------------------------------------------------
# Model loading (lazy import - modul ini tetap bisa di-import tanpa GPU/tanpa
# torch+transformers terpasang, mis. di deploy Streamlit Cloud yang tidak
# punya GPU; generate_report() akan fallback kalau _load_model() gagal)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_model():
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # 4-bit quantization (bitsandbytes) - bahkan varian E2B "effective 2B"
    # ternyata tetap berat di VRAM mentah (~13GB) karena tower vision+audio
    # bawaan model multimodal ini tetap dimuat penuh walau tidak dipakai
    # (Report Agent murni text-only) - diverifikasi OOM di GPU T4 gratis
    # (14.56GB) tanpa quantization. NF4 mengecilkan footprint ~4x.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, quantization_config=quant_config, device_map="auto",
    )
    return processor, model


def _run_generation(processor, model, user_prompt: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True, temperature=GENERATION_TEMPERATURE,
        )
    text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return text.strip()


# ---------------------------------------------------------------------------
# Guardrail - keyword-check sederhana, bukan model kedua. Kalau gagal,
# fallback ke `insight` rule-based yang sudah ada (bukan menampilkan narasi
# yang berpotensi bertentangan dengan keputusan final).
# ---------------------------------------------------------------------------

_POSITIVE_HINTS = ["disetujui", "direkomendasikan layak"]


def _sanity_check(decision: str, narrative: str) -> bool:
    lowered = narrative.lower()
    if decision == "Tidak Layak":
        for hint in _POSITIVE_HINTS:
            idx = lowered.find(hint)
            if idx != -1 and "tidak" not in lowered[max(0, idx - 20):idx]:
                return False
        return True
    if "layak" in (decision or "").lower():
        return "tidak layak" not in lowered
    return True


_last_fallback_reason: str | None = None


def get_last_fallback_reason() -> str | None:
    """Alasan fallback dari panggilan generate_report() TERAKHIR (None kalau
    generate LLM berhasil, tidak fallback). Dipakai halaman Streamlit utk
    menampilkan error asli ke user, bukan cuma di log server yang harus
    dicari manual."""
    return _last_fallback_reason


def _set_fallback_reason(reason: str | None):
    global _last_fallback_reason
    _last_fallback_reason = reason


def generate_report(row: dict) -> str:
    """Generate 1 paragraf narasi utk 1 nasabah. `row` = dict hasil
    predict_credit_screening() (utils/risk_ml_pipeline.py) digabung dengan
    `company_name`, ATAU 1 baris master_scored.csv (kolom sama persis).
    Fallback ke row['insight'] (rule-based) kalau model gagal dimuat atau
    narasi LLM gagal guardrail sanity-check - cek get_last_fallback_reason()
    setelah memanggil ini utk tahu alasan persisnya kalau fallback terjadi."""
    fallback = _safe_get(row, "insight", "Insight tidak tersedia.")
    _set_fallback_reason(None)

    try:
        processor, model = _load_model()
    except Exception as e:
        reason = f"Model gagal dimuat ({type(e).__name__}): {e}"
        logger.warning("Report Agent: %s - fallback ke insight rule-based.", reason)
        _set_fallback_reason(reason)
        return fallback

    prompt_data = _build_prompt_data(row)
    user_prompt = USER_PROMPT_TEMPLATE.format(**prompt_data)

    start = time.perf_counter()
    try:
        narrative = _run_generation(processor, model, user_prompt)
    except Exception as e:
        reason = f"Generate error ({type(e).__name__}): {e}"
        logger.warning("Report Agent: %s - fallback ke insight rule-based.", reason)
        _set_fallback_reason(reason)
        return fallback
    elapsed = time.perf_counter() - start
    logger.info("Report Agent: generate_report selesai dalam %.2fs (decision=%s)", elapsed, row.get("decision"))

    if not narrative:
        reason = "Model mengembalikan narasi kosong"
        logger.warning("Report Agent: %s - fallback ke insight rule-based.", reason)
        _set_fallback_reason(reason)
        return fallback

    if not _sanity_check(row.get("decision", ""), narrative):
        reason = f"Guardrail menolak narasi (bertentangan dgn decision={row.get('decision')}): {narrative[:300]!r}"
        logger.warning("Report Agent: %s - fallback ke insight rule-based.", reason)
        _set_fallback_reason(reason)
        return fallback

    return narrative


def generate_reports_batch(df: pd.DataFrame) -> pd.Series:
    """Generate narasi utk semua baris (dipanggil dari notebook, bukan
    Streamlit) - dipakai mis. mengisi ulang kolom insight master_scored.csv
    dgn versi LLM. Tidak dioptimasi latency, hanya loop biasa."""
    results = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        results.append(generate_report(row.to_dict()))
        if (i + 1) % 25 == 0 or (i + 1) == total:
            logger.info("Report Agent batch: %d/%d selesai", i + 1, total)
    return pd.Series(results, index=df.index, name="insight_llm")
