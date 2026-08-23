# Capstone Project — Screening Credit Agentic AI

Capstone project ODP Data Analyst Batch 367 (BNI) — Tim **Asek**.

Author: **Muhammad Nevin** dan **Indah Syafhyra Nurjanah**

Sistem *agentic AI* untuk membantu proses screening pengajuan kredit retail/UMKM,
menggabungkan model machine learning (skor kelayakan 5C) dengan beberapa agent
khusus yang masing-masing memeriksa satu aspek pengajuan (identitas, riwayat
kredit, keuangan, cashflow, agunan, daftar hitam nasional, risiko) lalu
merangkumnya jadi satu laporan keputusan.

## Latar Belakang

Analis kredit di bank harus mengecek banyak sumber data berbeda untuk satu
pengajuan — data kependudukan (Dukcapil), riwayat kredit (SLIK OJK), daftar
hitam nasional (DHN), legalitas agunan (ATR/BPN), laporan keuangan, dan mutasi
rekening — lalu menilainya berdasarkan prinsip 5C (*Character, Capacity,
Capital, Collateral, Condition*). Proyek ini mencoba mengotomatisasi sebagian
proses tersebut: model ML memberi skor awal kelayakan, sementara agent-agent AI
memeriksa tiap aspek 5C secara terpisah dan menyusun narasi/laporan yang bisa
jadi bahan pertimbangan analis manusia.

## Arsitektur

```
data mentah (7 tabel) → preprocessing/feature engineering → model ML (skor kelayakan)
                                                                    │
                                                                    ▼
                identity · credit_history · financial · cashflow · collateral · dhn · risk agent
                                                                    │
                                                                    ▼
                                                    orchestrator → report agent → laporan akhir
```

- **Model ML** — memprediksi skor kelayakan kredit dari fitur hasil olahan data
  (lihat [data_dictionary.md](data_dictionary.md) untuk skema sumber data).
- **Multi-agent layer** (`src/agents/`) — tiap agent fokus memeriksa satu aspek
  pengajuan dan bisa memakai GenAI (`src/genai.py`) untuk menyusun narasi.
- **Orchestrator** (`src/orchestrator.py`) — menjalankan agent-agent tersebut
  dan menggabungkan hasilnya jadi satu keputusan/laporan.

> ⚠️ Dataset yang dipakai adalah **data sintetis** (lihat
> [notebooks/01_dataset_generation.ipynb](notebooks/01_dataset_generation.ipynb)),
> dibuat untuk keperluan training model & demo — bukan data nasabah nyata.

## Struktur Proyek

```
├── app.py                      # entry point aplikasi (Streamlit)
├── data/
│   ├── generator/               # skrip pembuat dataset sintetis
│   ├── raw/                     # 7 tabel mentah (dukcapil, SLIK, DHN, ATR/BPN, dll.)
│   └── processed/                # feature table & hasil prediksi
├── data_dictionary.md          # skema & deskripsi seluruh tabel data
├── models/
│   ├── credit_model.pkl         # model terlatih
│   ├── preprocessor.pkl         # pipeline preprocessing
│   └── model_card.md            # dokumentasi model
├── notebooks/
│   ├── 01_dataset_generation.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── agents/                  # agent per-aspek 5C (identity, financial, cashflow,
│   │                             #   collateral, credit_history, dhn, risk, report)
│   ├── model/                   # train / predict / evaluate
│   ├── preprocessing/           # loaders, feature engineering, encoders
│   ├── orchestrator.py          # penggabung alur kerja antar-agent
│   ├── genai.py                 # integrasi GenAI untuk narasi laporan
│   └── schemas.py               # skema data/kontrak antar-modul
└── tests/                       # unit test (agents, model, orchestrator)
```

## Dataset

Tujuh tabel relasional terhubung lewat `NIK` — detail tiap kolom ada di
[data_dictionary.md](data_dictionary.md):

| Tabel | Isi |
|---|---|
| `retail_customer_profile` | Ringkasan pengajuan kredit + label keputusan (target) |
| `dukcapil` | Identitas pemohon (KTP) |
| `slik_credit_history` | Riwayat kredit di bank lain (SLIK/OJK) |
| `dhn` | Status Daftar Hitam Nasional |
| `agunan_atr_bpn` | Legalitas & valuasi agunan (ATR/BPN) |
| `laporan_keuangan` | Neraca & laba-rugi 2 tahun terakhir |
| `bank_account` | Mutasi & aktivitas rekening |

## Status Pengembangan

Repo ini masih dalam tahap awal — struktur proyek sudah dibuat, tapi sebagian
besar modul masih kosong (skeleton):

- [x] Skema data & data dictionary
- [x] Notebook pembuatan dataset sintetis
- [ ] EDA (`02_eda.ipynb`)
- [ ] Training model (`03_model_training.ipynb`, `src/model/`)
- [ ] Preprocessing pipeline (`src/preprocessing/`)
- [ ] Agent-agent 5C (`src/agents/`)
- [ ] Orchestrator & integrasi GenAI
- [ ] Aplikasi (`app.py`) & test suite

## Cara Menjalankan

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt

python app.py
```

> `requirements.txt` masih perlu dilengkapi seiring modul-modul di atas
> diimplementasikan.

## Tim

Tim **Asek** — ODP Data Analyst Batch 367, BNI.
