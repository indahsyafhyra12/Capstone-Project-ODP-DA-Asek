# Credit Screening Agentic AI — eLO

Capstone Project **ODP Data Analyst Batch 367 (BNI)** — Tim **eLO**.

**Author:** Muhammad Nevin & Indah Syafhyra Nurjanah

**🔗 Live Demo:** https://capstone-project-odp-da-asek-app-premium.streamlit.app/

Sistem *agentic AI* untuk membantu proses screening pengajuan kredit
retail/UMKM. Menggabungkan **model machine learning** (prediksi skor risiko),
**rule-based policy engine** (kebijakan bank yang eksplisit & auditable), dan
**LLM lokal** (narasi penjelasan berbahasa natural + OCR kwitansi) dalam satu
dashboard Streamlit multi-halaman.

---

## Daftar Isi

- [Latar Belakang](#latar-belakang)
- [Live Demo](#live-demo)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Sistem](#arsitektur-sistem)
- [Struktur Proyek](#struktur-proyek)
- [Dataset](#dataset)
- [Model Machine Learning](#model-machine-learning)
- [Cara Menjalankan Secara Lokal](#cara-menjalankan-secara-lokal)
- [Testing](#testing)
- [Notebooks](#notebooks)
- [Keterbatasan & Catatan Penting](#keterbatasan--catatan-penting)
- [Status Pengembangan](#status-pengembangan)
- [Tim](#tim)

---

## Latar Belakang

eLO BCM merupakan sistem yang digunakan oleh pengusul kredit untuk memproses
debitur baru, melakukan review, maupun menangani debitur interim dengan
plafon pembiayaan sampai dengan Rp10 miliar. Dalam prosesnya, pengusul perlu
mengumpulkan dan memverifikasi berbagai informasi sebelum analisis kredit dan
persetujuan dapat dilakukan.

Berdasarkan proses credit screening yang berjalan, terdapat tiga tantangan
utama:

1. **Input masih manual** — data dan dokumen pengajuan perlu dimasukkan serta
   diperiksa secara manual dari berbagai sumber.
2. **Assessment membutuhkan waktu** — proses pengumpulan data, verifikasi,
   analisis industri, analisis aspek manajemen, analisis keuangan, analisis
   risiko dan mitigasi, analisis proyeksi keuangan, evaluasi kebutuhan
   keuangan, hingga penetapan struktur fasilitas membutuhkan banyak waktu.
3. **Konteks pengajuan perlu dipahami secara menyeluruh** — pengusul perlu
   menggabungkan informasi dari SLIK, DHN, Dukcapil, ATR/BPN, laporan
   keuangan, mutasi rekening, dan sumber informasi lainnya agar rekomendasi
   kredit dapat dibuat secara tepat.

Untuk menjawab tantangan tersebut, project ini mengusulkan solusi berbasis
**Machine Learning dan Artificial Intelligence** yang terintegrasi dengan
sistem eLO. Solusi ini mengotomatisasi proses credit scoring secara end to
end melalui tiga kemampuan utama:

- **Ekstraksi informasi dokumen dan gambar** menggunakan VLM untuk mengambil
  informasi penting dari laporan keuangan dan hasil OTS.
- **Agentic AI credit scoring** yang menggunakan ML dan rule-based policy
  engine untuk menilai kelayakan kredit berdasarkan data yang telah
  tervalidasi.
- **Agentic AI reasoning dan dashboard monitoring** untuk menganalisis hasil
  scoring serta memberikan rekomendasi dan insight secara otomatis kepada
  Relationship Manager.

Dengan pendekatan ini, proses screening diharapkan menjadi lebih cepat,
konsisten, transparan, dan tetap dapat diaudit karena keputusan akhir tetap
dikendalikan oleh policy engine yang eksplisit.


## Overview Project

Analis kredit di bank harus mengecek banyak sumber data berbeda untuk satu
pengajuan — data kependudukan (Dukcapil), riwayat kredit (SLIK OJK), daftar
hitam nasional (DHN), legalitas agunan (ATR/BPN), laporan keuangan, dan mutasi
rekening — lalu menilainya berdasarkan prinsip **5C** (*Character, Capacity,
Capital, Collateral, Condition*). Proyek ini mengotomatisasi sebagian besar
proses tersebut:

1. **Agentic AI** menentukan urutan & kedalaman verifikasi yang perlu dilakukan
   (tidak semua pengajuan butuh pemeriksaan sedalam yang lain).
2. **Model ML** memberi skor risiko dari fitur-fitur hasil olahan data mentah.
3. **Policy Engine** (rule-based, eksplisit) menerjemahkan skor itu menjadi
   keputusan, zona risiko, jenis kredit, tenor, dan suku bunga.
4. **LLM** menyusun narasi penjelasan yang bisa langsung dibaca RM
   (Relationship Manager), tanpa pernah mengubah keputusan yang sudah final.
5. **VLM** mengkestrasi dokumen berupa foto kwitansi transaksi Retail untuk
   menghasilkan laporan keuangan secara otomatis.

## Live Demo

Versi premium aplikasi ini sudah di-deploy di Streamlit Community Cloud:

**🔗 https://capstone-project-odp-da-asek-app-premium.streamlit.app/**

Silakan eksplor menu di sidebar (Executive Overview, Daftar Pengajuan, Detail
Nasabah, Pengajuan Kredit Baru, Monitoring Portofolio, RM Monitoring, dan
Generate Laporan Keuangan dari Kwitansi). Perhatikan juga bagian
[Keterbatasan & Catatan Penting](#keterbatasan--catatan-penting) — dua fitur
yang butuh model AI berat (narasi Gemma & OCR kwitansi LightOnOCR) berjalan
dalam mode *fallback* rule-based di lingkungan cloud gratis ini karena tidak
tersedia GPU.

## Fitur Utama

Aplikasi ini punya **dua entry point** yang bisa dijalankan terpisah:

| Entry point | Folder halaman | Gaya | Dipakai untuk |
|---|---|---|---|
| [`app_premium.py`](app_premium.py) | [`pages_v2/`](pages_v2/) | UI premium (custom CSS, hero banner, palet warna wondr BNI) | **Live demo di atas** |
| [`app.py`](app.py) | [`pages/`](pages/) | UI Streamlit standar (multipage klasik) | Versi ringan/awal, tetap dipertahankan sebagai referensi |

Kedua versi punya cakupan fungsional yang sama, dibungkus tampilan berbeda:

- **🏠 Executive Overview** — KPI portofolio (total pengajuan, approval rate,
  rata-rata risk score, total nominal disetujui), distribusi zona risiko,
  validasi hard-rule DHN, fairness check approval rate per gender, komposisi
  nasabah, dan network graph klaster industri.
- **📋 Daftar Pengajuan** — tabel seluruh pengajuan dengan filter cabang,
  industri, dan tanggal.
- **👤 Detail Nasabah** — rincian 1 nasabah: breakdown skor 7-agent
  (Identity, Credit History, DHN, Collateral, Financial, Cashflow, Risk), dan
  kartu **Kesesuaian Jenis Kredit** (`recommend_credit_type()`) yang
  membandingkan jenis/tenor kredit yang diajukan dengan kemampuan bayar (DSR)
  nasabah.
- **📝 Pengajuan Kredit Baru** — form screening kredit baru, dua cara input:
  - Manual (1 nasabah, form lengkap), atau
  - Upload CSV (1 baris → auto-isi form untuk direview; >1 baris → diproses
    batch, hasil bisa didownload).
  - Bisa upload **ZIP foto kwitansi** untuk auto-isi 2 field profil finansial
    (estimasi omset & frekuensi transaksi bulanan) lewat OCR lokal.
  - Riwayat SLIK/DHN/rekening/keuangan ditelusuri otomatis lewat NIK dari
    `data/raw/*.csv` — field sistem (application_id, cif_number, dll) selalu
    diisi otomatis.
- **📊 Monitoring Portofolio** — dashboard monitoring portofolio kredit
  berjalan.
- **🏆 RM Monitoring** — performa per Relationship Manager + network graph
  kemiripan RM (graph analytics, `networkx`).
- **🧾 Generate Laporan Keuangan dari Kwitansi** — upload beberapa foto
  kwitansi penjualan/pembelian nasabah yang **sudah ada** di sistem, hasil
  OCR di-parse jadi laporan omset/pembelian/profit per tahun.

## Arsitektur Sistem

Ada **dua jalur skoring** yang hidup berdampingan di repo ini — penting untuk
dipahami karena keduanya dipakai di halaman berbeda:

```
┌─────────────────────────────────────────────────────────────────────┐
│  JALUR A — Dashboard historis (Overview, Daftar Pengajuan,          │
│  Detail Nasabah, Monitoring, RM Monitoring)                          │
│                                                                       │
│  data/processed/master_dataset.csv                                   │
│         │  utils/data_loader.py::load_master_data()                  │
│         ▼                                                             │
│  utils/agent_pipeline.py::score_dataframe()                          │
│  (7 fungsi rule-based murni: identity, credit_history, dhn,          │
│   collateral, financial, cashflow, risk_agent — dihitung LIVE         │
│   setiap data dimuat, bukan hasil training)                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  JALUR B — Pengajuan Kredit Baru (hybrid ML pipeline)                │
│                                                                       │
│  form / CSV / raw tables (by NIK) → utils/feature_builder.py          │
│         │                                                             │
│         ▼                                                             │
│  src/agents/planner_agent.py  — Adaptive Verification Planner        │
│  (rule engine + finite state machine, 12 rules; TANPA LLM) memutuskan │
│  agent rule-based mana yang jalan, urutan, dan flag review manual     │
│         │                                                             │
│         ▼                                                             │
│  utils/risk_ml_pipeline.py::predict_credit_screening()               │
│    Stage 1 — Hard-rule filter (identitas invalid / DHN / SLIK Macet) │
│    Stage 2 — Model ML memprediksi risk_score (models/risk_score_*)   │
│    Stage 3 — Policy Engine rule-based (threshold zona, jenis kredit,  │
│              tenor, bunga, narasi insight 6 kategori)                 │
│         │                                                             │
│         ▼                                                             │
│  src/orchestrator.py::run_screening() → src/schemas.py::ScreeningResult│
│         │                                                             │
│         ▼ (opsional, explain_with_gemma=True)                        │
│  src/genai.py + utils/report_agent.py — Gemma Explanation Layer      │
│  (google/gemma-4-E2B-it, single-shot, PUNYA guardrail: kalau narasi   │
│   kontradiksi/mengarang step → fallback ke teks rule-based)           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  JALUR C — OCR Kwitansi (independen, dipanggil dari 2 halaman)       │
│                                                                       │
│  foto kwitansi → utils/kwitansi_extractor.py                         │
│    1. LightOnOCR-2-1B (VLM lokal) → transkripsi teks mentah           │
│    2. Regex parser → nomor kwitansi, tanggal, nominal, pihak, dst.    │
│  dipakai di: Pengajuan Kredit Baru (auto-fill 2 field) dan            │
│  Generate Laporan Keuangan dari Kwitansi (rekap tahunan)              │
└─────────────────────────────────────────────────────────────────────┘
```

Governance yang dijaga ketat di seluruh pipeline: **satu kotak, satu jenis
keputusan** —

| Komponen | Tanggung jawab |
|---|---|
| Planner (`src/agents/planner_agent.py`) | Urutan & kedalaman verifikasi saja |
| ML (`models/risk_score_*.pkl`) | `risk_score` saja |
| Policy Engine (`utils/risk_ml_pipeline.py`) | Keputusan/zona/jenis kredit/bunga |
| Gemma (`src/genai.py`, `utils/report_agent.py`) | Narasi penjelasan saja — tidak pernah mengubah keputusan |

> Di UI live saat ini, pemanggilan `run_screening()` pada halaman Pengajuan
> Kredit Baru dilakukan dengan `explain_with_gemma=False` (lihat
> [Keterbatasan](#keterbatasan--catatan-penting)) — narasi yang tampil adalah
> `insight` rule-based dari Policy Engine, bukan hasil generate Gemma.

## Struktur Proyek

```
├── app.py                        # Entry point klasik (pakai pages/)
├── app_premium.py                 # Entry point premium (pakai pages_v2/) — dipakai live demo
├── pages/                         # Halaman UI klasik (auto-detect Streamlit multipage)
│   ├── 1_Daftar_Pengajuan.py
│   ├── 2_Detail_Nasabah.py
│   ├── 3_Pengajuan_Credit_Baru.py
│   └── 4_Monitoring_Portofolio.py
├── pages_v2/                      # Halaman UI premium (didaftarkan via st.navigation di app_premium.py)
│   ├── 00_Overview_Premium.py
│   ├── 01_Daftar_Pengajuan_Premium.py
│   ├── 02_Detail_Nasabah_Premium.py
│   ├── 03_Pengajuan_Credit_Baru_Premium.py
│   ├── 04_Monitoring_Portofolio_Premium.py
│   ├── 05_RM_Performance_Premium.py
│   └── 06_Laporan_Keuangan_Kwitansi_Premium.py
├── src/
│   ├── agents/planner_agent.py    # Adaptive Verification Planner (rule engine + FSM)
│   ├── orchestrator.py            # Menyatukan Planner + ML pipeline + Gemma
│   ├── genai.py                   # Gemma Explanation Layer (narasi proses & hasil)
│   └── schemas.py                 # Dataclass: PlannerStep, PlannerTrace, ScreeningResult, dll.
├── utils/
│   ├── agent_pipeline.py          # 7-agent rule-based (dipakai dashboard historis / Jalur A)
│   ├── risk_ml_pipeline.py        # Hybrid ML + Policy Engine 3-stage (Jalur B)
│   ├── feature_builder.py         # Bangun fitur dari raw tables (by NIK) untuk Jalur B
│   ├── kwitansi_extractor.py      # OCR kwitansi (LightOnOCR-2-1B) + regex parser
│   ├── report_agent.py            # Model loader Gemma + generate_report() + guardrail
│   ├── data_loader.py             # Load & cache master_dataset.csv + graph network
│   ├── ui_components.py           # Helper UI bersama (logo, warna zona, dll.) — dipakai pages/
│   └── ui_premium.py              # CSS & komponen UI premium (hero banner, KPI card, dll.) — dipakai pages_v2/
├── models/                        # Artefak model terlatih (lihat bagian Model ML)
├── data/
│   ├── generator/generate_dataset.py   # Skrip pembuat dataset sintetis
│   ├── raw/                       # 8 tabel mentah (dukcapil, SLIK, DHN, ATR/BPN, dll.)
│   ├── processed/                 # master_dataset.csv, master_scored.csv, feature table, graph nodes/edges
│   ├── kwitansi/, kwitansi_8/, kwitansi_small/  # Contoh foto kwitansi + ground truth untuk demo/uji OCR
│   └── img/                       # Aset logo
├── notebooks/                     # EDA, training model, graph analytics, demo planner (lihat bagian Notebooks)
├── tests/                         # Unit test (pytest) — planner, orchestrator, model, agents
├── data_dictionary_v2.md          # Skema & deskripsi lengkap seluruh tabel data
├── requirements.txt                # Dependensi dashboard Streamlit (ringan, tanpa torch/transformers)
├── requirements-notebooks.txt      # Dependensi tambahan untuk notebooks/ (training, SHAP, dll.)
└── Resources_Pendukung/            # Skrip & notebook eksperimen/referensi (bukan bagian aplikasi utama)
```

## Dataset

Data yang dipakai adalah **data sintetis** (dibuat lewat
[data/generator/generate_dataset.py](data/generator/generate_dataset.py) dan
[notebooks/01_updated_data_generation.ipynb](notebooks/01_updated_data_generation.ipynb))
untuk keperluan training model & demo — **bukan data nasabah nyata**.

8 tabel mentah terhubung lewat `NIK` (kecuali `rm_master` yang terhubung
lewat `rm_id`) — detail lengkap tiap kolom ada di
[data_dictionary_v2.md](data_dictionary_v2.md):

| Tabel | Isi |
|---|---|
| `retail_customer_profile` | Ringkasan pengajuan kredit (incl. jenis/tenor/tujuan kredit diajukan) |
| `dukcapil` | Identitas pemohon (KTP) |
| `slik_credit_history` | Riwayat kredit di bank lain (SLIK/OJK) |
| `dhn` | Status Daftar Hitam Nasional |
| `agunan_atr_bpn` | Legalitas & valuasi agunan (ATR/BPN) |
| `laporan_keuangan` | Neraca & laba-rugi 2 tahun terakhir |
| `bank_account` | Mutasi & aktivitas rekening (termasuk `current_balance`) |
| `rm_master` | Data Relationship Manager (untuk monitoring, bukan fitur skor) |

Ditambah dataset foto kwitansi sintetis (`data/kwitansi/`, `data/kwitansi_8/`,
`data/kwitansi_small/`) beserta ground truth-nya, untuk mendemokan/menguji
modul OCR.

## Model Machine Learning

Artefak model ada di [models/](models/):

- `risk_score_model.pkl` + `risk_score_preprocessor.pkl` +
  `risk_score_meta.pkl` — model yang **live dipakai** oleh
  `utils/risk_ml_pipeline.py` untuk memprediksi `risk_score` (Stage 2 dari
  Jalur B di atas).
- `risk_score_model_comparison.csv` — perbandingan performa 3 algoritma yang
  dicoba saat training (Logistic Regression, XGBoost, MLP PyTorch — lihat
  file `model_*_<timestamp>.pkl` untuk tiap kandidat).
- `credit_model.pkl` + `preprocessor.pkl` — model dari iterasi training lebih
  awal, dipertahankan sebagai referensi.

Proses training & evaluasi lengkap ada di
[notebooks/03_ml_risk_scoring.ipynb](notebooks/03_ml_risk_scoring.ipynb) dan
di-deploy lewat
[notebooks/04_deploy_predict_ml_risk_scoring.ipynb](notebooks/04_deploy_predict_ml_risk_scoring.ipynb).
Model ML ini **tidak pernah diberi tahu** formula bobot 5C manual yang dipakai
`utils/agent_pipeline.py` — ia belajar mapping fitur → risiko langsung dari
data historis.

## Cara Menjalankan Secara Lokal

Prasyarat: Python 3.11+ direkomendasikan (mengikuti versi `numpy`/`pandas`
yang dipakai).

```bash
# 1. Buat & aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# 2. Install dependensi dashboard
pip install -r requirements.txt

# 3. Jalankan versi premium (sama seperti live demo)
streamlit run app_premium.py

# — atau jalankan versi klasik —
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`. Semua data
(`data/raw/`, `data/processed/`, `models/`) sudah tersedia di repo ini —
tidak perlu setup database atau API key eksternal untuk menjalankan
dashboard.

### Menjalankan fitur AI lokal (opsional, butuh resource lebih besar)

Fitur narasi otomatis (Gemma) dan OCR kwitansi (LightOnOCR) **tidak**
termasuk di `requirements.txt` dasar karena berat (butuh `torch` +
`transformers`, idealnya dengan GPU). Untuk mencobanya secara lokal:

```bash
pip install torch transformers accelerate
```

Kedua modul (`utils/report_agent.py`, `utils/kwitansi_extractor.py`) sudah
didesain lazy-import — kalau `torch`/`transformers` tidak terpasang atau
GPU tidak tersedia, dashboard tetap berjalan normal dan otomatis fallback ke
narasi/hasil rule-based, bukan crash.

Jika ingin menjalankan Streamlit sekaligus mencoba fitur LLM dan VLM secara
langsung, gunakan environment GPU di Google Colab atau Kaggle. Upload atau
copy notebook yang sesuai ke platform tersebut, lalu pilih **Run all**:

- Google Colab: [`notebooks/streamlit_eLO_BNI_Colab.ipynb`](notebooks/streamlit_eLO_BNI_Colab.ipynb)
- Kaggle: [`notebooks/streamlit_eLO_BNI_kaggle.ipynb`](notebooks/streamlit_eLO_BNI_kaggle.ipynb)

Aktifkan GPU pada runtime Colab atau accelerator Kaggle sebelum menjalankan
semua cell. Notebook akan menyiapkan dan menjalankan aplikasi Streamlit dengan
dukungan LLM (Gemma) dan VLM/OCR (LightOnOCR).

## Testing

```bash
pip install -r requirements.txt
pytest tests/
```

Test suite (`tests/`) mencakup:

- `test_agents.py` — 6 rule-based agent (`utils/agent_pipeline.py`).
- `test_model.py` — pipeline ML + policy engine (`utils/risk_ml_pipeline.py`).
- `test_orchestrator.py` — Adaptive Verification Planner (12 rule, satu test
  per rule), orchestrator, dan guardrail Gemma layer. Tidak butuh GPU/model
  download — jalur LLM diuji lewat cabang fallback-nya saja.

## Notebooks

Semua proses riset & training ada di [notebooks/](notebooks/) (butuh
`pip install -r requirements-notebooks.txt` untuk beberapa di antaranya):

| Notebook | Isi |
|---|---|
| `01_updated_data_generation.ipynb` | Pembuatan dataset sintetis |
| `02_prepro_eda_agent.ipynb` | Preprocessing & EDA |
| `03_ml_risk_scoring.ipynb` | Training & evaluasi model risk score (model live) |
| `03_model_training.ipynb` | Training model iterasi awal |
| `04_deploy_predict_ml_risk_scoring.ipynb` | Deployment model risk score |
| `04_predict_deploy.ipynb` | Deployment model iterasi awal |
| `05_test_report_agent_colab.ipynb` | Uji coba Report Agent (Gemma) di Colab (GPU) |
| `06_extract_kwitansi_lightonocr.ipynb` | Eksperimen OCR kwitansi dengan LightOnOCR |
| `07_eval_kwitansi_extractor_colab.ipynb` | Evaluasi akurasi ekstraksi kwitansi |
| `08_adaptive_verification_planner_demo.ipynb` | Demo & bukti timing Adaptive Verification Planner |
| `streamlit_eLO_BNI_Colab.ipynb` | Menjalankan Streamlit dengan LLM dan VLM di Google Colab (GPU) |
| `streamlit_eLO_BNI_kaggle.ipynb` | Menjalankan Streamlit dengan LLM dan VLM di Kaggle (GPU) |
| `graph_analytics_rm_network.ipynb` | Graph analytics jaringan RM |
| `graph_analytics_industry*.ipynb` | Graph analytics klaster industri |
| `Streamlit_dashboard.ipynb` | Draf awal dashboard (referensi historis) |

## Keterbatasan & Catatan Penting

- **Data sintetis** — seluruh dataset dibuat sintetis untuk keperluan
  demo/training, bukan data nasabah nyata.
- **Live demo berjalan CPU-only** — Streamlit Community Cloud tidak
  menyediakan GPU dan `requirements.txt` sengaja tidak menyertakan
  `torch`/`transformers` (biar deploy ringan & cepat). Konsekuensinya:
  - Narasi Gemma (`src/genai.py`) di halaman Pengajuan Kredit Baru dipanggil
    dengan `explain_with_gemma=False` — narasi yang tampil adalah `insight`
    rule-based dari Policy Engine.
  - Modul OCR kwitansi (`utils/kwitansi_extractor.py`) & Report Agent
    (`utils/report_agent.py`) tetap bisa diuji **secara lokal** kalau
    `torch`/`transformers`/GPU tersedia (lihat
    [Cara Menjalankan Secara Lokal](#cara-menjalankan-secara-lokal)), atau
    lewat notebook Colab (`05_test_report_agent_colab.ipynb`,
    `06_extract_kwitansi_lightonocr.ipynb`).
- **Dua jalur skoring berbeda hidup berdampingan** (lihat
  [Arsitektur Sistem](#arsitektur-sistem)) — dashboard historis memakai
  formula rule-based murni (`utils/agent_pipeline.py`), sedangkan Pengajuan
  Kredit Baru memakai pipeline hybrid ML + Adaptive Verification Planner
  (`utils/risk_ml_pipeline.py` + `src/orchestrator.py`). Keduanya divalidasi
  silang saat pengembangan tapi **tidak identik secara desain** — jangan
  bandingkan angka `risk_score` dari kedua jalur secara langsung.

## Status Pengembangan

- [x] Skema data & data dictionary (`data_dictionary_v2.md`)
- [x] Pembuatan dataset sintetis (8 tabel + gambar kwitansi)
- [x] EDA & preprocessing
- [x] 7-agent rule-based pipeline (`utils/agent_pipeline.py`)
- [x] Model ML risk scoring + perbandingan algoritma (`models/`)
- [x] Hybrid ML + Policy Engine 3-stage (`utils/risk_ml_pipeline.py`)
- [x] Adaptive Verification Planner (`src/agents/planner_agent.py`) + orchestrator
- [x] Gemma Explanation Layer + guardrail (`src/genai.py`, `utils/report_agent.py`)
- [x] OCR kwitansi lokal (`utils/kwitansi_extractor.py`)
- [x] Dashboard Streamlit — versi klasik (`app.py`) & premium (`app_premium.py`)
- [x] Graph analytics (jaringan RM & klaster industri)
- [x] Unit test (`tests/`)
- [x] Deploy live demo (Streamlit Community Cloud)
- [ ] Full Publish for BNI (Coming Soon ...)

## Tim

**Tim eLO** — ODP Data Analyst Batch 367, BNI.

Produk: **eLO — Credit Screening Agentic AI**

Author: **Muhammad Nevin** & **Indah Syafhyra Nurjanah**
