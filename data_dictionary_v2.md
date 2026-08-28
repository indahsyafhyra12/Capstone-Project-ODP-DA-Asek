# Data Dictionary — Screening Credit Agentic AI (Update: + RM & Current Balance)

Versi terbaru, mencakup **8 tabel mentah** + **2 tabel hasil olahan**
(`master_dataset.csv`, `master_scored.csv`). Semua tabel terhubung lewat
**NIK**, kecuali `rm_master` yang terhubung lewat **`rm_id`**, dan
`retail_customer_profile` yang punya `application_id` sebagai primary key
sendiri.

---

## Tabel Mentah (Raw)

### 1. `dukcapil`
Identitas dasar pemohon, mengikuti struktur data KTP.
**Primary Key:** `dukcapil_id` | Relasi: 1 NIK = 1 baris

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| dukcapil_id | string | ID record dukcapil | DKC000001 |
| NIK | string | Nomor Induk Kependudukan (16 digit) | 3276010601750001 |
| nama | string | Nama lengkap | Budi Panjaitan |
| tempat_lahir | string | Kota kelahiran | Depok |
| tanggal_lahir | date | Tanggal lahir | 1975-01-06 |
| jenis_kelamin | string | Laki-Laki / Perempuan | Laki-Laki |
| golongan_darah | string | Golongan darah | B |
| alamat | string | Alamat sesuai KTP | Jl. Panjaitan No. 27 |
| rt_rw | string | RT/RW | 011/009 |
| kelurahan_desa | string | Kelurahan/desa | Sukajadi |
| kecamatan | string | Kecamatan | Sukmajaya |
| kota_kabupaten | string | Kota/kabupaten | Depok |
| provinsi | string | Provinsi | Jawa Barat |
| agama | string | Agama | HINDU |
| status_perkawinan | string | Status perkawinan | Menikah |
| pekerjaan | string | Pekerjaan (selalu Wiraswasta, krn UMKM) | Wiraswasta |
| kewarganegaraan | string | Kewarganegaraan | WNI |
| berlaku_hingga | string | Masa berlaku KTP | SEUMUR HIDUP |

---

### 2. `slik_credit_history`
Riwayat kredit di bank lain (data SLIK/OJK). Relasi: 1 NIK = 0–3 baris.
**Primary Key:** `slik_record_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| slik_record_id | string | ID record SLIK | SLK000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| inquiry_date | date | Tanggal data ditarik | 2024-03-06 |
| bank_name | string | Bank pemberi kredit (bank lain) | Bank BCA |
| loan_type | string | KMK/KI/KPR/KKB/KK | KPR |
| plafond | int (IDR) | Plafon kredit disetujui | 100.000.000 |
| outstanding_balance | int (IDR) | Sisa baki debet | 87.078.357 |
| installment_amount | int (IDR) | Cicilan bulanan | 2.882.804 |
| tenor_month | int | Jangka waktu (bulan) | 36 |
| collectability | int | Kode kolektibilitas 1–5 | 1 |
| collectability_label | string | Lancar / DPK / Kurang Lancar / Diragukan / Macet | Lancar |

---

### 3. `dhn` (Daftar Hitam Nasional)
Relasi: 1 NIK = 1 baris. **Primary Key:** `dhn_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| dhn_id | string | ID record DHN | DHN000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| status_dhn | string | Ya/Tidak | Tidak |
| alasan | string | Alasan jika Ya (kosong jika Tidak) | (kosong) |
| tanggal_input | date | Tanggal data dicatat | 2025-07-23 |

---

### 4. `agunan_atr_bpn`
Detail & legalitas agunan. Relasi: 1 NIK = 1 baris. **Primary Key:** `atr_bpn_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| atr_bpn_id | string | ID record ATR/BPN | ATR000001 |
| NIK | string | NIK pemilik agunan (FK) | 3276010601750001 |
| asset_type | string | Tanah/Rumah/Ruko/Gudang | Rumah |
| certificate_type | string | SHM/HGB | HGB |
| certificate_number | string | Nomor sertifikat | 14452/Poris Plawad |
| provinsi / kota / kecamatan / kelurahan | string | Lokasi agunan | Banten / Tangerang / Cipondoh / Poris Plawad |
| land_area_m2 | float | Luas tanah (m²) | 187.3 |
| building_area_m2 | float | Luas bangunan (m²) | 236.1 |
| nilai_tanah_per_m2 | int (IDR) | Estimasi harga tanah/m² | 8.020.000 |
| nilai_bangunan_per_m2 | int (IDR) | Estimasi harga bangunan/m² | 3.500.000 |
| nilai_tanah_total | int (IDR) | land_area_m2 × harga/m² | 1.502.146.000 |
| nilai_bangunan_total | int (IDR) | building_area_m2 × harga/m² | 826.350.000 |
| total_collateral_value | int (IDR) | Total nilai agunan | 2.328.496.000 |
| ownership_match | string | Nama sertifikat sesuai pemilik? Ya/Tidak | Ya |

> Nilai tanah/bangunan per m² adalah **estimasi sintetis** per tingkatan
> wilayah Jabodetabek, bukan data appraisal resmi.

---

### 5. `laporan_keuangan`
Neraca & laba-rugi 2 tahun (2024, 2025). Relasi: 1 NIK = 2 baris.
**Primary Key:** `laporan_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| laporan_id | string | ID record laporan | FIN000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| year | int | Tahun laporan | 2024 |
| revenue | int (IDR) | Omset tahunan | 29.666.491 |
| net_profit | int (IDR) | Laba bersih tahunan | 3.405.584 |
| total_asset | int (IDR) | Total aset | 48.148.218 |
| total_liability | int (IDR) | Total kewajiban | 27.015.813 |
| operating_cashflow | int (IDR) | Arus kas operasional | 3.039.455 |

---

### 6. `bank_account`
Rekening & mutasi transaksi. Relasi: 1 NIK = 1–2 baris.
**Primary Key:** `account_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| account_id | string | ID record rekening | ACC000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| account_number | string | Nomor rekening (10 digit, bukan angka matematis) | 6956650690 |
| bank_name | string | Nama bank | Bank BCA |
| account_type | string | Giro/Tabungan | Giro |
| account_status | string | Aktif/Dormant | Aktif |
| opened_date | date | Tanggal buka rekening | 2023-08-14 |
| average_balance_6m | int (IDR) | Rata-rata saldo 6 bulan terakhir | 10.342.365 |
| average_monthly_credit | int (IDR) | Rata-rata uang masuk/bulan | 2.196.060 |
| average_monthly_debit | int (IDR) | Rata-rata uang keluar/bulan | 2.092.204 |
| transaction_frequency_monthly | int | Frekuensi transaksi/bulan | 148 |
| overdraft_count_6m | int | Jumlah overdraft dalam 6 bulan | 0 |
| **current_balance** ⭐ | int (IDR) | **[BARU]** Saldo "hari ini" (real-time), berfluktuasi wajar 40–180% dari `average_balance_6m`; lebih rendah kalau Dormant, bisa negatif kalau baru kena overdraft | 13.581.790 |

---

### 7. `rm_master` ⭐ **[TABEL BARU]**
Data Relationship Banking Officer (RM) yang menangani pengajuan.
Relasi: 1 `rm_id` = 1 baris (4 RM per cabang × 10 cabang = 40 RM).
**Primary Key:** `rm_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| rm_id | string | ID unik RM | RM0001 |
| rm_name | string | Nama RM | Ani Tanjung |
| branch_name | string | Cabang tempat RM bertugas | KCP Tebet |
| region | string | Wilayah cabang | Region 1 |
| jabatan | string | Selalu "Relationship Banking Officer" | Relationship Banking Officer |
| level | string | Junior RB (60%) / Senior RB (40%) | Junior RB |
| join_date | date | Tanggal RM mulai bertugas | 2017-11-02 |

> ⚠️ **Kolom ini untuk monitoring/dashboard operasional saja — JANGAN
> dipakai sebagai fitur model scoring.** Menjadikan identitas RM sebagai
> fitur berisiko mengunci bias/favoritism RM individual ke dalam sistem,
> bukan menilai kelayakan nasabah murni.

---

### 8. `retail_customer_profile`
Tabel pengajuan kredit utama. **Primary Key:** `application_id` |
**Foreign Key:** `NIK` → dukcapil.NIK, **`rm_id`** → rm_master.rm_id ⭐

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| application_id | string | ID unik pengajuan | APP202600001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| cif_number | string | Nomor CIF nasabah | CIF1000001 |
| application_date | date | Tanggal pengajuan | 2025-07-08 |
| customer_type | string | Segmen nasabah | UMKM |
| company_name | string | Nama usaha | UD Santoso Abadi |
| legal_entity | string | PT/CV/UD | UD |
| owner_name | string | Nama pemilik | Budi Panjaitan |
| owner_gender | string | L/P | L |
| owner_age | int | Usia pemilik | 51 |
| owner_marital_status | string | Status perkawinan pemilik | Menikah |
| owner_education | string | Pendidikan terakhir pemilik | S2 |
| province / city / district / region | string | Lokasi usaha | Jawa Barat / Depok / Sukmajaya / Region 2 |
| branch_name | string | Cabang pengaju | KCP Bogor Baranangsiang |
| industry / sub_industry | string | Sektor & sub-sektor usaha | Manufaktur / Konveksi |
| business_age_year | int | Lama usaha (tahun) | 14 |
| employee_count | int | Jumlah karyawan | 9 |
| monthly_turnover_est | int (IDR) | Estimasi omset bulanan | 2.672.137 |
| transaction_frequency_monthly | int | Frekuensi transaksi/bulan | 66 |
| loan_requested | int (IDR) | Nominal pinjaman diajukan | 300.000.000 |
| jenis_kredit_diajukan ⭐ | string | **[Dipakai mulai versi ini]** KMK/KI/KUR yang diajukan nasabah — divalidasi terhadap DSR lewat `recommend_credit_type()` (`utils/agent_pipeline.py`) | KI |
| tenor_diajukan_bulan ⭐ | int | **[Dipakai mulai versi ini]** Tenor yang diajukan nasabah (bulan) — input `recommend_credit_type()` | 54 |
| tujuan_penggunaan_kredit ⭐ | string | **[Dipakai mulai versi ini]** Tujuan penggunaan kredit, tampilan teks saja — bukan fitur ML/rule | Pembelian mesin produksi tambahan usaha |
| collateral_type | string | Jenis agunan | Rumah |
| collateral_location/province/city | string | Lokasi agunan | Poris Plawad, Tangerang / Banten / Tangerang |
| collateral_size_m2 | float | Total luas agunan (m²) | 423.4 |
| collateral_market_value | int (IDR) | Nilai pasar agunan | 2.328.496.000 |
| collateral_liquidation_value | int (IDR) | Nilai likuidasi (~80% pasar) | 1.862.796.800 |
| collateral_ratio | float | Nilai agunan ÷ pinjaman | 7.76 |
| certificate_type | string | SHM/HGB | HGB |
| ownership_match | string | Nama sertifikat sesuai pemilik? | Ya |
| estimated_dsr | float | Estimasi Debt Service Ratio (dibatasi maks 3.0) | 3.0 |
| eligibility_score | float | Skor dari generator (⚠️ leakage, jangan jadi fitur) | 0.804 |
| label | string | Ground truth generator: Diterima/Ditolak | Diterima |
| **rm_id** ⭐ | string | **[BARU]** FK ke rm_master — RM yang menangani, dipilih dari cabang yang sama dengan `branch_name` | RM0020 |

---

## Tabel Hasil Olahan

### 9. `master_dataset.csv`
Hasil join 8 tabel + preprocessing (3.000 baris × 83 kolom). Berisi semua
kolom di atas (dengan `slik_*`, `bank_*`, `revenue_*` dll teragregasi/
ter-pivot dari relasi 1:banyak) + `rm_name`, `rm_branch_name`, `rm_region`,
`jabatan`, `level`, `join_date` (dari join `rm_master` lewat `rm_id`).
**Belum ada hasil skor/keputusan agent.**

Kolom tambahan hasil agregasi yang perlu dicatat:
- `bank_current_balance_total`, `bank_best_current_balance` ⭐ **[BARU]** —
  dari agregasi `current_balance` di `bank_account`
- `slik_worst_collectability`, `slik_has_macet`, dst — dari agregasi SLIK
- `revenue_growth_pct`, `profit_margin_2025`, dll — dari pivot laporan keuangan

---

### 10. `master_scored.csv`
Hasil `master_dataset.csv` **+ keluaran 7-agent pipeline** (3.000 baris ×
44 kolom — subset kolom terpilih, bukan semua 83). Ini yang berisi
keputusan akhir.

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| identity_passed | bool | Hasil Identity Agent | True |
| character_score | float (0–1) | Hasil Credit History Agent | 1.0 |
| character_notes | string | Catatan Credit History Agent | Kolektibilitas terburuk: Lancar di 1 bank |
| collateral_score | float (0–1) | Hasil Collateral Agent | 1.0 |
| collateral_notes | string | Catatan Collateral Agent | LTV agunan 776% dari pinjaman |
| financial_score | float (0–1) | Hasil Financial Agent | 0.625 |
| financial_notes | string | Catatan Financial Agent | Omset tumbuh +8.1%, margin 10.3% |
| cashflow_score | float (0–1) | Hasil Cashflow Agent ⭐ **[dikalibrasi ulang]** | 0.304 |
| cashflow_notes | string | Catatan Cashflow Agent | Saldo rata-rata 7.4x omset bulanan, real-time 5.1x |
| risk_score | float (0–1) | Skor gabungan dari Risk Agent (makin ke 1 makin layak) | 0.825 |
| **decision** | string | **Layak / Layak Bersyarat / Perlu Review Ulang / Tidak Layak** | Layak |
| **zone** | string | Hijau / Kuning / Merah (turunan dari decision) | Hijau |
| jenis_kredit_rekomendasi ⭐ | string | KMK / KI / KUR — divalidasi terhadap `jenis_kredit_diajukan` & DSR lewat `recommend_credit_type()` kalau data pengajuan tersedia, fallback ke tebakan dari skala pinjaman kalau tidak | KI |
| nominal_disetujui | int (IDR) | min(loan_requested, 70% nilai agunan) | 300.000.000 |
| jangka_waktu_bulan | int | Tenor rekomendasi | 36 |
| bunga_persen | float | Bunga berdasar zone (9.5/12.0/15.0) | 9.5 |
| **insight** | string | Narasi lengkap alasan keputusan | "Layak tapi Cashflow tergolong lemah (skor di bawah 0.5) — disarankan tetap dimonitor..." |
| **insight_kategori** ⭐ | string | Prefix dari `insight`, salah satu dari 6 kategori (lihat bawah) | Layak tapi |

**6 kategori `insight_kategori`:** Layak karena · **Layak tapi** ⭐ ·
Layak bersyarat karena · Perlu review ulang karena · **Tidak layak tapi** ⭐
· Tidak layak karena. *(⭐ = baru diimplementasikan lengkap di versi ini —
sebelumnya cuma 4 dari 6 kategori yang pernah muncul.)*

> ⚠️ **`decision`/`zone`/`risk_score` ini HASIL RULE-BASED (7 fungsi Python
> if/else + formula), BUKAN hasil model Machine Learning yang di-training.**
> Berbeda dari `label` (ground truth biner dari generator) — keduanya
> divalidasi silang (~92,7% kesesuaian) tapi dihasilkan dari proses yang
> sepenuhnya independen satu sama lain.

Kolom RM (`rm_id`, `rm_name`, `rm_branch_name`, `level`) juga disertakan
di tabel ini **untuk monitoring saja** — tidak pernah menjadi input ke
salah satu dari 7 agent.

---

### 11. Field Kesesuaian Jenis Kredit ⭐ (dihitung live, TIDAK tersimpan di `master_scored.csv`)

`recommend_credit_type()` (`utils/agent_pipeline.py`) memvalidasi
`jenis_kredit_diajukan`/`tenor_diajukan_bulan` terhadap kemampuan bayar
(DSR) nasabah. Dipanggil dari `score_application()`
(`utils/agent_pipeline.py`, dipakai `pages/2_Detail_Nasabah.py`) dan
`predict_credit_screening()` (`utils/risk_ml_pipeline.py`, dipakai halaman
Pengajuan Credit Baru) — **bukan kolom pre-computed** seperti tabel 9-10 di
atas, jadi tidak ikut ter-generate ulang ke `master_scored.csv` sampai file
itu diregenerasi ulang lewat notebook.

| Field | Tipe | Definisi | Contoh |
|---|---|---|---|
| jenis_kredit_sesuai | bool / None | `True` kalau pengajuan sesuai DSR (≤40%) apa adanya, `False` kalau perlu penyesuaian tenor/jenis, `None` kalau data pengajuan tidak diisi | False |
| dsr_pada_pengajuan | float / None | DSR dihitung dari nominal & tenor yang DIAJUKAN (beda dari `estimated_dsr` yang manual) | 0.65 |
| catatan_kesesuaian_kredit | string / None | Narasi penjelasan (mis. saran perpanjang tenor, alih jenis ke KI, atau plafon KUR terlampaui) | "Tenor 12 bulan terlalu berat (DSR 62%). Disarankan perpanjang tenor jadi 24 bulan (DSR turun ke 34%)." |

> ⚠️ **Catatan validasi:** DSR dihitung dari `cicilan_bulanan_pengajuan`
> (atau turunannya) dibagi `monthly_turnover_est` — pada distribusi
> `master_dataset.csv` yang ada saat ini, rasio itu median-nya ~12x
> (jauh di atas ambang `DSR_AMAN = 0.40`), sehingga mayoritas baris keluar
> `jenis_kredit_sesuai = False`. Ini konsisten dengan `estimated_dsr` yang
> juga mayoritas mepet ke cap 3.0 di kolom yang sama — bukan bug di
> `recommend_credit_type()`, tapi sinyal bahwa `DSR_AMAN` mungkin perlu
> dikalibrasi ulang ke skala dataset ini kalau field ini mau dipakai
> sebagai sinyal yang lebih seimbang (lih. kalibrasi serupa yang sudah
> dilakukan utk Cashflow Agent di atas).

---

## Ringkasan Perubahan dari Versi Sebelumnya

| Perubahan | Detail |
|---|---|
| Tabel baru | `rm_master` (relasi via `rm_id`, bukan NIK) |
| Kolom baru | `current_balance` di `bank_account`; `rm_id` di `retail_customer_profile` |
| Kalibrasi ulang | Cashflow Agent — skala saldo generator berubah (multiplier 40–100x, dulu 2–6x), rumus skor dinormalisasi pakai persentil ke-90 data historis |
| Narasi insight | Lengkap 6 kategori (dulu 4 dari 6) |
| Kesesuaian Jenis Kredit | `jenis_kredit_diajukan`/`tenor_diajukan_bulan`/`tujuan_penggunaan_kredit` (sudah ada di data, sebelumnya tidak dipakai) mulai divalidasi terhadap DSR lewat `recommend_credit_type()` — lihat tabel 11 |
| Kolom leakage/sensitif/operasional yang harus dihindari sbg fitur model | `eligibility_score`, `label` (leakage) · `owner_gender`, `owner_marital_status` (sensitif/fair-lending) · `rm_id`, `rm_name`, `rm_branch_name`, `level`, dll (operasional/governance) |
