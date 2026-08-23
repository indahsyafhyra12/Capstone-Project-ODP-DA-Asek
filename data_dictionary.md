# Data Dictionary — Screening Credit Agentic AI Dataset

Semua tabel terhubung lewat **NIK** (kecuali `retail_customer_profile` yang
punya `application_id` sebagai primary key sendiri, dengan `NIK` sebagai
foreign key ke `dukcapil`).

---

## 1. `retail_customer_profile`
Tabel utama pengajuan kredit — ringkasan lintas-tabel + label keputusan bank.
**Primary Key:** `application_id` | **Foreign Key:** `NIK` → `dukcapil.NIK`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| application_id | string | ID unik pengajuan kredit | APP202600001 |
| NIK | string | Nomor Induk Kependudukan pemohon (FK) | 3276010601750001 |
| cif_number | string | Nomor CIF nasabah di bank | CIF1000001 |
| application_date | date | Tanggal pengajuan | 2025-07-08 |
| customer_type | string | Segmen nasabah | UMKM |
| company_name | string | Nama usaha/badan hukum | UD Santoso Abadi |
| legal_entity | string | Bentuk badan usaha (PT/CV/UD) | UD |
| owner_name | string | Nama pemilik usaha | Budi Panjaitan |
| owner_gender | string | Jenis kelamin pemilik (L/P) | L |
| owner_age | int | Usia pemilik (tahun) | 51 |
| owner_marital_status | string | Status perkawinan pemilik | Menikah |
| owner_education | string | Pendidikan terakhir pemilik | S2 |
| province | string | Provinsi domisili usaha | Jawa Barat |
| city | string | Kota/kabupaten domisili usaha | Depok |
| district | string | Kecamatan domisili usaha | Sukmajaya |
| region | string | Wilayah operasional bank | Region 2 |
| branch_name | string | Cabang pengaju | KCP Bogor Baranangsiang |
| industry | string | Sektor industri | Manufaktur |
| sub_industry | string | Sub-sektor industri | Konveksi |
| business_age_year | int | Lama usaha berjalan (tahun) | 14 |
| employee_count | int | Jumlah karyawan | 9 |
| monthly_turnover_est | int (IDR) | Estimasi omset bulanan | 2.672.137 |
| transaction_frequency_monthly | int | Frekuensi transaksi rekening/bulan | 66 |
| loan_requested | int (IDR) | Nominal pinjaman yang diajukan | 300.000.000 |
| collateral_type | string | Jenis agunan | Rumah |
| collateral_location | string | Lokasi agunan (kelurahan, kota) | Poris Plawad, Tangerang |
| collateral_province | string | Provinsi agunan | Banten |
| collateral_city | string | Kota agunan | Tangerang |
| collateral_size_m2 | float | Total luas agunan (tanah+bangunan), m² | 423.4 |
| collateral_market_value | int (IDR) | Nilai pasar agunan | 2.328.496.000 |
| collateral_liquidation_value | int (IDR) | Nilai likuidasi agunan (~80% pasar) | 1.862.796.800 |
| collateral_ratio | float | Nilai agunan ÷ pinjaman diajukan | 7.76 |
| certificate_type | string | Jenis sertifikat agunan (SHM/HGB) | HGB |
| ownership_match | string | Kesesuaian nama sertifikat & pemilik (Ya/Tidak) | Ya |
| estimated_dsr | float | Debt Service Ratio estimasi (cicilan/omset, dibatasi maks 3.0) | 3.0 |
| eligibility_score | float | Skor kelayakan 5C gabungan, 0–1 (dipakai utk label, bukan buat dipakai sbg fitur ML — lihat catatan) | 0.804 |
| label | string | **Target/keputusan**: Diterima / Ditolak | Diterima |

> ⚠️ **Catatan modeling:** `eligibility_score` dihitung langsung dari rumus yang
> juga menentukan `label` — kalau dipakai sebagai fitur training, model akan
> "curang" (data leakage). Gunakan sebagai referensi/ground truth saja, bukan
> input model.

---

## 2. `dukcapil`
Identitas dasar pemohon, mengikuti struktur data KTP.
**Primary Key:** `dukcapil_id` (1 baris = 1 NIK, relasi 1:1)

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
| pekerjaan | string | Pekerjaan (semua Wiraswasta krn UMKM) | Wiraswasta |
| kewarganegaraan | string | Kewarganegaraan | WNI |
| berlaku_hingga | string | Masa berlaku KTP | SEUMUR HIDUP |

---

## 3. `slik_credit_history`
Riwayat fasilitas kredit di semua bank (data SLIK/OJK). 1 nasabah bisa punya
0–3 baris (relasi 1:banyak) — sebagian nasabah belum pernah punya kredit.
**Primary Key:** `slik_record_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| slik_record_id | string | ID record SLIK | SLK000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| inquiry_date | date | Tanggal data ditarik/dicek | 2024-03-06 |
| bank_name | string | Bank pemberi kredit (bank lain, bukan bank pengaju) | Bank BCA |
| loan_type | string | Jenis kredit: KMK/KI/KPR/KKB/KK | KPR |
| plafond | int (IDR) | Plafon kredit disetujui | 100.000.000 |
| outstanding_balance | int (IDR) | Sisa baki debet saat ini | 87.078.357 |
| installment_amount | int (IDR) | Cicilan bulanan | 2.882.804 |
| tenor_month | int | Jangka waktu kredit (bulan) | 36 |
| collectability | int | Kode kolektibilitas 1–5 | 1 |
| collectability_label | string | Label: Lancar / DPK / Kurang Lancar / Diragukan / Macet | Lancar |

---

## 4. `dhn` (Daftar Hitam Nasional)
Status blacklist nasabah. Relasi 1:1 dengan NIK.
**Primary Key:** `dhn_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| dhn_id | string | ID record DHN | DHN000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| status_dhn | string | Masuk daftar hitam? Ya/Tidak | Tidak |
| alasan | string | Alasan jika status = Ya (kosong jika Tidak) | (kosong) |
| tanggal_input | date | Tanggal data DHN dicatat/dicek | 2025-07-23 |

---

## 5. `agunan_atr_bpn`
Detail & validasi legalitas agunan lewat basis data ATR/BPN. Relasi 1:1
dengan NIK (versi awal — 1 nasabah 1 agunan utama).
**Primary Key:** `atr_bpn_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| atr_bpn_id | string | ID record ATR/BPN | ATR000001 |
| NIK | string | NIK pemilik agunan (FK) | 3276010601750001 |
| asset_type | string | Jenis aset: Tanah/Rumah/Ruko/Gudang | Rumah |
| certificate_type | string | Jenis sertifikat: SHM/HGB | HGB |
| certificate_number | string | Nomor sertifikat | 14452/Poris Plawad |
| provinsi | string | Provinsi lokasi agunan | Banten |
| kota | string | Kota lokasi agunan | Tangerang |
| kecamatan | string | Kecamatan lokasi agunan | Cipondoh |
| kelurahan | string | Kelurahan lokasi agunan | Poris Plawad |
| land_area_m2 | float | Luas tanah (m²) | 187.3 |
| building_area_m2 | float | Luas bangunan (m²), 0 jika aset = Tanah kosong | 236.1 |
| nilai_tanah_per_m2 | int (IDR) | Estimasi harga tanah per m² di kelurahan tsb | 8.020.000 |
| nilai_bangunan_per_m2 | int (IDR) | Estimasi harga bangunan per m² | 3.500.000 |
| nilai_tanah_total | int (IDR) | land_area_m2 × nilai_tanah_per_m2 | 1.502.146.000 |
| nilai_bangunan_total | int (IDR) | building_area_m2 × nilai_bangunan_per_m2 | 826.350.000 |
| total_collateral_value | int (IDR) | Total nilai agunan (tanah+bangunan) | 2.328.496.000 |
| ownership_match | string | Nama sertifikat sesuai pemilik? Ya/Tidak | Ya |

> Catatan: `nilai_tanah_per_m2` / `nilai_bangunan_per_m2` adalah **estimasi
> sintetis** per tingkatan wilayah Jabodetabek (bukan data appraisal resmi).

---

## 6. `laporan_keuangan`
Neraca & laba-rugi 2 tahun terakhir (2024, 2025). Relasi 1:2 dengan NIK.
**Primary Key:** `laporan_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| laporan_id | string | ID record laporan keuangan | FIN000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| year | int | Tahun laporan | 2024 |
| revenue | int (IDR) | Omset/pendapatan tahunan | 29.666.491 |
| net_profit | int (IDR) | Laba bersih tahunan | 3.405.584 |
| total_asset | int (IDR) | Total aset | 48.148.218 |
| total_liability | int (IDR) | Total kewajiban/utang | 27.015.813 |
| operating_cashflow | int (IDR) | Arus kas operasional | 3.039.455 |

---

## 7. `bank_account`
Data rekening & aktivitas transaksi nasabah (mutasi rekening). Relasi
1:banyak dengan NIK (1–2 rekening per nasabah).
**Primary Key:** `account_id`

| Kolom | Tipe | Definisi | Contoh |
|---|---|---|---|
| account_id | string | ID record rekening | ACC000001 |
| NIK | string | FK ke dukcapil | 3276010601750001 |
| account_number | string | Nomor rekening (10 digit, bukan angka matematis) | 6956650690 |
| bank_name | string | Nama bank (bisa BNI atau bank lain) | Bank BCA |
| account_type | string | Jenis rekening: Giro/Tabungan | Giro |
| account_status | string | Status rekening: Aktif/Dormant | Aktif |
| opened_date | date | Tanggal buka rekening | 2023-08-14 |
| average_balance_6m | int (IDR) | Rata-rata saldo 6 bulan terakhir | 620.254 |
| average_monthly_credit | int (IDR) | Rata-rata uang masuk per bulan | 2.196.060 |
| average_monthly_debit | int (IDR) | Rata-rata uang keluar per bulan | 2.092.204 |
| transaction_frequency_monthly | int | Frekuensi transaksi per bulan | 148 |
| overdraft_count_6m | int | Jumlah kejadian overdraft dalam 6 bulan | 0 |

---

## Diagram relasi (ringkas)

```
dukcapil (NIK) ──┬── retail_customer_profile (application_id, FK: NIK)
                  ├── slik_credit_history      (1:banyak)
                  ├── dhn                       (1:1)
                  ├── agunan_atr_bpn            (1:1)
                  ├── laporan_keuangan          (1:2, per tahun)
                  └── bank_account              (1:banyak)
```
