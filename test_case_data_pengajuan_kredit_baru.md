# Test Case Data — Customer Retail Pengajuan Kredit Baru

## Ringkasan

Telah dibuat **13 baris data pengajuan baru** pada file:

```
data/raw/retail_customer_profile_pengajuan_baru.csv
```

- Struktur kolom **identik** dengan `retail_customer_profile.csv` (file asli **tidak diubah sama sekali** — sudah diverifikasi via `git diff`, zero perubahan).
- 13 baris ini mencakup **12 dynamic routing path** yang perlu diuji pada Adaptive Verification Planner.

## Daftar Kasus Uji

| # | Kasus | NIK | Trigger Kondisi |
|---|-------|-----|------------------|
| 1a | NIK invalid | `317401050890300` (15 digit) | Format NIK salah → tanpa data pendukung sama sekali |
| 1b | NIK-nama ≠ Dukcapil | `3175011205883002` | Dukcapil = "Dedi Kurniawan", pengajuan = "Dedi Firmansyah" |
| 2 | DHN | `3276011503903003` | `status_dhn = "Ya"` |
| 3 | SLIK Macet | `3174012207853004` | `collectability = 5` |
| 4 | Character bersih | `3275014211923005` | Lancar, tidak ada DHN |
| 5 | Loan > 500 juta | `3671011001803006` | `loan_requested = 750jt` |
| 6 | Revenue turun > 30% | `3173012506873007` | -35% YoY |
| 7 | Overdraft ≥ 3 | `3276014809913008` | `overdraft_count_6m = 4` |
| 8 | Dormant account | `3271011904833009` | `account_status = "Dormant"` |
| 9 | Sertifikat beda nama | `3172013012793010` | `ownership_match = "Tidak"` |
| 10 | DSR tinggi | `3674015402903011` | `DSR = 1.95` |
| 11 | Confidence tinggi | `3174010508863012` | Semua faktor bersih & kuat |
| 12 | SHAP Financial dominan | `3171016710933013` | Revenue growth +60%, karakter/kolateral rata-rata |

## Data Pendukung yang Ditambahkan

NIK-NIK di atas juga ditambahkan sebagai baris baru ke seluruh sumber data pendukung berikut:

- `dukcapil.csv`
- `slik_credit_history.csv`
- `dhn.csv`
- `agunan_atr_bpn.csv`
- `laporan_keuangan.csv`
- `bank_account.csv`

> **Catatan:** Kasus **1a (NIK invalid)** sengaja **tidak** diberi baris pendukung di file-file di atas, karena validasi format NIK gagal lebih dulu sehingga proses tidak akan sampai ke tahap pengecekan data pendukung.

## Lokasi & Status Kode

- Kode generator data lengkap sudah dimasukkan ke **cell kosong** di bawah heading **"Generate Data TestCase"** pada notebook `01_updated_data_generation.ipynb`.
- Kode sudah diuji **idempotent** — aman dijalankan ulang (re-run) tanpa menghasilkan duplikasi data.

## Tujuan Test Case

Ke-12 kasus ini dirancang untuk memvalidasi bahwa **Adaptive Verification Planner** mampu mengarahkan setiap aplikasi ke jalur verifikasi yang sesuai (dynamic routing), termasuk jalur *hard-stop* (NIK invalid, DHN, SLIK macet) maupun jalur lanjutan (Financial, Collateral, Cashflow, Legal Verification, dsb.) sesuai kondisi data masing-masing nasabah.
