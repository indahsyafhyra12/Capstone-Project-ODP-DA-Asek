"""
Screening Credit Agentic AI - Synthetic Dataset Generator (v3)
============================================================
Generate 8 tabel relasional untuk training model screening kredit retail
banking (5C: Character, Capacity, Collateral, Condition + Capital/Identity).

Semua tabel terhubung lewat NIK (foreign key), KECUALI retail_customer_profile
yang punya application_id sebagai primary key.

UPDATE v3 (dari v2) - KHUSUS untuk kebutuhan graph analytics (node RM & node
industry), TIDAK menyentuh eligibility_score/label sama sekali:
  - rm_master: nama cabang & wilayah diganti pakai nama ASLI BNI area
    Jabodetabek (~48 cabang, bukan 10 fiktif), dikelompokkan ke 4 Kantor
    Wilayah asli. CATATAN JUJUR: pemetaan cabang -> wilayah ini best-effort
    dari data publik (bukan struktur organisasi internal BNI yang
    terverifikasi) - cukup untuk keperluan demo akademis.
  - rm_master: jumlah RM per cabang sekarang BERVARIASI (3-8), bukan fixed 4.
  - rm_master: kolom baru `kapasitas_max_nasabah` (5-25 per RM, senior
    cenderung lebih tinggi dari junior) - INPUT, bukan hasil hitungan.
  - retail_customer_profile: assignment RM sekarang pakai "slot pool"
    dibatasi kapasitas tiap RM (bukan random bebas per cabang seperti v2).
    Urutannya dibalik dari sebelumnya: RM dipilih dulu (dari slot yang
    tersedia), branch_name customer BARU ikut dari branch RM yang kepilih -
    supaya konsisten & tidak ada RM yang jumlah nasabahnya melebihi
    kapasitas_max_nasabah miliknya.
  - Kolom `region` generik di retail_customer_profile (Region 1-4) TETAP
    ADA & TIDAK diubah/dihapus (independen, by design pilihan user) -
    dashboard/graph disarankan pakai `rm_region` (dari join ke rm_master),
    bukan kolom `region` generik ini.
    
UPDATE v2 (dari v1):
  - Tambah 3 kolom baru di retail_customer_profile: jenis_kredit_diajukan,
    tenor_diajukan_bulan, tujuan_penggunaan_kredit (input pengajuan debitur,
    saling konsisten satu sama lain & dgn loan_requested/industry)
  - Fix DSR calculation: sebelumnya hardcode asumsi tenor 36 bulan & bunga
    flat 12%, sekarang pakai tenor & bunga yang BENERAN diajukan debitur
    (bunga beda per jenis kredit: KUR lebih rendah krn subsidi pemerintah)
  - NOTE: perubahan DSR ini berdampak ke eligibility_score & label di setiap
    baris (bukan random baru, tapi formula yang lebih akurat) - artinya
    master_dataset.csv, master_scored.csv, dan ML model Layer 1 kamu perlu
    di-regenerate/retrain ulang setelah pakai generator versi ini.
  - Fungsi kategorikan_kelayakan() (4-level: Layak/Layak Bersyarat/Perlu
    Review Ulang/Tidak Layak) disertakan di akhir file sebagai UTILITY
    terpisah - dipakai nanti saat membangun master_scored.csv, BUKAN
    bagian dari retail_customer_profile.csv.

Cara pakai di Google Colab:
    1. Copy semua isi file ini ke satu cell
    2. Run
    3. 8 file CSV akan tersimpan di /content/dataset/
       (retail_customer_profile.csv, dukcapil.csv, slik_credit_history.csv,
        dhn.csv, agunan_atr_bpn.csv, laporan_keuangan.csv, bank_account.csv,
        rm_master.csv)

PENTING saat load ulang CSV-nya nanti (termasuk di tahap join):
    Selalu paksa NIK dibaca sebagai teks, JANGAN biarkan pandas nebak tipenya,
    kalau tidak, 16 digit NIK bisa kepotong presisinya jadi angka:
        pd.read_csv("dukcapil.csv", dtype={"NIK": str})

Catatan penting: nilai tanah/bangunan per kelurahan di sini adalah ESTIMASI
SINTETIS yang dibuat plausible per tingkatan wilayah (bukan data appraisal
resmi/real) - cukup untuk keperluan training model & demo, BUKAN untuk
keputusan bisnis nyata.
"""

import random
import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Generator acak TERPISAH untuk field tambahan supaya penambahan-penambahan
# ini TIDAK menggeser urutan angka acak yang dipakai tabel/kolom lain yang
# sudah ada sebelumnya.
CB_RNG = np.random.default_rng(SEED + 1)         # khusus current_balance
RM_RNG = np.random.default_rng(SEED + 2)         # khusus rm_master & assignment RM
CREDIT_APP_RNG = np.random.default_rng(SEED + 3) # khusus kolom pengajuan kredit

N_CUSTOMERS = 3000          # jumlah nasabah/debitur unik
OUT_DIR = "/content/dataset" if os.path.isdir("/content") else "../data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================================
# 0. REFERENCE / LOOKUP DATA
# =========================================================================

FIRST_NAMES_M = ["Budi","Agus","Andi","Rizky","Dedi","Hendra","Yusuf","Fajar",
    "Wahyu","Bambang","Eko","Rudi","Slamet","Joko","Hadi","Ahmad","Dimas",
    "Arif","Taufik","Iwan","Gunawan","Sutrisno","Anton","Rian","Doni",
    "Yudi","Fauzi","Irfan","Bayu","Krisna"]
FIRST_NAMES_F = ["Siti","Dewi","Rina","Ani","Wulan","Sri","Yuni","Fitri",
    "Indah","Lestari","Ratna","Maya","Putri","Ika","Novi","Wati","Ayu",
    "Dian","Rita","Nina","Sari","Yanti","Lina","Desi","Tri","Retno",
    "Kartika","Anggi","Melati","Suryani"]
LAST_NAMES = ["Santoso","Wijaya","Kurniawan","Saputra","Setiawan","Pratama",
    "Hidayat","Nugroho","Firmansyah","Susanto","Gunawan","Halim","Wibowo",
    "Permana","Suryadi","Handoko","Kusuma","Rahman","Siregar","Simanjuntak",
    "Tanjung","Lubis","Hutapea","Panjaitan","Situmorang"]

RELIGIONS = ["ISLAM","KRISTEN","KATOLIK","HINDU","BUDDHA","KONGHUCU"]
MARITAL = ["Menikah","Belum Menikah","Cerai Hidup","Cerai Mati"]
EDUCATION = ["SMA/SMK","D3","S1","S2"]
BLOOD_TYPE = ["A","B","AB","O"]

INDUSTRIES = {
    "Perdagangan": ["Distributor Elektronik","Toko Sembako","Grosir Pakaian",
                    "Distributor Bahan Bangunan","Toko Alat Tulis"],
    "Kuliner": ["Restoran","Katering","Warung Makan","Bakery"],
    "Jasa": ["Bengkel","Laundry","Percetakan","Jasa Konstruksi Kecil"],
    "Manufaktur": ["Konveksi","Furniture","Pengolahan Makanan Ringan"],
    "Pertanian": ["Distributor Hasil Tani","Peternakan Ayam"],
    "Transportasi": ["Ekspedisi Kecil","Rental Kendaraan"],
}
INDUSTRY_RISK = {
    "Perdagangan": 0.05, "Kuliner": 0.10, "Jasa": 0.05,
    "Manufaktur": 0.08, "Pertanian": 0.15, "Transportasi": 0.12,
}

PROVINCES_CITIES = {
    "DKI Jakarta": ["Jakarta Selatan","Jakarta Pusat","Jakarta Timur","Jakarta Barat","Jakarta Utara"],
    "Jawa Barat": ["Bekasi","Depok","Bogor","Tangerang Selatan"],
    "Banten": ["Tangerang"],
}
# Kolom "region" GENERIK ini TETAP ADA & independen (by design, keputusan user) -
# dipakai di retail_customer_profile, TIDAK dihubungkan ke rm_region.
REGIONS = ["Region 1","Region 2","Region 3","Region 4"]

# =========================================================================
# BARU v3: cabang ASLI BNI area Jabodetabek, dikelompokkan ke 4 Kantor
# Wilayah asli. CATATAN: pemetaan cabang->wilayah ini best-effort dari data
# publik (nama cabang & wilayah nyata), BUKAN struktur organisasi internal
# BNI yang terverifikasi - representasi geografis masuk akal untuk demo,
# bukan klaim akurasi 100% terhadap struktur BNI sesungguhnya.
# =========================================================================
WILAYAH_BRANCHES = {
    "Kantor Wilayah Jakarta Senayan": [
        "KCP Melawai", "KCP Kyai Maja", "KC Gatot Subroto", "KCP Prof. Supomo Tebet",
        "KCP Blok M", "KCP Fatmawati", "KCP Pondok Indah", "KCP Cilandak",
        "KCP Pasar Minggu", "KCP Tebet Barat",
    ],
    "Kantor Wilayah Jakarta Kemayoran": [
        "KCP Menteng Raya", "KCP Gajah Mada", "KCP Pecenongan", "KCP Kramat Raya",
        "KC Sudirman", "KCP Cikini", "KCP Kelapa Gading", "KCP Pluit",
        "KCP Sunter", "KCP Kelapa Gading Bukit",
    ],
    "Kantor Wilayah Jakarta BSD": [
        "KCP Daan Mogot", "KCP Cengkareng", "KCP Kedoya", "KCP Kebon Jeruk",
        "KCP Puri Indah", "KCP Tangerang Kota", "KCP Karawaci", "KCP Cipondoh",
        "KC BSD", "KCP Ciputat", "KCP Bintaro", "KCP Serpong",
    ],
    "Kantor Wilayah 15 (Jakarta Timur)": [
        "KCP Rawamangun", "KCP Cawang", "KCP Jatinegara", "KCP Kramat Jati",
        "KCP Pulo Gadung", "KC Bekasi", "KCP Bekasi Timur", "KCP Cikarang",
        "KCP Jababeka", "KC Bogor", "KCP Cibinong", "KCP Bogor Baru",
        "KCP Dramaga", "KCP Depok Margonda", "KCP Depok Beji", "KCP Sawangan",
    ],
}
BRANCHES = [b for branches in WILAYAH_BRANCHES.values() for b in branches]
BRANCH_TO_REGION = {b: wilayah for wilayah, branches in WILAYAH_BRANCHES.items() for b in branches}

RM_LEVELS = ["Junior RB", "Senior RB"]

KELURAHAN_LOOKUP = [
    ("DKI Jakarta","Jakarta Selatan","Tebet","Tebet Timur", 28, 5.5),
    ("DKI Jakarta","Jakarta Selatan","Kebayoran Baru","Gunung",45, 6.0),
    ("DKI Jakarta","Jakarta Selatan","Pancoran","Duren Tiga", 30, 5.5),
    ("DKI Jakarta","Jakarta Pusat","Menteng","Menteng", 55, 6.5),
    ("DKI Jakarta","Jakarta Pusat","Cikini","Cikini", 40, 6.0),
    ("DKI Jakarta","Jakarta Timur","Kramat Jati","Kramat Jati", 18, 4.5),
    ("DKI Jakarta","Jakarta Timur","Cakung","Cakung Barat", 12, 4.0),
    ("DKI Jakarta","Jakarta Barat","Kebon Jeruk","Sukabumi Selatan", 22, 5.0),
    ("DKI Jakarta","Jakarta Barat","Cengkareng","Cengkareng Barat", 16, 4.2),
    ("DKI Jakarta","Jakarta Utara","Kelapa Gading","Kelapa Gading Barat", 25, 5.2),
    ("DKI Jakarta","Jakarta Utara","Pluit","Pluit", 27, 5.3),
    ("Jawa Barat","Bekasi","Bekasi Barat","Bintara", 9, 3.8),
    ("Jawa Barat","Bekasi","Bekasi Timur","Margahayu", 8, 3.6),
    ("Jawa Barat","Depok","Beji","Kemiri Muka", 10, 3.8),
    ("Jawa Barat","Depok","Sukmajaya","Mekarjaya", 8.5, 3.6),
    ("Jawa Barat","Bogor","Bogor Tengah","Paledang", 7, 3.4),
    ("Jawa Barat","Tangerang Selatan","Serpong","Rawa Buntu", 12, 4.0),
    ("Banten","Tangerang","Karawaci","Bojong Jaya", 9, 3.6),
    ("Banten","Tangerang","Cipondoh","Poris Plawad", 7.5, 3.4),
]

ASSET_TYPES = ["Tanah","Rumah","Ruko","Gudang"]
CERT_TYPES = ["SHM","HGB"]
LOAN_TYPES = ["KMK","KI","KPR","KKB","KK"]
COLLECT_MAP = {1:"Lancar", 2:"Dalam Perhatian Khusus (DPK)", 3:"Kurang Lancar",
                4:"Diragukan", 5:"Macet"}
OTHER_BANKS = ["Bank Mandiri","Bank BCA","Bank BRI","Bank BNI","Bank CIMB Niaga",
    "Bank Danamon","Bank Permata","Bank OCBC NISP","Bank Panin","BPR Mitra Usaha"]
DHN_REASONS = ["Tunggakan kredit >90 hari di bank lain","Terlibat kasus fraud dokumen",
    "Kredit macet yang belum diselesaikan","Cek/giro kosong berulang",
    "Laporan pihak ketiga terkait sengketa usaha"]

KUR_MIKRO_MAX = 50_000_000
KUR_KECIL_MAX = 500_000_000

TENOR_RANGE = {"KUR": (12, 36), "KMK": (12, 24), "KI": (36, 60)}
INTEREST_RATE = {"KUR": 0.06, "KMK": 0.11, "KI": 0.10}
INDUSTRY_KMK_WEIGHT = {
    "Perdagangan": 0.75, "Kuliner": 0.75, "Jasa": 0.60,
    "Manufaktur": 0.35, "Pertanian": 0.40, "Transportasi": 0.35,
}

TUJUAN_TEMPLATE = {
    ("KMK", "Perdagangan"): "Modal kerja pembelian stok barang dagangan {sub}",
    ("KMK", "Kuliner"): "Modal kerja pembelian bahan baku harian usaha {sub}",
    ("KMK", "Jasa"): "Modal kerja operasional usaha {sub}",
    ("KMK", "Manufaktur"): "Modal kerja pembelian bahan baku produksi {sub}",
    ("KMK", "Pertanian"): "Modal kerja operasional usaha {sub}",
    ("KMK", "Transportasi"): "Modal kerja operasional armada {sub}",
    ("KI", "Perdagangan"): "Perluasan/renovasi tempat usaha {sub}",
    ("KI", "Kuliner"): "Renovasi & penambahan peralatan usaha {sub}",
    ("KI", "Jasa"): "Pembelian peralatan/mesin usaha {sub}",
    ("KI", "Manufaktur"): "Pembelian mesin produksi tambahan usaha {sub}",
    ("KI", "Pertanian"): "Pengembangan lahan/kandang usaha {sub}",
    ("KI", "Transportasi"): "Pembelian armada kendaraan tambahan usaha {sub}",
    ("KUR", "Perdagangan"): "Tambahan modal kerja usaha {sub}",
    ("KUR", "Kuliner"): "Tambahan modal kerja usaha {sub}",
    ("KUR", "Jasa"): "Tambahan modal kerja usaha {sub}",
    ("KUR", "Manufaktur"): "Tambahan modal kerja usaha {sub}",
    ("KUR", "Pertanian"): "Tambahan modal kerja usaha {sub}",
    ("KUR", "Transportasi"): "Tambahan modal kerja usaha {sub}",
}


def generate_credit_application_details(loan_requested, industry, sub_industry, rng):
    kandidat = ["KMK", "KI"]
    if loan_requested <= KUR_KECIL_MAX:
        kandidat.append("KUR")

    if loan_requested <= KUR_MIKRO_MAX:
        weights_map = {"KUR": 0.65, "KMK": 0.25, "KI": 0.10}
    else:
        kmk_w = INDUSTRY_KMK_WEIGHT.get(industry, 0.5)
        if "KUR" in kandidat:
            weights_map = {"KUR": 0.30, "KMK": kmk_w * 0.70, "KI": (1 - kmk_w) * 0.70}
        else:
            weights_map = {"KMK": kmk_w, "KI": 1 - kmk_w}

    weights = np.array([weights_map[k] for k in kandidat])
    weights = weights / weights.sum()
    jenis_kredit = rng.choice(kandidat, p=weights)

    tenor_min, tenor_max = TENOR_RANGE[jenis_kredit]
    tenor_options = list(range(tenor_min, tenor_max + 1, 6))
    tenor = int(rng.choice(tenor_options))

    template = TUJUAN_TEMPLATE.get((jenis_kredit, industry), "Modal kerja usaha {sub}")
    tujuan = template.format(sub=sub_industry)

    return jenis_kredit, tenor, tujuan


def random_date(start_year, end_year):
    start = date(start_year, 1, 1)
    end = date(end_year, 8, 22)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

KODE_WILAYAH = {
    "Jakarta Selatan": "317401", "Jakarta Pusat": "317101", "Jakarta Timur": "317501",
    "Jakarta Barat": "317301", "Jakarta Utara": "317201",
    "Bekasi": "327501", "Depok": "327601", "Bogor": "327101",
    "Tangerang Selatan": "367401", "Tangerang": "367101",
}

def gen_nik(kota, tanggal_lahir, gender, idx):
    wilayah = KODE_WILAYAH.get(kota, "310101")
    d = tanggal_lahir.day + (40 if gender == "Perempuan" else 0)
    m, y = tanggal_lahir.month, tanggal_lahir.year % 100
    return f"{wilayah}{d:02d}{m:02d}{y:02d}{idx:04d}"


# =========================================================================
# 1. DUKCAPIL
# =========================================================================
def generate_dukcapil(n):
    rows = []
    for i in range(1, n+1):
        gender = random.choice(["Laki-Laki","Perempuan"])
        fname = random.choice(FIRST_NAMES_M if gender=="Laki-Laki" else FIRST_NAMES_F)
        lname = random.choice(LAST_NAMES)
        nama = f"{fname} {lname}"
        prov = random.choice(list(PROVINCES_CITIES.keys()))
        kota = random.choice(PROVINCES_CITIES[prov])
        tgl_lahir = random_date(1965, 2003)
        nik = gen_nik(kota, tgl_lahir, gender, i)
        rows.append({
            "dukcapil_id": f"DKC{i:06d}",
            "NIK": nik,
            "nama": nama,
            "tempat_lahir": kota,
            "tanggal_lahir": tgl_lahir.isoformat(),
            "jenis_kelamin": gender,
            "golongan_darah": random.choice(BLOOD_TYPE),
            "alamat": f"Jl. {random.choice(LAST_NAMES)} No. {random.randint(1,150)}",
            "rt_rw": f"{random.randint(1,12):03d}/{random.randint(1,10):03d}",
            "kelurahan_desa": random.choice(["Sukamaju","Sukajadi","Cempaka Putih",
                "Kebon Baru","Duren Sawit","Rawa Bunga","Cipete","Bintaro"]),
            "kecamatan": random.choice(["Tebet","Kramat Jati","Cengkareng",
                "Bekasi Timur","Sukmajaya","Serpong"]),
            "kota_kabupaten": kota,
            "provinsi": prov,
            "agama": random.choice(RELIGIONS),
            "status_perkawinan": random.choice(MARITAL),
            "pekerjaan": "Wiraswasta",
            "kewarganegaraan": "WNI",
            "berlaku_hingga": "SEUMUR HIDUP",
        })
    return pd.DataFrame(rows)


# =========================================================================
# 2. AGUNAN / ATR-BPN
# =========================================================================
def generate_agunan(dukcapil_df):
    rows = []
    agunan_lookup = {}
    for i, r in enumerate(dukcapil_df.itertuples(), start=1):
        nik = r.NIK
        prov, kota, kec, kel, harga_tanah, harga_bangunan = random.choice(KELURAHAN_LOOKUP)
        asset_type = random.choices(ASSET_TYPES, weights=[0.25,0.30,0.35,0.10])[0]
        land_area = round(np.random.uniform(60, 400), 1)
        building_area = 0.0 if asset_type == "Tanah" else round(land_area * np.random.uniform(0.5, 1.3), 1)
        htn = round(harga_tanah * np.random.uniform(0.85, 1.15), 2)
        hbg = round(harga_bangunan * np.random.uniform(0.85, 1.15), 2)
        nilai_tanah = round(land_area * htn * 1_000_000)
        nilai_bangunan = round(building_area * hbg * 1_000_000)
        total_value = nilai_tanah + nilai_bangunan
        ownership_match = np.random.choice(["Ya","Tidak"], p=[0.94, 0.06])
        row = {
            "atr_bpn_id": f"ATR{i:06d}",
            "NIK": nik,
            "asset_type": asset_type,
            "certificate_type": random.choice(CERT_TYPES),
            "certificate_number": f"{random.randint(10000,99999)}/{kel}",
            "provinsi": prov, "kota": kota, "kecamatan": kec, "kelurahan": kel,
            "land_area_m2": land_area,
            "building_area_m2": building_area,
            "nilai_tanah_per_m2": int(htn * 1_000_000),
            "nilai_bangunan_per_m2": int(hbg * 1_000_000),
            "nilai_tanah_total": nilai_tanah,
            "nilai_bangunan_total": nilai_bangunan,
            "total_collateral_value": total_value,
            "ownership_match": ownership_match,
        }
        rows.append(row)
        agunan_lookup[nik] = row
    return pd.DataFrame(rows), agunan_lookup


# =========================================================================
# 3. SLIK CREDIT HISTORY
# =========================================================================
def generate_slik(dukcapil_df):
    rows = []
    slik_summary = {}
    rid = 1
    for r in dukcapil_df.itertuples():
        nik = r.NIK
        n_loans = np.random.choice([0,1,2,3], p=[0.15,0.40,0.30,0.15])
        worst = 1
        total_installment = 0
        for _ in range(n_loans):
            plafond = int(np.random.choice([25,50,75,100,150,200,300,500]) * 1_000_000)
            outstanding = int(plafond * np.random.uniform(0.2, 0.95))
            tenor = int(np.random.choice([12,24,36,48,60]))
            installment = int(plafond / tenor * np.random.uniform(1.02,1.15))
            collect = np.random.choice([1,2,3,4,5], p=[0.72,0.14,0.07,0.04,0.03])
            worst = max(worst, collect)
            total_installment += installment
            rows.append({
                "slik_record_id": f"SLK{rid:06d}",
                "NIK": nik,
                "inquiry_date": random_date(2024,2026).isoformat(),
                "bank_name": random.choice(OTHER_BANKS),
                "loan_type": random.choice(LOAN_TYPES),
                "plafond": plafond,
                "outstanding_balance": outstanding,
                "installment_amount": installment,
                "tenor_month": tenor,
                "collectability": int(collect),
                "collectability_label": COLLECT_MAP[collect],
            })
            rid += 1
        slik_summary[nik] = {"worst_collect": worst, "total_installment": total_installment, "n_loans": n_loans}
    return pd.DataFrame(rows), slik_summary


# =========================================================================
# 4. DHN
# =========================================================================
def generate_dhn(dukcapil_df, slik_summary):
    rows = []
    dhn_lookup = {}
    for i, r in enumerate(dukcapil_df.itertuples(), start=1):
        nik = r.NIK
        worst = slik_summary[nik]["worst_collect"]
        p_blacklist = {1:0.01, 2:0.03, 3:0.10, 4:0.25, 5:0.45}[worst]
        status = np.random.choice(["Ya","Tidak"], p=[p_blacklist, 1-p_blacklist])
        reason = random.choice(DHN_REASONS) if status == "Ya" else ""
        row = {
            "dhn_id": f"DHN{i:06d}",
            "NIK": nik,
            "status_dhn": status,
            "alasan": reason,
            "tanggal_input": random_date(2023,2026).isoformat(),
        }
        rows.append(row)
        dhn_lookup[nik] = status
    return pd.DataFrame(rows), dhn_lookup


# =========================================================================
# 5. LAPORAN KEUANGAN
# =========================================================================
def generate_laporan_keuangan(dukcapil_df):
    rows = []
    fin_summary = {}
    rid = 1
    for r in dukcapil_df.itertuples():
        nik = r.NIK
        revenue_2024 = np.random.lognormal(mean=16.8, sigma=0.6)
        growth = np.random.normal(0.12, 0.20)
        revenue_2025 = revenue_2024 * (1 + growth)
        margin = np.clip(np.random.normal(0.11, 0.05), 0.01, 0.35)
        recs = []
        for yr, rev in [(2024, revenue_2024), (2025, revenue_2025)]:
            net_profit = rev * margin * np.random.uniform(0.85,1.15)
            total_asset = rev * np.random.uniform(1.1, 2.0)
            total_liability = total_asset * np.random.uniform(0.2, 0.7)
            op_cf = net_profit * np.random.uniform(0.8, 1.4)
            row = {
                "laporan_id": f"FIN{rid:06d}", "NIK": nik, "year": yr,
                "revenue": int(rev), "net_profit": int(net_profit),
                "total_asset": int(total_asset), "total_liability": int(total_liability),
                "operating_cashflow": int(op_cf),
            }
            rows.append(row); recs.append(row); rid += 1
        fin_summary[nik] = {
            "revenue_growth": growth,
            "latest_revenue": recs[1]["revenue"],
            "latest_net_profit": recs[1]["net_profit"],
            "latest_liability": recs[1]["total_liability"],
        }
    return pd.DataFrame(rows), fin_summary


# =========================================================================
# 6. BANK ACCOUNT
# =========================================================================
def generate_bank_account(dukcapil_df, fin_summary):
    rows = []
    cf_summary = {}
    aid = 1
    for r in dukcapil_df.itertuples():
        nik = r.NIK
        n_acc = np.random.choice([1,2], p=[0.65,0.35])
        monthly_rev = fin_summary[nik]["latest_revenue"] / 12
        best_avg_balance = 0
        for _ in range(n_acc):
            avg_credit = monthly_rev * np.random.uniform(0.6, 1.1)
            avg_debit = avg_credit * np.random.uniform(0.7, 0.98)
            avg_balance = max(avg_credit - avg_debit, 0) * np.random.uniform(40, 100)
            best_avg_balance = max(best_avg_balance, avg_balance)

            account = {
                "account_id": f"ACC{aid:06d}",
                "NIK": nik,
                "account_number": f"{random.randint(1000000000,9999999999):010d}",
                "bank_name": random.choice(["BNI"] + OTHER_BANKS),
                "account_type": random.choice(["Giro","Tabungan"]),
                "account_status": np.random.choice(["Aktif","Dormant"], p=[0.93,0.07]),
                "opened_date": random_date(2015,2025).isoformat(),
                "average_balance_6m": int(avg_balance),
                "average_monthly_credit": int(avg_credit),
                "average_monthly_debit": int(avg_debit),
                "transaction_frequency_monthly": int(np.random.uniform(20,200)),
                "overdraft_count_6m": int(np.random.choice([0,0,0,1,2,3], p=[0.6,0.15,0.1,0.08,0.04,0.03])),
            }

            if account["account_status"] == "Dormant":
                current_balance = avg_balance * CB_RNG.uniform(0.05, 0.25)
            elif account["overdraft_count_6m"] > 0 and CB_RNG.random() < 0.12:
                current_balance = -avg_debit * CB_RNG.uniform(0.02, 0.15)
            else:
                current_balance = avg_balance * CB_RNG.uniform(0.4, 1.8)
            account["current_balance"] = int(current_balance)

            rows.append(account)
            aid += 1
        cf_summary[nik] = {"best_avg_balance": best_avg_balance}
    return pd.DataFrame(rows), cf_summary


# =========================================================================
# 6b. RM MASTER (v3: cabang asli, RM per cabang variatif, kapasitas 5-25)
# =========================================================================
RM_PER_BRANCH_RANGE = (3, 8)       # jumlah RM per cabang, BUKAN fixed 4 lagi
KAPASITAS_JUNIOR_RANGE = (5, 18)   # kapasitas_max_nasabah utk Junior RB
KAPASITAS_SENIOR_RANGE = (10, 25)  # kapasitas_max_nasabah utk Senior RB

def generate_rm_master(n_customers):
    rows = []
    rid = 1
    for branch in BRANCHES:
        n_rm_branch = int(RM_RNG.integers(RM_PER_BRANCH_RANGE[0], RM_PER_BRANCH_RANGE[1] + 1))
        for _ in range(n_rm_branch):
            gender = RM_RNG.choice(["Laki-Laki", "Perempuan"])
            fname = RM_RNG.choice(FIRST_NAMES_M if gender == "Laki-Laki" else FIRST_NAMES_F)
            lname = RM_RNG.choice(LAST_NAMES)
            level = RM_RNG.choice(RM_LEVELS, p=[0.6, 0.4])
            kap_range = KAPASITAS_JUNIOR_RANGE if level == "Junior RB" else KAPASITAS_SENIOR_RANGE
            kapasitas = int(RM_RNG.integers(kap_range[0], kap_range[1] + 1))
            rows.append({
                "rm_id": f"RM{rid:04d}",
                "rm_name": f"{fname} {lname}",
                "branch_name": branch,
                "region": BRANCH_TO_REGION[branch],
                "jabatan": "Relationship Banking Officer",
                "level": level,
                "kapasitas_max_nasabah": kapasitas,
                "join_date": (date(2015, 1, 1) + timedelta(
                    days=int(RM_RNG.integers(0, (date(2025, 12, 31) - date(2015, 1, 1)).days)))).isoformat(),
            })
            rid += 1

    rm_df = pd.DataFrame(rows)

    # Safety check: total kapasitas harus >= N_CUSTOMERS, kalau kurang
    # (jarang terjadi tapi bisa krn randomness) top-up beberapa RM secara acak
    # sampai cukup - tidak ada RM yang "dipaksa" melebihi KAPASITAS_SENIOR_RANGE max.
    total_kapasitas = rm_df["kapasitas_max_nasabah"].sum()
    while total_kapasitas < n_customers:
        idx = RM_RNG.integers(0, len(rm_df))
        if rm_df.loc[idx, "kapasitas_max_nasabah"] < KAPASITAS_SENIOR_RANGE[1]:
            rm_df.loc[idx, "kapasitas_max_nasabah"] += 1
            total_kapasitas += 1

    print(f"  rm_master: {len(rm_df)} RM tersebar di {len(BRANCHES)} cabang, "
          f"total kapasitas {total_kapasitas} (kebutuhan {n_customers})")
    return rm_df


def build_rm_slot_pool(rm_df, n_customers):
    """
    Slot pool: tiap rm_id diulang sebanyak kapasitas_max_nasabah miliknya,
    lalu diacak. N_CUSTOMERS slot pertama dipakai untuk assignment -
    menjamin TIDAK ADA RM yang jumlah nasabahnya melebihi kapasitasnya.
    """
    slots = []
    for row in rm_df.itertuples():
        slots.extend([row.rm_id] * row.kapasitas_max_nasabah)
    RM_RNG.shuffle(slots)
    return slots[:n_customers]


# =========================================================================
# 7. RETAIL CUSTOMER PROFILE (application) + LABEL diterima/ditolak
# =========================================================================
def compute_label_score(worst_collect, dhn_status, growth, net_profit, dsr,
                          collateral_ratio, industry):
    s_character = {1:1.0, 2:0.8, 3:0.5, 4:0.25, 5:0.0}[worst_collect]
    if dhn_status == "Ya":
        s_character = min(s_character, 0.1)
    s_capacity = np.clip(0.5 + growth*1.2, 0, 1) * 0.5 + np.clip(1 - dsr, 0, 1) * 0.5
    s_capacity = np.clip(s_capacity, 0, 1)
    s_collateral = np.clip(collateral_ratio / 1.5, 0, 1)
    s_condition = 1 - INDUSTRY_RISK.get(industry, 0.1) * 4
    s_condition = np.clip(s_condition, 0, 1)

    score = 0.35*s_character + 0.30*s_capacity + 0.20*s_collateral + 0.15*s_condition
    score += np.random.normal(0, 0.05)
    return np.clip(score, 0, 1)

def generate_customer_profile(dukcapil_df, agunan_lookup, slik_summary, dhn_lookup,
                                fin_summary, cf_summary, rm_df, rm_slots):
    rm_id_to_branch = dict(zip(rm_df["rm_id"], rm_df["branch_name"]))
    rows = []
    for i, r in enumerate(dukcapil_df.itertuples(), start=1):
        nik = r.NIK
        prov, kota = r.provinsi, r.kota_kabupaten
        legal_entity = random.choice(["PT","CV","UD"])
        industry = random.choice(list(INDUSTRIES.keys()))
        sub_industry = random.choice(INDUSTRIES[industry])
        business_age = int(np.random.uniform(1, 20))
        employee_count = int(np.random.uniform(2, 80))
        monthly_turnover = fin_summary[nik]["latest_revenue"] / 12

        agunan = agunan_lookup[nik]
        loan_requested = int(np.random.choice([50,75,100,150,200,300,500,750,1000]) * 1_000_000)
        loan_requested = min(loan_requested, 10_000_000_000)
        collateral_ratio = round(agunan["total_collateral_value"] / max(loan_requested,1), 2)
        collateral_size_m2 = round(agunan["land_area_m2"] + agunan["building_area_m2"], 1)

        slik = slik_summary[nik]

        jenis_kredit_diajukan, tenor_diajukan_bulan, tujuan_penggunaan_kredit = \
            generate_credit_application_details(loan_requested, industry, sub_industry, CREDIT_APP_RNG)

        annual_rate = INTEREST_RATE[jenis_kredit_diajukan]
        new_installment = loan_requested / tenor_diajukan_bulan * (1 + annual_rate * tenor_diajukan_bulan / 12)
        dsr = (slik["total_installment"] + new_installment) / max(monthly_turnover, 1)

        score = compute_label_score(
            worst_collect=slik["worst_collect"], dhn_status=dhn_lookup[nik],
            growth=fin_summary[nik]["revenue_growth"], net_profit=fin_summary[nik]["latest_net_profit"],
            dsr=dsr, collateral_ratio=collateral_ratio, industry=industry,
        )
        label = "Diterima" if score >= 0.55 else "Ditolak"

        # --- BARU v3: RM diambil dari slot pool (kapasitas-terjamin),
        # branch_name customer IKUT dari cabang RM yang kepilih (bukan
        # dipilih independen lebih dulu seperti v2) ---
        rm_id = rm_slots[i - 1]
        branch_name = rm_id_to_branch[rm_id]

        row = {
            "application_id": f"APP{2026}{i:05d}",
            "NIK": nik,
            "cif_number": f"CIF{1000000+i}",
            "application_date": random_date(2025,2026).isoformat(),
            "customer_type": "UMKM",
            "company_name": f"{random.choice(['PT','CV','UD'])} {random.choice(LAST_NAMES)} {random.choice(['Jaya','Makmur','Sejahtera','Abadi','Mandiri'])}",
            "legal_entity": legal_entity,
            "owner_name": r.nama,
            "owner_gender": "L" if r.jenis_kelamin=="Laki-Laki" else "P",
            "owner_age": date.today().year - int(r.tanggal_lahir[:4]),
            "owner_marital_status": r.status_perkawinan,
            "owner_education": random.choice(EDUCATION),
            "province": prov, "city": kota,
            "district": r.kecamatan,
            "region": random.choice(REGIONS),  # kolom GENERIK, independen, TIDAK diubah
            "branch_name": branch_name,        # BARU v3: ikut dari cabang RM
            "industry": industry, "sub_industry": sub_industry,
            "business_age_year": business_age,
            "employee_count": employee_count,
            "monthly_turnover_est": int(monthly_turnover),
            "transaction_frequency_monthly": int(np.random.uniform(30,200)),
            "loan_requested": loan_requested,
            "jenis_kredit_diajukan": jenis_kredit_diajukan,
            "tenor_diajukan_bulan": tenor_diajukan_bulan,
            "tujuan_penggunaan_kredit": tujuan_penggunaan_kredit,
            "collateral_type": agunan["asset_type"],
            "collateral_location": f"{agunan['kelurahan']}, {agunan['kota']}",
            "collateral_province": agunan["provinsi"], "collateral_city": agunan["kota"],
            "collateral_size_m2": collateral_size_m2,
            "collateral_market_value": agunan["total_collateral_value"],
            "collateral_liquidation_value": int(agunan["total_collateral_value"] * 0.8),
            "collateral_ratio": collateral_ratio,
            "certificate_type": agunan["certificate_type"],
            "ownership_match": agunan["ownership_match"],
            "estimated_dsr": round(min(dsr,3.0), 2),
            "eligibility_score": round(float(score), 3),
            "label": label,
            "rm_id": rm_id,  # BARU v3: dari slot pool, bukan RM_RNG.choice bebas
        }

        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
# UTILITY (dipakai di tahap scoring/master_scored.csv, BUKAN generator ini)
# =========================================================================
def kategorikan_kelayakan(eligibility_score: float) -> str:
    if eligibility_score >= 0.80:
        return "Layak"
    elif eligibility_score >= 0.55:
        return "Layak Bersyarat"
    elif eligibility_score >= 0.40:
        return "Perlu Review Ulang"
    else:
        return "Tidak Layak"


# =========================================================================
# MAIN
# =========================================================================
def main():
    print(f"Generating {N_CUSTOMERS} customers...")
    dukcapil_df = generate_dukcapil(N_CUSTOMERS)
    agunan_df, agunan_lookup = generate_agunan(dukcapil_df)
    slik_df, slik_summary = generate_slik(dukcapil_df)
    dhn_df, dhn_lookup = generate_dhn(dukcapil_df, slik_summary)
    fin_df, fin_summary = generate_laporan_keuangan(dukcapil_df)
    bank_df, cf_summary = generate_bank_account(dukcapil_df, fin_summary)

    rm_df = generate_rm_master(N_CUSTOMERS)
    rm_slots = build_rm_slot_pool(rm_df, N_CUSTOMERS)

    profile_df = generate_customer_profile(dukcapil_df, agunan_lookup, slik_summary,
                                            dhn_lookup, fin_summary, cf_summary, rm_df, rm_slots)

    tables = {
        "retail_customer_profile": profile_df,
        "dukcapil": dukcapil_df,
        "slik_credit_history": slik_df,
        "dhn": dhn_df,
        "agunan_atr_bpn": agunan_df,
        "laporan_keuangan": fin_df,
        "bank_account": bank_df,
        "rm_master": rm_df,
    }
    for name, df in tables.items():
        if "NIK" in df.columns:
            df["NIK"] = df["NIK"].astype(str)
        if "account_number" in df.columns:
            df["account_number"] = df["account_number"].astype(str)
        path = os.path.join(OUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {name:28s} -> {len(df):6d} baris -> {path}")

    print("\nDistribusi label (retail_customer_profile):")
    print(profile_df["label"].value_counts(normalize=True).round(3))
    print("\nDistribusi jumlah nasabah per RM (harus semua <= kapasitas_max_nasabah):")
    nasabah_per_rm = profile_df["rm_id"].value_counts()
    kap_map = rm_df.set_index("rm_id")["kapasitas_max_nasabah"]
    over_capacity = (nasabah_per_rm > kap_map.reindex(nasabah_per_rm.index)).sum()
    print(f"  min={nasabah_per_rm.min()}, max={nasabah_per_rm.max()}, "
          f"median={nasabah_per_rm.median():.0f}, RM melebihi kapasitas={over_capacity} (harus 0)")
    print("\nSelesai. Semua file CSV ada di:", OUT_DIR)
    return tables

if __name__ == "__main__":
    tables = main()
