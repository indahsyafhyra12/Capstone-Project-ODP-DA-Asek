# Penyesuaian Sistem untuk Data Terbaru (v2)

Dokumen ini rangkuman lengkap: apa yang berubah di data, file mana yang
perlu disesuaikan, logic barunya gimana, dan kode implementasinya —
biar kamu paham dulu sebelum minta Claude Code eksekusi.

---

## 1. Apa yang berubah di data (v2)

| Kolom baru | Tabel | Sudah dipakai di mana? |
|---|---|---|
| `jenis_kredit_diajukan` (KMK/KI/KUR) | `retail_customer_profile.csv` | ❌ Belum dipakai di manapun |
| `tenor_diajukan_bulan` | `retail_customer_profile.csv` | ❌ Belum dipakai di manapun |
| `tujuan_penggunaan_kredit` | `retail_customer_profile.csv` | ❌ Belum dipakai di manapun (cuma tampilan teks) |
| `current_balance` / `bank_best_current_balance` | `bank_account.csv` → agregat | ✅ Sudah dipakai di `risk_ml_pipeline.py` (`_calibrated_cashflow`), ❌ belum di `agent_pipeline.py` |
| `rm_id`, `rm_master.csv` | `retail_customer_profile.csv` | ✅ Sudah untuk monitoring, sengaja TIDAK jadi fitur skor (governance) |

**Fokus dokumen ini: 3 kolom pertama** (jenis/tenor/tujuan) — itu yang paling belum tersentuh.

---

## 2. File yang perlu disesuaikan

| # | File | Perubahan |
|---|---|---|
| 1 | `utils/agent_pipeline.py` | Tambah fungsi `recommend_credit_type()` (baru); panggil dari `risk_agent()`; tambah KUR ke tabel tenor/bunga; **fix Cashflow Agent** (masih pakai `ratio/0.8`, belum dikalibrasi) |
| 2 | `utils/risk_ml_pipeline.py` | Tambah pemanggilan `recommend_credit_type()` yang sama (impor dari `agent_pipeline.py`, jangan duplikat); tambah KUR ke `apply_policy_engine()` |
| 3 | `pages/2_Detail_Nasabah.py` | Tambah kartu baru: "Kesesuaian Jenis Kredit" (nampilin `jenis_kredit_diajukan` vs rekomendasi + `catatan`) |
| 4 | `pages/3_Pengajuan_Credit_Baru.py` | Form manual perlu 3 field baru: dropdown jenis kredit diajukan, number input tenor diajukan, text area tujuan |
| 5 | `data_dictionary.md` / `data_dictionary_v2.md` | Update dokumentasi kolom & field output baru |

**File yang TIDAK perlu diubah:** `feature_builder.py` (kolom baru ini bukan fitur ML, jadi gak perlu ikut agregasi buat model), `src/agents/*.py` (kalau memang gak dipakai live, biarkan dulu — cek dulu apakah aktif).

---

## 3. Logic rule-based lengkap (INI YANG BARU)

### Prinsip dasar
Sistem **BUKAN** menebak dari nol, tapi **memvalidasi** apa yang diajukan
debitur (`jenis_kredit_diajukan`, `tenor_diajukan_bulan`), baru kasih
rekomendasi kalau ternyata gak sesuai kemampuan bayarnya.

### Langkah 1 — Cek plafon KUR (hard rule, gak bisa ditawar)
```
JIKA jenis_diajukan == "KUR" DAN loan_requested > Rp 500 juta:
    → TOLAK, ganti rekomendasi ke KMK/KI (sesuai rasio pinjaman/omset)
    → catatan: "KUR tidak dapat diberikan untuk plafon di atas Rp500 juta
      sesuai ketentuan pemerintah. Direkomendasikan [KMK/KI]."
```

### Langkah 2 — Hitung DSR di TERM yang diajukan (bukan asumsi sistem)
```
cicilan = loan_requested / tenor_diajukan × (1 + bunga[jenis_diajukan] × tenor_diajukan/12)
dsr_pengajuan = (cicilan_kredit_lain_dari_SLIK + cicilan) / monthly_turnover_est
```
Bunga per jenis: KUR=6% (subsidi), KMK=11%, KI=10% — sama persis konstanta
yang sudah ada di generator v2 (`INTEREST_RATE`), dipakai ulang di sini biar
konsisten.

### Langkah 3 — Bandingkan ke ambang aman (DSR ≤ 40%)
```
JIKA dsr_pengajuan ≤ 0.40:
    → jenis_kredit_rekomendasi = jenis_kredit_diajukan (SESUAI, disetujui apa adanya)

JIKA dsr_pengajuan > 0.40:
    a) Coba PERPANJANG TENOR dulu (masih di jenis kredit yang sama, dalam
       rentang tenor yang diizinkan) sampai DSR ≤ 0.40
    b) Kalau tenor maksimal pun masih berat, DAN jenis yang diajukan BUKAN
       KI → coba ALIH JENIS ke KI (tenor lebih panjang secara alami)
    c) Kalau masih berat juga → catatan "perlu review manual/tambahan agunan"
```

### Kenapa urutannya begini
Perpanjang tenor itu perubahan **paling kecil** (nasabah tetap dapet jenis
kredit yang sama, cuma jangka waktunya beda) — baru kalau itu gak cukup,
opsi yang lebih besar (ganti jenis kredit) dipertimbangkan. Prinsip
"perubahan minimal dulu" ini lazim dipakai di sistem rekomendasi kebijakan.

### Output field baru
| Field | Isi |
|---|---|
| `jenis_kredit_sesuai` | `True` / `False` |
| `dsr_pada_pengajuan` | DSR dihitung dari term yang diajukan nasabah |
| `catatan_kesesuaian_kredit` | Narasi penjelasan (lihat contoh di kode) |

---

## 4. Kode implementasi (sudah ditest, hasil di bawah kode)

Tambahkan ke **`utils/agent_pipeline.py`** (dekat konstanta lain seperti `TENOR_MONTHS`):

```python
# ---------------------------------------------------------------------------
# 8. Credit Type Recommendation (BARU — cek kesesuaian pengajuan vs kemampuan bayar)
# ---------------------------------------------------------------------------

KUR_KECIL_MAX = 500_000_000
TENOR_RANGE_DIAJUKAN = {"KUR": (12, 36), "KMK": (12, 24), "KI": (36, 60)}
INTEREST_RATE_DIAJUKAN = {"KUR": 0.06, "KMK": 0.11, "KI": 0.10}
DSR_AMAN = 0.40


def _hitung_dsr_pengajuan(loan_requested, jenis, tenor, slik_installment_lain, turnover):
    rate = INTEREST_RATE_DIAJUKAN[jenis]
    cicilan = loan_requested / tenor * (1 + rate * tenor / 12)
    dsr = (slik_installment_lain + cicilan) / max(turnover, 1)
    return dsr, cicilan


def recommend_credit_type(loan_requested, monthly_turnover_est, jenis_kredit_diajukan,
                            tenor_diajukan_bulan, slik_total_installment_other=0):
    """Bandingkan jenis kredit & tenor yang DIAJUKAN nasabah terhadap
    kemampuan bayarnya (DSR), bukan cuma menebak dari nominal seperti
    _recommend_loan_type() lama. Dipanggil terpisah dari risk_agent()
    supaya bisa dites/dipakai ulang independen."""
    ratio = loan_requested / max(monthly_turnover_est, 1)
    default_jenis = _recommend_loan_type(loan_requested, monthly_turnover_est)
    if loan_requested <= KUR_KECIL_MAX and ratio <= 6:
        default_jenis = "KUR"

    if not jenis_kredit_diajukan or not tenor_diajukan_bulan:
        # data pengajuan gak ada (mis. simulasi manual tanpa isi field ini)
        return {"jenis_kredit_rekomendasi": default_jenis, "jenis_kredit_sesuai": None,
                "dsr_pada_pengajuan": None, "catatan_kesesuaian_kredit": None}

    if jenis_kredit_diajukan == "KUR" and loan_requested > KUR_KECIL_MAX:
        return {
            "jenis_kredit_rekomendasi": default_jenis, "jenis_kredit_sesuai": False,
            "dsr_pada_pengajuan": None,
            "catatan_kesesuaian_kredit": (
                f"KUR tidak dapat diberikan untuk plafon di atas Rp500 juta sesuai "
                f"ketentuan pemerintah. Direkomendasikan {default_jenis}."),
        }

    dsr, _ = _hitung_dsr_pengajuan(loan_requested, jenis_kredit_diajukan, tenor_diajukan_bulan,
                                     slik_total_installment_other, monthly_turnover_est)
    if dsr <= DSR_AMAN:
        return {
            "jenis_kredit_rekomendasi": jenis_kredit_diajukan, "jenis_kredit_sesuai": True,
            "dsr_pada_pengajuan": round(dsr, 3),
            "catatan_kesesuaian_kredit": (
                f"Pengajuan {jenis_kredit_diajukan} tenor {tenor_diajukan_bulan} bulan sesuai, "
                f"DSR {dsr*100:.0f}%."),
        }

    tmin, tmax = TENOR_RANGE_DIAJUKAN[jenis_kredit_diajukan]
    for t in range(tenor_diajukan_bulan + 6, tmax + 1, 6):
        dsr2, _ = _hitung_dsr_pengajuan(loan_requested, jenis_kredit_diajukan, t,
                                          slik_total_installment_other, monthly_turnover_est)
        if dsr2 <= DSR_AMAN:
            return {
                "jenis_kredit_rekomendasi": jenis_kredit_diajukan, "jenis_kredit_sesuai": False,
                "dsr_pada_pengajuan": round(dsr, 3),
                "catatan_kesesuaian_kredit": (
                    f"Tenor {tenor_diajukan_bulan} bulan terlalu berat (DSR {dsr*100:.0f}%). "
                    f"Disarankan perpanjang tenor jadi {t} bulan (DSR turun ke {dsr2*100:.0f}%)."),
            }

    if jenis_kredit_diajukan != "KI":
        tmin_ki, tmax_ki = TENOR_RANGE_DIAJUKAN["KI"]
        for t in range(tmin_ki, tmax_ki + 1, 6):
            dsr3, _ = _hitung_dsr_pengajuan(loan_requested, "KI", t,
                                              slik_total_installment_other, monthly_turnover_est)
            if dsr3 <= DSR_AMAN:
                return {
                    "jenis_kredit_rekomendasi": "KI", "jenis_kredit_sesuai": False,
                    "dsr_pada_pengajuan": round(dsr, 3),
                    "catatan_kesesuaian_kredit": (
                        f"DSR pada {jenis_kredit_diajukan} terlalu tinggi ({dsr*100:.0f}%). "
                        f"Disarankan alih ke KI tenor {t} bulan (DSR turun ke {dsr3*100:.0f}%)."),
                }

    return {
        "jenis_kredit_rekomendasi": jenis_kredit_diajukan, "jenis_kredit_sesuai": False,
        "dsr_pada_pengajuan": round(dsr, 3),
        "catatan_kesesuaian_kredit": (
            f"DSR tetap tinggi ({dsr*100:.0f}%) meski tenor maksimal — "
            f"disarankan review manual/tambahan agunan/penjamin."),
    }
```

**Update tabel tenor & bunga existing** (tambah baris KUR):
```python
TENOR_MONTHS["KUR"] = {"Hijau": 36, "Kuning": 24}
INTEREST_RATE_PA["KUR"] = 6.0
```

**Panggil dari `score_application()`**, tambahkan setelah baris `risk = risk_agent(...)`:
```python
credit_type_check = recommend_credit_type(
    loan_requested=get("loan_requested") or 0,
    monthly_turnover_est=get("monthly_turnover_est") or 1,
    jenis_kredit_diajukan=get("jenis_kredit_diajukan"),
    tenor_diajukan_bulan=get("tenor_diajukan_bulan"),
    slik_total_installment_other=get("slik_total_installment_other") or 0,
)
```
lalu masukkan `credit_type_check` ke dict return `score_application()` dan
`_flatten()` (pola yang sama seperti field lain).

**Untuk `risk_ml_pipeline.py`**: cukup import & panggil fungsi yang sama:
```python
from utils.agent_pipeline import recommend_credit_type
# ...dalam predict_credit_screening(), setelah policy diterapkan:
credit_type_check = recommend_credit_type(
    row.get("loan_requested"), row.get("monthly_turnover_est"),
    row.get("jenis_kredit_diajukan"), row.get("tenor_diajukan_bulan"),
    row.get("slik_total_installment_other") or 0,
)
```
**Jangan tulis ulang logic-nya di file ini** — panggil fungsi yang sama dari
`agent_pipeline.py`, persis pola yang sudah dipakai untuk `identity_agent`
dkk. Ini penting supaya kedua sistem selalu konsisten untuk bagian ini,
walau `risk_score`-nya sendiri tetap boleh beda (ML vs formula).

---

## 5. Hasil test (sudah dijalankan, bukan teori)

| Skenario | Input | Output |
|---|---|---|
| KMK tenor pendek, DSR berat | Rp200jt, KMK, 12bln, omset 30jt | `sesuai=False`, DSR 65% → **saran perpanjang ke 24 bulan (DSR jadi 37%)** |
| KI tenor panjang, DSR aman | Rp200jt, KI, 60bln, omset 30jt | `sesuai=True`, DSR 20% — **disetujui sesuai pengajuan** |
| KUR plafon kelebihan | Rp600jt, KUR | `sesuai=False` — **KUR ditolak, diarahkan ke KMK** (melanggar plafon Rp500jt) |

---

## 6. Urutan eksekusi yang disarankan

1. Tambah fungsi `recommend_credit_type()` + update tabel KUR di `agent_pipeline.py`
2. Test lokal dulu (panggil fungsinya langsung dengan beberapa skenario, kayak di atas)
3. Wire ke `score_application()`/`_flatten()`/`score_dataframe()`
4. Import & panggil fungsi yang sama di `risk_ml_pipeline.py` (bukan duplikat logic)
5. Update UI: kartu baru di `pages/2_Detail_Nasabah.py`, field form baru di `pages/3_Pengajuan_Credit_Baru.py`
6. Update `data_dictionary.md`
7. Commit & push, cek di Streamlit Cloud

**Belum termasuk di scope ini** (perlu didiskusikan terpisah kalau mau
sekalian): fix Cashflow Agent di `agent_pipeline.py` (bug lama yang sudah
sering dibahas), konsolidasi 3 threshold berbeda jadi 1.
