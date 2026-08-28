"""7-agent rule-based screening pipeline (identity, credit_history, dhn,
collateral, financial, cashflow, risk).

All functions are pure and take raw values (not a whole dataframe row), so
they work identically for batch scoring (`score_dataframe`) and for the
manual Simulasi form (`score_application` with a plain dict).

Scoring design notes
---------------------
The 5C weight structure below (Character 0.30 / Capacity 0.30 /
Collateral 0.20 / Cashflow-as-Condition 0.20) is inspired by the
`compute_label_score` formula in notebooks/01_dataset_generation.ipynb
(Character 0.35 / Capacity 0.30 / Collateral 0.20 / Condition 0.15), but is
recomputed independently from raw columns and swaps the notebook's
industry-risk "Condition" component for a cashflow/liquidity component,
since that's the 6th agent this app requires. It intentionally does NOT
read `eligibility_score` / `label` (data_dictionary.md flags those as
leakage-prone ground truth, not model/agent input).

Decision thresholds, zones, and loan-term rules (interest rate, tenor,
approved amount, loan type) are not specified anywhere in the repo and
were designed for this app — see risk_agent() below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DUKCAPIL_PATH = Path(__file__).parent.parent / "data" / "raw" / "dukcapil.csv"
_dukcapil_niks_cache: set | None = None
_dukcapil_names_cache: dict | None = None


def _dukcapil_niks() -> set:
    """NIK-NIK yang terdaftar di data Dukcapil (dimuat & di-cache sekali)."""
    global _dukcapil_niks_cache
    if _dukcapil_niks_cache is None:
        try:
            _dukcapil_niks_cache = set(pd.read_csv(_DUKCAPIL_PATH, dtype={"NIK": str})["NIK"])
        except FileNotFoundError:
            _dukcapil_niks_cache = set()
    return _dukcapil_niks_cache


def _dukcapil_names() -> dict:
    """Mapping NIK -> nama sesuai Dukcapil (dimuat & di-cache sekali)."""
    global _dukcapil_names_cache
    if _dukcapil_names_cache is None:
        try:
            df = pd.read_csv(_DUKCAPIL_PATH, dtype={"NIK": str})
            _dukcapil_names_cache = dict(zip(df["NIK"], df["nama"]))
        except FileNotFoundError:
            _dukcapil_names_cache = {}
    return _dukcapil_names_cache


def _normalize_name(name) -> str:
    return " ".join(str(name).strip().split()).lower()


def _num(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _yes(value):
    return str(value).strip().lower() in ("ya", "yes", "true", "1")


def score_status(score, good=0.75, ok=0.5, low=0.3):
    if score >= good:
        return "Baik"
    if score >= ok:
        return "Cukup"
    if score >= low:
        return "Kurang"
    return "Buruk"


# ---------------------------------------------------------------------------
# 1. Identity agent
# ---------------------------------------------------------------------------

MIN_OWNER_AGE, MAX_OWNER_AGE = 21, 65


def identity_agent(nik, owner_age, owner_name):
    nik_str = str(nik).strip() if nik is not None else ""
    valid_format = nik_str.isdigit() and len(nik_str) == 16
    found_in_dukcapil = valid_format and nik_str in _dukcapil_niks()

    name_matches = False
    dukcapil_name = None
    if found_in_dukcapil:
        dukcapil_name = _dukcapil_names().get(nik_str)
        name_matches = _normalize_name(owner_name) == _normalize_name(dukcapil_name)

    age = None
    try:
        if owner_age is not None and not (isinstance(owner_age, float) and pd.isna(owner_age)):
            age = int(owner_age)
    except (TypeError, ValueError):
        age = None
    valid_age = age is not None and MIN_OWNER_AGE <= age <= MAX_OWNER_AGE

    notes = []
    if not valid_format:
        notes.append(f"NIK tidak valid (harus 16 digit angka): '{nik_str}'")
    elif not found_in_dukcapil:
        notes.append(f"NIK '{nik_str}' tidak ditemukan di data Dukcapil")
    elif not name_matches:
        notes.append(f"Nama '{owner_name}' tidak sesuai data Dukcapil untuk NIK ini")
    if not valid_age:
        shown = age if age is not None else "(tidak diisi)"
        notes.append(f"Usia pemohon {shown} di luar rentang layak {MIN_OWNER_AGE}-{MAX_OWNER_AGE} tahun")
    is_valid = valid_format and found_in_dukcapil and name_matches and valid_age
    if is_valid:
        notes.append(f"NIK 16 digit valid, nama & data Dukcapil sesuai, usia {age} tahun dalam rentang layak")

    return {
        "score": 1.0 if is_valid else 0.0,
        "status": "Valid" if is_valid else "Tidak Valid",
        "notes": notes,
        "hard_reject": not is_valid,
        "reject_reason": "Identitas tidak valid (NIK/nama/usia/tidak terdaftar Dukcapil)" if not is_valid else None,
    }


# ---------------------------------------------------------------------------
# 2. Credit history agent (Character, from SLIK)
# ---------------------------------------------------------------------------

COLLECT_SCORE_MAP = {0: 0.60, 1: 1.00, 2: 0.75, 3: 0.45, 4: 0.20, 5: 0.00}
COLLECT_LABEL_MAP = {
    0: "Belum ada riwayat",
    1: "Lancar",
    2: "Dalam Perhatian Khusus (DPK)",
    3: "Kurang Lancar",
    4: "Diragukan",
    5: "Macet",
}


def credit_history_agent(slik_worst_collectability, slik_has_macet=None, slik_n_banks=0):
    worst = int(_num(slik_worst_collectability, 0))
    worst = max(0, min(5, worst))
    n_banks = int(_num(slik_n_banks, 0))
    has_macet = bool(slik_has_macet) if slik_has_macet is not None else (worst == 5)
    has_macet = has_macet or worst == 5

    score = COLLECT_SCORE_MAP[worst]
    notes = [f"Kolektibilitas SLIK terburuk: {COLLECT_LABEL_MAP[worst]}"]
    if n_banks >= 3:
        score = max(0.0, score - 0.05)
        notes.append(f"Eksposur kredit aktif di {n_banks} bank lain — risiko konsentrasi tambahan")

    if has_macet:
        notes.append("Riwayat kredit Macet terdeteksi — otomatis Tidak Layak (hard rule)")

    return {
        "score": round(float(np.clip(score, 0, 1)), 3),
        "status": COLLECT_LABEL_MAP[worst],
        "notes": notes,
        "hard_reject": has_macet,
        "reject_reason": "Riwayat SLIK Macet" if has_macet else None,
    }


# ---------------------------------------------------------------------------
# 3. DHN agent (blacklist)
# ---------------------------------------------------------------------------


def dhn_agent(status_dhn, dhn_alasan=None):
    is_blacklisted = _yes(status_dhn)
    if is_blacklisted:
        alasan = dhn_alasan if isinstance(dhn_alasan, str) and dhn_alasan.strip() else "tidak ada keterangan"
        notes = [f"Terdaftar di Daftar Hitam Nasional — alasan: {alasan}"]
    else:
        notes = ["Tidak terdaftar di Daftar Hitam Nasional"]

    return {
        "score": 0.0 if is_blacklisted else 1.0,
        "status": "Blacklist" if is_blacklisted else "Bersih",
        "notes": notes,
        "hard_reject": is_blacklisted,
        "reject_reason": "Terdaftar di DHN" if is_blacklisted else None,
    }


# ---------------------------------------------------------------------------
# 4. Collateral agent
# ---------------------------------------------------------------------------


def collateral_agent(collateral_ratio, ownership_match):
    ratio = _num(collateral_ratio, 0.0)
    coverage_score = float(np.clip(ratio / 1.5, 0, 1))
    match_ok = _yes(ownership_match)

    notes = [f"Rasio nilai agunan terhadap pinjaman: {ratio:.2f}x"]
    score = coverage_score
    if not match_ok:
        score = min(score, 0.3)
        notes.append("Nama sertifikat agunan TIDAK sesuai pemilik — risiko legalitas")
    else:
        notes.append("Kepemilikan agunan sesuai (nama sertifikat cocok)")

    return {
        "score": round(float(np.clip(score, 0, 1)), 3),
        "status": score_status(score),
        "notes": notes,
        "hard_reject": False,
        "reject_reason": None,
    }


# ---------------------------------------------------------------------------
# 5. Financial agent (Capacity/Capital)
# ---------------------------------------------------------------------------


def financial_agent(revenue_growth_pct, profit_margin_2025):
    growth = _num(revenue_growth_pct, 0.0)  # fraction, e.g. 0.12 = +12%
    margin = _num(profit_margin_2025, 0.0)  # fraction, e.g. 0.11 = 11%

    growth_score = float(np.clip(0.5 + growth * 1.2, 0, 1))
    margin_score = float(np.clip(margin / 0.20, 0, 1))
    score = 0.5 * growth_score + 0.5 * margin_score

    notes = [
        f"Pertumbuhan omset YoY: {growth * 100:+.1f}%",
        f"Margin laba bersih 2025: {margin * 100:.1f}%",
    ]
    if growth < 0:
        notes.append("Omset mengalami kontraksi dibanding tahun sebelumnya")

    return {
        "score": round(float(np.clip(score, 0, 1)), 3),
        "status": score_status(score, good=0.6, ok=0.4, low=0.25),
        "notes": notes,
        "hard_reject": False,
        "reject_reason": None,
    }


# ---------------------------------------------------------------------------
# 6. Cashflow agent (Condition/liquidity, from bank account behaviour)
# ---------------------------------------------------------------------------


def cashflow_agent(bank_best_avg_balance_6m, monthly_turnover_est, bank_total_overdraft_6m=0, bank_any_dormant=0):
    balance = _num(bank_best_avg_balance_6m, 0.0)
    turnover = max(_num(monthly_turnover_est, 1.0), 1.0)
    ratio = balance / turnover

    balance_score = float(np.clip(ratio / 0.8, 0, 1))
    overdraft_count = int(_num(bank_total_overdraft_6m, 0))
    overdraft_penalty = min(overdraft_count * 0.10, 0.30)
    is_dormant = bool(_num(bank_any_dormant, 0))
    dormant_penalty = 0.10 if is_dormant else 0.0

    score = balance_score - overdraft_penalty - dormant_penalty

    notes = [f"Rata-rata saldo rekening 6 bulan setara {ratio * 100:.0f}% dari omset bulanan"]
    if overdraft_count > 0:
        notes.append(f"Tercatat {overdraft_count}x kondisi overdraft dalam 6 bulan terakhir")
    if is_dormant:
        notes.append("Terdapat rekening dormant (tidak aktif) di antara rekening nasabah")

    return {
        "score": round(float(np.clip(score, 0, 1)), 3),
        "status": score_status(score, good=0.6, ok=0.4, low=0.25),
        "notes": notes,
        "hard_reject": False,
        "reject_reason": None,
    }


# ---------------------------------------------------------------------------
# 7. Risk agent (orchestrator)
# ---------------------------------------------------------------------------

AGENT_WEIGHTS = {"credit_history": 0.30, "financial": 0.30, "collateral": 0.20, "cashflow": 0.20}

# (min_score_inclusive, decision, zone) — first match wins, checked high to low
DECISION_TIERS = [
    (0.75, "Layak", "Hijau"),
    (0.60, "Layak Bersyarat", "Kuning"),
    (0.45, "Perlu Review Ulang", "Kuning"),
    (0.00, "Tidak Layak", "Merah"),
]

INTEREST_RATE_PA = {"Hijau": 10.0, "Kuning": 13.0, "Merah": None, "KUR": 6.0}
TENOR_MONTHS = {
    "KMK": {"Hijau": 36, "Kuning": 24},
    "KI": {"Hijau": 60, "Kuning": 48},
    "KUR": {"Hijau": 36, "Kuning": 24},
}
APPROVAL_RATIO = {"Layak": 1.0, "Layak Bersyarat": 0.80, "Perlu Review Ulang": 0.50, "Tidak Layak": 0.0}

COMPONENT_LABELS = [
    ("credit_history", "Character / Riwayat Kredit"),
    ("financial", "Capacity / Keuangan"),
    ("collateral", "Collateral / Agunan"),
    ("cashflow", "Condition / Cashflow"),
]


def _recommend_loan_type(loan_requested, monthly_turnover_est):
    turnover = max(_num(monthly_turnover_est, 1.0), 1.0)
    ratio = _num(loan_requested, 0.0) / turnover
    return "KMK" if ratio <= 6 else "KI"


def risk_agent(identity, credit_history, dhn, collateral, financial, cashflow,
                loan_requested, monthly_turnover_est, collateral_liquidation_value=None):
    hard_reject_reasons = [
        res["reject_reason"]
        for res in (identity, credit_history, dhn)
        if res["hard_reject"]
    ]

    combined_score = (
        AGENT_WEIGHTS["credit_history"] * credit_history["score"]
        + AGENT_WEIGHTS["financial"] * financial["score"]
        + AGENT_WEIGHTS["collateral"] * collateral["score"]
        + AGENT_WEIGHTS["cashflow"] * cashflow["score"]
    )
    combined_score = round(float(np.clip(combined_score, 0, 1)), 3)

    if hard_reject_reasons:
        decision, zone = "Tidak Layak", "Merah"
    else:
        decision, zone = "Tidak Layak", "Merah"
        for threshold, dec, zn in DECISION_TIERS:
            if combined_score >= threshold:
                decision, zone = dec, zn
                break

    approval_ratio = APPROVAL_RATIO[decision]
    jenis_kredit = _recommend_loan_type(loan_requested, monthly_turnover_est) if approval_ratio > 0 else None

    nominal_disetujui = 0
    if approval_ratio > 0:
        nominal_disetujui = int(round(_num(loan_requested, 0) * approval_ratio, -6))
        liq_value = _num(collateral_liquidation_value, None) if collateral_liquidation_value is not None else None
        if liq_value:
            nominal_disetujui = int(min(nominal_disetujui, liq_value))

    jangka_waktu_bulan = TENOR_MONTHS[jenis_kredit][zone] if jenis_kredit else 0
    bunga_persen = INTEREST_RATE_PA.get(zone) if approval_ratio > 0 else None

    insight_parts = []
    if hard_reject_reasons:
        insight_parts.append("Ditolak otomatis oleh hard rule: " + "; ".join(hard_reject_reasons) + ".")
    else:
        insight_parts.append(f"Skor gabungan {combined_score:.2f} → keputusan {decision} (zona {zone}).")

    agent_map = {"credit_history": credit_history, "financial": financial, "collateral": collateral, "cashflow": cashflow}
    low_components = [label for key, label in COMPONENT_LABELS if agent_map[key]["score"] < 0.5]
    low_detail = [f"{label} ({agent_map[key]['score']:.2f})" for key, label in COMPONENT_LABELS if agent_map[key]["score"] < 0.5]
    if low_detail and decision in ("Layak", "Layak Bersyarat"):
        insight_parts.append(
            "Catatan: meski keputusan akhir " + decision + ", skor komponen berikut tergolong rendah (<0.50) "
            "dan tetap perlu perhatian analis: " + ", ".join(low_detail) + "."
        )
    elif low_detail:
        insight_parts.append("Komponen skor rendah (<0.50): " + ", ".join(low_detail) + ".")

    return {
        "combined_score": combined_score,
        "decision": decision,
        "zone": zone,
        "jenis_kredit_rekomendasi": jenis_kredit,
        "nominal_disetujui": nominal_disetujui,
        "jangka_waktu_bulan": jangka_waktu_bulan,
        "bunga_persen": bunga_persen,
        "insight": " ".join(insight_parts),
        "hard_reject_reasons": hard_reject_reasons,
        "low_components": low_components,
    }


# ---------------------------------------------------------------------------
# 8. Credit Type Recommendation (memvalidasi jenis/tenor yang DIAJUKAN
# nasabah terhadap kemampuan bayarnya, bukan cuma menebak dari nominal
# seperti _recommend_loan_type() di atas)
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
    supaya bisa dites/dipakai ulang independen.

    `jenis_kredit_rekomendasi` di hasil ini SENGAJA menggantikan field
    dengan nama sama dari risk_agent() (lihat pemanggilan di
    score_application()) - itu tebakan naif dari nominal pinjaman saja,
    ini rekomendasi yang sudah divalidasi terhadap DSR. Field
    nominal_disetujui/jangka_waktu_bulan/bunga_persen risk_agent() TIDAK
    ikut diubah di sini (lihat Resources_Pendukung/penyesuaian_data_terbaru.md)."""
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


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def score_application(data) -> dict:
    """Run all 7 agents on one application. `data` is a dict-like object
    (pd.Series row or plain dict) exposing at least the raw columns used
    below via `.get(key)`."""
    get = data.get

    identity = identity_agent(get("NIK"), get("owner_age"), get("owner_name"))
    credit_history = credit_history_agent(
        get("slik_worst_collectability"), get("slik_has_macet"), get("slik_n_banks")
    )
    dhn = dhn_agent(get("status_dhn"), get("dhn_alasan"))
    collateral = collateral_agent(get("collateral_ratio"), get("ownership_match"))
    financial = financial_agent(get("revenue_growth_pct"), get("profit_margin_2025"))
    cashflow = cashflow_agent(
        get("bank_best_avg_balance_6m"), get("monthly_turnover_est"),
        get("bank_total_overdraft_6m"), get("bank_any_dormant"),
    )
    risk = risk_agent(
        identity, credit_history, dhn, collateral, financial, cashflow,
        loan_requested=get("loan_requested") or 0,
        monthly_turnover_est=get("monthly_turnover_est") or 1,
        collateral_liquidation_value=get("collateral_liquidation_value"),
    )
    credit_type_check = recommend_credit_type(
        loan_requested=get("loan_requested") or 0,
        monthly_turnover_est=get("monthly_turnover_est") or 1,
        jenis_kredit_diajukan=get("jenis_kredit_diajukan"),
        tenor_diajukan_bulan=get("tenor_diajukan_bulan"),
        slik_total_installment_other=get("slik_total_installment_other") or 0,
    )

    return {
        "identity": identity,
        "credit_history": credit_history,
        "dhn": dhn,
        "collateral": collateral,
        "financial": financial,
        "cashflow": cashflow,
        "risk": risk,
        "credit_type_check": credit_type_check,
    }


def _flatten(result: dict) -> dict:
    flat = {}
    for agent_name in ("identity", "credit_history", "dhn", "collateral", "financial", "cashflow"):
        res = result[agent_name]
        flat[f"{agent_name}_score"] = res["score"]
        flat[f"{agent_name}_status"] = res["status"]
        flat[f"{agent_name}_notes"] = " | ".join(res["notes"])
        flat[f"{agent_name}_hard_reject"] = res["hard_reject"]
    risk = result["risk"]
    flat.update({
        "risk_score": risk["combined_score"],
        "decision": risk["decision"],
        "zone": risk["zone"],
        "jenis_kredit_rekomendasi": risk["jenis_kredit_rekomendasi"],
        "nominal_disetujui": risk["nominal_disetujui"],
        "jangka_waktu_bulan": risk["jangka_waktu_bulan"],
        "bunga_persen": risk["bunga_persen"],
        "insight": risk["insight"],
    })
    # credit_type_check["jenis_kredit_rekomendasi"] SENGAJA menimpa nilai dari
    # risk di atas (tebakan naif dari nominal) - lihat docstring
    # recommend_credit_type(). jenis_kredit_sesuai/dsr_pada_pengajuan/
    # catatan_kesesuaian_kredit adalah field baru.
    flat.update(result["credit_type_check"])
    return flat


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Batch-run the pipeline over every row and return df + new agent columns."""
    scored_rows = [_flatten(score_application(row)) for _, row in df.iterrows()]
    scored_df = pd.DataFrame(scored_rows, index=df.index)
    return pd.concat([df, scored_df], axis=1)
