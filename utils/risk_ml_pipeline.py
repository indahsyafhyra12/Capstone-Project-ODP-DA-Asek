"""Layer 1 (ML) + Layer 2 (rule-based policy engine) hybrid credit screening.

Three explicit stages, run in this order by `predict_credit_screening()`:

  STAGE 1 - PRE-ML HARD RULE FILTER (deterministic, no model involved)
      Reuses identity_agent / credit_history_agent / dhn_agent from
      utils/agent_pipeline.py (already-existing, tested pure functions on
      raw values) to check the 3 kill-switches: invalid identity, DHN
      blacklist, SLIK Macet. Verified to produce the EXACT same 220/220
      hard-reject rows as the notebook-02 pipeline that generated
      master_scored.csv (Definisi A) on the full training pool, so it's
      safe to reuse even though this module's downstream threshold table
      (STAGE 3) is Definisi A's, not agent_pipeline.py's Definisi B.
      Rows that fail here NEVER reach the ML model - risk_score stays
      undefined, decision is forced to "Tidak Layak". This is also why the
      ML model (STAGE 2) was trained only on rows that passed this filter.

  STAGE 2 - ML-PREDICTED risk_score
      A regression model (see notebooks/03_ml_risk_scoring.ipynb) predicts
      the combined 0-1 risk_score directly from raw applicant features -
      it was never told the 0.35/0.25/0.20/0.10/0.10 weight formula or the
      per-agent scoring rules, it learned the mapping from data.

  STAGE 3 - POST-ML POLICY ENGINE (rule-based, unchanged from Definisi A)
      Given risk_score (now ML-sourced instead of formula-sourced), apply
      Definisi A's decision thresholds (0.70 / 0.55 / 0.40), loan-type /
      tenor / interest-rate / approved-amount rules, and 6-category insight
      narrative logic (notebooks/02_prepro&eda fix.ipynb). The 4 rule
      sub-agents (credit_history/collateral/financial/cashflow from
      utils/agent_pipeline.py) are also called here - NOT as ML features,
      but purely so the insight narrative can still say which pillar is
      weak/strong, same as before.

This intentionally reuses Definisi A's thresholds/weights/interest rates
for the STAGE 3 decision table, NOT utils/agent_pipeline.py's risk_agent()
(a separate, later rewrite with different numbers that is what's currently
live in app.py/pages/*.py) - only its per-agent building blocks (identity/
credit_history/dhn/collateral/financial/cashflow) are reused, since those
were verified equivalent on the training data. Wiring this module into the
Streamlit dashboard is a separate decision that changes the live decision/
zone/bunga_persen numbers - see the training notebook's summary.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.agent_pipeline import (
    identity_agent,
    credit_history_agent,
    dhn_agent,
    collateral_agent,
    financial_agent,
    recommend_credit_type,
)

MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_preprocessor = None
_meta = None


def _load_artifacts():
    global _model, _preprocessor, _meta
    if _model is None:
        _model = joblib.load(MODEL_DIR / "risk_score_model.pkl")
        _preprocessor = joblib.load(MODEL_DIR / "risk_score_preprocessor.pkl")
        _meta = joblib.load(MODEL_DIR / "risk_score_meta.pkl")
    return _model, _preprocessor, _meta


def _num(value, default=0.0):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# STAGE 3 constants - Definisi A's policy table (not in agent_pipeline.py)
# ---------------------------------------------------------------------------

INTEREST_BY_ZONE = {"Hijau": 9.5, "Kuning": 12.0, "Merah": 15.0}
TENOR_BY_LOAN = {"KMK": 12, "KI": 36, "KPR": 120, "KKB": 48, "KK": 24, "KUR": 36}
STRONG, WEAK = 0.7, 0.5

# agent_pipeline.py's cashflow_agent() is NOT calibrated for this dataset's
# balance/turnover scale (40-100x multiplier in the generator) - verified
# 73% of applicants score >=0.99 with it, making the "Cashflow lemah" insight
# flag never fire. Definisi A's notebook explicitly recalibrated against this
# exact problem using the 90th-percentile of the historical distribution as
# the "full score" reference, so that's kept here for insight purposes only
# (does NOT feed the ML model or the risk_score/decision).
_BALANCE_RATIO_P90 = 19.2
_CURRENT_RATIO_P90 = 22.8


def _calibrated_cashflow(row):
    turnover = max(_num(row.get("monthly_turnover_est"), 1.0), 1.0)
    balance_ratio = _num(row.get("bank_best_avg_balance_6m"), 0.0) / turnover
    current_ratio = _num(row.get("bank_best_current_balance"), 0.0) / turnover
    base = 0.5 * np.clip(balance_ratio / _BALANCE_RATIO_P90, 0, 1) + 0.5 * np.clip(current_ratio / _CURRENT_RATIO_P90, 0, 1)
    overdraft_count = int(_num(row.get("bank_total_overdraft_6m"), 0))
    overdraft_penalty = min(overdraft_count * 0.1, 0.3)
    is_dormant = _num(row.get("bank_any_dormant"), 0) == 1
    score = round(float(np.clip(base - overdraft_penalty - (0.15 if is_dormant else 0.0), 0, 1)), 3)

    notes = f"Saldo rata-rata {balance_ratio:.1f}x omset bulanan, saldo real-time {current_ratio:.1f}x omset bulanan"
    if overdraft_count > 0:
        notes += f", overdraft {overdraft_count}x dalam 6 bulan"
    if is_dormant:
        notes += ", memiliki rekening dormant"
    return {"score": score, "notes": notes}


_INSIGHT_KATEGORI_PREFIXES = [
    "Layak bersyarat karena", "Perlu review ulang karena", "Layak tapi",
    "Layak karena", "Tidak layak tapi", "Tidak layak karena",
]


def _kategori_insight(insight: str) -> str:
    lowered = insight.lower()
    for prefix in _INSIGHT_KATEGORI_PREFIXES:
        if lowered.startswith(prefix.lower()):
            return prefix
    return "Lainnya"


def _compose_insight(decision, scores, score, hard_rule_reason=None):
    weak = [k for k, v in scores.items() if v < WEAK]
    strong = [k for k, v in scores.items() if v >= STRONG]

    if hard_rule_reason:
        return f"Tidak layak karena {hard_rule_reason}."

    if decision == "Layak":
        if weak:
            return (f"Layak tapi {', '.join(weak)} tergolong lemah (skor di bawah 0.5) "
                     "— disarankan tetap dimonitor meski keputusan akhir disetujui.")
        return ("Layak karena seluruh komponen (" +
                ", ".join(f"{k} {v:.2f}" for k, v in scores.items()) + ") berada di zona aman.")

    if decision == "Layak Bersyarat":
        alasan = weak if weak else ["beberapa indikator berada di batas ambang"]
        return f"Layak bersyarat karena {', '.join(alasan)} — disarankan tambahan agunan/penjamin atau plafon diturunkan."

    if decision == "Perlu Review Ulang":
        return (f"Perlu review ulang karena skor gabungan ({score:.2f}) berada di area abu-abu "
                 "— disarankan OTS/wawancara lanjutan sebelum keputusan final.")

    if strong:
        return (f"Tidak layak tapi {', '.join(strong)} tergolong kuat — skor gabungan "
                 f"({score:.2f}) masih di bawah ambang, bisa dipertimbangkan ulang jika "
                 "ada mitigasi risiko dari sisi lain.")
    return f"Tidak layak karena skor gabungan ({score:.2f}) di bawah ambang batas kelayakan pada hampir seluruh komponen."


def _score_to_decision_zone(score):
    if score >= 0.70:
        return "Layak", "Hijau"
    if score >= 0.55:
        return "Layak Bersyarat", "Kuning"
    if score >= 0.40:
        return "Perlu Review Ulang", "Kuning"
    return "Tidak Layak", "Merah"


def apply_policy_engine(risk_score, loan_requested, collateral_market_value) -> dict:
    """STAGE 3 - decision/zone/jenis/nominal/tenor/bunga, as a pure function of
    risk_score + the 2 loan-sizing inputs. Factored out of
    predict_credit_screening() so a loan officer's manual risk_score override
    (pages/3_Pengajuan_Credit_Baru.py) can cascade through the exact same
    rules instead of leaving the other fields stuck at stale AI-derived
    values next to an edited score."""
    decision, zone = _score_to_decision_zone(risk_score)

    loan_requested = _num(loan_requested, 0)
    collateral_market_value = _num(collateral_market_value, 0)
    max_by_collateral = collateral_market_value * 0.7
    nominal = int(min(loan_requested, max_by_collateral)) if decision != "Tidak Layak" else 0
    jenis = "KMK" if loan_requested < 200_000_000 else "KI"
    jenis = jenis if decision != "Tidak Layak" else "-"
    tenor = TENOR_BY_LOAN.get(jenis, 24) if decision != "Tidak Layak" else 0
    bunga = INTEREST_BY_ZONE[zone] if decision != "Tidak Layak" else None

    return {
        "decision": decision, "zone": zone, "jenis_kredit_rekomendasi": jenis,
        "nominal_disetujui": nominal, "jangka_waktu_bulan": tenor, "bunga_persen": bunga,
    }


def _hard_reject_result(insight):
    return {
        "risk_score": None, "decision": "Tidak Layak", "zone": "Merah",
        "jenis_kredit_rekomendasi": "-", "nominal_disetujui": 0,
        "jangka_waktu_bulan": 0, "bunga_persen": None, "insight": insight,
        "insight_kategori": _kategori_insight(insight),
        "jenis_kredit_sesuai": None, "dsr_pada_pengajuan": None,
        "catatan_kesesuaian_kredit": None,
        "character_score": None, "character_notes": None,
        "financial_score": None, "financial_notes": None,
        "collateral_score": None, "collateral_notes": None,
        "cashflow_score": None, "cashflow_notes": None,
        "shap_top_factors": [],
    }


def run_hard_rule_agents(row: dict) -> dict:
    """STAGE 1, exposed standalone so callers can inspect/test each agent's
    verdict independently of a full predict_credit_screening() call. Reuses
    utils.agent_pipeline's pure per-agent functions (raw value in, dict
    with score/status/notes/hard_reject/reject_reason out)."""
    return {
        "identity": identity_agent(row.get("NIK"), row.get("owner_age"), row.get("owner_name")),
        "credit_history": credit_history_agent(
            row.get("slik_worst_collectability"), row.get("slik_has_macet"), row.get("slik_n_banks")
        ),
        "dhn": dhn_agent(row.get("status_dhn"), row.get("dhn_alasan")),
    }


def predict_credit_screening(row_raw_features: dict) -> dict:
    """Run the 3-stage hybrid pipeline for one applicant.

    `row_raw_features` is a dict/Series exposing the raw columns from
    master_dataset.csv (NIK, owner_age, status_dhn, slik_*, revenue_*,
    collateral_*, bank_*, industry, loan_requested, ...).

    Returns the 7 policy fields (risk_score, decision, zone,
    jenis_kredit_rekomendasi, nominal_disetujui, jangka_waktu_bulan,
    bunga_persen) + insight + shap_top_factors.
    """
    row = row_raw_features

    # === STAGE 1: pre-ML hard rule filter (deterministic) ===================
    # Reuses utils.agent_pipeline's agents instead of duplicating the checks -
    # verified to match Definisi A's hard-reject set exactly (220/220) on the
    # full training pool.
    hard = run_hard_rule_agents(row)

    if hard["identity"]["hard_reject"]:
        reason = "; ".join(hard["identity"]["notes"]) or hard["identity"]["reject_reason"]
        return _hard_reject_result(_compose_insight("Tidak Layak", {}, 0, reason.lower()))

    if hard["dhn"]["hard_reject"]:
        alasan = row.get("dhn_alasan") or "tidak ada keterangan"
        reason = f"nasabah terdaftar di Daftar Hitam Nasional ({alasan})"
        return _hard_reject_result(_compose_insight("Tidak Layak", {}, 0, reason))

    if hard["credit_history"]["hard_reject"]:
        return _hard_reject_result(_compose_insight("Tidak Layak", {}, 0, "memiliki riwayat kredit Macet pada SLIK"))

    # === STAGE 2: ML-predicted risk_score ====================================
    model, preprocessor, meta = _load_artifacts()
    feature_cols = meta["numeric_features"] + meta["categorical_features"]
    X = pd.DataFrame([{c: row.get(c) for c in feature_cols}])
    X_proc = preprocessor.transform(X)
    risk_score = float(np.clip(model.predict(X_proc)[0], 0, 1))

    # === STAGE 3: post-ML policy engine (rule-based, unchanged) ==============
    policy = apply_policy_engine(risk_score, row.get("loan_requested"), row.get("collateral_market_value"))
    decision, zone = policy["decision"], policy["zone"]
    nominal, jenis = policy["nominal_disetujui"], policy["jenis_kredit_rekomendasi"]
    tenor, bunga = policy["jangka_waktu_bulan"], policy["bunga_persen"]

    # Validasi jenis/tenor yang DIAJUKAN nasabah terhadap DSR (bukan ML
    # feature - panggil fungsi yang sama dari agent_pipeline.py, jangan
    # duplikat, supaya kedua sistem konsisten untuk bagian ini). Field
    # "jenis_kredit_rekomendasi" di credit_type_check SENGAJA menimpa
    # `jenis` dari policy di atas (tebakan naif dari nominal) - lihat
    # docstring recommend_credit_type(); nominal/tenor/bunga TIDAK ikut
    # diubah.
    credit_type_check = recommend_credit_type(
        row.get("loan_requested"), row.get("monthly_turnover_est"),
        row.get("jenis_kredit_diajukan"), row.get("tenor_diajukan_bulan"),
        row.get("slik_total_installment_other") or 0,
    )
    jenis = credit_type_check["jenis_kredit_rekomendasi"]

    # Sub-scores/notes purely for the insight narrative (NOT fed into the ML model)
    character = hard["credit_history"]
    financial = financial_agent(row.get("revenue_growth_pct"), row.get("profit_margin_2025"))
    collateral = collateral_agent(row.get("collateral_ratio"), row.get("ownership_match"))
    cashflow = _calibrated_cashflow(row)
    scores = {
        "Character": character["score"], "Financial": financial["score"],
        "Collateral": collateral["score"], "Cashflow": cashflow["score"],
    }
    insight = _compose_insight(decision, scores, risk_score)
    insight_kategori = _kategori_insight(insight)

    # SHAP top factors, for extra transparency on top of the rule-style insight
    top_factors = []
    try:
        import shap
        explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else None
        if explainer is not None:
            sv = explainer(X_proc)
            feat_names = preprocessor.get_feature_names_out()
            contrib = pd.Series(sv.values[0], index=feat_names)
            top_factors = [
                {"feature": f, "shap_value": round(float(v), 4)}
                for f, v in contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5).items()
            ]
    except ImportError:
        pass

    return {
        "risk_score": round(risk_score, 3),
        "decision": decision,
        "zone": zone,
        "jenis_kredit_rekomendasi": jenis,
        "nominal_disetujui": nominal,
        "jangka_waktu_bulan": tenor,
        "bunga_persen": bunga,
        "insight": insight,
        "insight_kategori": insight_kategori,
        "character_score": character["score"], "character_notes": "; ".join(character["notes"]),
        "financial_score": financial["score"], "financial_notes": "; ".join(financial["notes"]),
        "collateral_score": collateral["score"], "collateral_notes": "; ".join(collateral["notes"]),
        "cashflow_score": cashflow["score"], "cashflow_notes": cashflow["notes"],
        "shap_top_factors": top_factors,
        "jenis_kredit_sesuai": credit_type_check["jenis_kredit_sesuai"],
        "dsr_pada_pengajuan": credit_type_check["dsr_pada_pengajuan"],
        "catatan_kesesuaian_kredit": credit_type_check["catatan_kesesuaian_kredit"],
    }
