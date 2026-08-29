"""Tests for the Adaptive Verification Planner (src/agents/planner_agent.py),
the orchestrator (src/orchestrator.py) and the Gemma layer's guardrails
(src/genai.py) - no GPU/model download required, LLM paths are exercised only
through their fallback branches (monkeypatched to fail fast).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import genai
from src.agents import planner_agent as pa
from src.schemas import PlannerStep, PlannerTrace, ScreeningResult
from utils.risk_ml_pipeline import predict_credit_screening

VALID_NIK = "3276010601750001"
VALID_NAME = "Budi Panjaitan"
VALID_AGE = 45

BASE_APPLICANT = {
    "NIK": VALID_NIK, "owner_age": VALID_AGE, "owner_name": VALID_NAME, "company_name": "CV Contoh",
    "status_dhn": "Tidak", "dhn_alasan": None,
    "slik_worst_collectability": 1, "slik_has_macet": False, "slik_n_banks": 1,
    "collateral_ratio": 1.5, "ownership_match": "Ya",
    "revenue_growth_pct": 0.1, "profit_margin_2025": 0.1,
    "bank_best_avg_balance_6m": 40_000_000, "bank_best_current_balance": 30_000_000,
    "monthly_turnover_est": 50_000_000, "bank_total_overdraft_6m": 0, "bank_any_dormant": 0,
    "loan_requested": 100_000_000, "estimated_dsr": 0.2,
}


def applicant(**overrides) -> dict:
    return {**BASE_APPLICANT, **overrides}


# ---------------------------------------------------------------------------
# One test per rule (12) - acceptance criterion.
# ---------------------------------------------------------------------------


def test_rule_01_invalid_nik_triggers():
    identity = pa.identity_agent("123", VALID_AGE, VALID_NAME)
    assert pa.rule_01_invalid_nik(identity) is not None
    valid_identity = pa.identity_agent(VALID_NIK, VALID_AGE, VALID_NAME)
    assert pa.rule_01_invalid_nik(valid_identity) is None


def test_rule_02_dhn_hit_triggers():
    dhn = pa.dhn_agent("Ya", "riwayat macet")
    assert pa.rule_02_dhn_hit(dhn) is not None
    assert pa.rule_02_dhn_hit(pa.dhn_agent("Tidak")) is None


def test_rule_03_slik_macet_triggers():
    macet = pa.credit_history_agent(slik_worst_collectability=5)
    assert pa.rule_03_slik_macet(macet) is not None
    lancar = pa.credit_history_agent(slik_worst_collectability=1)
    assert pa.rule_03_slik_macet(lancar) is None


def test_rule_04_character_clean_triggers_only_when_all_clean():
    identity = pa.identity_agent(VALID_NIK, VALID_AGE, VALID_NAME)
    dhn = pa.dhn_agent("Tidak")
    clean_history = pa.credit_history_agent(slik_worst_collectability=1)
    assert pa.rule_04_character_clean(identity, dhn, clean_history) is not None

    macet_history = pa.credit_history_agent(slik_worst_collectability=5)
    assert pa.rule_04_character_clean(identity, dhn, macet_history) is None


def test_rule_05_large_loan_prioritizes_collateral():
    assert pa.rule_05_large_loan_collateral_priority(600_000_000) is not None
    assert pa.rule_05_large_loan_collateral_priority(100_000_000) is None


def test_rule_06_revenue_drop_triggers():
    assert pa.rule_06_revenue_drop(-0.35) is not None
    assert pa.rule_06_revenue_drop(0.05) is None


def test_rule_07_overdraft_deep_check_triggers():
    assert pa.rule_07_overdraft_deep_check(3) is not None
    assert pa.rule_07_overdraft_deep_check(1) is None


def test_rule_08_dormant_investigation_triggers():
    assert pa.rule_08_dormant_investigation(1) is not None
    assert pa.rule_08_dormant_investigation(0) is None


def test_rule_09_ownership_mismatch_triggers():
    assert pa.rule_09_ownership_mismatch_legal("Tidak") is not None
    assert pa.rule_09_ownership_mismatch_legal("Ya") is None


def test_rule_10_high_dsr_triggers():
    assert pa.rule_10_high_dsr_review(0.6) is not None
    assert pa.rule_10_high_dsr_review(0.2) is None


def test_rule_11_fast_track_only_when_no_flags():
    assert pa.rule_11_fast_track([]) is not None
    assert pa.rule_11_fast_track(["LEGAL_VERIFICATION"]) is None


def test_rule_12_shap_emphasis_maps_feature_to_pillar():
    assert pa.infer_shap_emphasis([{"feature": "num__revenue_growth_pct", "shap_value": 0.2}]) == "Financial"
    assert pa.infer_shap_emphasis([{"feature": "num__collateral_ratio", "shap_value": 0.1}]) == "Collateral"
    assert pa.infer_shap_emphasis([]) is None


# ---------------------------------------------------------------------------
# Hard-reject stops early - must never reach financial/collateral/cashflow.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_field,bad_value", [
    ("NIK", "0000000000000000"),
    ("status_dhn", "Ya"),
    ("slik_worst_collectability", 5),
])
def test_hard_reject_stops_before_financial_collateral_cashflow(monkeypatch, bad_field, bad_value):
    def _boom(*a, **k):
        raise AssertionError("should not be called on a hard-reject path")

    monkeypatch.setattr(pa, "financial_agent", _boom)
    monkeypatch.setattr(pa, "collateral_agent", _boom)
    monkeypatch.setattr(pa, "_calibrated_cashflow", _boom)

    trace, results = pa.plan(applicant(**{bad_field: bad_value}))

    assert trace.stopped_early is True
    assert trace.final_state == "STOPPED"
    assert "financial" not in results and "collateral" not in results and "cashflow" not in results


def test_clean_application_reaches_ready_for_ml_and_calls_all_agents():
    trace, results = pa.plan(applicant())
    assert trace.stopped_early is False
    assert trace.final_state == "READY_FOR_ML"
    assert set(results.keys()) == {"identity", "dhn", "credit_history", "financial", "collateral", "cashflow"}
    assert trace.flags == []  # nothing risky -> Fast Track, no review flags
    assert any(s.step_name == "FAST_TRACK" for s in trace.steps)


def test_large_loan_reorders_collateral_before_financial():
    trace, _ = pa.plan(applicant(loan_requested=600_000_000))
    step_order = [s.step_name for s in trace.steps]
    assert step_order.index("COLLATERAL_CHECK") < step_order.index("FINANCIAL_CHECK")


# ---------------------------------------------------------------------------
# Evidence Completeness Check
# ---------------------------------------------------------------------------


def test_evidence_completeness_complete():
    trace, results = pa.plan(applicant())
    assert trace.evidence_completeness == "Complete"


def test_evidence_completeness_missing():
    status = pa.evaluate_evidence_completeness(applicant(revenue_growth_pct=None), {}, [])
    assert status == "Missing"


def test_evidence_completeness_contradiction():
    status = pa.evaluate_evidence_completeness(
        applicant(slik_has_macet=False, slik_worst_collectability=5), {}, [],
    )
    assert status == "Contradiction"


# ---------------------------------------------------------------------------
# Orchestrator wiring: precomputed agent results actually reach the ML stage
# (i.e. predict_credit_screening never recomputes what the planner already ran).
# ---------------------------------------------------------------------------


def test_predict_credit_screening_accepts_precomputed_results(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("should not recompute - orchestrator already ran this agent")

    trace, results = pa.plan(applicant())
    monkeypatch.setattr("utils.risk_ml_pipeline.financial_agent", _boom)
    monkeypatch.setattr("utils.risk_ml_pipeline.collateral_agent", _boom)
    monkeypatch.setattr("utils.risk_ml_pipeline._calibrated_cashflow", _boom)
    monkeypatch.setattr("utils.risk_ml_pipeline.run_hard_rule_agents", _boom)

    pytest.importorskip("xgboost")
    result = predict_credit_screening(
        applicant(),
        precomputed_hard={k: results[k] for k in ("identity", "credit_history", "dhn")},
        precomputed_financial=results["financial"],
        precomputed_collateral=results["collateral"],
        precomputed_cashflow=results["cashflow"],
    )
    assert result["decision"] in ("Layak", "Layak Bersyarat", "Perlu Review Ulang", "Tidak Layak")


# ---------------------------------------------------------------------------
# Planner Summary guardrail - narrative must never mention a step outside the
# trace it was generated from (>= 5 cases, acceptance criterion).
# ---------------------------------------------------------------------------

_MINIMAL_TRACE = PlannerTrace(steps=[
    PlannerStep(1, "IDENTITY_CHECK", "-", "ok", {}),
    PlannerStep(2, "DHN_CHECK", "-", "ok", {}),
    PlannerStep(3, "SLIK_CHECK", "-", "ok", {}),
    PlannerStep(4, "CHARACTER_CHECK", "Rule 4", "bersih", {}),
    PlannerStep(5, "FINANCIAL_CHECK", "-", "ok", {}),
    PlannerStep(6, "COLLATERAL_CHECK", "-", "ok", {}),
    PlannerStep(7, "CASHFLOW_CHECK", "-", "ok", {}),
    PlannerStep(8, "FAST_TRACK", "Rule 11", "no flags", {}),
], final_state="READY_FOR_ML")

_STOPPED_TRACE = PlannerTrace(
    steps=[PlannerStep(1, "IDENTITY_CHECK", "-", "ok", {}), PlannerStep(2, "STOPPED", "Rule 1", "NIK invalid", {}, "STOPPED")],
    final_state="STOPPED", stopped_early=True, stop_reason="NIK invalid",
)


@pytest.mark.parametrize("trace,narrative,expected", [
    (_MINIMAL_TRACE, "Planner menjalankan verifikasi Financial dan Collateral secara berurutan tanpa flag tambahan.", True),
    (_MINIMAL_TRACE, "Planner melakukan verifikasi legal tambahan karena kepemilikan agunan tidak sesuai.", False),
    (_MINIMAL_TRACE, "Planner melakukan retrieval data pendukung karena omzet turun tajam.", False),
    (_MINIMAL_TRACE, "Planner menjalankan review rekomendasi kredit karena DSR tinggi.", False),
    (_STOPPED_TRACE, "Planner berhenti pada tahap awal karena NIK tidak valid, tanpa melanjutkan ke Financial atau Collateral.", True),
    (_STOPPED_TRACE, "Planner menjalankan cashflow deep check sebelum menghentikan proses.", False),
])
def test_planner_summary_guardrail(trace, narrative, expected):
    assert genai._planner_summary_guardrail(narrative, trace) is expected


# ---------------------------------------------------------------------------
# Gemma unavailable -> full result still produced, no crash.
# ---------------------------------------------------------------------------


def test_explain_falls_back_when_model_fails_to_load(monkeypatch):
    import utils.report_agent as report_agent

    def _raise(*a, **k):
        raise RuntimeError("no GPU available")

    monkeypatch.setattr(genai, "_load_model", _raise)
    monkeypatch.setattr(report_agent, "_load_model", _raise)

    ml_result = {"decision": "Layak", "insight": "Layak karena semua komponen aman.", "risk_score": 0.8}
    explanation = genai.explain(_MINIMAL_TRACE, ml_result, applicant())

    assert explanation.fallback_used is True
    assert explanation.guardrail_passed is False
    assert explanation.final_decision_narrative == ml_result["insight"]
    assert explanation.planner_summary  # deterministic template, never empty


# ---------------------------------------------------------------------------
# Schema compatibility: ScreeningResult must carry every field the old flat
# dict from predict_credit_screening() has, unchanged - dashboard compatibility.
# ---------------------------------------------------------------------------


def test_screening_result_field_parity_with_predict_credit_screening():
    old_pipeline_fields = {
        "risk_score", "decision", "zone", "jenis_kredit_rekomendasi", "nominal_disetujui",
        "jangka_waktu_bulan", "bunga_persen", "insight", "insight_kategori",
        "character_score", "character_notes", "financial_score", "financial_notes",
        "collateral_score", "collateral_notes", "cashflow_score", "cashflow_notes",
        "shap_top_factors", "jenis_kredit_sesuai", "dsr_pada_pengajuan", "catatan_kesesuaian_kredit",
    }
    new_fields = set(ScreeningResult.__dataclass_fields__.keys())
    assert old_pipeline_fields <= new_fields
