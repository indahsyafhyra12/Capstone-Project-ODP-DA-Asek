"""Adaptive Verification Planner - deterministic Rule Engine + Finite State Machine.

Decides the ORDER and DEPTH of verification (which of the 6 existing rule-based
agents in utils/agent_pipeline.py run, in what order, and which ones get flagged for
extra manual review) from evidence already available on the applicant row. It never
scores anything itself and never touches the ML model - it purely routes.

HARD REQUIREMENT: no LLM import anywhere in this file. `plan()` must be callable
standalone, with no network/GPU/model dependency, so it can run in a unit test in
milliseconds.

Design note on the 4 "no matching agent" rules (Supporting Data Retrieval, Legal
Verification, Cashflow Investigation, Credit Recommendation Review): the codebase has
no data-retrieval/legal-verification module to call, so these are recorded as
trace-only flags (PlannerStep.action == "FLAGGED_FOR_MANUAL_REVIEW") - governance
metadata surfaced to the RM and to the Gemma narrative, not a new computation.

Design note on Rule 11 (Fast Track): "skip deep checks" means skip the flag-raising
branches (rules 6-10), not skip financial_agent/collateral_agent/cashflow_agent
themselves - those 3 are still needed downstream for Stage 3's insight narrative
(utils/risk_ml_pipeline.py's `_compose_insight`). The measurable efficiency win this
planner buys is the hard-reject path (rules 1-3), which stops before financial/
collateral/cashflow run at all - see notebooks/08_adaptive_verification_planner_demo.ipynb
for the timing proof against the old unconditional utils.agent_pipeline.score_application().
"""
from __future__ import annotations

import pandas as pd

from utils.agent_pipeline import (
    DSR_AMAN,
    _num,
    _yes,
    collateral_agent,
    credit_history_agent,
    dhn_agent,
    financial_agent,
    identity_agent,
    recommend_credit_type,
)
from utils.risk_ml_pipeline import _calibrated_cashflow

from src.schemas import EvidenceStatus, PlannerStep, PlannerTrace

LARGE_LOAN_THRESHOLD = 500_000_000
REVENUE_DROP_THRESHOLD = -0.30
OVERDRAFT_DEEP_THRESHOLD = 3

# ---------------------------------------------------------------------------
# Rule Engine - one pure, independently-testable function per rule. Each
# returns (triggered: bool, reason: str) if evaluated, or None if not applicable.
# ---------------------------------------------------------------------------


def rule_01_invalid_nik(identity_result: dict) -> tuple[bool, str] | None:
    if identity_result["hard_reject"]:
        return True, identity_result["reject_reason"] or "Identitas tidak valid."
    return None


def rule_02_dhn_hit(dhn_result: dict) -> tuple[bool, str] | None:
    if dhn_result["hard_reject"]:
        return True, dhn_result["reject_reason"] or "Terdaftar di DHN."
    return None


def rule_03_slik_macet(credit_history_result: dict) -> tuple[bool, str] | None:
    if credit_history_result["hard_reject"]:
        return True, credit_history_result["reject_reason"] or "Riwayat SLIK Macet."
    return None


def rule_04_character_clean(identity_result: dict, dhn_result: dict, credit_history_result: dict) -> tuple[bool, str] | None:
    if identity_result["hard_reject"] or dhn_result["hard_reject"] or credit_history_result["hard_reject"]:
        return None
    return True, "Character (Identity + SLIK + DHN) bersih - lanjut ke verifikasi Financial/Collateral/Cashflow."


def rule_05_large_loan_collateral_priority(loan_requested) -> tuple[bool, str] | None:
    amount = _num(loan_requested, 0)
    if amount > LARGE_LOAN_THRESHOLD:
        return True, f"Pinjaman diajukan Rp{amount:,.0f} > Rp{LARGE_LOAN_THRESHOLD:,.0f} - Collateral diprioritaskan sebelum Financial."
    return None


def rule_06_revenue_drop(revenue_growth_pct) -> tuple[bool, str] | None:
    growth = _num(revenue_growth_pct, 0.0)
    if growth < REVENUE_DROP_THRESHOLD:
        return True, f"Pertumbuhan omset {growth * 100:.1f}% (turun > {abs(REVENUE_DROP_THRESHOLD) * 100:.0f}%) - perlu retrieval data pendukung."
    return None


def rule_07_overdraft_deep_check(bank_total_overdraft_6m) -> tuple[bool, str] | None:
    count = int(_num(bank_total_overdraft_6m, 0))
    if count >= OVERDRAFT_DEEP_THRESHOLD:
        return True, f"{count}x overdraft dalam 6 bulan (>= {OVERDRAFT_DEEP_THRESHOLD}) - Cashflow Deep Check."
    return None


def rule_08_dormant_investigation(bank_any_dormant) -> tuple[bool, str] | None:
    if _yes(bank_any_dormant) or bool(_num(bank_any_dormant, 0)):
        return True, "Ditemukan rekening dormant - Cashflow Investigation."
    return None


def rule_09_ownership_mismatch_legal(ownership_match) -> tuple[bool, str] | None:
    if not _yes(ownership_match):
        return True, "Nama sertifikat agunan tidak sesuai pemilik - Legal Verification."
    return None


def rule_10_high_dsr_review(dsr_pada_pengajuan: float | None) -> tuple[bool, str] | None:
    """`dsr_pada_pengajuan` is the DSR at the applicant's PROPOSED jenis/tenor,
    from utils.agent_pipeline.recommend_credit_type() - NOT the raw
    `estimated_dsr`/`dsr_capped` dataset column, which is a differently-scaled
    field (capped at 3.0, ~unrelated to the 0-1 DSR_AMAN threshold below;
    verified median 3.0 across the training data, so comparing it against
    DSR_AMAN would flag almost every application)."""
    if dsr_pada_pengajuan is None:
        return None
    if dsr_pada_pengajuan > DSR_AMAN:
        return True, f"DSR pada pengajuan {dsr_pada_pengajuan * 100:.0f}% > ambang aman {DSR_AMAN * 100:.0f}% - Credit Recommendation Review."
    return None


def rule_11_fast_track(review_flags: list[str]) -> tuple[bool, str] | None:
    if not review_flags:
        return True, "Evidence lengkap, tidak ada flag risiko tambahan - Fast Track (tanpa deep-review step)."
    return None


_SHAP_PILLAR_KEYWORDS = {
    "Character": ("slik", "dhn", "identity", "collectability", "nik"),
    "Financial": ("revenue", "profit", "margin", "turnover", "asset", "liability"),
    "Collateral": ("collateral",),
    "Cashflow": ("bank_", "overdraft", "dormant", "balance", "cashflow"),
}


def infer_shap_emphasis(shap_top_factors: list[dict]) -> str | None:
    """Rule 12: kalau SHAP menunjukkan faktor dari satu pilar 5C paling dominan,
    kembalikan nama pilar itu (dipakai orchestrator utk menambah 1 trace step
    post-ML, dan oleh genai.py utk menekankan pilar itu di narasi)."""
    if not shap_top_factors:
        return None
    top_feature = str(shap_top_factors[0].get("feature", "")).lower()
    for pillar, keywords in _SHAP_PILLAR_KEYWORDS.items():
        if any(kw in top_feature for kw in keywords):
            return pillar
    return None


# ---------------------------------------------------------------------------
# Evidence Completeness Check - asks "is evidence for this decision sufficient?",
# not "did every step run?".
# ---------------------------------------------------------------------------

_REQUIRED_FOR_ML_STAGE = [
    "revenue_growth_pct", "profit_margin_2025",
    "collateral_ratio", "ownership_match",
    "bank_best_avg_balance_6m", "monthly_turnover_est",
]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def evaluate_evidence_completeness(applicant: dict, agent_results: dict, flags: list[str]) -> EvidenceStatus:
    get = applicant.get

    for field in _REQUIRED_FOR_ML_STAGE:
        if _is_missing(get(field)):
            return "Missing"

    slik_has_macet = get("slik_has_macet")
    slik_worst = int(_num(get("slik_worst_collectability"), 0))
    if slik_has_macet is not None and not _yes(slik_has_macet) and slik_worst == 5:
        return "Contradiction"  # SLIK worst-collectability says Macet, has_macet flag disagrees

    if _yes(get("ownership_match")) and _num(get("collateral_ratio"), 0) <= 0:
        return "Contradiction"  # kepemilikan dibilang sesuai tapi tidak ada nilai agunan sama sekali

    return "Complete"


# ---------------------------------------------------------------------------
# The Planner (Observe -> Decide -> Replan loop, expressed as one straight-line
# FSM walk since each observation here is only ever re-evaluated once - a real
# "replan" trigger, evidence changing mid-flight, isn't possible with this
# dataset's static raw columns, so the loop degenerates to a single pass. Kept
# as a sequence of discrete decide-steps, not one big if/else block, so each
# step/rule stays independently readable and testable.)
# ---------------------------------------------------------------------------


def plan(applicant: dict) -> tuple[PlannerTrace, dict]:
    """Run the planner for one applicant. Returns (PlannerTrace, agent_results) -
    agent_results contains only the sub-agent outputs actually computed, keyed by
    'identity'/'dhn'/'credit_history'/'financial'/'collateral'/'cashflow'."""
    get = applicant.get
    trace = PlannerTrace()
    agent_results: dict = {}
    _counter = [0]

    def add_step(step_name, triggered_rule, reason, evidence_used, action="EXECUTED"):
        _counter[0] += 1
        trace.steps.append(PlannerStep(_counter[0], step_name, triggered_rule, reason, dict(evidence_used), action))

    # --- Character: Identity -> DHN -> SLIK, stop immediately on hard-reject ---
    identity = identity_agent(get("NIK"), get("owner_age"), get("owner_name"))
    agent_results["identity"] = identity
    add_step("IDENTITY_CHECK", "-", "Verifikasi NIK/nama/usia via Dukcapil.", {"NIK": get("NIK")})
    hit = rule_01_invalid_nik(identity)
    if hit:
        add_step("STOPPED", "Rule 1", hit[1], {"identity": identity["status"]}, action="STOPPED")
        trace.final_state, trace.stopped_early, trace.stop_reason = "STOPPED", True, hit[1]
        return trace, agent_results

    dhn = dhn_agent(get("status_dhn"), get("dhn_alasan"))
    agent_results["dhn"] = dhn
    add_step("DHN_CHECK", "-", "Cek status Daftar Hitam Nasional.", {"status_dhn": get("status_dhn")})
    hit = rule_02_dhn_hit(dhn)
    if hit:
        add_step("STOPPED", "Rule 2", hit[1], {"dhn": dhn["status"]}, action="STOPPED")
        trace.final_state, trace.stopped_early, trace.stop_reason = "STOPPED", True, hit[1]
        return trace, agent_results

    credit_history = credit_history_agent(get("slik_worst_collectability"), get("slik_has_macet"), get("slik_n_banks"))
    agent_results["credit_history"] = credit_history
    add_step("SLIK_CHECK", "-", "Cek riwayat kolektibilitas SLIK.", {"slik_worst_collectability": get("slik_worst_collectability")})
    hit = rule_03_slik_macet(credit_history)
    if hit:
        add_step("STOPPED", "Rule 3", hit[1], {"credit_history": credit_history["status"]}, action="STOPPED")
        trace.final_state, trace.stopped_early, trace.stop_reason = "STOPPED", True, hit[1]
        return trace, agent_results

    hit4 = rule_04_character_clean(identity, dhn, credit_history)
    add_step("CHARACTER_CHECK", "Rule 4", hit4[1], {}, action="EXECUTED")

    review_flags: list[str] = []

    # --- Rule 5: large loan -> run Collateral before Financial ---
    loan_requested = get("loan_requested")
    hit5 = rule_05_large_loan_collateral_priority(loan_requested)
    collateral = None
    if hit5:
        collateral = collateral_agent(get("collateral_ratio"), get("ownership_match"))
        agent_results["collateral"] = collateral
        add_step("COLLATERAL_CHECK", "Rule 5", hit5[1], {"loan_requested": loan_requested})

    # --- Financial (always run - Capacity/Capital is never skipped) ---
    financial = financial_agent(get("revenue_growth_pct"), get("profit_margin_2025"))
    agent_results["financial"] = financial
    add_step(
        "FINANCIAL_CHECK", "Rule 4" if not hit5 else "-",
        "Evaluasi Capacity/Capital (pertumbuhan omset & margin laba).",
        {"revenue_growth_pct": get("revenue_growth_pct"), "profit_margin_2025": get("profit_margin_2025")},
    )

    # --- Rule 6: revenue drop -> Supporting Data Retrieval (trace-only flag) ---
    hit6 = rule_06_revenue_drop(get("revenue_growth_pct"))
    if hit6:
        add_step("SUPPORTING_DATA_RETRIEVAL", "Rule 6", hit6[1], {"revenue_growth_pct": get("revenue_growth_pct")}, action="FLAGGED_FOR_MANUAL_REVIEW")
        trace.flags.append("SUPPORTING_DATA_RETRIEVAL")
        review_flags.append("SUPPORTING_DATA_RETRIEVAL")

    # --- Collateral, if rule 5 didn't already run it ---
    if collateral is None:
        collateral = collateral_agent(get("collateral_ratio"), get("ownership_match"))
        agent_results["collateral"] = collateral
        add_step("COLLATERAL_CHECK", "-", "Evaluasi Collateral (rasio agunan & kepemilikan).",
                  {"collateral_ratio": get("collateral_ratio"), "ownership_match": get("ownership_match")})

    # --- Rule 9: ownership mismatch -> Legal Verification (trace-only flag) ---
    hit9 = rule_09_ownership_mismatch_legal(get("ownership_match"))
    if hit9:
        add_step("LEGAL_VERIFICATION", "Rule 9", hit9[1], {"ownership_match": get("ownership_match")}, action="FLAGGED_FOR_MANUAL_REVIEW")
        trace.flags.append("LEGAL_VERIFICATION")
        review_flags.append("LEGAL_VERIFICATION")

    # --- Cashflow (always run - Condition is never skipped). Uses the same
    # p90-calibrated scorer as utils.risk_ml_pipeline's Stage 3 (NOT
    # agent_pipeline.cashflow_agent, which isn't calibrated for this dataset's
    # balance/turnover scale - see risk_ml_pipeline.py's module docstring) so
    # the precomputed result plugs into predict_credit_screening() unchanged.
    cashflow = _calibrated_cashflow(applicant)
    agent_results["cashflow"] = cashflow

    hit7 = rule_07_overdraft_deep_check(get("bank_total_overdraft_6m"))
    hit8 = rule_08_dormant_investigation(get("bank_any_dormant"))
    cashflow_reasons, cashflow_rules = [], []
    if hit7:
        cashflow_reasons.append(hit7[1]); cashflow_rules.append("Rule 7")
        trace.flags.append("CASHFLOW_DEEP_CHECK"); review_flags.append("CASHFLOW_DEEP_CHECK")
    if hit8:
        cashflow_reasons.append(hit8[1]); cashflow_rules.append("Rule 8")
        trace.flags.append("CASHFLOW_INVESTIGATION"); review_flags.append("CASHFLOW_INVESTIGATION")
    add_step(
        "CASHFLOW_CHECK", "+".join(cashflow_rules) or "-",
        " ".join(cashflow_reasons) or "Evaluasi Condition/likuiditas (saldo, overdraft, dormant).",
        {"bank_total_overdraft_6m": get("bank_total_overdraft_6m"), "bank_any_dormant": get("bank_any_dormant")},
        action="FLAGGED_FOR_MANUAL_REVIEW" if cashflow_reasons else "EXECUTED",
    )

    # --- Rule 10: high DSR at the proposed jenis/tenor -> Credit Recommendation
    # Review (trace-only flag). Cheap pure arithmetic (no data lookups), so
    # recomputing it again in Stage 3's credit_type_check isn't worth plumbing
    # through a precomputed_* kwarg like the 6 heavier agents above.
    credit_type_check = recommend_credit_type(
        get("loan_requested") or 0, get("monthly_turnover_est") or 1,
        get("jenis_kredit_diajukan"), get("tenor_diajukan_bulan"),
        get("slik_total_installment_other") or 0,
    )
    hit10 = rule_10_high_dsr_review(credit_type_check["dsr_pada_pengajuan"])
    if hit10:
        add_step("CREDIT_RECOMMENDATION_REVIEW", "Rule 10", hit10[1],
                  {"dsr_pada_pengajuan": credit_type_check["dsr_pada_pengajuan"]}, action="FLAGGED_FOR_MANUAL_REVIEW")
        trace.flags.append("CREDIT_RECOMMENDATION_REVIEW")
        review_flags.append("CREDIT_RECOMMENDATION_REVIEW")

    # --- Rule 11: fast track (informational - no extra flag raised anywhere) ---
    hit11 = rule_11_fast_track(review_flags)
    if hit11:
        add_step("FAST_TRACK", "Rule 11", hit11[1], {"review_flags": review_flags})

    trace.final_state = "READY_FOR_ML"
    trace.evidence_completeness = evaluate_evidence_completeness(applicant, agent_results, trace.flags)
    return trace, agent_results
