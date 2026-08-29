"""Adaptive Verification Planner orchestrator.

Replaces the STATIC calling pattern of utils.agent_pipeline.score_application()
(which always runs all 7 agents, unconditionally, in fixed order) with an ADAPTIVE
one: src.agents.planner_agent.plan() decides which of the 6 rule-based agents run,
in what order, and which get flagged for extra review - then this module feeds
whatever the planner already computed into the existing ML risk_score + Policy
Engine (utils.risk_ml_pipeline.predict_credit_screening(), unmodified except for the
precomputed_* kwargs it now accepts), and finally asks the Gemma Explanation Layer
(src/genai.py) to narrate the already-final result.

Governance table (each box owns exactly one kind of decision):
    Planner        -> urutan & kedalaman verifikasi   (src/agents/planner_agent.py)
    ML             -> risk_score                      (utils/risk_ml_pipeline.py, unchanged model)
    Policy Engine  -> approval / decision              (utils/risk_ml_pipeline.py, unchanged thresholds)
    Gemma          -> penjelasan (narasi) saja         (src/genai.py)
"""
from __future__ import annotations

from src import genai
from src.agents.planner_agent import infer_shap_emphasis, plan
from src.schemas import PlannerStep, PlannerTrace, ScreeningResult
from utils.risk_ml_pipeline import _compose_insight, _hard_reject_result, predict_credit_screening


def _build_hard_reject_result(applicant: dict, agent_results: dict) -> dict:
    """Mirrors predict_credit_screening()'s Stage-1 short-circuit text exactly
    (same _compose_insight/_hard_reject_result calls), using the hard-rule results
    the planner already computed instead of recomputing them."""
    identity = agent_results.get("identity")
    dhn = agent_results.get("dhn")
    credit_history = agent_results.get("credit_history")

    if identity and identity["hard_reject"]:
        reason = "; ".join(identity["notes"]) or identity["reject_reason"]
        return _hard_reject_result(_compose_insight("Tidak Layak", {}, 0, reason.lower()))
    if dhn and dhn["hard_reject"]:
        alasan = applicant.get("dhn_alasan") or "tidak ada keterangan"
        reason = f"nasabah terdaftar di Daftar Hitam Nasional ({alasan})"
        return _hard_reject_result(_compose_insight("Tidak Layak", {}, 0, reason))
    if credit_history and credit_history["hard_reject"]:
        return _hard_reject_result(_compose_insight("Tidak Layak", {}, 0, "memiliki riwayat kredit Macet pada SLIK"))
    raise ValueError("_build_hard_reject_result dipanggil tapi trace tidak stopped_early oleh hard_reject apapun")


def _append_post_ml_emphasis_step(trace: PlannerTrace, ml_result: dict) -> None:
    """Rule 12: kalau SHAP menunjukkan 1 pilar 5C paling dominan, catat sebagai
    trace step tambahan setelah ML - dipakai genai.py utk menekankan pilar itu."""
    emphasis = infer_shap_emphasis(ml_result.get("shap_top_factors", []))
    if not emphasis:
        return
    trace.steps.append(PlannerStep(
        order=len(trace.steps) + 1,
        step_name="POST_ML_EMPHASIS",
        triggered_rule="Rule 12",
        reason=f"SHAP menunjukkan faktor {emphasis} paling dominan terhadap risk_score - narasi ditekankan pada aspek {emphasis}.",
        evidence_used={"shap_top_factors": ml_result.get("shap_top_factors", [])[:1]},
        action="EXECUTED",
    ))


def run_screening(applicant: dict, explain_with_gemma: bool = True) -> ScreeningResult:
    """Run the full adaptive pipeline for one applicant (dict/Series exposing the
    same raw columns as utils.risk_ml_pipeline.predict_credit_screening()).

    Set `explain_with_gemma=False` to skip loading/calling the LLM (e.g. in fast
    unit tests) - `gemma_explanation` is then left `None`."""
    trace, agent_results = plan(applicant)

    if trace.stopped_early:
        ml_result = _build_hard_reject_result(applicant, agent_results)
    else:
        precomputed_hard = {k: agent_results[k] for k in ("identity", "credit_history", "dhn")}
        ml_result = predict_credit_screening(
            applicant,
            precomputed_hard=precomputed_hard,
            precomputed_financial=agent_results.get("financial"),
            precomputed_collateral=agent_results.get("collateral"),
            precomputed_cashflow=agent_results.get("cashflow"),
        )
        _append_post_ml_emphasis_step(trace, ml_result)

    explanation = genai.explain(trace, ml_result, applicant) if explain_with_gemma else None
    return ScreeningResult.from_pipeline(ml_result, trace, explanation)


def run_screening_dataframe(df, explain_with_gemma: bool = True) -> list[ScreeningResult]:
    """Batch helper mirroring utils.agent_pipeline.score_dataframe() - one
    ScreeningResult per row, same order as `df`."""
    return [run_screening(row.to_dict(), explain_with_gemma=explain_with_gemma) for _, row in df.iterrows()]
