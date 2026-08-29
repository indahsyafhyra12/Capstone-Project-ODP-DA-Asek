"""Gemma Explanation Layer - narrates results that are ALREADY FINAL.

Two outputs, both pure narration (never a decision):
  1. Planner Summary       - the PROCESS story (why steps were skipped/prioritized/
                              flagged), built from the Adaptive Verification
                              Planner's PlannerTrace (src/agents/planner_agent.py).
  2. Final Decision Narrative - the RESULT story (why approved/rejected/review),
                              delegated straight to utils.report_agent.generate_report()
                              which already does exactly this job against the
                              existing rule-based `insight` categories.

Same model as utils/report_agent.py (google/gemma-4-E2B-it, 4-bit quantized),
loaded through that module's `_load_model()`/`_run_generation()` so both narrations
share one cached model instance instead of loading Gemma twice. Single-shot
generation, never a ReAct/agentic loop - this module never decides anything, and
never imports the planner (no dependency in the other direction, matches the
governance table: Planner decides routing, ML decides risk_score, Policy Engine
decides approval, Gemma explains).

Guardrails (2, both independent, either failing -> deterministic fallback):
  - Final Decision Narrative: utils.report_agent's own keyword sanity-check
    (decision vs. narrative contradiction) - reused as-is.
  - Planner Summary (new): the generated text must not name a verification step
    that isn't actually present in the PlannerTrace it was given - checked via
    `_STEP_KEYWORDS` below. Fails -> `_fallback_planner_summary()`, a deterministic
    template rendered directly from the trace (no LLM).
"""
from __future__ import annotations

import logging

from src.schemas import GemmaExplanation, PlannerTrace
from utils.report_agent import _load_model, _run_generation, generate_report, get_last_fallback_reason

logger = logging.getLogger(__name__)

SYSTEM_PLANNER_SUMMARY_PROMPT = """Anda adalah asisten yang menjelaskan PROSES verifikasi kredit
yang dijalankan oleh Adaptive Verification Planner (rule engine + finite state
machine, BUKAN LLM) kepada RM (Relationship Manager).

ATURAN KETAT:
1. Anda HANYA boleh menyebut tahap/step verifikasi yang ADA di daftar "Jejak
   Planner" yang diberikan. JANGAN PERNAH menyebut, menyiratkan, atau mengarang
   tahap verifikasi lain (mis. jangan sebut "retrieval data pendukung" kalau
   step SUPPORTING_DATA_RETRIEVAL tidak ada di daftar).
2. Tugas Anda HANYA menjelaskan proses (urutan, apa yang diprioritaskan/dilewati/
   ditandai untuk review) - BUKAN menjelaskan hasil akhir (layak/tidak layak),
   itu ditangani narasi terpisah.
3. 2-4 kalimat, Bahasa Indonesia profesional, tanpa mengarang angka/fakta baru.
4. Kalau planner berhenti lebih awal (hard-reject), jelaskan itu sebagai proses
   yang berhenti otomatis di tahap awal, bukan hasil analisis mendalam.

Contoh:
Jejak Planner: IDENTITY_CHECK (EXECUTED), DHN_CHECK (EXECUTED), SLIK_CHECK
(EXECUTED) -> STOPPED (Rule 3: Riwayat SLIK Macet).
Narasi: Planner menjalankan verifikasi Character secara berurutan (Identitas ->
DHN -> SLIK) dan menghentikan proses secara otomatis begitu SLIK menunjukkan
riwayat Macet, tanpa perlu melanjutkan ke verifikasi Financial, Collateral, atau
Cashflow sama sekali.

Jejak Planner: CHARACTER_CHECK (EXECUTED) -> COLLATERAL_CHECK (Rule 5, pinjaman
>Rp500jt) -> FINANCIAL_CHECK (EXECUTED) -> COLLATERAL_CHECK sudah dijalankan ->
CASHFLOW_CHECK (EXECUTED) -> FAST_TRACK (Rule 11).
Narasi: Karena nominal pinjaman di atas Rp500 juta, planner memprioritaskan
verifikasi Collateral lebih awal sebelum Financial. Tidak ditemukan flag risiko
tambahan pada tahap manapun, sehingga planner melanjutkan langsung ke penilaian
ML tanpa tahapan review tambahan (Fast Track).

Sekarang tulis narasi untuk jejak planner berikut, ikuti aturan di atas persis."""

_STEP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "SUPPORTING_DATA_RETRIEVAL": ("retrieval data pendukung", "supporting data", "data pendukung tambahan"),
    "LEGAL_VERIFICATION": ("verifikasi legal", "legal verification"),
    "CASHFLOW_DEEP_CHECK": ("cashflow deep check", "pemeriksaan arus kas mendalam", "arus kas mendalam"),
    "CASHFLOW_INVESTIGATION": ("investigasi arus kas", "cashflow investigation"),
    "CREDIT_RECOMMENDATION_REVIEW": ("review rekomendasi kredit", "credit recommendation review"),
    "FINANCIAL_CHECK": ("financial", "keuangan"),
    "COLLATERAL_CHECK": ("collateral", "agunan"),
    "CASHFLOW_CHECK": ("cashflow", "arus kas"),
    "FAST_TRACK": ("fast track", "jalur cepat"),
}


def _build_planner_summary_prompt(trace: PlannerTrace) -> str:
    lines = []
    for step in trace.steps:
        rule = f"{step.triggered_rule}: " if step.triggered_rule != "-" else ""
        lines.append(f"- {step.step_name} ({step.action}) {rule}{step.reason}")
    header = "Jejak Planner (HANYA gunakan step di daftar ini):" if not trace.stopped_early else \
        "Jejak Planner (planner BERHENTI lebih awal - HANYA gunakan step di daftar ini):"
    return header + "\n" + "\n".join(lines) + "\n\nNarasi:"


def _fallback_planner_summary(trace: PlannerTrace) -> str:
    if trace.stopped_early:
        return f"Planner menghentikan proses verifikasi lebih awal: {trace.stop_reason}"
    flagged = [s.reason for s in trace.steps if s.action == "FLAGGED_FOR_MANUAL_REVIEW"]
    if flagged:
        return ("Planner menjalankan verifikasi Character, Financial, Collateral, dan Cashflow, "
                 "lalu menandai tahap tambahan untuk review manual: " + " ".join(flagged))
    return ("Planner menjalankan seluruh tahap verifikasi standar (Character, Financial, Collateral, "
            "Cashflow) tanpa menemukan flag risiko tambahan, sehingga langsung diteruskan ke penilaian ML.")


_NEGATION_WORDS = ("tanpa", "tidak", "bukan", "belum")
_NEGATION_WINDOW = 50  # chars looked back from a keyword hit, e.g. "tanpa melanjutkan ke Financial"


def _planner_summary_guardrail(narrative: str, trace: PlannerTrace) -> bool:
    """Reject a keyword hit only when it's an unqualified claim the step
    happened - a hit preceded (within `_NEGATION_WINDOW` chars) by a negation
    word (mirrors utils.report_agent's own `_sanity_check` window trick) means
    the narrative is correctly saying that step did NOT run, so it's allowed."""
    present_steps = {s.step_name for s in trace.steps}
    lowered = narrative.lower()
    for step_name, keywords in _STEP_KEYWORDS.items():
        if step_name in present_steps:
            continue
        for kw in keywords:
            idx = lowered.find(kw)
            if idx == -1:
                continue
            window = lowered[max(0, idx - _NEGATION_WINDOW):idx]
            if not any(neg in window for neg in _NEGATION_WORDS):
                return False
    return True


def _generate_planner_summary(trace: PlannerTrace) -> tuple[str, bool, str | None]:
    """Returns (text, fallback_used, fallback_reason)."""
    fallback = _fallback_planner_summary(trace)
    try:
        processor, model = _load_model()
    except Exception as e:
        reason = f"Model gagal dimuat ({type(e).__name__}): {e}"
        logger.warning("genai.explain: %s - fallback ke template planner summary.", reason)
        return fallback, True, reason

    prompt = _build_planner_summary_prompt(trace)
    try:
        narrative = _run_generation(processor, model, prompt, system_prompt=SYSTEM_PLANNER_SUMMARY_PROMPT)
    except Exception as e:
        reason = f"Generate error ({type(e).__name__}): {e}"
        logger.warning("genai.explain: %s - fallback ke template planner summary.", reason)
        return fallback, True, reason

    if not narrative:
        return fallback, True, "Model mengembalikan planner summary kosong"

    if not _planner_summary_guardrail(narrative, trace):
        reason = f"Guardrail menolak planner summary (menyebut step di luar trace): {narrative[:300]!r}"
        logger.warning("genai.explain: %s - fallback ke template planner summary.", reason)
        return fallback, True, reason

    return narrative, False, None


def explain(trace: PlannerTrace, ml_result: dict, applicant: dict) -> GemmaExplanation:
    """Build both narrations for one applicant's already-final result.

    `ml_result` is predict_credit_screening()'s (or the hard-reject equivalent's)
    output dict; `applicant` is the same raw dict passed to the planner/orchestrator
    (only `company_name` is read from it here, for the Final Decision Narrative)."""
    planner_summary, planner_fallback, planner_fallback_reason = _generate_planner_summary(trace)

    row = {"company_name": applicant.get("company_name", "Nasabah"), **ml_result}
    final_decision_narrative = generate_report(row)
    final_fallback_reason = get_last_fallback_reason()
    final_fallback = final_fallback_reason is not None

    fallback_used = planner_fallback or final_fallback
    fallback_reason = "; ".join(r for r in (planner_fallback_reason, final_fallback_reason) if r) or None

    return GemmaExplanation(
        planner_summary=planner_summary,
        final_decision_narrative=final_decision_narrative,
        guardrail_passed=not planner_fallback,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )
