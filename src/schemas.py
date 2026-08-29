"""Typed records for the Adaptive Verification Planner + Gemma Explanation Layer.

Plain stdlib `dataclasses` (no pydantic dependency in requirements.txt). These are
purely additive: `ScreeningResult` mirrors every field
`utils.risk_ml_pipeline.predict_credit_screening()` already returns, plus the two new
fields `planner_trace`/`gemma_explanation` - so a Streamlit page keyed off the old
flat dict still finds every field it expects if it ever adopts `ScreeningResult`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvidenceStatus = Literal["Complete", "Missing", "Contradiction"]


@dataclass
class PlannerStep:
    order: int
    step_name: str
    triggered_rule: str
    reason: str
    evidence_used: dict[str, Any]
    action: str = "EXECUTED"  # EXECUTED | FLAGGED_FOR_MANUAL_REVIEW | STOPPED | SKIPPED


@dataclass
class PlannerTrace:
    steps: list[PlannerStep] = field(default_factory=list)
    final_state: str = "INIT"
    stopped_early: bool = False
    stop_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    evidence_completeness: EvidenceStatus | None = None

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(s) for s in self.steps]


@dataclass
class GemmaExplanation:
    planner_summary: str
    final_decision_narrative: str
    guardrail_passed: bool
    fallback_used: bool
    fallback_reason: str | None = None


@dataclass
class ScreeningResult:
    # Same field names/types as predict_credit_screening()'s return dict.
    risk_score: float | None
    decision: str
    zone: str
    jenis_kredit_rekomendasi: str
    nominal_disetujui: int
    jangka_waktu_bulan: int
    bunga_persen: float | None
    insight: str
    insight_kategori: str
    character_score: float | None
    character_notes: str | None
    financial_score: float | None
    financial_notes: str | None
    collateral_score: float | None
    collateral_notes: str | None
    cashflow_score: float | None
    cashflow_notes: str | None
    shap_top_factors: list[dict[str, Any]]
    jenis_kredit_sesuai: bool | None
    dsr_pada_pengajuan: float | None
    catatan_kesesuaian_kredit: str | None
    # Additive fields (not present in the old flat dict).
    planner_trace: PlannerTrace
    gemma_explanation: GemmaExplanation | None = None

    @classmethod
    def from_pipeline(
        cls, ml_result: dict[str, Any], planner_trace: PlannerTrace,
        gemma_explanation: GemmaExplanation | None = None,
    ) -> "ScreeningResult":
        base_fields = {f: ml_result.get(f) for f in ml_result if f in cls.__dataclass_fields__}
        return cls(**base_fields, planner_trace=planner_trace, gemma_explanation=gemma_explanation)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for backward-compatible merging with the old pipeline's output -
        every original key/type preserved, `planner_trace`/`gemma_explanation` added
        as nested dataclass instances (not flattened, so callers opt in explicitly)."""
        d = asdict(self)
        d["planner_trace"] = self.planner_trace
        d["gemma_explanation"] = self.gemma_explanation
        return d
