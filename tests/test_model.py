"""Tests for the light patch to utils/risk_ml_pipeline.py (optional
precomputed_hard/precomputed_financial/precomputed_collateral/precomputed_cashflow
kwargs on predict_credit_screening()) - the model/preprocessor/thresholds
themselves are untouched, only whether Stage 1/Stage 3's sub-agent calls get
recomputed or reused is patched.
"""
import pandas as pd
import pytest

from utils import risk_ml_pipeline as rmp

pytest.importorskip("xgboost", reason="risk_score_model.pkl was trained with xgboost")

MASTER = pd.read_csv("data/processed/master_dataset.csv", dtype={"NIK": str})
DUKCAPIL_NIKS = set(pd.read_csv("data/raw/dukcapil.csv", dtype={"NIK": str})["NIK"])


def _clean_row() -> dict:
    clean = MASTER[
        (MASTER["status_dhn"] == "Tidak")
        & (MASTER["slik_worst_collectability"] < 5)
        & (MASTER["NIK"].isin(DUKCAPIL_NIKS))
    ]
    return clean.iloc[0].to_dict()


def test_model_artifacts_load():
    model, preprocessor, meta = rmp._load_artifacts()
    assert model is not None
    assert preprocessor is not None
    assert "numeric_features" in meta and "categorical_features" in meta


def test_predict_without_precomputed_args_unchanged_behavior():
    row = _clean_row()
    result = rmp.predict_credit_screening(row)
    assert result["decision"] in ("Layak", "Layak Bersyarat", "Perlu Review Ulang", "Tidak Layak")
    assert 0.0 <= result["risk_score"] <= 1.0


def test_precomputed_hard_bypasses_run_hard_rule_agents(monkeypatch):
    row = _clean_row()
    hard = rmp.run_hard_rule_agents(row)

    def _boom(*a, **k):
        raise AssertionError("run_hard_rule_agents should not be called when precomputed_hard is given")

    monkeypatch.setattr(rmp, "run_hard_rule_agents", _boom)
    result = rmp.predict_credit_screening(row, precomputed_hard=hard)
    assert result["decision"] in ("Layak", "Layak Bersyarat", "Perlu Review Ulang", "Tidak Layak")


def test_precomputed_subscores_bypass_recomputation(monkeypatch):
    row = _clean_row()
    financial = rmp.financial_agent(row.get("revenue_growth_pct"), row.get("profit_margin_2025"))
    collateral = rmp.collateral_agent(row.get("collateral_ratio"), row.get("ownership_match"))
    cashflow = rmp._calibrated_cashflow(row)

    def _boom(*a, **k):
        raise AssertionError("should not recompute - precomputed value was provided")

    monkeypatch.setattr(rmp, "financial_agent", _boom)
    monkeypatch.setattr(rmp, "collateral_agent", _boom)
    monkeypatch.setattr(rmp, "_calibrated_cashflow", _boom)

    result = rmp.predict_credit_screening(
        row, precomputed_financial=financial, precomputed_collateral=collateral, precomputed_cashflow=cashflow,
    )
    assert result["financial_score"] == financial["score"]
    assert result["collateral_score"] == collateral["score"]
    assert result["cashflow_score"] == cashflow["score"]


def test_precomputed_and_recomputed_give_identical_result():
    """Same inputs, computed either inline or precomputed, must produce an
    identical result - the patch must not change the model/threshold logic."""
    row = _clean_row()
    baseline = rmp.predict_credit_screening(row)

    hard = rmp.run_hard_rule_agents(row)
    financial = rmp.financial_agent(row.get("revenue_growth_pct"), row.get("profit_margin_2025"))
    collateral = rmp.collateral_agent(row.get("collateral_ratio"), row.get("ownership_match"))
    cashflow = rmp._calibrated_cashflow(row)
    patched = rmp.predict_credit_screening(
        row, precomputed_hard=hard, precomputed_financial=financial,
        precomputed_collateral=collateral, precomputed_cashflow=cashflow,
    )

    assert baseline["risk_score"] == patched["risk_score"]
    assert baseline["decision"] == patched["decision"]
    assert baseline["insight"] == patched["insight"]
