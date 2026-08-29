"""Unit tests for the 6 pure rule-based agents in utils/agent_pipeline.py.

These are the building blocks the Adaptive Verification Planner (src/agents/
planner_agent.py) routes between - each is a pure function of raw values, so no
model/dataset fixtures are needed here beyond a couple of real NIK/nama pairs from
data/raw/dukcapil.csv (needed for identity_agent's Dukcapil lookup).
"""
from utils.agent_pipeline import (
    cashflow_agent,
    collateral_agent,
    credit_history_agent,
    dhn_agent,
    financial_agent,
    identity_agent,
)

VALID_NIK = "3276010601750001"
VALID_NAME = "Budi Panjaitan"
VALID_AGE = 45


def test_identity_agent_valid():
    result = identity_agent(VALID_NIK, VALID_AGE, VALID_NAME)
    assert result["hard_reject"] is False
    assert result["status"] == "Valid"
    assert result["score"] == 1.0


def test_identity_agent_invalid_format():
    result = identity_agent("123", VALID_AGE, VALID_NAME)
    assert result["hard_reject"] is True
    assert result["score"] == 0.0


def test_identity_agent_not_in_dukcapil():
    result = identity_agent("9999999999999999", VALID_AGE, VALID_NAME)
    assert result["hard_reject"] is True


def test_identity_agent_name_mismatch():
    result = identity_agent(VALID_NIK, VALID_AGE, "Nama Salah")
    assert result["hard_reject"] is True


def test_identity_agent_age_out_of_range():
    result = identity_agent(VALID_NIK, 15, VALID_NAME)
    assert result["hard_reject"] is True


def test_credit_history_agent_macet_is_hard_reject():
    result = credit_history_agent(slik_worst_collectability=5)
    assert result["hard_reject"] is True
    assert result["score"] == 0.0


def test_credit_history_agent_lancar_not_hard_reject():
    result = credit_history_agent(slik_worst_collectability=1)
    assert result["hard_reject"] is False
    assert result["score"] == 1.0


def test_credit_history_agent_many_banks_penalized():
    baseline = credit_history_agent(slik_worst_collectability=1, slik_n_banks=1)
    many_banks = credit_history_agent(slik_worst_collectability=1, slik_n_banks=5)
    assert many_banks["score"] < baseline["score"]


def test_dhn_agent_blacklisted():
    result = dhn_agent("Ya", "Riwayat kredit macet PT Bank ABC 2023")
    assert result["hard_reject"] is True
    assert result["score"] == 0.0


def test_dhn_agent_clean():
    result = dhn_agent("Tidak")
    assert result["hard_reject"] is False
    assert result["score"] == 1.0


def test_collateral_agent_ownership_mismatch_caps_score():
    matched = collateral_agent(collateral_ratio=1.5, ownership_match="Ya")
    mismatched = collateral_agent(collateral_ratio=1.5, ownership_match="Tidak")
    assert mismatched["score"] <= 0.3
    assert mismatched["score"] < matched["score"]
    assert mismatched["hard_reject"] is False  # ownership mismatch is a flag, not a hard reject


def test_financial_agent_growth_and_margin_drive_score():
    growing = financial_agent(revenue_growth_pct=0.20, profit_margin_2025=0.15)
    shrinking = financial_agent(revenue_growth_pct=-0.35, profit_margin_2025=0.02)
    assert growing["score"] > shrinking["score"]
    assert "kontraksi" in " ".join(shrinking["notes"]).lower()


def test_cashflow_agent_overdraft_and_dormant_penalized():
    clean = cashflow_agent(bank_best_avg_balance_6m=40_000_000, monthly_turnover_est=50_000_000)
    risky = cashflow_agent(
        bank_best_avg_balance_6m=40_000_000, monthly_turnover_est=50_000_000,
        bank_total_overdraft_6m=4, bank_any_dormant=1,
    )
    assert risky["score"] < clean["score"]
