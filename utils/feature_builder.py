"""Build ML-ready raw feature rows from data/raw/*.csv, by NIK.

Joins/aggregates retail_customer_profile.csv + slik_credit_history.csv +
dhn.csv + bank_account.csv + laporan_keuangan.csv exactly like
notebooks/02_prepro&eda fix.ipynb does to build master_dataset.csv -
except scoped to a handful of target application_id's instead of the
whole 3,000-row dataset, and defensive against rows/NIKs that have no
match in one of the other tables (never happens in the shipped synthetic
data, but does for a NIK a loan officer types into the Simulasi form that
has no SLIK/bank/financial history yet, or none at all if the applicant
is brand new to the bank).

Single source of truth for this join, shared by
notebooks/04_deploy_predict_ml_risk_scoring.ipynb and pages/3_Simulasi.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
NIK_STR = {"NIK": str}

SLIK_NUM_COLS = ["slik_n_loans", "slik_worst_collectability", "slik_n_banks",
                  "slik_total_outstanding", "slik_total_installment_other",
                  "slik_avg_tenor_month", "slik_has_macet"]
BANK_NUM_COLS = ["bank_n_accounts", "bank_best_avg_balance_6m", "bank_total_avg_credit",
                  "bank_total_avg_debit", "bank_total_overdraft_6m",
                  "bank_current_balance_total", "bank_best_current_balance", "bank_any_dormant"]
FIN_NUM_COLS = ["revenue_2024", "revenue_2025", "net_profit_2024", "net_profit_2025",
                 "total_asset_2024", "total_asset_2025", "total_liability_2024", "total_liability_2025",
                 "operating_cashflow_2024", "operating_cashflow_2025",
                 "revenue_growth_pct", "profit_margin_2025", "liability_to_asset_2025"]


def load_raw_tables():
    """Load the 5 raw tables needed for feature building. Callers in a
    Streamlit page should wrap this with @st.cache_data; notebooks can
    call it directly."""
    profile = pd.read_csv(RAW_DIR / "retail_customer_profile.csv", dtype=NIK_STR)
    slik = pd.read_csv(RAW_DIR / "slik_credit_history.csv", dtype=NIK_STR)
    dhn = pd.read_csv(RAW_DIR / "dhn.csv", dtype=NIK_STR)
    bank = pd.read_csv(RAW_DIR / "bank_account.csv", dtype={**NIK_STR, "account_number": str})
    fin = pd.read_csv(RAW_DIR / "laporan_keuangan.csv", dtype=NIK_STR)
    return profile, slik, dhn, bank, fin


def build_features_from_raw(application_ids, profile, slik, dhn, bank, fin):
    """Build 1 raw feature row per application_id, purely from raw tables.

    `profile` only needs to contain the target application_id rows - it
    can be a slice of the real retail_customer_profile.csv (existing
    applicant) or a freshly-constructed 1-row DataFrame from a Streamlit
    form (brand new application, never persisted anywhere). `slik`/`dhn`/
    `bank`/`fin` should normally be the real, full raw tables so a NIK's
    genuine history (if any) is picked up automatically; a NIK with no
    rows in one of them (new-to-that-table customer) gets neutral
    defaults (0 / "Tidak" / no history), not an error - matches how
    predict_credit_screening()'s STAGE 1 treats "belum ada riwayat SLIK"
    as a normal, non-hard-reject case.

    Raises ValueError if any application_id isn't found in `profile`.
    """
    if isinstance(application_ids, str):
        application_ids = [application_ids]

    rows = profile[profile["application_id"].isin(application_ids)].copy()
    missing_ids = set(application_ids) - set(rows["application_id"])
    if missing_ids:
        raise ValueError(f"application_id tidak ditemukan di profile: {sorted(missing_ids)}")
    rows["application_id"] = pd.Categorical(rows["application_id"], categories=application_ids, ordered=True)
    rows = rows.sort_values("application_id").reset_index(drop=True)

    target_niks = rows["NIK"].tolist()

    slik_f = slik[slik["NIK"].isin(target_niks)]
    slik_agg = (slik_f.groupby("NIK")
        .agg(slik_n_loans=("slik_record_id", "count"), slik_worst_collectability=("collectability", "max"),
             slik_n_banks=("bank_name", "nunique"), slik_total_outstanding=("outstanding_balance", "sum"),
             slik_total_installment_other=("installment_amount", "sum"), slik_avg_tenor_month=("tenor_month", "mean"))
        .reset_index())
    slik_agg["slik_has_macet"] = (slik_agg["slik_worst_collectability"] == 5).astype(int)
    slik_agg["slik_has_credit_history"] = 1

    dhn_f = dhn[dhn["NIK"].isin(target_niks)]
    dhn_slim = dhn_f[["NIK", "status_dhn", "alasan"]].rename(columns={"alasan": "dhn_alasan"})

    bank_f = bank[bank["NIK"].isin(target_niks)]
    bank_agg = (bank_f.groupby("NIK")
        .agg(bank_n_accounts=("account_id", "count"), bank_best_avg_balance_6m=("average_balance_6m", "max"),
             bank_total_avg_credit=("average_monthly_credit", "sum"), bank_total_avg_debit=("average_monthly_debit", "sum"),
             bank_total_overdraft_6m=("overdraft_count_6m", "sum"), bank_current_balance_total=("current_balance", "sum"),
             bank_best_current_balance=("current_balance", "max"))
        .reset_index())
    if len(bank_f):
        bank_agg["bank_any_dormant"] = bank_f.groupby("NIK")["account_status"].apply(lambda s: int((s == "Dormant").any())).values
    else:
        bank_agg["bank_any_dormant"] = pd.Series(dtype=int)

    fin_f = fin[fin["NIK"].isin(target_niks)]
    fin_pivot = fin_f.pivot_table(index="NIK", columns="year",
        values=["revenue", "net_profit", "total_asset", "total_liability", "operating_cashflow"])
    fin_pivot.columns = [f"{c}_{y}" for c, y in fin_pivot.columns]
    fin_pivot = fin_pivot.reset_index()
    if len(fin_pivot):
        fin_pivot["revenue_growth_pct"] = ((fin_pivot["revenue_2025"] - fin_pivot["revenue_2024"]) / fin_pivot["revenue_2024"]).round(4)
        fin_pivot["profit_margin_2025"] = (fin_pivot["net_profit_2025"] / fin_pivot["revenue_2025"]).round(4)
        fin_pivot["liability_to_asset_2025"] = (fin_pivot["total_liability_2025"] / fin_pivot["total_asset_2025"]).round(4)

    result = (rows
        .merge(dhn_slim, on="NIK", how="left")
        .merge(slik_agg, on="NIK", how="left")
        .merge(bank_agg, on="NIK", how="left")
        .merge(fin_pivot, on="NIK", how="left"))

    result[SLIK_NUM_COLS] = result[SLIK_NUM_COLS].fillna(0)
    result["slik_has_credit_history"] = result["slik_has_credit_history"].fillna(0).astype(int)
    result["status_dhn"] = result["status_dhn"].fillna("Tidak")
    result["dhn_alasan"] = result["dhn_alasan"].fillna("Tidak Berlaku")
    # Kolomnya sendiri bisa tidak tercipta sama sekali (bukan cuma NaN) kalau
    # groupby/pivot_table sumbernya kosong total - jadi dibuat dulu kalau
    # belum ada, baru fillna.
    for col in BANK_NUM_COLS + FIN_NUM_COLS:
        if col not in result.columns:
            result[col] = 0.0
        else:
            result[col] = result[col].fillna(0)
    result["dsr_capped"] = result["estimated_dsr"].clip(upper=3.0)

    return result
