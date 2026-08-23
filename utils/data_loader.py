"""Load master_dataset.csv and score it with the 7-agent pipeline (cached)."""
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.agent_pipeline import score_dataframe

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "master_dataset.csv"


@st.cache_data
def load_master_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype={"NIK": str})
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df = score_dataframe(df)
    return df


def get_filtered_data(df, branch=None, industry=None, date_range=None):
    filtered = df.copy()
    if branch and branch != "Semua Cabang":
        filtered = filtered[filtered["branch_name"] == branch]
    if industry and industry != "Semua Industri":
        filtered = filtered[filtered["industry"] == industry]
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["application_date"] >= pd.Timestamp(start_date))
            & (filtered["application_date"] <= pd.Timestamp(end_date))
        ]
    return filtered


def get_unique_values(df, column):
    return sorted(df[column].dropna().unique().tolist())
