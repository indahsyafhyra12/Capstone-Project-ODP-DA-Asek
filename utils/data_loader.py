"""Load master_dataset.csv and score it with the 7-agent pipeline (cached)."""
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from utils.agent_pipeline import score_dataframe

DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "master_dataset.csv"
NETWORK_NODES_PATH = Path(__file__).parent.parent / "data" / "processed" / "rm_network_nodes.csv"
NETWORK_EDGES_PATH = Path(__file__).parent.parent / "data" / "processed" / "rm_network_edges.csv"


@st.cache_data
def load_master_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype={"NIK": str})
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df = score_dataframe(df)
    return df


@st.cache_data
def load_rm_network():
    """Load node & edge table hasil graph analytics RM
    (notebooks/graph_analytics_rm_network.ipynb) dan hitung posisi layout
    (spring layout, sama seperti Tahap 7 notebook) — dicache karena graph-nya
    kecil (~40 RM) tapi tetap sayang dihitung ulang tiap rerun."""
    nodes = pd.read_csv(NETWORK_NODES_PATH)
    edges = pd.read_csv(NETWORK_EDGES_PATH)

    G = nx.Graph()
    G.add_nodes_from(nodes["rm_id"])
    for _, r in edges.iterrows():
        G.add_edge(r["rm_id_a"], r["rm_id_b"], jaccard=r["jaccard"])
    pos = nx.spring_layout(G, seed=42, k=0.9, weight="jaccard")

    nodes["pos_x"] = nodes["rm_id"].map(lambda n: pos[n][0])
    nodes["pos_y"] = nodes["rm_id"].map(lambda n: pos[n][1])
    return nodes, edges


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
