"""Detail Nasabah — 7 kartu agent untuk satu pengajuan."""
import streamlit as st

from utils.agent_pipeline import score_application
from utils.data_loader import load_master_data
from utils.ui_components import apply_logo, render_full_result

st.set_page_config(page_title="Detail Nasabah", page_icon="🔍", layout="wide")
apply_logo()
st.title("🔍 Detail Nasabah")

df = load_master_data()
ids = df["application_id"].tolist()

default_id = st.session_state.get("selected_application_id")
default_index = ids.index(default_id) if default_id in ids else 0

picked_id = st.selectbox(
    "Pilih ID Pengajuan",
    ids,
    index=default_index,
    format_func=lambda aid: f"{aid} — {df.loc[df['application_id'] == aid, 'company_name'].values[0]}",
)
st.session_state["selected_application_id"] = picked_id

row = df[df["application_id"] == picked_id].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Perusahaan", row["company_name"])
c2.metric("Pemilik", row["owner_name"])
c3.metric("Cabang", row["branch_name"])
c4.metric("Pinjaman Diajukan", f"Rp {row['loan_requested']:,.0f}".replace(",", "."))
st.caption(f"Industri: {row['industry']} · {row['sub_industry']} — Badan usaha: {row['legal_entity']} — Diajukan: {row['application_date'].date()}")

st.divider()

result = score_application(row)
render_full_result(result)
