"""Daftar Pengajuan — tabel semua pengajuan dengan filter & pencarian."""
import streamlit as st

from utils.data_loader import load_master_data
from utils.ui_components import DECISION_ZONE, ZONE_COLORS, apply_logo

st.set_page_config(page_title="Daftar Pengajuan", page_icon="📋", layout="wide")
apply_logo()
st.title("📋 Daftar Pengajuan")

df = load_master_data()

# --- Sidebar filters -------------------------------------------------------
st.sidebar.header("Filter")
branch = st.sidebar.multiselect("Cabang", sorted(df["branch_name"].unique().tolist()))
industry = st.sidebar.multiselect("Industri", sorted(df["industry"].unique().tolist()))
decision = st.sidebar.multiselect("Keputusan", sorted(df["decision"].unique().tolist()))
zone = st.sidebar.multiselect("Zona", sorted(df["zone"].unique().tolist()))
score_range = st.sidebar.slider("Risk Score", 0.0, 1.0, (0.0, 1.0), step=0.01)
search = st.sidebar.text_input("Cari nama / ID pengajuan")

filtered = df.copy()
if branch:
    filtered = filtered[filtered["branch_name"].isin(branch)]
if industry:
    filtered = filtered[filtered["industry"].isin(industry)]
if decision:
    filtered = filtered[filtered["decision"].isin(decision)]
if zone:
    filtered = filtered[filtered["zone"].isin(zone)]
filtered = filtered[filtered["risk_score"].between(score_range[0], score_range[1])]
if search:
    s = search.strip().lower()
    mask = (
        filtered["application_id"].str.lower().str.contains(s)
        | filtered["company_name"].str.lower().str.contains(s)
        | filtered["owner_name"].str.lower().str.contains(s)
    )
    filtered = filtered[mask]

st.caption(f"{len(filtered):,} dari {len(df):,} pengajuan".replace(",", "."))

display_cols = [
    "application_id", "company_name", "owner_name", "branch_name", "industry",
    "loan_requested", "risk_score", "decision", "zone",
    "nominal_disetujui", "jangka_waktu_bulan", "bunga_persen",
]
show_df = filtered[display_cols].rename(columns={
    "application_id": "ID Pengajuan", "company_name": "Perusahaan", "owner_name": "Pemilik",
    "branch_name": "Cabang", "industry": "Industri", "loan_requested": "Pinjaman Diajukan",
    "risk_score": "Risk Score", "decision": "Keputusan", "zone": "Zona",
    "nominal_disetujui": "Nominal Disetujui", "jangka_waktu_bulan": "Tenor (bulan)", "bunga_persen": "Bunga (%)",
})


def _style_decision(val):
    color = ZONE_COLORS.get(DECISION_ZONE.get(val, ""), "#6b7280")
    return f"background-color:{color}22;color:{color};font-weight:600;"


def _style_zone(val):
    color = ZONE_COLORS.get(val, "#6b7280")
    return f"background-color:{color}22;color:{color};font-weight:600;"


styled = show_df.style.map(_style_decision, subset=["Keputusan"]).map(_style_zone, subset=["Zona"]).format({
    "Pinjaman Diajukan": "Rp {:,.0f}", "Nominal Disetujui": "Rp {:,.0f}", "Risk Score": "{:.2f}",
})
st.dataframe(styled, use_container_width=True, hide_index=True, height=480)

st.divider()
st.subheader("Lihat Detail Nasabah")
c1, c2 = st.columns([3, 1])
with c1:
    options = filtered["application_id"] + " — " + filtered["company_name"]
    picked = st.selectbox("Pilih pengajuan", options.tolist() if not options.empty else [])
with c2:
    st.write("")
    st.write("")
    if st.button("Lihat Detail →", use_container_width=True, disabled=not picked):
        st.session_state["selected_application_id"] = picked.split(" — ")[0]
        st.switch_page("pages/2_Detail_Nasabah.py")
