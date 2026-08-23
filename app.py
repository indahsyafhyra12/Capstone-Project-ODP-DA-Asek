"""Overview — Screening Credit Agentic AI."""
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import get_filtered_data, load_master_data
from utils.ui_components import ZONE_COLORS

st.set_page_config(page_title="Screening Credit Agentic AI", page_icon="🏦", layout="wide")

ZONE_ORDER = ["Hijau", "Kuning", "Merah"]
DECISION_ORDER = ["Layak", "Layak Bersyarat", "Perlu Review Ulang", "Tidak Layak"]
DECISION_COLORS = {
    "Layak": ZONE_COLORS["Hijau"],
    "Layak Bersyarat": ZONE_COLORS["Kuning"],
    "Perlu Review Ulang": ZONE_COLORS["Kuning"],
    "Tidak Layak": ZONE_COLORS["Merah"],
}

def format_rupiah_compact(value: float) -> str:
    value = float(value)
    if abs(value) >= 1e12:
        return f"Rp {value / 1e12:,.2f} T".replace(",", "X").replace(".", ",").replace("X", ".")
    if abs(value) >= 1e9:
        return f"Rp {value / 1e9:,.2f} M".replace(",", "X").replace(".", ",").replace("X", ".")
    if abs(value) >= 1e6:
        return f"Rp {value / 1e6:,.1f} Jt".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Rp {value:,.0f}".replace(",", ".")


st.title("🏦 Screening Credit Agentic AI — Overview")
st.caption("Sistem multi-agent untuk screening awal kelayakan kredit UMKM (prinsip 5C) — hasil scoring dihitung live oleh 7-agent pipeline setiap data dimuat.")

df = load_master_data()

# --- Filters -----------------------------------------------------------
f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    branch = st.selectbox("Cabang", ["Semua Cabang"] + sorted(df["branch_name"].unique().tolist()))
with f2:
    industry = st.selectbox("Industri", ["Semua Industri"] + sorted(df["industry"].unique().tolist()))
with f3:
    min_date, max_date = df["application_date"].min(), df["application_date"].max()
    date_range = st.date_input("Rentang Tanggal Pengajuan", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = get_filtered_data(df, branch=branch, industry=industry, date_range=date_range if len(date_range) == 2 else None)

if filtered.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# --- KPI cards -----------------------------------------------------------
approved = filtered["decision"].isin(["Layak", "Layak Bersyarat"])
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Pengajuan", f"{len(filtered):,}".replace(",", "."))
k2.metric("Approval Rate", f"{approved.mean() * 100:.1f}%")
k3.metric("Rata-rata Risk Score", f"{filtered['risk_score'].mean():.2f}")
k4.metric("Total Nominal Disetujui", format_rupiah_compact(filtered["nominal_disetujui"].sum()))
k5.metric("Nasabah DHN", int((filtered["status_dhn"] == "Ya").sum()))

st.divider()

# --- Distribusi zona & validasi hard rule --------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Distribusi Zona Risiko")
    zone_counts = filtered["zone"].value_counts().reindex(ZONE_ORDER).fillna(0).reset_index()
    zone_counts.columns = ["zone", "count"]
    fig = px.bar(zone_counts, x="zone", y="count", color="zone", color_discrete_map=ZONE_COLORS, text="count")
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Jumlah Pengajuan")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Validasi Hard Rule — DHN")
    dhn_yes = filtered[filtered["status_dhn"] == "Ya"]
    if len(dhn_yes) > 0:
        pct_rejected = (dhn_yes["decision"] == "Tidak Layak").mean() * 100
    else:
        pct_rejected = 100.0
    st.metric("Nasabah DHN 'Ya' otomatis Tidak Layak", f"{pct_rejected:.1f}%", help="Idealnya 100% — memvalidasi hard rule risk_agent.")
    if len(dhn_yes) > 0 and pct_rejected < 100:
        st.error(f"{(dhn_yes['decision'] != 'Tidak Layak').sum()} nasabah berstatus DHN 'Ya' TIDAK otomatis ditolak — periksa hard rule.")
    else:
        st.success("Hard rule DHN berjalan konsisten pada seluruh data terfilter.")

st.divider()

# --- Kesehatan keuangan & fairness check ----------------------------------
c3, c4 = st.columns(2)
with c3:
    st.subheader("Kesehatan Keuangan Portofolio")
    growth_health = pd.Series(["Growth Positif" if g > 0 else "Growth Negatif" for g in filtered["revenue_growth_pct"]])
    counts = growth_health.value_counts().reset_index()
    counts.columns = ["kategori", "count"]
    fig = px.pie(counts, names="kategori", values="count", color="kategori",
                 color_discrete_map={"Growth Positif": ZONE_COLORS["Hijau"], "Growth Negatif": ZONE_COLORS["Merah"]}, hole=0.45)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    st.subheader("Fairness Check — Approval Rate per Gender")
    st.caption("owner_gender tidak dipakai sebagai fitur scoring — panel ini murni untuk monitoring bias, bukan input agent.")
    gender_stats = filtered.groupby("owner_gender")["decision"].apply(lambda s: s.isin(["Layak", "Layak Bersyarat"]).mean() * 100).reset_index()
    gender_stats.columns = ["owner_gender", "approval_rate"]
    fig = px.bar(gender_stats, x="owner_gender", y="approval_rate", text_auto=".1f", color="owner_gender")
    fig.update_layout(showlegend=False, yaxis_title="Approval Rate (%)", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Komposisi nasabah -----------------------------------------------------
st.subheader("Komposisi Nasabah")
c5, c6, c7 = st.columns(3)
with c5:
    fig = px.pie(filtered, names="legal_entity", title="Badan Usaha", hole=0.45)
    st.plotly_chart(fig, use_container_width=True)
with c6:
    fig = px.pie(filtered, names="owner_education", title="Pendidikan Pemilik", hole=0.45)
    st.plotly_chart(fig, use_container_width=True)
with c7:
    fig = px.histogram(filtered, x="owner_age", nbins=20, title="Usia Pemilik")
    fig.update_layout(yaxis_title="Jumlah", xaxis_title="Usia")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Top cabang table -------------------------------------------------------
st.subheader("Ringkasan per Cabang")
branch_summary = (
    filtered.groupby("branch_name")
    .agg(
        total_pengajuan=("application_id", "count"),
        approval_rate=("decision", lambda s: s.isin(["Layak", "Layak Bersyarat"]).mean() * 100),
        avg_risk_score=("risk_score", "mean"),
        total_nominal_disetujui=("nominal_disetujui", "sum"),
    )
    .reset_index()
    .sort_values("total_pengajuan", ascending=False)
)
branch_summary["approval_rate"] = branch_summary["approval_rate"].round(1)
branch_summary["avg_risk_score"] = branch_summary["avg_risk_score"].round(2)
st.dataframe(branch_summary, use_container_width=True, hide_index=True)
