"""Premium RM Performance V1"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_master_data
from utils.ui_components import apply_logo
from utils.ui_premium import (
    inject_css, hero_banner, section_header, chart_title, kpi_card, rupiah_short,
    WONDR_COLORS, ZONE_COLORS
)

st.set_page_config(page_title="RM Performance", page_icon="🏆", layout="wide")
apply_logo()
inject_css()

hero_banner(
    "RM Performance",
    "Ringkasan performa Relationship Banking Officer."
)

# master_dataset.csv sudah punya rm_name/rm_branch_name/rm_region/jabatan/
# level/join_date ter-join per baris (dicek langsung ke data) — jadi
# load_master_data() harusnya sudah cukup, tanpa perlu merge rm_master.csv
# terpisah. Kalau ternyata kolom-kolom ini kosong pas dijalanin, artinya
# load_master_data() versi kamu drop kolom itu — kabari aku.
df = load_master_data()

# ======================================================
# FILTER
# ======================================================

section_header("🔎", "Filter")

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        f_region = st.selectbox("Region", ["Semua"] + sorted(df.rm_region.dropna().unique().tolist()))
    with c2:
        f_branch = st.selectbox("Cabang", ["Semua"] + sorted(df.rm_branch_name.dropna().unique().tolist()))
    with c3:
        f_level = st.selectbox("Jabatan/Level", ["Semua"] + sorted(df.level.dropna().unique().tolist()))

filtered = df.copy()
if f_region != "Semua":
    filtered = filtered[filtered.rm_region == f_region]
if f_branch != "Semua":
    filtered = filtered[filtered.rm_branch_name == f_branch]
if f_level != "Semua":
    filtered = filtered[filtered.level == f_level]

# ======================================================
# RM AGGREGATION
# ======================================================

rm_stats = (
    filtered.groupby(["rm_id", "rm_name", "rm_branch_name", "rm_region", "jabatan", "level", "join_date"])
    .agg(
        total_nasabah=("application_id", "count"),
        approval_rate=("decision", lambda x: x.isin(["Layak", "Layak Bersyarat"]).mean()),
        avg_eligibility=("risk_score", "mean"),
        total_nominal=("nominal_disetujui", "sum"),
    )
    .reset_index()
)

# ======================================================
# TEAM SUMMARY
# ======================================================

section_header("📌", "Team Summary")

team_approval = (filtered.decision.isin(["Layak", "Layak Bersyarat"])).mean() if len(filtered) else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total RM Aktif", rm_stats.rm_id.nunique(), "👥")
with c2:
    kpi_card("Avg Approval Rate", f"{team_approval:.1%}", "✅")
with c3:
    kpi_card("Avg Pengajuan/RM", f"{rm_stats.total_nasabah.mean():.0f}" if len(rm_stats) else "0", "📋")
with c4:
    kpi_card("Total Nominal Disetujui", rupiah_short(filtered.nominal_disetujui.sum()), "💰")

# ======================================================
# RM LEADERBOARD
# ======================================================

section_header("🏅", "RM Leaderboard")

leaderboard = rm_stats.sort_values("approval_rate", ascending=False).copy()
leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
leaderboard["Nominal"] = leaderboard["total_nominal"].map(rupiah_short)
leaderboard_display = leaderboard[["Rank", "rm_name", "rm_branch_name", "total_nasabah", "approval_rate", "Nominal"]].rename(columns={
    "rm_name": "RM", "rm_branch_name": "Cabang", "total_nasabah": "Nasabah", "approval_rate": "Approval Rate"
})

styled = leaderboard_display.style.format({"Approval Rate": "{:.1%}"}).background_gradient(subset=["Approval Rate"], cmap="RdYlGn")

with st.container(border=True):
    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

# ======================================================
# DETAIL PER RM
# ======================================================

section_header("👤", "Detail per RM")

rm_stats["label"] = rm_stats["rm_id"] + " — " + rm_stats["rm_name"]
selected_rm_label = st.selectbox("Pilih RM", rm_stats["label"])
rm_row = rm_stats[rm_stats["label"] == selected_rm_label].iloc[0]
rm_id_selected = rm_row["rm_id"]
rm_detail_df = filtered[filtered.rm_id == rm_id_selected]

delta_pp = (rm_row["approval_rate"] - team_approval) * 100

d1, d2, d3 = st.columns(3)

with d1:
    with st.container(border=True):
        chart_title("Approval Rate")
        st.markdown(
            f'<div style="font-size:20px;font-weight:700;color:#173404;">{rm_row["approval_rate"]:.1%}</div>'
            f'<div style="font-size:10px;color:#9CA3AF;margin-top:2px;">{delta_pp:+.1f}pp dari rata-rata tim</div>',
            unsafe_allow_html=True
        )

with d2:
    with st.container(border=True):
        chart_title("Decision Breakdown")
        dec_counts = rm_detail_df["decision"].value_counts()
        st.markdown(
            " · ".join([f"{v} {k}" for k, v in dec_counts.items()]),
            unsafe_allow_html=False
        )

with d3:
    with st.container(border=True):
        chart_title("Bergabung Sejak")
        st.markdown(
            f'<div style="font-size:15px;font-weight:700;">{rm_row["join_date"]}</div>'
            f'<div style="font-size:10px;color:#9CA3AF;margin-top:2px;">{rm_row["jabatan"]} — {rm_row["level"]} · {rm_row["rm_region"]}</div>',
            unsafe_allow_html=True
        )

# ======================================================
# PERUSAHAAN YANG DITANGANI
# ======================================================

section_header("🏢", "Perusahaan yang Ditangani")

companies = rm_detail_df[["company_name", "industry", "loan_requested", "decision"]].copy()
companies.columns = ["Perusahaan", "Industri", "Pinjaman", "Keputusan"]
companies["Pinjaman"] = companies["Pinjaman"].map(rupiah_short)

with st.container(border=True):
    st.dataframe(companies, use_container_width=True, hide_index=True, height=320)

# ======================================================
# KATEGORI NOMINAL & TENURE VS PERFORMA
# ======================================================

section_header("📊", "Kategori Nominal & Tenure vs Performa")

k1, k2 = st.columns(2)

with k1:
    with st.container(border=True):
        chart_title("Kategori Nominal Ditangani (RM Terpilih)")
        kat = rm_detail_df["kategori_nominal_diajukan"].value_counts()
        fig = go.Figure(go.Bar(
            x=kat.values, y=kat.index, orientation="h",
            marker_color=WONDR_COLORS["turquoise"]["core"],
        ))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with k2:
    with st.container(border=True):
        chart_title("Tenure vs Approval Rate (Semua RM)")

        rm_stats["tenure_tahun"] = (pd.Timestamp.now() - pd.to_datetime(rm_stats["join_date"])).dt.days / 365.25
        bins = [0, 1, 2, 4, 6, 100]
        labels = ["<1th", "1-2th", "2-4th", "4-6th", ">6th"]
        rm_stats["tenure_bucket"] = pd.cut(rm_stats["tenure_tahun"], bins=bins, labels=labels)

        tenure_perf = rm_stats.groupby("tenure_bucket", observed=True)["approval_rate"].mean().reindex(labels)

        fig = go.Figure(go.Bar(
            x=tenure_perf.index.astype(str), y=tenure_perf.values,
            marker_color=WONDR_COLORS["purple"]["core"],
        ))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), yaxis_tickformat=".0%",
                           font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# DISTRIBUSI JABATAN/LEVEL
# ======================================================

section_header("🎓", "Distribusi Jabatan/Level")

with st.container(border=True):
    level_perf = rm_stats.groupby("level").agg(
        jumlah_rm=("rm_id", "count"),
        avg_approval=("approval_rate", "mean"),
    ).reset_index()

    l1, l2 = st.columns(2)
    with l1:
        fig = go.Figure(go.Pie(
            labels=level_perf["level"], values=level_perf["jumlah_rm"], hole=0.6,
            marker_colors=[WONDR_COLORS["orange"]["core"], WONDR_COLORS["turquoise"]["core"]],
            textinfo="percent",
        ))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), showlegend=True,
                           font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)
    with l2:
        for _, r in level_perf.iterrows():
            st.metric(f"Avg Approval Rate — {r['level']}", f"{r['avg_approval']:.1%}")

# ======================================================
# TARGET VS AKTUAL (PLACEHOLDER)
# ======================================================

st.markdown(
    '<div style="border:1px dashed #D1D5DB; border-radius:12px; padding:14px; margin-top:16px; text-align:center;">'
    '<span style="font-size:11px; color:#9CA3AF;">🎯 Target vs Aktual — <b>Coming Soon</b>, '
    'menunggu konfirmasi angka target dari transkrip wawancara.</span></div>',
    unsafe_allow_html=True
)