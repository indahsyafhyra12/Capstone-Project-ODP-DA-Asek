# ============================================================
# 04_Monitoring_Portofolio_Premium.py
# Premium Portfolio Monitoring Dashboard
# ============================================================

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_master_data, get_filtered_data
from utils.ui_components import apply_logo
from utils.ui_premium import (
    inject_css, hero_banner, section_header, chart_title, kpi_card,
    WONDR_COLORS, ZONE_COLORS
)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Monitoring Portofolio",
    page_icon="📊",
    layout="wide"
)

apply_logo()
inject_css()

ZONE_ORDER = ["Hijau", "Kuning", "Merah"]

# PENTING: sebelumnya file ini import ZONE_COLORS dari utils.ui_components
# (hex: #16a34a/#d97706/#dc2626) — beda dari ZONE_COLORS di ui_premium.py
# (#22C55E/#EAB308/#F97316-EF4444) yang dipakai halaman lain. Disatuin ke
# satu sumber (ui_premium) biar warna zone konsisten persis di semua
# halaman, bukan cuma mirip-mirip.
ZONE_LABEL_COLORS = {
    "Hijau": ZONE_COLORS["layak"],
    "Kuning": ZONE_COLORS["layak_bersyarat"],
    "Merah": ZONE_COLORS["tidak_layak"],
}

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = load_master_data()

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
hero_banner(
    "Monitoring Portofolio",
    "Portfolio Quality Dashboard — Monitoring kualitas portofolio kredit UMKM seluruh cabang."
)

# ------------------------------------------------------------
# FILTER CARD
# ------------------------------------------------------------
section_header("🗂️", "Portfolio Filter")

with st.container(border=True):
    f1, f2, f3 = st.columns([1, 1, 2])

    with f1:
        branch = st.selectbox(
            "Cabang",
            ["Semua Cabang"] + sorted(df["branch_name"].unique())
        )

    with f2:
        industry = st.selectbox(
            "Industri",
            ["Semua Industri"] + sorted(df["industry"].unique())
        )

    with f3:
        min_date = df["application_date"].min()
        max_date = df["application_date"].max()

        date_range = st.date_input(
            "Periode",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

filtered = get_filtered_data(
    df,
    branch=branch,
    industry=industry,
    date_range=date_range if len(date_range) == 2 else None
)

# ------------------------------------------------------------
# KPI
# ------------------------------------------------------------

section_header("📌", "Executive Summary")

approval = filtered["decision"].isin(["Layak", "Layak Bersyarat"])

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    kpi_card("Total Pengajuan", f"{len(filtered):,}".replace(",", "."), "📄")
with c2:
    kpi_card("Approval Rate", f"{approval.mean()*100:.1f}%", "✅")
with c3:
    kpi_card("Avg Credit Eligibility", f"{filtered['risk_score'].mean():.2f}", "🟢")
with c4:
    kpi_card("Cabang Aktif", filtered["branch_name"].nunique(), "🏦")
with c5:
    kpi_card("Industri Aktif", filtered["industry"].nunique(), "🏭")

# ------------------------------------------------------------
# BREAKDOWN
# ------------------------------------------------------------

section_header("📊", "Breakdown Zona")

left, right = st.columns(2)

with left:
    with st.container(border=True):
        chart_title("Breakdown Zona per Cabang")

        branch_zone = (
            filtered.groupby(["branch_name", "zone"])
            .size()
            .reset_index(name="jumlah")
        )

        fig = px.bar(
            branch_zone, x="branch_name", y="jumlah", color="zone",
            category_orders={"zone": ZONE_ORDER},
            color_discrete_map=ZONE_LABEL_COLORS,
        )
        fig.update_layout(
            height=340, xaxis_title="", yaxis_title="Jumlah Pengajuan", legend_title="",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    with st.container(border=True):
        chart_title("Breakdown Zona per Industri")

        industry_zone = (
            filtered.groupby(["industry", "zone"])
            .size()
            .reset_index(name="jumlah")
        )

        fig = px.bar(
            industry_zone, x="industry", y="jumlah", color="zone",
            category_orders={"zone": ZONE_ORDER},
            color_discrete_map=ZONE_LABEL_COLORS,
        )
        fig.update_layout(
            height=340, xaxis_title="", yaxis_title="Jumlah Pengajuan", legend_title="",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

section_header("🌡️", "Cabang × Industri")

with st.container(border=True):
    pivot = (
        filtered
        .groupby(["branch_name", "industry"])["risk_score"]
        .mean()
        .reset_index()
        .pivot(index="branch_name", columns="industry", values="risk_score")
    )

    # Pakai RdYlGn (sama kayak Industry Intelligence & Geographic Insights
    # di Overview) daripada custom 4-stop scale — biar konsisten satu
    # gradient risk/eligibility di semua halaman.
    fig = px.imshow(
        pivot, color_continuous_scale="RdYlGn", aspect="auto",
        labels=dict(color="Eligibility"),
    )
    fig.update_layout(height=400, coloraxis_colorbar_title="Eligibility", font=dict(family="Inter, sans-serif"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TOP & BOTTOM
# ------------------------------------------------------------

section_header("🏆", "Branch Performance")

summary = (
    filtered
    .groupby("branch_name")
    .agg(
        total_pengajuan=("application_id", "count"),
        approval_rate=("decision", lambda x: x.isin(["Layak", "Layak Bersyarat"]).mean() * 100),
        eligibility=("risk_score", "mean"),
    )
    .reset_index()
)

summary["approval_rate"] = summary["approval_rate"].round(1)
summary["eligibility"] = summary["eligibility"].round(2)

team_avg_approval = summary["approval_rate"].mean()

top5 = summary.nlargest(5, "approval_rate")
bottom5 = summary.nsmallest(5, "approval_rate")

l, r = st.columns(2)

with l:
    with st.container(border=True):
        chart_title("Top Performing Branch")
        st.dataframe(
            top5.rename(columns={"branch_name": "Cabang", "approval_rate": "Approval %", "eligibility": "Eligibility"}),
            use_container_width=True, hide_index=True,
        )

with r:
    with st.container(border=True):
        chart_title("Cabang Perlu Perhatian")
        st.caption(f"Approval rate di bawah rata-rata seluruh cabang ({team_avg_approval:.1f}%)")
        bottom5_display = bottom5.rename(columns={"branch_name": "Cabang", "approval_rate": "Approval %", "eligibility": "Eligibility"}).copy()
        bottom5_display["Selisih dari Rata-rata"] = (bottom5_display["Approval %"] - team_avg_approval).round(1).astype(str) + "pp"
        st.dataframe(bottom5_display, use_container_width=True, hide_index=True)

with st.container(border=True):
    chart_title("Top 10 Approval Rate")

    top10 = summary.sort_values("approval_rate", ascending=False).head(10)

    fig = px.bar(
        top10, x="approval_rate", y="branch_name", orientation="h",
        color="approval_rate", color_continuous_scale="RdYlGn", text="approval_rate",
    )
    fig.update_layout(height=360, xaxis_title="Approval Rate (%)", yaxis_title="", font=dict(family="Inter, sans-serif"))
    fig.update_traces(texttemplate="%{text:.1f}%")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# INSIGHT
# ------------------------------------------------------------

section_header("💡", "Portfolio Insights")

best_branch = top5.iloc[0]
worst_branch = bottom5.iloc[0]
top_industry = filtered["industry"].value_counts().idxmax()
yellow = (filtered["zone"] == "Kuning").sum()

a, b = st.columns(2)

with a:
    st.success(
        f"**{best_branch['branch_name']}** memiliki Approval Rate tertinggi "
        f"({best_branch['approval_rate']:.1f}%) dengan Credit Eligibility rata-rata "
        f"{best_branch['eligibility']:.2f}."
    )

with b:
    st.warning(
        f"**{worst_branch['branch_name']}** memiliki Approval Rate terendah "
        f"({worst_branch['approval_rate']:.1f}%), {(worst_branch['approval_rate']-team_avg_approval):.1f}pp dari rata-rata."
    )

c, d = st.columns(2)

with c:
    st.info(
        f"**{top_industry}** merupakan industri dengan volume pengajuan terbesar "
        f"pada periode terpilih."
    )

with d:
    st.warning(
        f"Terdapat **{yellow:,}** nasabah di Zona Kuning yang layak menjadi prioritas monitoring."
        .replace(",", ".")
    )

# ------------------------------------------------------------
# RINGKASAN CABANG
# ------------------------------------------------------------

section_header("📋", "Ringkasan Portofolio Cabang")

table = summary.sort_values("approval_rate", ascending=False).rename(columns={
    "branch_name": "Cabang",
    "total_pengajuan": "Total Pengajuan",
    "approval_rate": "Approval Rate (%)",
    "eligibility": "Avg Credit Eligibility",
})

with st.container(border=True):
    st.dataframe(table, use_container_width=True, hide_index=True)