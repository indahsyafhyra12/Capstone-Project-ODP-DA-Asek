# ============================================================
# 04_Monitoring_Portofolio_Premium.py
# Premium Portfolio Monitoring Dashboard
# ============================================================

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_master_data, get_filtered_data
from utils.ui_components import apply_logo, ZONE_COLORS
from utils.ui_premium import inject_css, hero_banner

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

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = load_master_data()

# ------------------------------------------------------------
# HERO
# ------------------------------------------------------------
hero_banner(
    "📊 Monitoring Portofolio",
    "Portfolio Quality Dashboard • Monitoring kualitas portofolio kredit UMKM seluruh cabang."
)

# ------------------------------------------------------------
# FILTER CARD
# ------------------------------------------------------------
st.markdown("### 🗂 Portfolio Filter")

f1, f2, f3 = st.columns([1,1,2])

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
        value=(min_date,max_date),
        min_value=min_date,
        max_value=max_date
    )

filtered = get_filtered_data(
    df,
    branch=branch,
    industry=industry,
    date_range=date_range if len(date_range)==2 else None
)

st.markdown("---")

# ------------------------------------------------------------
# KPI
# ------------------------------------------------------------

approval = filtered["decision"].isin(["Layak","Layak Bersyarat"])

c1,c2,c3,c4,c5 = st.columns(5)

with c1:
    st.metric("📄 Total Pengajuan", f"{len(filtered):,}".replace(",", "."))

with c2:
    st.metric("✅ Approval Rate", f"{approval.mean()*100:.1f}%")

with c3:
    st.metric("🟢 Avg Credit Eligibility", f"{filtered['risk_score'].mean():.2f}")

with c4:
    st.metric("🏦 Cabang Aktif", filtered["branch_name"].nunique())

with c5:
    st.metric("🏭 Industri Aktif", filtered["industry"].nunique())

st.markdown("---")

# ------------------------------------------------------------
# BREAKDOWN
# ------------------------------------------------------------

left,right = st.columns(2)

with left:

    st.markdown("#### Breakdown Zona per Cabang")

    branch_zone = (
        filtered.groupby(["branch_name","zone"])
        .size()
        .reset_index(name="jumlah")
    )

    fig = px.bar(
        branch_zone,
        x="branch_name",
        y="jumlah",
        color="zone",
        category_orders={"zone":ZONE_ORDER},
        color_discrete_map=ZONE_COLORS
    )

    fig.update_layout(
        height=360,
        xaxis_title="",
        yaxis_title="Jumlah Pengajuan",
        legend_title=""
    )

    st.plotly_chart(fig,use_container_width=True)

with right:

    st.markdown("#### Breakdown Zona per Industri")

    industry_zone = (
        filtered.groupby(["industry","zone"])
        .size()
        .reset_index(name="jumlah")
    )

    fig = px.bar(
        industry_zone,
        x="industry",
        y="jumlah",
        color="zone",
        category_orders={"zone":ZONE_ORDER},
        color_discrete_map=ZONE_COLORS
    )

    fig.update_layout(
        height=360,
        xaxis_title="",
        yaxis_title="Jumlah Pengajuan",
        legend_title=""
    )

    st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

st.markdown("### 🌡 Cabang × Industri")

pivot = (
    filtered
    .groupby(["branch_name","industry"])["risk_score"]
    .mean()
    .reset_index()
    .pivot(index="branch_name",columns="industry",values="risk_score")
)

fig = px.imshow(
    pivot,
    color_continuous_scale=[
        "#E53935",
        "#F57C00",
        "#8BC34A",
        "#16A34A"
    ],
    aspect="auto",
    labels=dict(color="Eligibility")
)

fig.update_layout(
    height=420,
    coloraxis_colorbar_title="Eligibility"
)

st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# TOP & BOTTOM
# ------------------------------------------------------------

summary = (
    filtered
    .groupby("branch_name")
    .agg(
        total_pengajuan=("application_id","count"),
        approval_rate=("decision",lambda x:x.isin(["Layak","Layak Bersyarat"]).mean()*100),
        eligibility=("risk_score","mean")
    )
    .reset_index()
)

summary["approval_rate"] = summary["approval_rate"].round(1)
summary["eligibility"] = summary["eligibility"].round(2)

top5 = summary.nlargest(5,"approval_rate")
bottom5 = summary.nsmallest(5,"approval_rate")

l,r = st.columns(2)

with l:

    st.markdown("### 🏆 Top Performing Branch")

    st.dataframe(
        top5.rename(columns={
            "branch_name":"Cabang",
            "approval_rate":"Approval %",
            "eligibility":"Eligibility"
        }),
        use_container_width=True,
        hide_index=True
    )

with r:

    st.markdown("### ⚠ Need Attention")

    st.dataframe(
        bottom5.rename(columns={
            "branch_name":"Cabang",
            "approval_rate":"Approval %",
            "eligibility":"Eligibility"
        }),
        use_container_width=True,
        hide_index=True
    )

st.markdown("#### Top 10 Approval Rate")

top10 = summary.sort_values("approval_rate",ascending=False).head(10)

fig = px.bar(
    top10,
    x="approval_rate",
    y="branch_name",
    orientation="h",
    color="approval_rate",
    color_continuous_scale=["#C49A00","#16A34A"],
    text="approval_rate"
)

fig.update_layout(
    height=380,
    xaxis_title="Approval Rate (%)",
    yaxis_title=""
)

fig.update_traces(texttemplate="%{text:.1f}%")

st.plotly_chart(fig,use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# INSIGHT
# ------------------------------------------------------------

st.markdown("### 💡 Portfolio Insights")

best_branch = top5.iloc[0]
worst_branch = bottom5.iloc[0]
top_industry = filtered["industry"].value_counts().idxmax()
yellow = (filtered["zone"]=="Kuning").sum()

a,b = st.columns(2)

with a:

    st.success(
        f"**{best_branch['branch_name']}** memiliki Approval Rate tertinggi "
        f"({best_branch['approval_rate']:.1f}%) dengan Credit Eligibility rata-rata "
        f"{best_branch['eligibility']:.2f}."
    )

with b:

    st.warning(
        f"**{worst_branch['branch_name']}** memiliki Approval Rate terendah "
        f"({worst_branch['approval_rate']:.1f}%)."
    )

c,d = st.columns(2)

with c:

    st.info(
        f"**{top_industry}** merupakan industri dengan volume pengajuan terbesar "
        f"pada periode terpilih."
    )

with d:

    st.warning(
        f"Terdapat **{yellow:,}** nasabah di Zona Kuning yang layak menjadi prioritas monitoring."
        .replace(",",".")
    )

st.markdown("---")

# ------------------------------------------------------------
# RINGKASAN CABANG
# ------------------------------------------------------------

st.markdown("### 📋 Ringkasan Portofolio Cabang")

table = summary.sort_values("approval_rate",ascending=False).rename(columns={
    "branch_name":"Cabang",
    "total_pengajuan":"Total Pengajuan",
    "approval_rate":"Approval Rate (%)",
    "eligibility":"Avg Credit Eligibility"
})

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)