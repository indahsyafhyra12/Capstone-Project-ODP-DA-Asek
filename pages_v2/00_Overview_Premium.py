
"""Overview Premium V2
Simpan sebagai: pages_v2/00_Overview_Premium.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_loader import load_master_data, get_filtered_data
from utils.ui_components import apply_logo, ZONE_COLORS
from utils.ui_premium import inject_css, hero_banner, metric_card, status_badge

st.set_page_config(page_title="Overview Premium", page_icon="🏦", layout="wide")
apply_logo()
inject_css()

ZONE_ORDER = ["Hijau","Kuning","Merah"]

CITY_COORDS = {
    "Jakarta Selatan": (-6.2615,106.8106),
    "Jakarta Pusat": (-6.1805,106.8284),
    "Jakarta Timur": (-6.2250,106.9004),
    "Jakarta Barat": (-6.1352,106.8133),
    "Jakarta Utara": (-6.1481,106.8998),
    "Bekasi": (-6.2383,106.9756),
    "Depok": (-6.4025,106.7942),
    "Bogor": (-6.5971,106.8060),
    "Tangerang Selatan": (-6.2897,106.7186),
    "Tangerang": (-6.1783,106.6319),
}

def rupiah(v):
    v=float(v)
    if abs(v)>=1e12: return f"Rp {v/1e12:,.2f} T".replace(",", "X").replace(".", ",").replace("X",".")
    if abs(v)>=1e9: return f"Rp {v/1e9:,.2f} M".replace(",", "X").replace(".", ",").replace("X",".")
    if abs(v)>=1e6: return f"Rp {v/1e6:,.1f} Jt".replace(",", "X").replace(".", ",").replace("X",".")
    return f"Rp {v:,.0f}".replace(",", ".")

hero_banner(
    "🏦 Credit Screening Agentic AI eLO",
    "Executive Dashboard • Decision Support System untuk screening awal kelayakan kredit UMKM berbasis 5C & Multi-Agent AI."
)

df = load_master_data()

# ================= FILTER =================
with st.container(border=True):
    st.markdown("#### 🎛 Portfolio Filter")
    c1,c2,c3,c4 = st.columns([1,1,1,2])

    with c1:
        branch = st.selectbox("Cabang",["Semua Cabang"]+sorted(df.branch_name.unique().tolist()))

    with c2:
        industry = st.selectbox("Industri",["Semua Industri"]+sorted(df.industry.unique().tolist()))

    with c3:
        if "sub_industry" in df.columns:
            sub_opts=["Semua Sub-Industry"]+sorted(df.sub_industry.dropna().unique().tolist())
            sub=st.selectbox("Sub-Industry",sub_opts)
        else:
            sub="Semua Sub-Industry"

    with c4:
        dr=st.date_input("Tanggal",value=(df.application_date.min(),df.application_date.max()))

filtered=get_filtered_data(df,branch=branch,industry=industry,date_range=dr if len(dr)==2 else None)

if "sub_industry" in filtered.columns and sub!="Semua Sub-Industry":
    filtered=filtered[filtered.sub_industry==sub]

if filtered.empty:
    st.warning("Tidak ada data.")
    st.stop()

# ================= KPI =================
approved=filtered.decision.isin(["Layak","Layak Bersyarat"])

st.markdown("### 📌 Executive Summary")

cols=st.columns(5)
items=[
("📋","Total Pengajuan",f"{len(filtered):,}".replace(",", ".")),
("✅","Approval Rate",f"{approved.mean()*100:.1f}%"),
("🟢","Avg Eligibility Score",f"{filtered.risk_score.mean():.2f}"),
("💰","Disetujui",rupiah(filtered.nominal_disetujui.sum())),
("🚫","DHN",int((filtered.status_dhn=="Ya").sum()))
]
for col,(i,t,v) in zip(cols,items):
    with col:
        metric_card(i,t,v)

# ================= HERO SECTION =================
st.markdown("### 🤖 AI Screening Outcome")

left,right=st.columns([1.3,1])

with left:
    with st.container(border=True):
        st.markdown("#### AI Decision Funnel")

        stage=pd.DataFrame({
            "Stage":[
                "Total Pengajuan",
                "Layak + Bersyarat",
                "Review",
                "Tidak Layak"
            ],
            "Count":[
                len(filtered),
                filtered.decision.isin(["Layak","Layak Bersyarat"]).sum(),
                (filtered.decision=="Perlu Review Ulang").sum(),
                (filtered.decision=="Tidak Layak").sum()
            ]
        })

        fig=px.funnel(
            stage,
            y="Stage",
            x="Count",
            color="Stage",
            color_discrete_sequence=["#F36F21","#16A34A","#F59E0B","#DC2626"]
        )

        fig.update_layout(height=360,showlegend=False,margin=dict(t=20,b=20))
        st.plotly_chart(fig,width="stretch")

with right:
    with st.container(border=True):
        st.markdown("#### Portfolio Quality")

        zone=filtered.zone.value_counts().reindex(ZONE_ORDER).fillna(0).reset_index()
        zone.columns=["Zone","Count"]

        fig=px.pie(zone,names="Zone",values="Count",hole=.55,color="Zone",color_discrete_map=ZONE_COLORS)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=260,margin=dict(t=10,b=10))
        st.plotly_chart(fig,width="stretch")

        dhn=filtered[filtered.status_dhn=="Ya"]
        pct=100 if len(dhn)==0 else (dhn.decision=="Tidak Layak").mean()*100

        st.metric("DHN Auto Reject",f"{pct:.1f}%")
        status_badge("Hijau" if pct==100 else "Merah")

# ================= FINANCIAL =================
st.markdown("### 💼 Financial Snapshot")

c1,c2,c3=st.columns(3)

with c1:
    with st.container(border=True):
        growth=pd.Series(["Growth Positif" if x>0 else "Growth Negatif" for x in filtered.revenue_growth_pct]).value_counts().reset_index()
        growth.columns=["kategori","count"]
        fig=px.pie(growth,names="kategori",values="count",hole=.45,color="kategori",
                   color_discrete_map={"Growth Positif":"#16A34A","Growth Negatif":"#DC2626"})
        fig.update_layout(height=280)
        st.plotly_chart(fig,width="stretch")

with c2:
    with st.container(border=True):
        fig=px.histogram(filtered["monthly_turnover_est"]/1e6,nbins=24,title="Monthly Turnover")
        fig.update_layout(height=280,xaxis_title="Juta Rupiah")
        st.plotly_chart(fig,width="stretch")

with c3:
    with st.container(border=True):
        fig=px.histogram(filtered,x="employee_count",nbins=20,title="Employee Count")
        fig.update_layout(height=280)
        st.plotly_chart(fig,width="stretch")

# ================= INDUSTRY =================
st.markdown("### 🏭 Industry Intelligence")

i1,i2=st.columns([1.3,1])

with i1:
    with st.container(border=True):
        treemap=filtered.groupby("industry").agg(
            Exposure=("loan_requested","sum"),
            AvgEligibility=("risk_score","mean")
        ).reset_index()

        fig=px.treemap(
            treemap,
            path=["industry"],
            values="Exposure",
            color="AvgEligibility",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig,width="stretch")

with i2:
    with st.container(border=True):
        if "sub_industry" in filtered.columns:
            sub_rank=(filtered.groupby("sub_industry")
                      .agg(AvgEligibility=("risk_score","mean"),Applications=("application_id","count"))
                      .sort_values("AvgEligibility",ascending=True)
                      .head(10)
                      .reset_index())

            fig=px.bar(sub_rank,y="sub_industry",x="AvgEligibility",orientation="h",color="AvgEligibility",color_continuous_scale="RdYlGn")
            fig.update_layout(height=360,yaxis_title="")
            st.plotly_chart(fig,width="stretch")
        else:
            st.info("Kolom sub_industry tidak tersedia.")

# ================= MAP =================
st.markdown("### 🌍 Geographic Insights")

with st.container(border=True):
    city=(filtered.groupby("city")
          .agg(Customers=("application_id","count"),AvgEligibility=("risk_score","mean"))
          .reset_index())

    city["lat"]=city.city.map(lambda x:CITY_COORDS.get(x,(None,None))[0])
    city["lon"]=city.city.map(lambda x:CITY_COORDS.get(x,(None,None))[1])
    city=city.dropna()

    fig=px.scatter_mapbox(city,lat="lat",lon="lon",size="Customers",color="AvgEligibility",
                          hover_name="city",color_continuous_scale="RdYlGn",zoom=8.5,size_max=45)
    fig.update_layout(mapbox_style="open-street-map",height=420,margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig,width="stretch")


# ================= WATCHLIST =================
st.markdown("### 🚨 Priority Review Queue")

with st.container(border=True):
    priority = filtered["decision"].map({
        "Tidak Layak": 0,
        "Perlu Review Ulang": 1,
        "Layak Bersyarat": 2,
        "Layak": 3
    }).fillna(4)

    watch = (
        filtered.assign(priority=priority)
        .sort_values(
            by=["status_dhn", "priority", "risk_score"],
            ascending=[False, True, True]
        )
        .head(10)[[
            "company_name",
            "branch_name",
            "industry",
            "risk_score",
            "decision"
        ]]
        .rename(columns={
            "company_name": "Company",
            "branch_name": "Branch",
            "industry": "Industry",
            "risk_score": "Eligibility Score",
            "decision": "Decision"
        })
    )

    st.dataframe(watch, width="stretch", hide_index=True)

# ================= BRANCH =================
st.markdown("### 🏢 Branch Performance")

b1,b2=st.columns([1,1.3])

summary=(filtered.groupby("branch_name")
         .agg(Applications=("application_id","count"),
              Approval=("decision",lambda s:s.isin(["Layak","Layak Bersyarat"]).mean()*100),
              AvgEligibility=("risk_score","mean"))
         .reset_index())

summary["Approval"]=summary["Approval"].round(1)
summary["AvgEligibility"]=summary["AvgEligibility"].round(2)

with b1:
    with st.container(border=True):
        top=summary.sort_values("Applications",ascending=False).head(10)
        fig=px.bar(top,x="Applications",y="branch_name",orientation="h",color="Applications",color_continuous_scale="RdYlGn")
        fig.update_layout(height=380,yaxis_title="")
        st.plotly_chart(fig,width="stretch")

with b2:
    with st.container(border=True):
        styled=(summary.sort_values("Applications",ascending=False)
                .style.background_gradient(subset=["Approval"],cmap="Greens")
                .background_gradient(subset=["AvgEligibility"],cmap="RdYlGn"))
        st.dataframe(styled,width="stretch",hide_index=True)
