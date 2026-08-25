"""Monitoring Portofolio — breakdown zona per cabang/industri & ranking cabang."""
import plotly.express as px
import streamlit as st

from utils.data_loader import load_master_data
from utils.ui_components import ZONE_COLORS, apply_logo

st.set_page_config(page_title="Monitoring Portofolio", page_icon="📈", layout="wide")
apply_logo()
st.title("📈 Monitoring Portofolio")

df = load_master_data()
ZONE_ORDER = ["Hijau", "Kuning", "Merah"]

st.subheader("Breakdown Zona Risiko per Cabang")
branch_zone = (
    df.groupby(["branch_name", "zone"]).size().reset_index(name="count")
)
branch_order = df["branch_name"].value_counts().index.tolist()
fig = px.bar(
    branch_zone, x="branch_name", y="count", color="zone",
    color_discrete_map=ZONE_COLORS, category_orders={"zone": ZONE_ORDER, "branch_name": branch_order},
    barmode="stack",
)
fig.update_layout(xaxis_title=None, yaxis_title="Jumlah Pengajuan", legend_title=None)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Breakdown Zona Risiko per Industri")
industry_zone = df.groupby(["industry", "zone"]).size().reset_index(name="count")
industry_order = df["industry"].value_counts().index.tolist()
fig2 = px.bar(
    industry_zone, x="industry", y="count", color="zone",
    color_discrete_map=ZONE_COLORS, category_orders={"zone": ZONE_ORDER, "industry": industry_order},
    barmode="stack",
)
fig2.update_layout(xaxis_title=None, yaxis_title="Jumlah Pengajuan", legend_title=None)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("Heatmap Zona: Cabang x Industri (Rata-rata Risk Score)")
heat = df.pivot_table(index="branch_name", columns="industry", values="risk_score", aggfunc="mean")
fig3 = px.imshow(heat, color_continuous_scale=["#dc2626", "#d97706", "#16a34a"], aspect="auto",
                  labels=dict(color="Avg Risk Score"))
st.plotly_chart(fig3, use_container_width=True)

st.divider()

st.subheader("Ranking Approval Rate per Cabang")
branch_summary = (
    df.groupby("branch_name")
    .agg(
        total_pengajuan=("application_id", "count"),
        approval_rate=("decision", lambda s: s.isin(["Layak", "Layak Bersyarat"]).mean() * 100),
        avg_risk_score=("risk_score", "mean"),
    )
    .reset_index()
    .sort_values("approval_rate", ascending=False)
)
branch_summary["approval_rate"] = branch_summary["approval_rate"].round(1)
branch_summary["avg_risk_score"] = branch_summary["avg_risk_score"].round(2)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Approval rate tertinggi**")
    st.dataframe(branch_summary.head(5), use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Approval rate terendah**")
    st.dataframe(branch_summary.tail(5).sort_values("approval_rate"), use_container_width=True, hide_index=True)

fig4 = px.bar(branch_summary.sort_values("approval_rate"), x="approval_rate", y="branch_name", orientation="h",
              text_auto=".1f", color="approval_rate", color_continuous_scale=["#dc2626", "#d97706", "#16a34a"])
fig4.update_layout(yaxis_title=None, xaxis_title="Approval Rate (%)", coloraxis_showscale=False)
st.plotly_chart(fig4, use_container_width=True)
