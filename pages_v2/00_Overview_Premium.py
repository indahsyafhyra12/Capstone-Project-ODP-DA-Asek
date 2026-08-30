import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import load_industry_cluster_network
from utils.ui_components import apply_logo
from utils.ui_premium import hero_banner, kpi_card, section_header, chart_title, rupiah_short, inject_css, WONDR_COLORS, ZONE_COLORS

# PENTING: ini yang sebelumnya kelewat — 01_Daftar_Pengajuan_Premium.py
# (dan halaman lain) manggil apply_logo() + inject_css() + set_page_config()
# sendiri di atas tiap file, bukan cuma dari app_premium.py. Tanpa ini,
# logo sidebar-nya render beda (kepotong) khusus di halaman Overview.
st.set_page_config(page_title="Executive Overview Premium", page_icon="🏠", layout="wide")
apply_logo()
inject_css()

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data():

    master = pd.read_csv("data/processed/master_dataset.csv")
    scored = pd.read_csv("data/processed/master_scored.csv")

    df = master.merge(
        scored[
            [
                "application_id",
                "risk_score",
                "decision",
                "zone",
                "nominal_disetujui",
                "jenis_kredit_rekomendasi"
            ]
        ],
        on="application_id",
        how="left"
    )

    # PENTING: risk_score di master_scored.csv itu SUDAH "semakin tinggi
    # semakin layak" (rata-rata Layak=0.81, Tidak Layak=0.36) — BUKAN skor
    # risiko yang perlu dibalik. Versi sebelumnya salah nulis
    # `1 - risk_score`, itu yang bikin Avg Eligibility kelihatan 0.24
    # padahal harusnya ~0.76. Jangan dibalik lagi di sini.
    df["credit_eligibility"] = df["risk_score"]

    return df


df = load_data()

# ======================================================
# SAFE COLUMN MAPPING (ANTI ERROR)
# ======================================================

DATE_COL = "application_date"
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# ======================================================
# HERO
# ======================================================

hero_banner(
    "Credit Screening Agentic AI eLO",
    "Executive Dashboard • Decision Support System untuk screening awal kelayakan kredit UMKM berbasis 5C & Multi-Agent AI."
)

# ======================================================
# PORTFOLIO FILTER
# ======================================================

section_header("🗂️", "Portfolio Filter")

with st.container(border=True):

    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.5])

    with c1:
        branch = st.selectbox(
            "Cabang",
            ["Semua Cabang"] + sorted(df.branch_name.dropna().unique().tolist())
        )

    with c2:
        industry = st.selectbox(
            "Industri",
            ["Semua Industri"] + sorted(df.industry.dropna().unique().tolist())
        )

    with c3:
        sub = st.selectbox(
            "Sub-Industry",
            ["Semua Sub-Industry"] + sorted(df.sub_industry.dropna().unique().tolist())
        )

    with c4:
        date_range = st.date_input(
            "Periode",
            value=(df[DATE_COL].min().date(), df[DATE_COL].max().date()),
            min_value=df[DATE_COL].min().date(),
            max_value=df[DATE_COL].max().date(),
        )

# ======================================================
# APPLY FILTER
# ======================================================

filtered = df.copy()

if branch != "Semua Cabang":
    filtered = filtered[filtered.branch_name == branch]

if industry != "Semua Industri":
    filtered = filtered[filtered.industry == industry]

if sub != "Semua Sub-Industry":
    filtered = filtered[filtered.sub_industry == sub]

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered[DATE_COL] >= pd.to_datetime(start_date))
        & (filtered[DATE_COL] <= pd.to_datetime(end_date))
    ]

# ======================================================
# FORMAT RUPIAH
# ======================================================

# ======================================================
# FORMAT RUPIAH
# ======================================================
# rupiah_short() sekarang dari utils/ui_premium.py — dipake bareng
# sama halaman lain (Daftar Pengajuan), jangan definisiin ulang di sini.


# ======================================================
# EXECUTIVE SUMMARY
# ======================================================

total = len(filtered)

layak = (filtered.decision == "Layak").sum()
layak_bersyarat = (filtered.decision == "Layak Bersyarat").sum()
review = (filtered.decision == "Perlu Review Ulang").sum()
tidak_layak = (filtered.decision == "Tidak Layak").sum()

approval = (layak + layak_bersyarat) / total if total else 0

avg_eligibility = filtered.credit_eligibility.mean()

nominal = filtered.nominal_disetujui.fillna(0).sum()

# "Cabang Aktif" dihitung dari kombinasi branch_name + region, BUKAN
# branch_name doang — karena 1 nama cabang (misal "KCP Cibubur") muncul
# di ke-4 region sebagai 4 kantor fisik berbeda. branch_name.nunique()
# cuma ngasih 10 (salah), branch_name+region ngasih 40 (bener, cocok
# sama jumlah RM di rm_master).
cabang = filtered[["branch_name", "region"]].drop_duplicates().shape[0]

section_header("📌", "Executive Summary")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    kpi_card("Total Pengajuan", f"{total:,}".replace(",", "."), "📋")
with c2:
    kpi_card("Approval Rate", f"{approval:.1%}", "✅")
with c3:
    kpi_card("Avg Eligibility", f"{avg_eligibility:.2f}", "🟢")
with c4:
    kpi_card("Disetujui", rupiah_short(nominal), "💰")
with c5:
    kpi_card("Cabang Aktif", str(cabang), "🏦")

# ======================================================
# RINGKASAN HASIL SCREENING
# ======================================================
st.markdown("<br>", unsafe_allow_html=True)
section_header("✅", "Ringkasan Hasil Screening")

s1, s2, s3 = st.columns([1.2, 1, 1])

with s1:
    with st.container(border=True):

        chart_title("AI Decision Funnel")

        funnel = pd.DataFrame({
            "Stage": ["Total Pengajuan", "Layak + Bersyarat", "Tidak Layak", "Review"],
            "Value": [total, layak + layak_bersyarat, tidak_layak, review]
        })

        # Warna dibalikin ke gaya original: Total pakai oranye wondr
        # (bukan abu-abu netral), Layak+Bersyarat/Tidak Layak/Review
        # tetap pakai warna zone biar konsisten sama Portfolio Quality
        # donut di sebelahnya. Review pakai kuning (bukan oranye
        # perlu_review) biar ga mirip sama warna Total.
        fig = go.Figure(go.Bar(
            x=funnel["Value"],
            y=funnel["Stage"],
            orientation="h",
            text=funnel["Value"].apply(lambda v: f"{v:,}".replace(",", ".")),
            textposition="outside",
            textfont=dict(size=13),
            marker_color=[
                WONDR_COLORS["orange"]["core"],
                ZONE_COLORS["layak"],
                ZONE_COLORS["tidak_layak"],
                ZONE_COLORS["layak_bersyarat"],
            ],
        ))

        fig.update_layout(
            height=280,
            margin=dict(l=0, r=40, t=10, b=10),
            xaxis=dict(visible=False, range=[0, total * 1.25]),
            yaxis=dict(categoryorder="array", categoryarray=funnel.Stage[::-1]),
            showlegend=False,
            font=dict(family="Inter, sans-serif"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Legend eksplisit — warna di sini juga dipakai buat makna zone
        # di chart lain, jadi perlu ditulis jelas apa maksud tiap warna.
        st.markdown(
            f"""
            <div style="display:flex; flex-wrap:wrap; gap:12px; font-size:10px; color:#6B7280; padding-top:6px; border-top:1px solid #F3F4F6;">
                <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{WONDR_COLORS["orange"]["core"]};margin-right:4px;"></span>Total pengajuan</span>
                <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{ZONE_COLORS["layak"]};margin-right:4px;"></span>Layak + Bersyarat</span>
                <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{ZONE_COLORS["tidak_layak"]};margin-right:4px;"></span>Tidak Layak</span>
                <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{ZONE_COLORS["layak_bersyarat"]};margin-right:4px;"></span>Perlu Review</span>
            </div>
            """,
            unsafe_allow_html=True
        )

with s2:
    with st.container(border=True):

        chart_title("Portfolio quality (zone)")

        quality = (
            filtered["zone"]
            .value_counts()
            .reindex(["Hijau", "Kuning", "Merah"])
            .fillna(0)
        )

        fig = go.Figure(go.Pie(
            labels=quality.index,
            values=quality.values,
            hole=0.6,
            marker_colors=[ZONE_COLORS["layak"], ZONE_COLORS["layak_bersyarat"], ZONE_COLORS["tidak_layak"]],
            textinfo="percent",
        ))

        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10), showlegend=True,
                           font=dict(family="Inter, sans-serif"))

        st.plotly_chart(fig, use_container_width=True)

with s3:
    with st.container(border=True):

        chart_title("Jenis kredit diajukan")

        kredit = filtered["jenis_kredit_diajukan"].value_counts()

        fig = go.Figure(go.Pie(
            labels=kredit.index,
            values=kredit.values,
            hole=0.6,
            marker_colors=[
                WONDR_COLORS["orange"]["core"],
                WONDR_COLORS["turquoise"]["core"],
                WONDR_COLORS["pink"]["core"],
            ],
            textinfo="percent",
        ))

        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10), showlegend=True,
                           font=dict(family="Inter, sans-serif"))

        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# PROFIL NASABAH
# ======================================================

section_header("👤", "Profil Nasabah")

n1, n2, n3, n4 = st.columns(4)

with n1:
    with st.container(border=True):
        chart_title("Badan usaha")
        entity = filtered["legal_entity"].value_counts()
        fig = go.Figure(go.Pie(
            labels=entity.index, values=entity.values, hole=0.6,
            marker_colors=[WONDR_COLORS["turquoise"][t] for t in ["core", "shadow1", "shadow2"]],
            textinfo="percent",
        ))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), showlegend=True,
                           font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with n2:
    with st.container(border=True):
        chart_title("Usia bisnis (tahun)")
        fig = px.histogram(
            filtered, x="business_age_year", nbins=10,
            color_discrete_sequence=[WONDR_COLORS["purple"]["core"]],
        )
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), showlegend=False,
                           yaxis_title=None, xaxis_title=None, font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with n3:
    with st.container(border=True):
        chart_title("Pendidikan pemilik")
        edu_order = ["SMA/SMK", "D3", "S1", "S2"]
        edu = filtered["owner_education"].value_counts().reindex(edu_order).fillna(0)
        fig = go.Figure(go.Bar(
            x=edu.values, y=edu.index, orientation="h",
            marker_color=WONDR_COLORS["pink"]["core"],
        ))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), showlegend=False,
                           xaxis_title=None, font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with n4:
    with st.container(border=True):
        chart_title("Gender pemilik")
        # Kolom aslinya owner_gender, bukan "gender"/"jenis_kelamin" (2 nama
        # itu ga ada di data — versi sebelumnya salah nebak nama kolom).
        gender = filtered["owner_gender"].value_counts()
        fig = go.Figure(go.Pie(
            labels=gender.index.map({"P": "Perempuan", "L": "Laki-laki"}),
            values=gender.values, hole=0.6,
            marker_colors=[WONDR_COLORS["orange"]["core"], WONDR_COLORS["orange"]["highlight"]],
            textinfo="percent",
        ))
        fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=10), showlegend=True,
                           font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# GEOGRAPHIC INSIGHTS
# ======================================================

section_header("🌍", "Geographic Insights")

# PENTING: branch_coords ini isinya nama KOTA, jadi lookup-nya harus
# pakai kolom `city`, bukan `branch_name`. Versi sebelumnya nge-lookup
# pakai branch_name (isinya "KCP Cibubur" dst) yang ga match sama
# sekali ke dictionary ini — hasilnya semua baris ke-drop dan peta
# kosong.
branch_coords = {
    "Jakarta Pusat": (-6.175, 106.827),
    "Jakarta Selatan": (-6.261, 106.811),
    "Jakarta Barat": (-6.167, 106.763),
    "Jakarta Timur": (-6.225, 106.900),
    "Jakarta Utara": (-6.138, 106.880),
    "Bekasi": (-6.238, 106.975),
    "Depok": (-6.402, 106.794),
    "Bogor": (-6.595, 106.816),
    "Tangerang": (-6.178, 106.631),
    "Tangerang Selatan": (-6.286, 106.718),
}

geo = (
    filtered.groupby("city", as_index=False)
    .agg(
        total_pengajuan=("application_id", "count"),
        avg_eligibility=("credit_eligibility", "mean"),
        nominal_disetujui=("nominal_disetujui", "sum"),
    )
)

geo["lat"] = geo["city"].map(lambda x: branch_coords.get(x, (None, None))[0])
geo["lon"] = geo["city"].map(lambda x: branch_coords.get(x, (None, None))[1])
geo["nominal_disetujui_short"] = geo["nominal_disetujui"].map(rupiah_short)
geo = geo.dropna(subset=["lat", "lon"])

with st.container(border=True):
    if geo.empty:
        st.info("Tidak ada data lokasi untuk filter saat ini.")
    else:
        fig = px.scatter_mapbox(
            geo, lat="lat", lon="lon",
            size="total_pengajuan", color="nominal_disetujui",
            hover_name="city",
            hover_data={
                "nominal_disetujui": False,
                "nominal_disetujui_short": True,
                "avg_eligibility": ":.2f",
                "lat": False,
                "lon": False,
            },
            color_continuous_scale="RdYlGn", zoom=8, height=480,
        )
        fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# FINANCIAL SNAPSHOT
# ======================================================

section_header("💰", "Financial Snapshot")

f1, f2, f3 = st.columns(3)

with f1:
    with st.container(border=True):
        chart_title("Growth revenue (2024→2025)")
        growth = (filtered["revenue_growth_pct"] > 0).map({True: "Growth Positif", False: "Growth Negatif"})
        counts = growth.value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.6,
            marker_colors=[ZONE_COLORS["layak"], ZONE_COLORS["tidak_layak"]],
            textinfo="percent",
        ))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10), showlegend=True,
                           font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with f2:
    with st.container(border=True):
        chart_title("Monthly turnover")
        fig = px.histogram(
            filtered, x="monthly_turnover_est", nbins=25,
            color_discrete_sequence=[WONDR_COLORS["turquoise"]["core"]],
        )
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10), showlegend=False,
                           yaxis_title=None, xaxis_title="Rp", font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with f3:
    with st.container(border=True):
        chart_title("Employee count")
        fig = px.histogram(
            filtered, x="employee_count", nbins=20,
            color_discrete_sequence=[WONDR_COLORS["purple"]["core"]],
        )
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10), showlegend=False,
                           yaxis_title=None, xaxis_title=None, font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# INDUSTRY INTELLIGENCE
# ======================================================

section_header("🏭", "Industry Intelligence")

i1, i2 = st.columns([1.2, 1])

ind = (
    filtered.groupby("industry")
    .agg(
        Applications=("application_id", "count"),
        Eligibility=("credit_eligibility", "mean")
    )
    .reset_index()
)

with i1:
    with st.container(border=True):
        fig = px.treemap(
            ind, path=["industry"], values="Applications",
            color="Eligibility", color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=400, font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with i2:
    with st.container(border=True):
        sub_top = (
            filtered.groupby("sub_industry")["credit_eligibility"]
            .mean().nlargest(10).sort_values().reset_index()
        )
        fig = px.bar(
            sub_top, x="credit_eligibility", y="sub_industry", orientation="h",
            color="credit_eligibility", color_continuous_scale="RdYlGn",
        )
        fig.update_layout(height=400, coloraxis_showscale=False, font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

if not ind.empty:
    top_ind = ind.sort_values("Eligibility", ascending=False).iloc[0]
    st.info(
        f"**{top_ind.industry}** memiliki rata-rata Credit Eligibility tertinggi "
        f"({top_ind.Eligibility:.2f}) dengan **{top_ind.Applications:,}** pengajuan."
    )

# ------------------------------------------------------
# Cluster Risiko Lintas-Segmen (Graph Analytics)
# hasil notebooks/graph_analytics_industry_part 2.ipynb, diekspor ke
# data/processed/industry_cluster_nodes.csv & industry_cluster_edges.csv.
# Nasabah di-cluster (Louvain) bukan cuma berdasarkan sub_industry|region
# yang sama persis, tapi juga kemiripan risk_score lintas segmen — jadi
# 1 cluster bisa memotong batas industri. Ini analisis atas SELURUH
# portofolio (tidak ikut filter Cabang/Industri/Sub-Industry di atas),
# karena cluster-nya dihitung dari komposisi lintas-segmen.
# ------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):
    chart_title("🕸️ Cluster Risiko Lintas-Segmen (Graph Analytics)")
    st.caption(
        "Nasabah dikelompokkan berdasarkan kombinasi segmen (sub-industry × region) **dan** kemiripan "
        "risk_score, sehingga satu cluster bisa berisi lebih dari satu industri kalau profil risikonya "
        "menyatu. Analisis ini mencakup seluruh portofolio (tidak mengikuti filter di atas)."
    )

    ic_nodes, ic_edges = load_industry_cluster_network()

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        kpi_card("Jumlah Cluster", ic_nodes["cluster_id"].nunique(), "🧩")
    with ic2:
        kpi_card("Cluster Risiko Tinggi", int((ic_nodes["low_score_share"] >= 0.5).sum()), "🚨")
    with ic3:
        kpi_card("Avg Nasabah/Cluster", f"{ic_nodes['n_nasabah'].mean():.0f}", "👥")
    with ic4:
        kpi_card("Pasangan Cluster Terhubung", len(ic_edges), "🔗")

    g1, g2 = st.columns([1.4, 1])

    with g1:
        edges_to_draw = ic_edges.nlargest(60, "weight")
        pos_lookup = ic_nodes.set_index("cluster_id")[["pos_x", "pos_y"]].to_dict("index")

        edge_x, edge_y = [], []
        for _, r in edges_to_draw.iterrows():
            p0, p1 = pos_lookup[r["cluster_a"]], pos_lookup[r["cluster_b"]]
            edge_x += [p0["pos_x"], p1["pos_x"], None]
            edge_y += [p0["pos_y"], p1["pos_y"], None]

        fig_ic = go.Figure()
        fig_ic.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1, color="rgba(156,163,175,0.45)"),
            hoverinfo="none", showlegend=False,
        ))
        fig_ic.add_trace(go.Scatter(
            x=ic_nodes["pos_x"], y=ic_nodes["pos_y"], mode="markers",
            marker=dict(
                size=(ic_nodes["n_nasabah"] / 3).clip(lower=10),
                color=ic_nodes["avg_score"], colorscale="RdYlGn", cmin=0, cmax=1,
                showscale=True, colorbar=dict(title="Avg Score"),
                line=dict(width=1, color="white"),
            ),
            text=ic_nodes.apply(
                lambda r: (
                    f"Cluster {r['cluster_id']}<br>{r['top_segments']}<br>"
                    f"Nasabah: {r['n_nasabah']} · Avg Score: {r['avg_score']:.2f}<br>"
                    f"Risiko Rendah: {r['low_score_share']:.0%}"
                ), axis=1,
            ),
            hoverinfo="text", showlegend=False,
        ))
        fig_ic.update_layout(
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            font=dict(family="Inter, sans-serif"), plot_bgcolor="white",
        )
        st.plotly_chart(fig_ic, use_container_width=True)

    with g2:
        chart_title("Cluster Risiko Tertinggi")
        top_risk_ic = ic_nodes.sort_values("low_score_share", ascending=False).head(5).copy()
        top_risk_ic["Risiko Rendah"] = top_risk_ic["low_score_share"].map("{:.0%}".format)
        top_risk_ic["Avg Score"] = top_risk_ic["avg_score"].map("{:.2f}".format)
        st.dataframe(
            top_risk_ic[["cluster_id", "top_segments", "n_nasabah", "Avg Score", "Risiko Rendah"]]
            .rename(columns={"cluster_id": "Cluster", "top_segments": "Segmen Dominan", "n_nasabah": "Nasabah"}),
            use_container_width=True, hide_index=True, height=230,
        )

        chart_title("Pasangan Cluster Paling Erat Terhubung")
        seg_lookup = ic_nodes.set_index("cluster_id")["top_segments"].to_dict()
        top_pairs_ic = ic_edges.nlargest(5, "weight").copy()
        top_pairs_ic["Cluster A"] = top_pairs_ic["cluster_a"].map(lambda c: f"{c} — {seg_lookup.get(c, '')}")
        top_pairs_ic["Cluster B"] = top_pairs_ic["cluster_b"].map(lambda c: f"{c} — {seg_lookup.get(c, '')}")
        st.dataframe(
            top_pairs_ic[["Cluster A", "Cluster B", "weight"]].rename(columns={"weight": "Kedekatan"}),
            use_container_width=True, hide_index=True, height=230,
        )

# ======================================================
# BRANCH INTELLIGENCE
# ======================================================

section_header("🏦", "Branch Intelligence")

# Group by branch_name + region (bukan branch_name doang) — 1 nama
# cabang muncul di 4 region berbeda sebagai kantor fisik yang beda.
branch_perf = (
    filtered.groupby(["branch_name", "region"])
    .agg(
        Approval=("decision", lambda x: x.isin(["Layak", "Layak Bersyarat"]).mean()),
        Eligibility=("credit_eligibility", "mean")
    )
    .reset_index()
)
branch_perf["label"] = branch_perf["branch_name"] + " (" + branch_perf["region"] + ")"

b1, b2 = st.columns([1.4, 1])

with b1:
    with st.container(border=True):
        top5 = branch_perf.nlargest(5, "Approval")
        fig = px.bar(
            top5, x="Approval", y="label", orientation="h",
            color="Eligibility", color_continuous_scale="RdYlGn", text="Approval",
        )
        fig.update_traces(texttemplate="%{text:.0%}")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=10), yaxis_title=None,
                           font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

with b2:
    with st.container(border=True):
        chart_title(f"🔍 Cabang perlu perhatian")
        st.markdown(
            f'<div style="font-size:10px;color:#9CA3AF;margin-top:-6px;margin-bottom:10px;">'
            f'5 cabang dengan approval rate di bawah rata-rata portofolio ({approval:.1%})</div>',
            unsafe_allow_html=True
        )
        worst = branch_perf.nsmallest(5, "Approval")
        for _, r in worst.iterrows():
            delta_pp = (r.Approval - approval) * 100
            st.markdown(
                f"""
                <div style="border:1px solid #E5E7EB; border-radius:12px; padding:10px 12px; margin-bottom:10px; background:white;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:600;color:#1F2937;font-size:13px;">{r.label}</span>
                        <span style="color:#C64827;font-size:13px;font-weight:700;">{r.Approval:.1%}</span>
                    </div>
                    <div style="color:#9CA3AF;font-size:11px;margin-top:2px;">{delta_pp:.1f}pp dari rata-rata</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ======================================================
# Insight Portofolio
# ======================================================

section_header("📌", "Insight Portofolio")

top_credit = filtered["jenis_kredit_diajukan"].mode()[0] if not filtered.empty else "-"
top_industry = ind.sort_values("Eligibility", ascending=False).iloc[0]["industry"] if not ind.empty else "-"
yellow = (filtered.zone == "Kuning").sum()
best_branch = branch_perf.sort_values("Eligibility", ascending=False).iloc[0] if not branch_perf.empty else None

r1, r2 = st.columns(2)
r3, r4 = st.columns(2)

with r1:
    st.success(f"**{top_credit}** menjadi jenis kredit yang paling banyak diajukan oleh debitur.")
with r2:
    st.success(f"Approval Rate mencapai **{approval:.1%}**, menunjukkan mayoritas pengajuan memenuhi kriteria kelayakan.")
with r3:
    st.info(f"**{top_industry}** merupakan industri dengan rata-rata Credit Eligibility tertinggi.")
with r4:
    st.warning(f"Terdapat **{yellow}** debitur pada **Zona Kuning** yang perlu diprioritaskan untuk monitoring lanjutan.")

st.markdown("---")

if best_branch is not None:
    st.markdown(
        f"""
        <div style="background:#FFF7ED; border-left:5px solid #F87336; border-radius:12px; padding:16px 18px;">
            <div style="font-weight:700;color:#1F2937;margin-bottom:6px;">Executive Takeaway</div>
            <div style="color:#4B5563;line-height:1.6;">
                Cabang <b>{best_branch.label}</b> menunjukkan performa terbaik berdasarkan
                Credit Eligibility. Fokus peningkatan berikutnya adalah melakukan pendampingan pada
                debitur Zona Kuning serta mengoptimalkan portofolio di industri dengan eligibility yang masih rendah.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )