
"""Daftar Pengajuan Premium V2
Simpan sebagai: pages_v2/01_Daftar_Pengajuan_Premium.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd

from utils.data_loader import load_master_data
from utils.ui_components import apply_logo
from utils.ui_premium import inject_css, hero_banner

st.set_page_config(page_title="Daftar Pengajuan Premium", page_icon="📋", layout="wide")
apply_logo()
inject_css()

df = load_master_data()

hero_banner(
    "📋 Daftar Pengajuan Kredit",
    "Queue operasional untuk Credit Analyst & Relationship Manager."
)

# ---------- FILTER ----------
with st.container(border=True):
    st.markdown("#### 🔎 Smart Search & Filter")

    c1,c2,c3,c4 = st.columns([1.1,1,1,1.2])

    with c1:
        search = st.text_input(
            "Cari",
            placeholder="ID / Perusahaan / Pemilik"
        )

    with c2:
        branch = st.selectbox(
            "Cabang",
            ["Semua"] + sorted(df.branch_name.unique().tolist())
        )

    with c3:
        decision = st.selectbox(
            "Keputusan",
            ["Semua","Layak","Layak Bersyarat","Perlu Review Ulang","Tidak Layak"]
        )

    with c4:
        zone = st.selectbox(
            "Zona",
            ["Semua","Hijau","Kuning","Merah"]
        )

# ---------- FILTER LOGIC ----------
filtered = df.copy()

if search:
    s = search.lower()
    mask = (
        filtered["application_id"].astype(str).str.lower().str.contains(s)
        | filtered["company_name"].str.lower().str.contains(s)
        | filtered["owner_name"].str.lower().str.contains(s)
    )
    filtered = filtered[mask]

if branch != "Semua":
    filtered = filtered[filtered.branch_name == branch]

if decision != "Semua":
    filtered = filtered[filtered.decision == decision]

if zone != "Semua":
    filtered = filtered[filtered.zone == zone]

# ---------- SUMMARY ----------
st.markdown("### 📊 Queue Summary")

a,b,c,d = st.columns(4)

with a:
    st.metric("Total Queue", len(filtered))

with b:
    st.metric("Need Review", int((filtered.decision=="Perlu Review Ulang").sum()))

with c:
    st.metric("DHN Cases", int((filtered.status_dhn=="Ya").sum()))

with d:
    st.metric("Avg Eligibility", f"{filtered.risk_score.mean():.2f}")

# ---------- QUICK INSIGHT ----------
st.markdown("### 💡 AI Queue Insight")

i1,i2,i3 = st.columns(3)

with i1:
    with st.container(border=True):
        top_branch = filtered.branch_name.mode().iloc[0] if len(filtered) else "-"
        st.markdown("**Cabang Teraktif**")
        st.markdown(f"### {top_branch}")
        st.caption("Volume pengajuan tertinggi pada filter saat ini.")

with i2:
    with st.container(border=True):
        top_industry = filtered.industry.mode().iloc[0] if len(filtered) else "-"
        st.markdown("**Industri Dominan**")
        st.markdown(f"### {top_industry}")
        st.caption("Sektor dengan jumlah aplikasi terbanyak.")

with i3:
    with st.container(border=True):
        review_pct = (filtered.decision=="Perlu Review Ulang").mean()*100 if len(filtered) else 0
        st.markdown("**Manual Review Rate**")
        st.markdown(f"### {review_pct:.1f}%")
        st.caption("Semakin kecil semakin efisien screening AI.")

# ---------- TABLE ----------
st.markdown("### 📄 Daftar Pengajuan")

table = filtered[[
    "application_id",
    "company_name",
    "owner_name",
    "branch_name",
    "industry",
    "loan_requested",
    "risk_score",
    "decision"
]].copy()

table.rename(columns={
    "application_id":"ID Pengajuan",
    "company_name":"Perusahaan",
    "owner_name":"Pemilik",
    "branch_name":"Cabang",
    "industry":"Industri",
    "loan_requested":"Pinjaman",
    "risk_score":"Eligibility Score",
    "decision":"Keputusan"
}, inplace=True)

table["Pinjaman"] = table["Pinjaman"].map(lambda x:f"Rp {x:,.0f}".replace(",","."))

def decision_badge(val):
    colors={
        "Layak":"#16A34A",
        "Layak Bersyarat":"#F59E0B",
        "Perlu Review Ulang":"#EA580C",
        "Tidak Layak":"#DC2626"
    }
    c=colors.get(val,"#64748B")
    return f'background:{c}20;color:{c};font-weight:700;border:1px solid {c}55;border-radius:999px;text-align:center;'

styled=(table.style
    .format({"Eligibility Score":"{:.2f}"})
    .background_gradient(subset=["Eligibility Score"],cmap="RdYlGn")
    .map(decision_badge,subset=["Keputusan"]))

st.dataframe(styled,width="stretch",hide_index=True,height=520)

# ---------- PRIORITY ACTION ----------
st.markdown("### 🚨 Priority Action")

priority = (
    filtered.assign(priority=filtered["decision"].map({
        "Tidak Layak":0,
        "Perlu Review Ulang":1,
        "Layak Bersyarat":2,
        "Layak":3
    }).fillna(4))
    .sort_values(["status_dhn","priority","risk_score"],ascending=[False,True,True])
    .head(5)
)

p1,p2 = st.columns([1.2,1])

with p1:
    with st.container(border=True):
        st.markdown("#### Top Priority Cases")
        for _,row in priority.iterrows():
            st.markdown(
                f"""
                <div style='padding:12px;border-bottom:1px solid #E5E7EB'>
                    <b>{row['company_name']}</b><br>
                    <span style='color:#64748B'>{row['branch_name']} • {row['industry']}</span><br>
                    Eligibility Score: <b>{row['risk_score']:.2f}</b> • <span style='color:#DC2626'><b>{row['decision']}</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )

with p2:
    with st.container(border=True):
        st.markdown("#### Workflow Selanjutnya")

        st.markdown("""
        **Credit Analyst**
        - Review seluruh status **Perlu Review Ulang**.
        - Validasi kasus DHN sebelum approval.

        **Relationship Manager**
        - Hubungi nasabah dengan status Layak Bersyarat.
        - Lengkapi dokumen pendukung.
        """)
