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
from utils.ui_premium import inject_css, hero_banner, kpi_card, section_header, chart_title, rupiah_short

st.set_page_config(page_title="Daftar Pengajuan Premium", page_icon="📋", layout="wide")
apply_logo()
inject_css()

df = load_master_data()

hero_banner(
    "Daftar Pengajuan Kredit",
    "Queue operasional untuk Credit Analyst & Relationship Manager."
)

# ---------- FILTER ----------
section_header("🔎", "Smart Search & Filter")

with st.container(border=True):

    c1, c2, c3, c4, c5 = st.columns([1.2, 0.9, 0.9, 0.9, 0.9])

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
        entity = st.selectbox(
            "Badan Usaha",
            ["Semua"] + sorted(df.legal_entity.unique().tolist())
        )

    with c4:
        # Industri dipilih daripada Zona — Zona (Hijau/Kuning/Merah) itu
        # turunan langsung dari Keputusan, jadi infonya udah kecover di
        # filter Keputusan. Industri belum ada filter-nya sama sekali
        # di halaman ini sebelumnya.
        industry = st.selectbox(
            "Industri",
            ["Semua"] + sorted(df.industry.dropna().unique().tolist())
        )

    with c5:
        decision = st.selectbox(
            "Keputusan",
            ["Semua", "Layak", "Layak Bersyarat", "Perlu Review Ulang", "Tidak Layak"]
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

if industry != "Semua":
    filtered = filtered[filtered.industry == industry]

if entity != "Semua":
    filtered = filtered[filtered.legal_entity == entity]

# ---------- SUMMARY ----------
section_header("📊", "Queue Summary")

a, b, c, d = st.columns(4)

with a:
    kpi_card("Total Queue", f"{len(filtered):,}".replace(",", "."), "📋")
with b:
    kpi_card("Need Review", int((filtered.decision == "Perlu Review Ulang").sum()), "🟡")
with c:
    kpi_card("DHN Cases", int((filtered.status_dhn == "Ya").sum()), "🚫")
with d:
    # risk_score sudah "makin tinggi makin layak" — jangan dibalik (1 - risk_score),
    # sama seperti fix di 00_Overview_Premium.py.
    kpi_card("Avg Eligibility", f"{filtered.risk_score.mean():.2f}", "🟢")

# ---------- QUICK INSIGHT ----------
section_header("💡", "Queue Snapshot")

i1, i2, i3 = st.columns(3)

with i1:
    with st.container(border=True):
        top_branch = filtered.branch_name.mode().iloc[0] if len(filtered) else "-"
        chart_title("Cabang Teraktif")
        st.markdown(f"### {top_branch}")
        st.caption("Volume pengajuan tertinggi pada filter saat ini.")

with i2:
    with st.container(border=True):
        top_industry = filtered.industry.mode().iloc[0] if len(filtered) else "-"
        chart_title("Industri Dominan")
        st.markdown(f"### {top_industry}")
        st.caption("Sektor dengan jumlah aplikasi terbanyak.")

with i3:
    with st.container(border=True):
        review_pct = (filtered.decision == "Perlu Review Ulang").mean() * 100 if len(filtered) else 0
        chart_title("Manual Review Rate")
        st.markdown(f"### {review_pct:.1f}%")
        st.caption("Semakin kecil semakin efisien screening AI.")

# ---------- TABLE ----------
section_header("📄", "Daftar Pengajuan")

table = filtered[[
    "application_id",
    "company_name",
    "owner_name",
    "branch_name",
    "industry",
    "sub_industry",
    "loan_requested",
    "risk_score",
    "decision"
]].copy()

table.rename(columns={
    "application_id": "ID Pengajuan",
    "company_name": "Perusahaan",
    "owner_name": "Pemilik",
    "branch_name": "Cabang",
    "industry": "Industri",
    "sub_industry": "Sub-Industri",
    "loan_requested": "Pinjaman",
    "risk_score": "Eligibility Score",
    "decision": "Keputusan"
}, inplace=True)

table["Pinjaman"] = table["Pinjaman"].map(rupiah_short)


def decision_badge(val):
    colors = {
        "Layak": "#16A34A",
        "Layak Bersyarat": "#F59E0B",
        "Perlu Review Ulang": "#EA580C",
        "Tidak Layak": "#DC2626"
    }
    c = colors.get(val, "#64748B")
    return f'background:{c}20;color:{c};font-weight:700;border:1px solid {c}55;border-radius:999px;text-align:center;'


styled = (table.style
          .format({"Eligibility Score": "{:.2f}"})
          .background_gradient(subset=["Eligibility Score"], cmap="RdYlGn")
          .map(decision_badge, subset=["Keputusan"]))

with st.container(border=True):
    st.dataframe(styled, width="stretch", hide_index=True, height=520)

# ---------- PRIORITY ACTION ----------
section_header("🚨", "Priority Action")


def priority_reason(row):
    # Kasus yang BENERAN butuh tindakan manusia — bukan yang udah final
    # ditolak. "Tidak Layak" polos (tanpa DHN flag) sengaja ga masuk sini
    # karena statusnya udah final, ga ada action yang perlu diambil lagi.
    if row["status_dhn"] == "Ya":
        return "DHN Flag", "🚫", 0
    if row["decision"] == "Perlu Review Ulang":
        return "Perlu Review", "🟡", 1
    if row["decision"] == "Layak Bersyarat" and row["loan_requested"] >= 500_000_000:
        return "Nominal Besar", "💰", 2
    return None, None, None


priority = filtered.copy()
priority[["reason", "reason_icon", "tier"]] = priority.apply(
    lambda r: pd.Series(priority_reason(r)), axis=1
)
priority = priority.dropna(subset=["tier"]).sort_values(["tier", "risk_score"]).head(5)

REASON_COLORS = {
    "DHN Flag": ("#FCEBEB", "#791F1F"),
    "Perlu Review": ("#FAEEDA", "#412402"),
    "Nominal Besar": ("#EAF3DE", "#173404"),
}

p1, p2 = st.columns([1.2, 1])

with p1:
    with st.container(border=True):
        chart_title("Top Priority Cases")
        st.caption("Kasus yang benar-benar butuh tindakan — bukan yang sudah final ditolak.")
        if priority.empty:
            st.info("Tidak ada kasus yang perlu diprioritaskan pada filter saat ini.")
        for _, row in priority.iterrows():
            bg, fg = REASON_COLORS[row["reason"]]
            st.markdown(
                f"""
                <div style='padding:12px;border-bottom:1px solid #E5E7EB'>
                    <span style="font-size:11px;font-weight:700;background:{bg};color:{fg};padding:2px 8px;border-radius:6px;">{row['reason_icon']} {row['reason'].upper()}</span>
                    <b style="margin-left:6px;">{row['company_name']}</b><br>
                    <span style='color:#64748B'>{row['branch_name']} • {row['industry']}</span><br>
                    Eligibility Score: <b>{row['risk_score']:.2f}</b> • <b>{row['decision']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

with p2:
    with st.container(border=True):
        chart_title("Workflow Selanjutnya")

        st.markdown("""
        **Credit Analyst**
        - Review seluruh status **Perlu Review Ulang**.
        - Validasi kasus DHN sebelum approval.

        **Relationship Manager**
        - Hubungi nasabah dengan status Layak Bersyarat.
        - Lengkapi dokumen pendukung.
        """)