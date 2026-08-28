"""Premium Detail Nasabah V6"""

import streamlit as st

from utils.agent_pipeline import recommend_credit_type
from utils.data_loader import load_master_data
from utils.ui_components import apply_logo
from utils.ui_premium import (
    inject_css, hero_banner, section_header, chart_title, rupiah_short,
    WONDR_COLORS, ZONE_COLORS
)

st.set_page_config(page_title="Detail Nasabah", page_icon="👤", layout="wide")
apply_logo()
inject_css()

hero_banner(
    "Detail Nasabah",
    "Customer 360° View — evaluasi 5-Agent AI berdasarkan skema 5C."
)

df = load_master_data()
df["label"] = df["application_id"] + " — " + df["company_name"]
selected = st.selectbox("Pilih ID Pengajuan", df["label"])
row = df[df["label"] == selected].iloc[0]


def pick(*cols, default="-"):
    for c in cols:
        if c in row.index and row[c] is not None:
            return row[c]
    return default


# Kolom "aman" yang udah diverifikasi ada di master_dataset.csv/master_scored.csv
# (application_id, decision, zone dari master_scored — hasil keputusan akhir).
score = float(pick("risk_score", default=0))
loan = float(pick("loan_requested", default=0))
nominal_disetujui = float(pick("nominal_disetujui", default=0))
tenor = pick("jangka_waktu_bulan", default="-")
bunga = pick("bunga_persen", default=None)
jenis_kredit = pick("jenis_kredit_rekomendasi", default="-")
decision = pick("decision", default="-")
zone = pick("zone", default="-")

# PENTING: sengaja ga bikin ulang sistem klasifikasi 4-tier terpisah dari
# risk_score (kayak versi sebelumnya) — itu bisa ga sinkron sama kolom
# `decision`/`zone` yang udah jadi sumber kebenaran resmi. Warna & label
# di sini semuanya nurut zone asli.
ZONE_TO_HEX = {"Hijau": ZONE_COLORS["layak"], "Kuning": ZONE_COLORS["layak_bersyarat"], "Merah": ZONE_COLORS["tidak_layak"]}
zone_color = ZONE_TO_HEX.get(zone, "#6B7280")

# ==========================================================
# HEADER: INFO NASABAH + ELIGIBILITY BADGE
# ==========================================================

left, right = st.columns([3.3, 1])

with left:
    with st.container(border=True):
        st.markdown(f"### 🏢 {pick('company_name')}")
        st.caption(
            f"{pick('application_id')} • {pick('industry')} • "
            f"{pick('sub_industry')} • {pick('legal_entity')}"
        )

with right:
    st.markdown(f"""
    <div style="background:{zone_color}12;border:1px solid {zone_color};border-radius:18px;padding:18px;text-align:center;">
      <div style="font-size:13px;color:#64748B;">Eligibility</div>
      <div style="font-size:36px;font-weight:700;color:{zone_color};margin:8px 0;">{score:.2f}</div>
      <span style="background:{zone_color};color:white;padding:5px 12px;border-radius:999px;font-size:12px;">{decision}</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pemilik", pick("owner_name"))
k2.metric("Cabang", pick("branch_name"))
k3.metric("Pinjaman Diajukan", rupiah_short(loan))
k4.metric("Tanggal", str(pick("application_date"))[:10])

# ==========================================================
# RISK AGENT — KEPUTUSAN AKHIR
# ==========================================================

section_header("🧭", "Risk Agent — Keputusan Akhir")

with st.container(border=True):

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div style="font-size:26px;font-weight:800;color:{zone_color};">{score:.2f}</div><div style="font-size:11px;color:#6B7280;">Skor gabungan</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<span style="background:{zone_color}22;color:{zone_color};border:1px solid {zone_color};padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;">{decision}</span><div style="font-size:11px;color:#6B7280;margin-top:6px;">Keputusan</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<span style="background:{zone_color}22;color:{zone_color};border:1px solid {zone_color};padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;">{zone}</span><div style="font-size:11px;color:#6B7280;margin-top:6px;">Zona risiko</div>', unsafe_allow_html=True)

    if decision != "Tidak Layak":
        st.markdown('<div style="border-top:0.5px solid #E5E7EB;margin:12px 0;"></div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Jenis Kredit", jenis_kredit)
        d2.metric("Nominal Disetujui", rupiah_short(nominal_disetujui))
        d3.metric("Jangka Waktu", f"{tenor} bulan" if tenor != "-" else "-")
        d4.metric("Bunga", f"{bunga:.1f}% p.a." if isinstance(bunga, (int, float)) else "-")

    st.markdown(
        f'<div style="background:#E6F1FB;color:#0C447C;border-radius:8px;padding:8px 12px;margin-top:8px;font-size:13px;">'
        f'Skor gabungan {score:.2f} → keputusan {decision} (zona {zone}).</div>',
        unsafe_allow_html=True
    )

# ==========================================================
# 5-AGENT ASSESSMENT
# ==========================================================
# PENTING: real data cuma punya 5 kelompok skor (identity_passed,
# character_score, collateral_score, financial_score, cashflow_score) —
# BUKAN 6 (Identity/Credit History/DHN terpisah kayak versi sebelumnya).
# character_score itu gabungan SLIK + DHN jadi satu. Kalau ternyata
# utils/data_loader.py punya breakdown 6 terpisah, kasih tau biar aku
# sesuaikan lagi.

section_header("🤖", "5-Agent Assessment")


def agent_status(v):
    if v >= 0.75:
        return ZONE_COLORS["layak"], "Baik"
    if v >= 0.5:
        return ZONE_COLORS["layak_bersyarat"], "Cukup"
    return ZONE_COLORS["tidak_layak"], "Perlu Perhatian"


identity_ok = pick("identity_passed", default=None)
identity_score = 1.0 if identity_ok in (True, "Ya", 1, "1") else (0.0 if identity_ok is not None else None)

agents = [
    ("🪪", "Identity", identity_score, "NIK & data Dukcapil tervalidasi." if identity_score == 1.0 else "Perlu verifikasi identitas."),
    ("📊", "Character", pick("character_score", default=None), pick("character_notes", default="")),
    ("🏠", "Collateral", pick("collateral_score", default=None), pick("collateral_notes", default="")),
    ("💰", "Financial", pick("financial_score", default=None), pick("financial_notes", default="")),
    ("💳", "Cashflow", pick("cashflow_score", default=None), pick("cashflow_notes", default="")),
]

cols = st.columns(3)
for i, (icon, name, v, note) in enumerate(agents):
    with cols[i % 3]:
        with st.container(border=True):
            if v is None:
                chart_title(f"{icon} {name}")
                st.caption("Data tidak tersedia untuk agent ini.")
            else:
                v = float(v)
                c, status = agent_status(v)
                st.markdown(
                    f'<div style="font-size:11px;font-weight:600;">{icon} {name}</div>'
                    f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:6px;">'
                    f'<span style="font-size:18px;font-weight:700;color:{c};">{v:.2f}</span>'
                    f'<span style="background:{c}22;color:{c};font-size:9px;padding:1px 8px;border-radius:99px;">{status}</span></div>'
                    f'<div style="height:5px;background:#E5E7EB;border-radius:99px;margin-top:6px;">'
                    f'<div style="width:{v*100:.0f}%;height:5px;background:{c};border-radius:99px;"></div></div>',
                    unsafe_allow_html=True
                )
                if note:
                    st.caption(note)

# ==========================================================
# FINANCIAL SNAPSHOT
# ==========================================================

section_header("💼", "Financial Snapshot")

f1, f2, f3 = st.columns(3)
f1.metric("Omzet Bulanan", rupiah_short(float(pick("monthly_turnover_est", default=0))))
growth = pick("revenue_growth_pct", default=None)
f2.metric("Revenue Growth (YoY)", f"{float(growth):.1f}%" if isinstance(growth, (int, float)) else "-")
f3.metric("Jumlah Karyawan", int(float(pick("employee_count", default=0))))

# ==========================================================
# KESESUAIAN JENIS KREDIT
# ==========================================================
# Bandingkan jenis kredit yang DIAJUKAN nasabah (jenis_kredit_diajukan/
# tenor_diajukan_bulan, kolom master_scored.csv) terhadap kemampuan
# bayarnya (DSR) - dihitung live di sini (fungsi murni, tidak butuh ML),
# BUKAN dari kolom pre-computed karena field ini baru ada mulai
# predict_credit_screening() versi terbaru, belum ikut di-generate ulang
# ke master_scored.csv. Tidak ditampilkan sama sekali kalau nasabah tidak
# mengisi jenis/tenor pengajuan (jenis_kredit_sesuai == None).

credit_type_check = recommend_credit_type(
    loan, float(pick("monthly_turnover_est", default=0)) or 1,
    pick("jenis_kredit_diajukan", default=None), pick("tenor_diajukan_bulan", default=None),
)

if credit_type_check["jenis_kredit_sesuai"] is not None:
    section_header("📑", "Kesesuaian Jenis Kredit")

    sesuai = credit_type_check["jenis_kredit_sesuai"]
    sesuai_color = ZONE_COLORS["layak"] if sesuai else ZONE_COLORS["layak_bersyarat"]
    dsr = credit_type_check["dsr_pada_pengajuan"]

    with st.container(border=True):
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Jenis Diajukan", pick("jenis_kredit_diajukan", default="-"))
        tenor_diajukan = pick("tenor_diajukan_bulan", default=None)
        q2.metric("Tenor Diajukan", f"{tenor_diajukan} bulan" if tenor_diajukan else "-")
        q3.metric("Jenis Direkomendasikan", credit_type_check["jenis_kredit_rekomendasi"])
        q4.metric("DSR Pengajuan", f"{dsr*100:.0f}%" if dsr is not None else "-")

        st.markdown(
            f'<span style="background:{sesuai_color}22;color:{sesuai_color};border:1px solid {sesuai_color};'
            f'padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;">'
            f'{"Sesuai" if sesuai else "Perlu Penyesuaian"}</span>',
            unsafe_allow_html=True,
        )
        st.caption(credit_type_check["catatan_kesesuaian_kredit"])

# ==========================================================
# SUPPORTING INFORMATION
# ==========================================================

section_header("📋", "Supporting Information")

s1, s2, s3 = st.columns(3)
s1.metric("Status DHN", pick("status_dhn"))
s2.metric("Badan Usaha", pick("legal_entity"))
s3.metric("Pendidikan", pick("owner_education"))

# ==========================================================
# NEXT ACTION
# ==========================================================

section_header("➡️", "Next Action")

a, b, c = st.columns(3)
if decision == "Layak":
    a.success("Lanjut ke Analisis Kredit")
    b.info("Review Dokumen")
    c.warning("Monitoring Berkala")
elif decision == "Layak Bersyarat":
    a.warning("Lengkapi Dokumen Pendukung")
    b.info("Follow-up Relationship Manager")
    c.success("Lanjut Bersyarat")
elif decision == "Perlu Review Ulang":
    a.warning("Review Manual oleh Credit Analyst")
    b.info("Validasi Ulang Data Finansial")
    c.warning("Tahan Sementara")
else:
    a.error("Tidak Direkomendasikan")
    b.info("Informasikan ke Nasabah")
    c.warning("Arsipkan Kasus")