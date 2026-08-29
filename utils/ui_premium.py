from base64 import b64encode
from pathlib import Path

import streamlit as st

LOGO_PATH = Path(__file__).parent.parent / "data" / "img" / "elo_bni_logo_crop.png"
LOGO_DATA_URI = "data:image/png;base64," + b64encode(LOGO_PATH.read_bytes()).decode("ascii")

# ==========================================================
# WONDR COLOR PALETTE
# ==========================================================
# Dipakai buat chart NON-zone (industri, RM performance, trend, dsb).
# Sengaja cuma 4 family ini (bukan semua 6) — Green & Yellow wondr
# sengaja di-exclude karena mirip warna zone-status (hijau/kuning
# risk), biar ga ketuker maknanya sama chart zone yang pakai
# traffic-light merah/kuning/oranye/hijau.
WONDR_COLORS = {
    "orange":    {"highlight": "#FEEBDD", "core": "#FF8736", "shadow1": "#F3650B", "shadow2": "#C64827"},
    "turquoise": {"highlight": "#D9F4F1", "core": "#71D8D3", "shadow1": "#52B5AB", "shadow2": "#44A095"},
    "pink":      {"highlight": "#FDE1FF", "core": "#FDA9FF", "shadow1": "#D279D7", "shadow2": "#AB59B2"},
    "purple":    {"highlight": "#E7E3FF", "core": "#9B7EDC", "shadow1": "#7863C6", "shadow2": "#5B4A99"},
}

# Urutan "core" tone buat dipakai langsung sebagai list warna di
# Plotly/matplotlib kalau butuh palet kategorikal cepat, misal:
# fig = px.bar(df, x=..., y=..., color=..., color_discrete_sequence=WONDR_CATEGORICAL)
WONDR_CATEGORICAL = [
    WONDR_COLORS["orange"]["core"],
    WONDR_COLORS["turquoise"]["core"],
    WONDR_COLORS["pink"]["core"],
    WONDR_COLORS["purple"]["core"],
]

# Zone-status tetap terpisah, JANGAN diganti pakai wondr palette.
ZONE_COLORS = {
    "layak": "#22C55E",
    "layak_bersyarat": "#EAB308",
    "perlu_review": "#F97316",
    "tidak_layak": "#EF4444",
}


# ==========================================================
# GLOBAL CSS
# ==========================================================
def inject_css():

    st.markdown("""
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    .block-container{
        padding-top:1.2rem;
        padding-bottom:2rem;
        font-family:'Inter','Segoe UI',sans-serif;
    }

    /* ======================================================
       ST.METRIC() — dipakai di beberapa halaman (Detail Nasabah,
       dst.) buat data ringkas. Default Streamlit gede banget
       (~32px value), disamain ke skala chart-title/kpi-value.
    ====================================================== */

    [data-testid="stMetricValue"]{
        font-size:18px !important;
        font-weight:700 !important;
        color:#1F2937 !important;
    }

    [data-testid="stMetricLabel"]{
        font-size:11px !important;
        color:#6B7280 !important;
    }

    html, body, [class*="css"]{
        font-family:'Inter','Segoe UI',sans-serif;
    }

    /* ======================================================
       SIDEBAR — SHELL
    ====================================================== */

    section[data-testid="stSidebar"]{
        background:#F8FAFC;
        border-right:1px solid #E5E7EB;
    }

    /* Streamlit membungkus tiap block markdown/nav dalam
       container yang punya gap default. Ini matiin gap itu
       khusus di sidebar biar spacing sepenuhnya dikontrol
       CSS custom di bawah, bukan flex-gap bawaan Streamlit. */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{
        gap:0 !important;
    }

    /* ======================================================
       SIDEBAR — NAV LIST
       (fix utama: <ul>/<li> bawaan Streamlit punya margin/padding
       sendiri yang bikin jarak antar menu kelihatan renggang)
    ====================================================== */

    section[data-testid="stSidebarNav"]{
        padding-top:.5rem;
    }

    section[data-testid="stSidebarNav"] ul{
        gap:0 !important;
        margin:0 !important;
        padding:0 !important;
    }

    section[data-testid="stSidebarNav"] li{
        margin:0 !important;
        padding:0 !important;
    }

    section[data-testid="stSidebarNav"] li div{
        margin:0 !important;
    }

    section[data-testid="stSidebarNav"] a{
        border-radius:12px;
        margin:2px 10px;
        padding:9px 12px;
        display:flex;
        align-items:center;
        gap:10px;
        font-weight:500;
        transition:.15s;
    }

    section[data-testid="stSidebarNav"] a:hover{
        background:#F3F4F6;
    }

    section[data-testid="stSidebar"] [aria-current="page"]{
        background:#EEF2F7;
        font-weight:700;
    }

    .sidebar-divider{
        border-top:1px solid #E5E7EB;
        margin:16px 0 0 0;
    }

    /* ======================================================
       SYSTEM STATUS CARD
    ====================================================== */

    .status-card{
        background:white;
        border:1px solid #E5E7EB;
        border-radius:18px;
        padding:18px;
        margin-top:16px;
    }

    .status-title{
        font-size:15px;
        font-weight:700;
        color:#1F2937;
        margin-bottom:14px;
    }

    .status-item{
        display:flex;
        align-items:flex-start;
        gap:10px;
        margin-bottom:12px;
    }

    .status-icon{
        font-size:16px;
        width:20px;
    }

    .status-label{
        font-size:12px;
        font-weight:600;
        color:#374151;
    }

    .status-value{
        font-size:12px;
        color:#6B7280;
        margin-top:1px;
        white-space:nowrap;
    }

    /* ======================================================
       QUICK ACTIONS PANEL
    ====================================================== */

    .quick-actions-card{
        background:white;
        border:1px solid #E5E7EB;
        border-radius:18px;
        padding:16px;
        margin-top:16px;
    }

    .quick-actions-title{
        font-size:15px;
        font-weight:700;
        color:#1F2937;
        margin-bottom:12px;
        display:flex;
        align-items:center;
        gap:6px;
    }

    .quick-action-btn{
        display:flex;
        align-items:center;
        gap:10px;
        border:1px solid #E5E7EB;
        border-radius:12px;
        padding:10px 12px;
        margin-bottom:8px;
        color:#9CA3AF;
        font-size:13px;
        font-weight:500;
    }

    .quick-action-btn:last-child{
        margin-bottom:0;
    }

    /* ======================================================
       KPI / METRIC CARD — kotak jelas, bukan flat/transparan
    ====================================================== */

    .kpi-card{
        background:white;
        border:1px solid #D1D5DB;
        border-radius:14px;
        padding:16px;
    }

    .kpi-label{
        font-size:12px;
        color:#6B7280;
        white-space:nowrap;
    }

    .kpi-value{
        font-size:24px;
        font-weight:700;
        color:#111827;
        margin-top:4px;
    }

    /* ======================================================
       SECTION CARD
    ====================================================== */

    .section-card{
        background:white;
        border:1px solid #D1D5DB;
        border-radius:14px;
        padding:20px;
        margin-bottom:20px;
    }

    .section-title{
        font-size:16px;
        font-weight:700;
        color:#1F2937;
        margin-bottom:14px;
    }

    /* ======================================================
       SECTION HEADER — pembatas antar section (bukan st.markdown('##..'),
       itu defaultnya kegedean, ~28px). Ini dipanggil lewat section_header().
    ====================================================== */

    .section-header{
        display:flex;
        align-items:center;
        gap:8px;
        margin:6px 0 14px 0;
        padding-top:18px;
        border-top:2px solid #E5E7EB;
    }

    .section-header-icon{
        width:26px;
        height:26px;
        border-radius:7px;
        background:#F3F4F6;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:13px;
        flex-shrink:0;
    }

    .section-header-title{
        font-size:15px;
        font-weight:700;
        color:#1F2937;
    }

    /* ======================================================
       CHART TITLE — pengganti st.markdown('#### ...') di dalam
       tiap card chart. Default h4 Streamlit itu ~28px, kegedean
       dibanding section-header-title (15px) yg jadi induknya.
    ====================================================== */

    .chart-title{
        font-size:12px;
        font-weight:600;
        color:#4B5563;
        margin-bottom:8px;
    }

    /* ======================================================
       HERO BANNER
    ====================================================== */

    .hero-banner{
        position:relative;
        background:#F3650B;
        border-radius:18px;
        padding:22px 28px;
        margin-top:12px;
        color:white;
        overflow:hidden;
        box-sizing:border-box;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:16px;
        margin-bottom:16px;
    }

    .hero-banner-circle{
        position:absolute;
        right:-30px;
        top:-30px;
        width:150px;
        height:150px;
        border-radius:50%;
        background:rgba(255,255,255,.08);
    }

    .hero-banner-left{
        display:flex;
        align-items:center;
        gap:16px;
        z-index:2;
        min-width:0;
    }

    .hero-banner-icon{
        
        border-radius:14px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:28px;
        flex-shrink:0;
    }

    .hero-banner-title{
        font-size:22px;
        font-weight:800;
        line-height:1.25;
    }

    .hero-banner-subtitle{
        margin-top:4px;
        font-size:13px;
        line-height:1.4;
        opacity:.95;
    }

    .hero-banner-badge{
        font-size:20px;
        font-weight:800;
        z-index:2;
        flex-shrink:0;
    }

    </style>
    """, unsafe_allow_html=True)


# ==========================================================
# HERO BANNER
# ==========================================================
def hero_banner(title, subtitle):
    # PENTING: sebelumnya pakai components.html() yang render di iframe
    # terpisah — elemen di dalam iframe ga bisa position:sticky ke scroll
    # halaman utama Streamlit (iframe punya scroll context sendiri).
    # Diganti st.markdown() biar jadi bagian asli halaman, baru sticky-nya
    # beneran nempel pas discroll.
    #
    # HTML ini sengaja ditulis flush-left tanpa baris kosong (concatenation
    # string), sama kayak status card di sidebar — kalau ada baris kosong +
    # indentasi, Markdown salah baca ini sebagai code block, bukan HTML.
    html = (
        '<div class="hero-banner">'
        '<div class="hero-banner-circle"></div>'
        '<div class="hero-banner-left">'
        '<div class="hero-banner-icon">'
        f'<img src="{LOGO_DATA_URI}" style="width:105px;object-fit:contain;">'
        '</div>'
        f'<div><div class="hero-banner-title">{title}</div>'
        f'<div class="hero-banner-subtitle">{subtitle}</div></div>'
        '</div>'
        '<div class="hero-banner-badge">BNI</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ==========================================================
# SECTION HEADER HELPER
# ==========================================================
def section_header(icon, title):
    """Ganti st.markdown('## icon title') — versi ini punya divider di
    atas dan font-size yang proporsional (15px), bukan default h2
    Streamlit yang gede banget (~28px)."""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-header-icon">{icon}</div>
        <div class="section-header-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)


def chart_title(text):
    """Ganti st.markdown('#### ...') di dalam card chart — versi ini
    12px, ga lebih gede dari section_header di atasnya."""
    st.markdown(f'<div class="chart-title">{text}</div>', unsafe_allow_html=True)


def rupiah_short(x):
    """Format angka rupiah jadi singkatan: Jt = Juta, M = Miliar, T = Triliun.
    Dipakai bareng di semua halaman biar formatnya konsisten."""
    if x is None or (isinstance(x, float) and x != x):  # None / NaN check tanpa import pandas
        return "Rp 0"
    if x >= 1_000_000_000_000:
        return f"Rp {x/1e12:.2f} T"
    if x >= 1_000_000_000:
        return f"Rp {x/1e9:.1f} M"
    if x >= 1_000_000:
        return f"Rp {x/1e6:.0f} Jt"
    return f"Rp {x:,.0f}".replace(",", ".")




# ==========================================================
# KPI CARD HELPER
# ==========================================================
def kpi_card(label, value, icon=""):
    """Render satu KPI card. Panggil dalam st.columns() buat bikin baris KPI."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon} {label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)