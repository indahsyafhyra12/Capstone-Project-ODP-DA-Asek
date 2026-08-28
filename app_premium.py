import streamlit as st
from utils.ui_components import apply_logo
from utils.ui_premium import inject_css

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(
    page_title="eLO • Credit Screening Agentic AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_logo()

# ==========================================================
# STYLE
# ==========================================================
# Semua CSS sekarang tinggal di utils/ui_premium.py (inject_css()).
# Jangan tambahin <style> block lagi di sini — dulu ada 2 sumber CSS
# yang isinya bentrok (app_premium.py vs ui_premium.py), bikin edit
# style kelihatan "gak ngefek" karena ketimpa diam-diam.
inject_css()

# ==========================================================
# PLACEHOLDER PAGES
# ==========================================================
# rm_page() dan industry_page() dihapus — udah ada halaman beneran di
# pages_v2/05_RM_Performance_Premium.py dan
# pages_v2/06_Laporan_Keuangan_Kwitansi_Premium.py

# ==========================================================
# NAVIGATION
# ==========================================================

pg = st.navigation([

    st.Page(
        "pages_v2/00_Overview_Premium.py",
        title="Executive Overview",
        icon="🏠"
    ),

    st.Page(
        "pages_v2/01_Daftar_Pengajuan_Premium.py",
        title="Daftar Pengajuan",
        icon="📋"
    ),

    st.Page(
        "pages_v2/02_Detail_Nasabah_Premium.py",
        title="Detail Nasabah",
        icon="👤"
    ),

    st.Page(
        "pages_v2/03_Pengajuan_Credit_Baru_Premium.py",
        title="Pengajuan Kredit Baru",
        icon="📝"
    ),

    st.Page(
        "pages_v2/04_Monitoring_Portofolio_Premium.py",
        title="Monitoring Portofolio",
        icon="📊"
    ),

    st.Page(
        "pages_v2/05_RM_Performance_Premium.py",
        title="RM Performance",
        icon="🏆"
    ),

    st.Page(
        "pages_v2/06_Laporan_Keuangan_Kwitansi_Premium.py",
        title="Generate Laporan Keuangan",
        icon="🧾"
    )
])

# ==========================================================
# SIDEBAR FOOTER
# ==========================================================
# Divider + status card digabung jadi SATU st.markdown call.
# Kalau dipisah jadi 2 call (seperti sebelumnya), tiap call dibungkus
# element-container sendiri oleh Streamlit yang punya gap default —
# itu sumber jarak "aneh" di antara divider dan card kemarin.

with st.sidebar:

    # PENTING: string HTML ini sengaja ditulis sebagai concatenation
    # flush-left TANPA baris kosong di antara div. Kalau ada baris
    # kosong + indentasi (seperti versi sebelumnya), Markdown salah
    # baca ini sebagai code block, bukan HTML -> muncul sebagai teks
    # mentah, bukan ke-render.
    status_card_html = (
        '<div class="sidebar-divider"></div>'
        '<div class="status-card">'
        '<div class="status-title">⚙️ System Status</div>'
        '<div class="status-item">'
        '<div class="status-icon">🟢</div>'
        '<div><div class="status-label">AI Pipeline</div>'
        '<div class="status-value">LLM: gemma-4-E2B-it</div>'
        '<div class="status-value">VLM: LightOnOCR-2-1B</div></div>'
        '</div>'
        '<div class="status-item">'
        '<div class="status-icon">🤖</div>'
        '<div><div class="status-label">ML Model</div>'
        '<div class="status-value">XGBoost v3.4.1</div></div>'
        '</div>'
        '<div class="status-item" style="margin-bottom:0;">'
        '<div class="status-icon">🗓️</div>'
        '<div><div class="status-label">Last Refresh</div>'
        '<div class="status-value">Today</div></div>'
        '</div>'
        '</div>'
    )

    # Placeholder UI dulu (belum ada logic) — nanti kalau file lama
    # yang punya logic Export Report/Refresh Dashboard/Documentation
    # ketemu, tinggal ganti div ini jadi st.button beneran.
    quick_actions_html = (
        '<div class="quick-actions-card">'
        '<div class="quick-actions-title">⚡ Quick Actions</div>'
        '<div class="quick-action-btn">📄 Export Report</div>'
        '<div class="quick-action-btn">🔄 Refresh Dashboard</div>'
        '<div class="quick-action-btn">📘 Documentation</div>'
        '</div>'
    )

    st.markdown(status_card_html + quick_actions_html, unsafe_allow_html=True)

# ==========================================================
# RUN
# ==========================================================

pg.run()