import streamlit as st

from utils.ui_components import apply_logo
from utils.ui_premium import inject_css, hero_banner

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Credit Screening Agentic AI eLO",
    page_icon="🏦",
    layout="wide",
)

apply_logo()
inject_css()

# ==========================================================
# SIDEBAR PREMIUM
# ==========================================================

with st.sidebar:

    st.divider()

    # ---------- System Status ----------
    with st.container(border=True):
        st.markdown("### ⚙️ System Status")

        c1, c2 = st.columns([1, 5])
        c1.markdown("🟢")
        c2.markdown("**AI Pipeline**\n\nActive")

        c1, c2 = st.columns([1, 5])
        c1.markdown("🤖")
        c2.markdown("**ML Model**\n\nv1.2")

        c1, c2 = st.columns([1, 5])
        c1.markdown("📅")
        c2.markdown("**Last Refresh**\n\nToday")

    st.write("")

    # ---------- Quick Actions ----------
    with st.container(border=True):
        st.markdown("### ⚡ Quick Actions")

        st.button("📄 Export Report", use_container_width=True, disabled=True)
        st.button("🔄 Refresh Dashboard", use_container_width=True, disabled=True)
        st.button("📘 Documentation", use_container_width=True, disabled=True)


# ==========================================================
# PLACEHOLDER PREMIUM PAGES
# ==========================================================

def monitoring_page():
    hero_banner(
        "📊 Monitoring Portofolio",
        "Portfolio Quality Dashboard • Monitoring kualitas portofolio seluruh cabang."
    )

    st.info("Demo Mode • Halaman premium siap dihubungkan ke data portofolio.")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Outstanding", "Rp 0")
    k2.metric("High Risk", "0")
    k3.metric("Approval", "0%")
    k4.metric("Coverage", "0 Cabang")

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Portfolio Trend")
        st.line_chart({"Outstanding": [0, 0, 0, 0, 0]})

    with c2:
        st.subheader("Risk Distribution")
        st.bar_chart({"Hijau": [0], "Kuning": [0], "Merah": [0]})


def rm_page():
    hero_banner(
        "🏆 RM Performance",
        "Relationship Manager Performance Dashboard."
    )

    st.info("Demo Mode • KPI akan otomatis terhubung ke data RM.")

    k1, k2, k3 = st.columns(3)

    k1.metric("RM Aktif", "0")
    k2.metric("Approval Rate", "0%")
    k3.metric("Avg Eligibility", "0.00")

    st.divider()

    st.subheader("Leaderboard")

    st.dataframe(
        {
            "RM": [],
            "Cabang": [],
            "Approval": [],
        },
        use_container_width=True,
        hide_index=True,
    )


def industry_page():
    hero_banner(
        "🏭 Industry Intelligence",
        "Analisis sektor usaha berdasarkan Credit Eligibility Score."
    )

    st.info("Demo Mode • Akan menggunakan data industry & sub_industry.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Industry Composition")
        st.bar_chart({"Industry": [0]})

    with c2:
        st.subheader("Sub Industry")
        st.bar_chart({"Sub Industry": [0]})


# ==========================================================
# NAVIGATION
# ==========================================================

pages = [

    st.Page(
        "pages_v2/00_Overview_Premium.py",
        title="Executive Overview",
        icon="🏠",
        default=True,
    ),

    st.Page(
        "pages_v2/01_Daftar_Pengajuan_Premium.py",
        title="Daftar Pengajuan",
        icon="📋",
    ),

    st.Page(
        "pages_v2/02_Detail_Nasabah_Premium.py",
        title="Detail Nasabah",
        icon="👤",
    ),

    st.Page(
        "pages_v2/03_Pengajuan_Credit_Baru_Premium.py",
        title="Pengajuan Kredit Baru",
        icon="📝",
    ),

    st.Page(
        "pages_v2/04_Monitoring_Portofolio_Premium.py",
        title="Monitoring Portofolio",
        icon="📊",
    ),

    st.Page(
        rm_page,
        title="RM Performance",
        icon="🏆",
    ),

    st.Page(
        industry_page,
        title="Industry Intelligence",
        icon="🏭",
    ),
]

pg = st.navigation(pages)
pg.run()