import streamlit as st

# ===== BNI Design System =====
BNI_ORANGE = "#F36F21"
BNI_TEAL = "#00A3AD"
BG = "#F8FAFC"
BORDER = "#E5E7EB"
TEXT = "#1F2937"

SUCCESS = "#16A34A"
WARNING = "#F59E0B"
DANGER = "#DC2626"


def inject_css():
    st.markdown(f"""
    <style>
    .block-container {{
        padding-top: 1.8rem;
        max-width: 1450px;
    }}

    [data-testid="stSidebar"] {{
        background: {BG};
        border-right:1px solid {BORDER};
    }}

    .hero {{
        background: linear-gradient(135deg,{BNI_ORANGE},#FF8A3D);
        border-radius:22px;
        padding:28px;
        color:white;
        margin-bottom:22px;
    }}

    .hero h1 {{
        margin:0;
        font-size:42px;
        font-weight:800;
    }}

    .hero p {{
        margin-top:8px;
        font-size:16px;
        opacity:.95;
    }}

    .metric-card {{
        background:white;
        border:1px solid {BORDER};
        border-radius:18px;
        padding:18px;
        box-shadow:0 3px 12px rgba(0,0,0,.05);
    }}

    .metric-label {{
        color:#6B7280;
        font-size:13px;
    }}

    .metric-value {{
        font-size:30px;
        font-weight:700;
        color:{TEXT};
    }}

    .section-card {{
        background:white;
        border:1px solid {BORDER};
        border-radius:18px;
        padding:20px;
        margin-bottom:18px;
    }}

    div[data-testid="stDataFrame"] {{
        border-radius:16px;
        overflow:hidden;
    }}

    div[data-testid="stPlotlyChart"] {{
        border-radius:16px;
    }}
    </style>
    """, unsafe_allow_html=True)


def hero_banner(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(icon, title, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(status):
    colors = {
        "Layak": SUCCESS,
        "Layak Bersyarat": WARNING,
        "Perlu Review Ulang": WARNING,
        "Tidak Layak": DANGER,
        "Hijau": SUCCESS,
        "Kuning": WARNING,
        "Merah": DANGER
    }

    color = colors.get(status, "#64748B")

    st.markdown(
        f"""
        <span style="
            background:{color}20;
            color:{color};
            padding:6px 12px;
            border-radius:999px;
            font-weight:600;
            font-size:14px;
            border:1px solid {color}55;
            display:inline-block;
        ">
            {status}
        </span>
        """,
        unsafe_allow_html=True,
    )