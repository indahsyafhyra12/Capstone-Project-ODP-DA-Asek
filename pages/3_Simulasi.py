"""Simulasi — input manual, agent_pipeline dijalankan live (rule-based, tanpa LLM)."""
import streamlit as st

from utils.agent_pipeline import COLLECT_LABEL_MAP, score_application
from utils.data_loader import load_master_data
from utils.ui_components import apply_logo, render_full_result

st.set_page_config(page_title="Simulasi", page_icon="🧪", layout="wide")
apply_logo()
st.title("🧪 Simulasi Pengajuan Baru")
st.caption("Logic agent sepenuhnya rule-based/deterministik — tidak ada pemanggilan LLM/API eksternal.")

df = load_master_data()
industry_options = sorted(df["industry"].unique().tolist())

with st.form("simulasi_form"):
    st.subheader("Identitas")
    c1, c2 = st.columns(2)
    with c1:
        nik = st.text_input("NIK (16 digit)", max_chars=16, placeholder="3276010601750001")
    with c2:
        owner_age = st.number_input("Usia Pemilik", min_value=17, max_value=90, value=35)

    st.subheader("Riwayat Kredit (SLIK) & DHN")
    c3, c4, c5 = st.columns(3)
    with c3:
        worst_label = st.selectbox("Kolektibilitas SLIK Terburuk", options=list(COLLECT_LABEL_MAP.items()),
                                    format_func=lambda kv: kv[1], index=1)
        slik_worst = worst_label[0]
    with c4:
        slik_n_banks = st.number_input("Jumlah Bank Lain (SLIK)", min_value=0, max_value=10, value=1)
    with c5:
        status_dhn = st.selectbox("Status DHN", ["Tidak", "Ya"])
    dhn_alasan = st.text_input("Alasan DHN (jika Ya)", disabled=(status_dhn == "Tidak"))

    st.subheader("Agunan")
    c6, c7, c8 = st.columns(3)
    with c6:
        collateral_market_value = st.number_input("Nilai Pasar Agunan (Rp)", min_value=0, value=500_000_000, step=10_000_000)
    with c7:
        ownership_match = st.selectbox("Kepemilikan Sesuai Sertifikat?", ["Ya", "Tidak"])
    with c8:
        loan_requested = st.number_input("Nominal Pinjaman Diajukan (Rp)", min_value=1_000_000, value=100_000_000, step=10_000_000)

    st.subheader("Keuangan")
    c9, c10, c11 = st.columns(3)
    with c9:
        revenue_growth_pct = st.number_input("Pertumbuhan Omset YoY (%)", min_value=-100.0, max_value=200.0, value=10.0, step=1.0)
    with c10:
        profit_margin_2025 = st.number_input("Margin Laba Bersih 2025 (%)", min_value=0.0, max_value=100.0, value=11.0, step=0.5)
    with c11:
        monthly_turnover_est = st.number_input("Estimasi Omset Bulanan (Rp)", min_value=1_000_000, value=2_000_000, step=100_000)

    st.subheader("Rekening")
    c12, c13, c14 = st.columns(3)
    with c12:
        bank_best_avg_balance_6m = st.number_input("Rata-rata Saldo 6 Bulan (Rp)", min_value=0, value=1_000_000, step=100_000)
    with c13:
        bank_total_overdraft_6m = st.number_input("Jumlah Overdraft (6 bulan)", min_value=0, max_value=20, value=0)
    with c14:
        bank_any_dormant = st.checkbox("Ada rekening dormant?")

    st.subheader("Lainnya")
    industry = st.selectbox("Industri", industry_options)
    st.caption("Industri ditampilkan untuk konteks laporan — tidak dipakai sebagai input skor pada versi agent ini.")

    submitted = st.form_submit_button("Jalankan Screening", type="primary", use_container_width=True)

if submitted:
    data = {
        "NIK": nik,
        "owner_age": owner_age,
        "slik_worst_collectability": slik_worst,
        "slik_has_macet": slik_worst == 5,
        "slik_n_banks": slik_n_banks,
        "status_dhn": status_dhn,
        "dhn_alasan": dhn_alasan,
        "collateral_ratio": (collateral_market_value / loan_requested) if loan_requested else 0,
        "ownership_match": ownership_match,
        "revenue_growth_pct": revenue_growth_pct / 100.0,
        "profit_margin_2025": profit_margin_2025 / 100.0,
        "bank_best_avg_balance_6m": bank_best_avg_balance_6m,
        "monthly_turnover_est": monthly_turnover_est,
        "bank_total_overdraft_6m": bank_total_overdraft_6m,
        "bank_any_dormant": bank_any_dormant,
        "loan_requested": loan_requested,
        "collateral_liquidation_value": collateral_market_value * 0.8,
    }
    result = score_application(data)
    st.divider()
    st.subheader(f"Hasil Screening — {industry}")
    render_full_result(result)
