"""Screening Pengajuan Credit Baru — predict screening lewat pipeline hybrid ML.

Dua cara input:
  1. Manual (1 nasabah) — form lengkap.
  2. Upload CSV — 1 baris otomatis ngisi form manual utk direview sebelum
     submit; >1 baris diproses batch langsung, hasilnya tabel + download.

Field sistem (application_id, cif_number, application_date, dst.) selalu
diisi otomatis, tidak pernah diminta dari user maupun CSV. Riwayat SLIK/
DHN/rekening/keuangan ditelusuri otomatis lewat NIK dari data/raw/*.csv
(build_features_from_raw) — kalau NIK sudah ada di sistem, riwayat asli
dipakai; kalau NIK baru, dipakai default netral (bukan hard-reject),
konsisten dengan notebooks/04_deploy_predict_ml_risk_scoring.ipynb.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils.agent_pipeline import _dukcapil_names, _normalize_name
from utils.feature_builder import build_features_from_raw, load_raw_tables
from utils.report_agent import generate_report, get_last_fallback_reason
from utils.risk_ml_pipeline import predict_credit_screening
from utils.ui_components import apply_logo

st.set_page_config(page_title="Pengajuan Credit Baru", page_icon="🧪", layout="wide")
apply_logo()
st.title("Screening Pengajuan Credit Baru")
st.caption("Lengkapi data pengajuan di bawah untuk mendapatkan rekomendasi kelayakan kredit secara otomatis dari sistem.")


@st.cache_data
def _load_raw():
    return load_raw_tables()


@st.cache_data
def _load_rm_master():
    return pd.read_csv("data/raw/rm_master.csv")


@st.cache_data
def _load_dukcapil():
    return pd.read_csv("data/raw/dukcapil.csv", dtype={"NIK": str})


profile_full, slik_full, dhn_full, bank_full, fin_full = _load_raw()
rm_master = _load_rm_master()
dukcapil_full = _load_dukcapil()

LEGAL_ENTITY_OPTIONS = sorted(profile_full["legal_entity"].unique().tolist())
EDUCATION_OPTIONS = sorted(profile_full["owner_education"].unique().tolist())
MARITAL_OPTIONS = sorted(profile_full["owner_marital_status"].unique().tolist())
REGION_OPTIONS = sorted(profile_full["region"].unique().tolist())
BRANCH_OPTIONS = sorted(rm_master["branch_name"].unique().tolist())
INDUSTRY_SUBINDUSTRY = profile_full.groupby("industry")["sub_industry"].unique().apply(sorted).to_dict()
INDUSTRY_OPTIONS = sorted(INDUSTRY_SUBINDUSTRY.keys())
COLLATERAL_TYPE_OPTIONS = sorted(profile_full["collateral_type"].unique().tolist())
CERTIFICATE_TYPE_OPTIONS = sorted(profile_full["certificate_type"].unique().tolist())
GENDER_OPTIONS = ["L", "P"]
YA_TIDAK_OPTIONS = ["Ya", "Tidak"]

DUKCAPIL_NIKS = set(dukcapil_full["NIK"])
KNOWN_NIKS = set(profile_full["NIK"]) | set(slik_full["NIK"]) | set(bank_full["NIK"]) | set(fin_full["NIK"])

# Kolom CSV upload = persis nama kolom retail_customer_profile.csv (boleh
# upload slice CSV itu apa adanya - kolom sistem seperti application_id/
# cif_number/eligibility_score/label kalau ada akan diabaikan & dibuat ulang).
REQUIRED_CSV_COLUMNS = [
    "NIK", "owner_age", "legal_entity", "owner_education", "industry", "region", "branch_name",
    "business_age_year", "employee_count", "monthly_turnover_est", "transaction_frequency_monthly",
    "loan_requested", "collateral_type", "collateral_market_value", "collateral_liquidation_value",
    "certificate_type", "ownership_match", "estimated_dsr",
]
OPTIONAL_CSV_DEFAULTS = {
    "company_name": "", "owner_name": "", "owner_gender": "L", "owner_marital_status": "Menikah",
    "province": "", "city": "", "district": "", "sub_industry": None,
    "collateral_location": "", "collateral_province": "", "collateral_city": "",
    "collateral_size_m2": 0.0, "rm_id": None,
}
FORM_FIELDS_ORDER = [
    "NIK", "owner_name", "owner_age", "owner_gender", "owner_marital_status", "owner_education",
    "company_name", "legal_entity", "industry", "sub_industry", "business_age_year", "employee_count",
    "branch_name", "province", "city", "district", "region",
    "monthly_turnover_est", "transaction_frequency_monthly", "loan_requested", "estimated_dsr",
    "collateral_type", "certificate_type", "collateral_market_value", "collateral_liquidation_value",
    "collateral_size_m2", "ownership_match", "collateral_location", "collateral_province", "collateral_city",
]


def _apply_defaults(row: dict) -> dict:
    """Isi kolom opsional yang tidak ada di CSV dengan default masuk akal."""
    row = dict(row)
    for col, default in OPTIONAL_CSV_DEFAULTS.items():
        if col not in row or pd.isna(row.get(col)):
            row[col] = default
    if not row.get("sub_industry"):
        opts = INDUSTRY_SUBINDUSTRY.get(row.get("industry"), [])
        row["sub_industry"] = opts[0] if opts else ""
    if not row.get("company_name"):
        row["company_name"] = f"Nasabah NIK {row.get('NIK')}"
    return row


def _validate_csv_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]


def _build_csv_template() -> bytes:
    example = {
        "NIK": "3276010601750099", "company_name": "CV Contoh Sejahtera", "legal_entity": "CV",
        "owner_name": "Contoh Pemilik", "owner_gender": "L", "owner_age": 40,
        "owner_marital_status": "Menikah", "owner_education": "S1",
        "province": "Jawa Barat", "city": "Depok", "district": "Sukmajaya", "region": "Region 2",
        "branch_name": BRANCH_OPTIONS[0], "industry": "Perdagangan",
        "sub_industry": INDUSTRY_SUBINDUSTRY.get("Perdagangan", [""])[0],
        "business_age_year": 5, "employee_count": 10, "monthly_turnover_est": 50_000_000,
        "transaction_frequency_monthly": 80, "loan_requested": 200_000_000,
        "collateral_type": "Rumah", "collateral_location": "Sukmajaya, Depok",
        "collateral_province": "Jawa Barat", "collateral_city": "Depok", "collateral_size_m2": 100.0,
        "collateral_market_value": 500_000_000, "collateral_liquidation_value": 400_000_000,
        "certificate_type": "SHM", "ownership_match": "Ya", "estimated_dsr": 1.0,
    }
    cols = REQUIRED_CSV_COLUMNS + [c for c in OPTIONAL_CSV_DEFAULTS if c != "rm_id"]
    cols = list(dict.fromkeys(cols))  # dedupe, keep order
    template_df = pd.DataFrame([{c: example.get(c, "") for c in cols}])
    return template_df.to_csv(index=False).encode("utf-8")


def run_ml_screening(user_fields: dict, application_id: str) -> dict:
    """Bagian CEPAT (ML only) - system-fill field non-user, build fitur dari
    raw tables, jalankan predict_credit_screening(). TIDAK memanggil LLM,
    jadi risk_score/decision bisa langsung dilihat tanpa menunggu narasi.
    Narasi "Alasan" diisi lewat run_llm_narrative() secara terpisah, dipicu
    tombol lain."""
    nik = str(user_fields["NIK"]).strip()
    existing_cif = profile_full.loc[profile_full["NIK"] == nik, "cif_number"]
    cif_number = existing_cif.iloc[0] if len(existing_cif) else f"CIF-SIM-{nik[-6:] if len(nik) >= 6 else nik}"

    loan_requested = float(user_fields["loan_requested"])
    collateral_market_value = float(user_fields["collateral_market_value"])
    collateral_ratio = round(collateral_market_value / loan_requested, 4) if loan_requested else 0.0

    branch_name = user_fields.get("branch_name")
    rm_id = user_fields.get("rm_id")
    if not rm_id:
        default_rm = rm_master.loc[rm_master["branch_name"] == branch_name, "rm_id"]
        rm_id = default_rm.iloc[0] if len(default_rm) else None

    new_row = {
        "application_id": application_id, "NIK": nik, "cif_number": cif_number,
        "application_date": datetime.now().strftime("%Y-%m-%d"), "customer_type": "UMKM",
        "collateral_ratio": collateral_ratio, "rm_id": rm_id,
        "eligibility_score": np.nan, "label": np.nan,
    }
    for col in profile_full.columns:
        if col not in new_row:
            new_row[col] = user_fields.get(col)
    new_profile_row = pd.DataFrame([new_row])[profile_full.columns.tolist()]

    features = build_features_from_raw([application_id], new_profile_row, slik_full, dhn_full, bank_full, fin_full)
    result = predict_credit_screening(features.iloc[0].to_dict())

    company_name = user_fields.get("company_name", "")
    return {
        "application_id": application_id,
        "company_name": company_name,
        "NIK": nik,
        "Decision (Eligibility Recommendation)": result["decision"],
        "Risk Score / Eligibility Score": result["risk_score"],
        "Zone": result["zone"],
        "Jenis Kredit": result["jenis_kredit_rekomendasi"],
        "Nominal Disetujui": result["nominal_disetujui"],
        "Jangka Waktu (bulan)": result["jangka_waktu_bulan"],
        "Bunga (% p.a.)": result["bunga_persen"],
        "Alasan": None,
        "_fallback_reason": None,
        "_insight": result["insight"],
        "_shap": result["shap_top_factors"],
        "_is_existing_nik": nik in KNOWN_NIKS,
        "_result": result,
    }


def run_llm_narrative(row: dict) -> dict:
    """Bagian LAMBAT (LLM) - isi kolom "Alasan" dari row hasil
    run_ml_screening() (butuh key "_result" mentah dari predict_credit_screening).
    Dipanggil terpisah lewat tombol sendiri supaya risk_score bisa dilihat
    duluan tanpa menunggu ini."""
    row = dict(row)
    row["Alasan"] = generate_report({"company_name": row["company_name"], **row["_result"]})
    row["_fallback_reason"] = get_last_fallback_reason()
    return row


def _prefill_form_from_row(row: dict):
    """Set st.session_state utk tiap widget form manual, lalu rerun supaya
    Tab Input Manual langsung menampilkan nilai dari CSV."""
    row = _apply_defaults(row)
    for field in FORM_FIELDS_ORDER:
        st.session_state[f"f_{field}"] = row.get(field)
    st.session_state["_prefill_notice"] = row.get("company_name", "")


tab_manual, tab_csv = st.tabs(["📝 Input Manual (1 Nasabah)", "📤 Upload CSV (1 atau Banyak Nasabah)"])

# ---------------------------------------------------------------------------
# TAB: Upload CSV
# ---------------------------------------------------------------------------
with tab_csv:
    st.subheader("Upload CSV Pengajuan")
    st.caption(
        "Kolom mengikuti skema `retail_customer_profile.csv`. Kolom sistem "
        "(application_id, cif_number, eligibility_score, label, dst.) tidak perlu "
        "disertakan — kalau ada di file, akan diabaikan dan dibuat ulang oleh sistem."
    )

    dl_col, up_col = st.columns([1, 2])
    with dl_col:
        st.download_button(
            "⬇️ Download Template CSV", data=_build_csv_template(),
            file_name="template_pengajuan_kredit.csv", mime="text/csv", use_container_width=True,
        )
        st.caption(f"Kolom wajib diisi: {len(REQUIRED_CSV_COLUMNS)}. Kolom lain otomatis diberi nilai default kalau kosong.")

    uploaded = st.file_uploader("Pilih file CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            upload_df = pd.read_csv(uploaded, dtype={"NIK": str})
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            upload_df = None

        if upload_df is not None:
            missing = _validate_csv_columns(upload_df)
            if missing:
                st.error(f"Kolom wajib berikut tidak ditemukan di CSV: {', '.join(missing)}")
            else:
                st.success(f"CSV terbaca: {len(upload_df)} baris pengajuan.")
                st.dataframe(upload_df, use_container_width=True, height=min(300, 60 + 35 * len(upload_df)))

                if len(upload_df) == 1:
                    st.info("CSV berisi 1 pengajuan — form di tab **Input Manual** akan diisi otomatis untuk direview sebelum di-submit.")
                    if st.button("➡️ Isi ke Form Manual", type="primary"):
                        _prefill_form_from_row(upload_df.iloc[0].to_dict())
                        st.rerun()
                else:
                    st.info(f"CSV berisi {len(upload_df)} pengajuan — akan diproses langsung sebagai batch (tanpa form individual).")
                    if st.button(f"🔍 Jalankan Screening ML untuk {len(upload_df)} Nasabah", type="primary"):
                        progress = st.progress(0.0, text="Memproses...")
                        batch_results = []
                        for i, (_, row) in enumerate(upload_df.iterrows()):
                            row_filled = _apply_defaults(row.to_dict())
                            app_id = f"SIM{datetime.now().strftime('%Y%m%d%H%M%S')}{i:03d}"
                            try:
                                batch_results.append(run_ml_screening(row_filled, app_id))
                            except Exception as e:
                                batch_results.append({
                                    "application_id": app_id, "company_name": row_filled.get("company_name", ""),
                                    "NIK": row_filled.get("NIK"), "Decision (Eligibility Recommendation)": "ERROR",
                                    "Risk Score / Eligibility Score": None, "Zone": "-", "Jenis Kredit": "-",
                                    "Nominal Disetujui": None, "Jangka Waktu (bulan)": None, "Bunga (% p.a.)": None,
                                    "Alasan": str(e), "_fallback_reason": None, "_insight": "", "_shap": [], "_is_existing_nik": False,
                                    "_result": None,
                                })
                            progress.progress((i + 1) / len(upload_df), text=f"Memproses {i + 1}/{len(upload_df)}...")
                        progress.empty()
                        st.session_state["_batch_results"] = batch_results

    if "_batch_results" in st.session_state:
        st.divider()
        st.subheader("Hasil Screening Batch (ML)")
        results_df = pd.DataFrame(st.session_state["_batch_results"])
        display_df = results_df.drop(columns=["_insight", "_shap", "_is_existing_nik", "_fallback_reason", "_result"])

        n_error = (display_df["Decision (Eligibility Recommendation)"] == "ERROR").sum()
        n_layak = display_df["Decision (Eligibility Recommendation)"].isin(["Layak", "Layak Bersyarat"]).sum()
        belum_narasi_idx = [i for i, r in enumerate(st.session_state["_batch_results"]) if r.get("_result") is not None and not r.get("Alasan")]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Diproses", len(display_df))
        m2.metric("Layak / Layak Bersyarat", int(n_layak))
        m3.metric("Gagal Diproses", int(n_error))
        m4.metric("Belum Ada Narasi LLM", len(belum_narasi_idx))

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download Hasil (CSV)", data=display_df.to_csv(index=False).encode("utf-8"),
            file_name=f"hasil_screening_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        if belum_narasi_idx:
            if st.button(f"🤖 Generate Narasi (LLM) untuk {len(belum_narasi_idx)} Nasabah"):
                progress = st.progress(0.0, text="Menyusun narasi...")
                updated = list(st.session_state["_batch_results"])
                for done, i in enumerate(belum_narasi_idx):
                    updated[i] = run_llm_narrative(updated[i])
                    progress.progress((done + 1) / len(belum_narasi_idx), text=f"Menyusun narasi {done + 1}/{len(belum_narasi_idx)}...")
                progress.empty()
                st.session_state["_batch_results"] = updated
                st.rerun()

        n_llm_fallback = sum(1 for r in st.session_state["_batch_results"] if r.get("_fallback_reason"))
        if n_llm_fallback:
            st.caption(f"{n_llm_fallback} baris fallback ke narasi rule-based (LLM gagal) — lihat detail di bawah.")

        with st.expander("Detail insight & narasi per nasabah"):
            for r in st.session_state["_batch_results"]:
                st.markdown(f"**{r['application_id']} — {r['company_name']}**: {r.get('Alasan') or r['_insight']}")
                if r.get("_fallback_reason"):
                    st.caption(f"⚠️ Fallback: {r['_fallback_reason']}")

        if st.button("🗑️ Bersihkan hasil batch"):
            del st.session_state["_batch_results"]
            st.rerun()

# ---------------------------------------------------------------------------
# TAB: Input Manual
# ---------------------------------------------------------------------------
with tab_manual:
    if st.session_state.get("_prefill_notice"):
        st.success(f"Form terisi otomatis dari CSV untuk **{st.session_state['_prefill_notice']}** — silakan cek/ubah lalu jalankan screening.")

    def sv(field, default):
        return st.session_state.get(f"f_{field}", default)

    def sb_index(options, field, default_val):
        val = sv(field, default_val)
        return options.index(val) if val in options else 0

    st.subheader("Industri")
    industry = st.selectbox("Industri", INDUSTRY_OPTIONS, index=sb_index(INDUSTRY_OPTIONS, "industry", INDUSTRY_OPTIONS[0]), key="f_industry")
    sub_industry_options = INDUSTRY_SUBINDUSTRY.get(industry, [])

    with st.form("manual_form"):
        st.subheader("Identitas & Profil Usaha")
        c1, c2, c3 = st.columns(3)
        with c1:
            nik = st.text_input("NIK (16 digit)", value=sv("NIK", ""), max_chars=16, placeholder="3276010601750001")
            owner_name = st.text_input("Nama Pemilik", value=sv("owner_name", ""))
        with c2:
            owner_age = st.number_input("Usia Pemilik", min_value=17, max_value=90, value=int(sv("owner_age", 35)))
            owner_gender = st.selectbox("Jenis Kelamin", GENDER_OPTIONS, index=sb_index(GENDER_OPTIONS, "owner_gender", "L"))
            st.caption("Tidak dipakai sebagai fitur skor model (fair-lending) — hanya data identitas.")
        with c3:
            owner_marital_status = st.selectbox("Status Perkawinan", MARITAL_OPTIONS, index=sb_index(MARITAL_OPTIONS, "owner_marital_status", MARITAL_OPTIONS[0]))
            owner_education = st.selectbox("Pendidikan Terakhir", EDUCATION_OPTIONS, index=sb_index(EDUCATION_OPTIONS, "owner_education", EDUCATION_OPTIONS[0]))

        c4, c5, c6 = st.columns(3)
        with c4:
            company_name = st.text_input("Nama Usaha", value=sv("company_name", ""))
            legal_entity = st.selectbox("Badan Usaha", LEGAL_ENTITY_OPTIONS, index=sb_index(LEGAL_ENTITY_OPTIONS, "legal_entity", LEGAL_ENTITY_OPTIONS[0]))
        with c5:
            sub_industry_default = sv("sub_industry", sub_industry_options[0] if sub_industry_options else "")
            sub_industry_idx = sub_industry_options.index(sub_industry_default) if sub_industry_default in sub_industry_options else 0
            sub_industry = st.selectbox("Sub-Industri", sub_industry_options, index=sub_industry_idx)
            business_age_year = st.number_input("Lama Usaha (tahun)", min_value=0, max_value=80, value=int(sv("business_age_year", 5)))
        with c6:
            employee_count = st.number_input("Jumlah Karyawan", min_value=0, max_value=1000, value=int(sv("employee_count", 10)))
            branch_name = st.selectbox("Cabang Pengaju", BRANCH_OPTIONS, index=sb_index(BRANCH_OPTIONS, "branch_name", BRANCH_OPTIONS[0]))

        c7, c8, c9 = st.columns(3)
        with c7:
            province = st.text_input("Provinsi (lokasi usaha)", value=sv("province", ""))
        with c8:
            city = st.text_input("Kota/Kabupaten (lokasi usaha)", value=sv("city", ""))
        with c9:
            district = st.text_input("Kecamatan (lokasi usaha)", value=sv("district", ""))
        region = st.selectbox("Region", REGION_OPTIONS, index=sb_index(REGION_OPTIONS, "region", REGION_OPTIONS[0]))

        st.subheader("Keuangan & Pinjaman")
        c10, c11, c12 = st.columns(3)
        with c10:
            monthly_turnover_est = st.number_input("Estimasi Omset Bulanan (Rp)", min_value=1_000_000, value=int(sv("monthly_turnover_est", 50_000_000)), step=1_000_000)
        with c11:
            transaction_frequency_monthly = st.number_input("Frekuensi Transaksi/Bulan", min_value=0, max_value=2000, value=int(sv("transaction_frequency_monthly", 80)))
        with c12:
            loan_requested = st.number_input("Nominal Pinjaman Diajukan (Rp)", min_value=1_000_000, value=int(sv("loan_requested", 200_000_000)), step=10_000_000)
        estimated_dsr = st.number_input("Estimasi DSR (Debt Service Ratio)", min_value=0.0, max_value=3.0, value=float(sv("estimated_dsr", 1.0)), step=0.1,
                                         help="Dibatasi maks 3.0 (dsr_capped) sesuai skema data training.")

        st.subheader("Agunan")
        c13, c14, c15 = st.columns(3)
        with c13:
            collateral_type = st.selectbox("Jenis Agunan", COLLATERAL_TYPE_OPTIONS, index=sb_index(COLLATERAL_TYPE_OPTIONS, "collateral_type", COLLATERAL_TYPE_OPTIONS[0]))
            certificate_type = st.selectbox("Jenis Sertifikat", CERTIFICATE_TYPE_OPTIONS, index=sb_index(CERTIFICATE_TYPE_OPTIONS, "certificate_type", CERTIFICATE_TYPE_OPTIONS[0]))
        with c14:
            collateral_market_value = st.number_input("Nilai Pasar Agunan (Rp)", min_value=0, value=int(sv("collateral_market_value", 500_000_000)), step=10_000_000)
            collateral_liquidation_value = st.number_input("Nilai Likuidasi Agunan (Rp)", min_value=0, value=int(sv("collateral_liquidation_value", 400_000_000)), step=10_000_000,
                                                             help="Umumnya sekitar 80% dari nilai pasar.")
        with c15:
            collateral_size_m2 = st.number_input("Luas Agunan (m²)", min_value=0.0, value=float(sv("collateral_size_m2", 100.0)), step=5.0)
            ownership_match = st.selectbox("Kepemilikan Sesuai Sertifikat?", YA_TIDAK_OPTIONS, index=sb_index(YA_TIDAK_OPTIONS, "ownership_match", "Ya"))

        c16, c17, c18 = st.columns(3)
        with c16:
            collateral_location = st.text_input("Lokasi Agunan", value=sv("collateral_location", ""))
        with c17:
            collateral_province = st.text_input("Provinsi Agunan", value=sv("collateral_province", ""))
        with c18:
            collateral_city = st.text_input("Kota Agunan", value=sv("collateral_city", ""))

        submitted = st.form_submit_button("🔍 Jalankan Screening (ML)", type="primary", use_container_width=True)

    if submitted:
        nik_clean = nik.strip()
        if len(nik_clean) != 16 or not nik_clean.isdigit():
            st.warning("NIK bukan 16 digit angka — akan otomatis ditolak oleh hard-rule identitas (Stage 1), sesuai desain sistem.")
        elif nik_clean not in DUKCAPIL_NIKS:
            st.error(f"NIK `{nik}` TIDAK terdaftar di Dukcapil — akan otomatis ditolak (hard-rule identitas, Stage 1), model ML tidak akan dipanggil.")
        elif _normalize_name(owner_name) != _normalize_name(_dukcapil_names().get(nik_clean)):
            st.error(f"Nama `{owner_name}` TIDAK sesuai data Dukcapil untuk NIK `{nik}`.")
        elif nik_clean in KNOWN_NIKS:
            st.info(f"NIK `{nik}` terverifikasi Dukcapil dan sudah dikenal sistem — riwayat SLIK/DHN/rekening/keuangan asli akan dipakai otomatis.")
        else:
            st.info(f"NIK `{nik}` terverifikasi Dukcapil, tapi belum ada riwayat kredit — diproses sebagai nasabah baru dengan data netral (bukan hard-reject).")

        user_fields = {
            "NIK": nik.strip(), "owner_name": owner_name, "owner_age": owner_age, "owner_gender": owner_gender,
            "owner_marital_status": owner_marital_status, "owner_education": owner_education,
            "company_name": company_name, "legal_entity": legal_entity, "industry": industry, "sub_industry": sub_industry,
            "business_age_year": business_age_year, "employee_count": employee_count, "branch_name": branch_name,
            "province": province, "city": city, "district": district, "region": region,
            "monthly_turnover_est": monthly_turnover_est, "transaction_frequency_monthly": transaction_frequency_monthly,
            "loan_requested": loan_requested, "estimated_dsr": estimated_dsr,
            "collateral_type": collateral_type, "certificate_type": certificate_type,
            "collateral_market_value": collateral_market_value, "collateral_liquidation_value": collateral_liquidation_value,
            "collateral_size_m2": collateral_size_m2, "ownership_match": ownership_match,
            "collateral_location": collateral_location, "collateral_province": collateral_province, "collateral_city": collateral_city,
        }
        application_id = f"SIM{datetime.now().strftime('%Y%m%d%H%M%S')}"
        with st.spinner("Menjalankan model screening (ML)..."):
            st.session_state["_manual_result"] = run_ml_screening(user_fields, application_id)

    if st.session_state.get("_manual_result"):
        result_row = st.session_state["_manual_result"]

        st.divider()
        st.subheader("Hasil Screening (ML) — risk_score, decision, dst.")
        display_row = {k: v for k, v in result_row.items() if not k.startswith("_") and k not in ("NIK", "Alasan")}
        st.dataframe(pd.DataFrame([display_row]), hide_index=True, use_container_width=True)

        with st.expander("Insight teknis dari model (rule-based, tersedia instan)"):
            st.write(result_row["_insight"])
            if result_row["_shap"]:
                st.caption("Faktor paling berpengaruh terhadap skor (SHAP):")
                st.dataframe(pd.DataFrame(result_row["_shap"]), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Narasi (LLM)")
        if result_row.get("Alasan"):
            st.write(result_row["Alasan"])
            if result_row.get("_fallback_reason"):
                st.warning(
                    f"⚠️ Narasi di atas adalah **fallback** ke insight rule-based — narasi LLM (Gemma) "
                    f"gagal dihasilkan. Alasan teknis: `{result_row['_fallback_reason']}`"
                )
        else:
            st.caption("Belum digenerate — klik tombol di bawah kalau perlu narasi natural untuk laporan (opsional, risk_score di atas sudah final).")
            if st.button("🤖 Generate Narasi (LLM)"):
                with st.spinner("Menyusun narasi dengan LLM..."):
                    st.session_state["_manual_result"] = run_llm_narrative(result_row)
                st.rerun()
