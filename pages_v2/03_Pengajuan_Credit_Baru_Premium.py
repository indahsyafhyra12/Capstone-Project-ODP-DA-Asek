"""Screening Pengajuan Credit Baru (Premium UI) — predict screening lewat pipeline hybrid ML.

Fungsionalitas identik dengan pages/3_Pengajuan_Credit_Baru.py, dibungkus
tampilan premium (hero banner, section-title styling) utk dipakai lewat
app_premium.py + pages_v2. Kalau ada perubahan fungsional, terapkan di kedua
file supaya tidak drift.

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

Screening ML (risk_score/decision/dst.) dan narasi LLM sengaja dipisah jadi
2 tombol - ML instan, narasi LLM opsional & dipicu terpisah supaya loan
officer tidak perlu menunggu LLM cuma utk lihat risk_score. Field turunan
risk_score (decision/zone/bunga/nominal/tenor/jenis kredit) bisa dioverride
manual lewat apply_policy_engine() (utils/risk_ml_pipeline.py) - kalau cuma
Credit Eligibility Score yang diedit, field turunannya ikut di-cascade
otomatis pakai policy engine yang sama dengan AI.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src import genai
from src.orchestrator import run_screening
from utils.agent_pipeline import _dukcapil_names, _normalize_name
from utils.feature_builder import build_features_from_raw, load_raw_tables
from utils.kwitansi_extractor import build_raw_text_export, compute_monthly_estimates, extract_zip_bytes
from utils.risk_ml_pipeline import apply_policy_engine
from utils.ui_components import apply_logo
from utils.ui_premium import inject_css, hero_banner

st.set_page_config(page_title="Pengajuan Kredit Baru", page_icon="📝", layout="wide")
apply_logo()
inject_css()

st.markdown("""
<style>
html,body,[class*="css"]{font-family:Inter,sans-serif;}
.section-title{font-size:24px;font-weight:700;color:#1F2937;margin:26px 0 12px;}
[data-testid="stMetricLabel"]{font-size:13px;color:#64748B;}
[data-testid="stMetricValue"]{font-size:22px;font-weight:600;color:#1F2937;}
div[data-testid="stNumberInput"]:has(input[aria-label="Nominal Pinjaman Diajukan (Rp)"]) label p{
    font-size:20px;font-weight:700;color:#1F2937;
}
div[data-testid="stNumberInput"]:has(input[aria-label="Nominal Pinjaman Diajukan (Rp)"]) input{
    font-size:30px;font-weight:700;color:#B91C1C;height:56px;
}
</style>
""", unsafe_allow_html=True)

hero_banner(
    "Pengajuan Kredit Baru",
    "Credit Eligibility AI • Screening UMKM menggunakan ML Pipeline (risk_ml_pipeline)."
)

c1, c2 = st.columns([3, 1])
with c2:
    st.success("🤖 ML Pipeline Active")


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

# Opsi utk override manual hasil screening AI (lihat run_ml_screening() /
# utils/risk_ml_pipeline.py) - dipakai kalau RM tidak setuju dgn rekomendasi
# model dan mau menyesuaikan sendiri sebelum hasil dianggap final.
DECISION_OPTIONS = ["Layak", "Layak Bersyarat", "Perlu Review Ulang", "Tidak Layak"]
ZONE_OPTIONS = ["Hijau", "Kuning", "Merah"]
JENIS_KREDIT_OPTIONS = ["KMK", "KI", "KUR", "KPR", "KKB", "KK", "-"]
# Jenis yang bisa dipilih nasabah SAAT MENGAJUKAN (beda dari JENIS_KREDIT_OPTIONS
# di atas, yang juga menampung hasil rekomendasi/override) - dibatasi ke 3 jenis
# yang divalidasi recommend_credit_type() (utils/agent_pipeline.py).
JENIS_KREDIT_DIAJUKAN_OPTIONS = ["KMK", "KI", "KUR"]
MANUAL_EDIT_FIELDS = [
    "Decision (Eligibility Recommendation)", "Credit Eligibility Score", "Zone",
    "Jenis Kredit", "Nominal Disetujui", "Jangka Waktu (bulan)", "Bunga (% p.a.)",
]

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
    "jenis_kredit_diajukan": None, "tenor_diajukan_bulan": 0, "tujuan_penggunaan_kredit": "",
}
FORM_FIELDS_ORDER = [
    "NIK", "owner_name", "owner_age", "owner_gender", "owner_marital_status", "owner_education",
    "company_name", "legal_entity", "industry", "sub_industry", "business_age_year", "employee_count",
    "branch_name", "province", "city", "district", "region",
    "monthly_turnover_est", "transaction_frequency_monthly", "loan_requested", "estimated_dsr",
    "jenis_kredit_diajukan", "tenor_diajukan_bulan", "tujuan_penggunaan_kredit",
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
        "jenis_kredit_diajukan": "KMK", "tenor_diajukan_bulan": 24,
        "tujuan_penggunaan_kredit": "Tambahan modal kerja operasional usaha",
    }
    cols = REQUIRED_CSV_COLUMNS + [c for c in OPTIONAL_CSV_DEFAULTS if c != "rm_id"]
    cols = list(dict.fromkeys(cols))  # dedupe, keep order
    template_df = pd.DataFrame([{c: example.get(c, "") for c in cols}])
    return template_df.to_csv(index=False).encode("utf-8")


def run_ml_screening(user_fields: dict, application_id: str) -> dict:
    """Bagian CEPAT (Planner + ML, tanpa LLM) - system-fill field non-user,
    build fitur dari raw tables, jalankan src.orchestrator.run_screening()
    dengan explain_with_gemma=False. Itu menjalankan Adaptive Verification
    Planner (src/agents/planner_agent.py) lalu ML risk_score + Policy Engine
    yang sama seperti sebelumnya (utils.risk_ml_pipeline.predict_credit_screening()),
    tapi TIDAK memanggil Gemma, jadi risk_score/decision bisa langsung dilihat
    tanpa menunggu narasi. PlannerTrace-nya disimpan (kunci "_planner_trace")
    supaya run_llm_narrative() bisa memakainya utk Planner Summary tanpa
    menjalankan ulang planner-nya. Narasi "Alasan" diisi lewat
    run_llm_narrative() secara terpisah, dipicu tombol lain."""
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
    screening = run_screening(features.iloc[0].to_dict(), explain_with_gemma=False)
    result = screening.to_dict()
    planner_trace = result.pop("planner_trace")
    result.pop("gemma_explanation", None)

    company_name = user_fields.get("company_name", "")
    return {
        "application_id": application_id,
        "company_name": company_name,
        "NIK": nik,
        "Decision (Eligibility Recommendation)": result["decision"],
        "Credit Eligibility Score": result["risk_score"],
        "Zone": result["zone"],
        "Jenis Kredit": result["jenis_kredit_rekomendasi"],
        "Nominal Disetujui": result["nominal_disetujui"],
        "Jangka Waktu (bulan)": result["jangka_waktu_bulan"],
        "Bunga (% p.a.)": result["bunga_persen"],
        "Alasan": None,
        "_planner_summary": None,
        "_fallback_reason": None,
        "_insight": result["insight"],
        "_shap": result["shap_top_factors"],
        "_is_existing_nik": nik in KNOWN_NIKS,
        "_result": result,
        "_planner_trace": planner_trace,
        "_manual_override": False,
        "_loan_requested": loan_requested,
        "_collateral_market_value": collateral_market_value,
        "_jenis_kredit_diajukan": user_fields.get("jenis_kredit_diajukan"),
        "_tenor_diajukan_bulan": user_fields.get("tenor_diajukan_bulan"),
        "_jenis_kredit_sesuai": result["jenis_kredit_sesuai"],
        "_dsr_pada_pengajuan": result["dsr_pada_pengajuan"],
        "_catatan_kesesuaian_kredit": result["catatan_kesesuaian_kredit"],
    }


def run_llm_narrative(row: dict) -> dict:
    """Bagian LAMBAT (LLM) - isi kolom "Alasan" + "_planner_summary" dari row
    hasil run_ml_screening() lewat src.genai.explain(), yaitu Gemma Explanation
    Layer yang menghasilkan DUA narasi dari trace/hasil yang SAMA dengan yang
    sudah dilihat di layar (PlannerTrace dari "_planner_trace" + ml_result dari
    "_result"): Planner Summary (proses verifikasi) dan Final Decision
    Narrative (hasil akhir, via utils.report_agent.generate_report() seperti
    sebelumnya). Dipanggil terpisah lewat tombol sendiri supaya risk_score
    bisa dilihat duluan tanpa menunggu LLM."""
    row = dict(row)
    explanation = genai.explain(row["_planner_trace"], row["_result"], {"company_name": row["company_name"]})
    row["Alasan"] = explanation.final_decision_narrative
    row["_planner_summary"] = explanation.planner_summary
    row["_fallback_reason"] = explanation.fallback_reason
    return row


DERIVED_FROM_RISK_SCORE = [
    "Decision (Eligibility Recommendation)", "Zone", "Jenis Kredit",
    "Nominal Disetujui", "Jangka Waktu (bulan)", "Bunga (% p.a.)",
]
_POLICY_FIELD_MAP = {
    "Decision (Eligibility Recommendation)": "decision", "Zone": "zone",
    "Jenis Kredit": "jenis_kredit_rekomendasi", "Nominal Disetujui": "nominal_disetujui",
    "Jangka Waktu (bulan)": "jangka_waktu_bulan", "Bunga (% p.a.)": "bunga_persen",
}


def _cascade_edit(new_vals: dict, orig: dict) -> dict:
    """Kalau Credit Eligibility Score diedit tapi field turunannya
    (decision/zone/jenis kredit/nominal/tenor/bunga) TIDAK ikut disentuh
    user di edit yang sama, hitung ulang otomatis pakai apply_policy_engine()
    - policy engine yang sama dengan yang dipakai predict_credit_screening()
    - supaya field2 itu tidak nyangkut di nilai lama yang sudah tidak
    konsisten dengan skor baru. Field yang memang sengaja diedit manual oleh
    user tetap dihormati, tidak ditimpa cascade."""
    new_vals = dict(new_vals)
    score_changed = new_vals["Credit Eligibility Score"] != orig.get("Credit Eligibility Score")
    if not score_changed:
        return new_vals
    untouched = [c for c in DERIVED_FROM_RISK_SCORE if new_vals[c] == orig.get(c)]
    if not untouched:
        return new_vals
    cascaded = apply_policy_engine(
        new_vals["Credit Eligibility Score"],
        orig.get("_loan_requested", 0), orig.get("_collateral_market_value", 0),
    )
    for c in untouched:
        new_vals[c] = cascaded[_POLICY_FIELD_MAP[c]]
    return new_vals


def _prefill_form_from_row(row: dict):
    """Set st.session_state utk tiap widget form manual, lalu rerun supaya
    Tab Input Manual langsung menampilkan nilai dari CSV."""
    row = _apply_defaults(row)
    for field in FORM_FIELDS_ORDER:
        st.session_state[f"f_{field}"] = row.get(field)
    st.session_state["_prefill_notice"] = row.get("company_name", "")


if st.session_state.get("_open_manual_after_prefill"):
    tab_csv, tab_manual = st.tabs(["📂 Bulk Screening CSV", "✍️ Input Manual"])
else:
    tab_manual, tab_csv = st.tabs(["✍️ Input Manual", "📂 Bulk Screening CSV"])

# ---------------------------------------------------------------------------
# TAB: Upload CSV
# ---------------------------------------------------------------------------
with tab_csv:
    st.markdown('<div class="section-title">📂 Bulk Screening CSV</div>', unsafe_allow_html=True)
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
                        st.session_state["_open_manual_after_prefill"] = True
                        st.rerun()
                else:
                    st.info(f"CSV berisi {len(upload_df)} pengajuan — akan diproses langsung sebagai batch (tanpa form individual).")
                    if st.button(f"🤖 Jalankan AI Screening (ML) untuk {len(upload_df)} Nasabah", type="primary"):
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
                                    "Credit Eligibility Score": None, "Zone": "-", "Jenis Kredit": "-",
                                    "Nominal Disetujui": None, "Jangka Waktu (bulan)": None, "Bunga (% p.a.)": None,
                                    "Alasan": str(e), "_fallback_reason": None, "_insight": "", "_shap": [], "_is_existing_nik": False,
                                    "_result": None, "_manual_override": False,
                                })
                            progress.progress((i + 1) / len(upload_df), text=f"Memproses {i + 1}/{len(upload_df)}...")
                        progress.empty()
                        st.session_state["_batch_results"] = batch_results
                        st.session_state["_batch_edit_version"] = 0

    if "_batch_results" in st.session_state:
        st.divider()
        st.markdown('<div class="section-title">📈 Hasil Batch Screening (ML)</div>', unsafe_allow_html=True)
        results_df = pd.DataFrame(st.session_state["_batch_results"])
        internal_cols = [c for c in results_df.columns if c.startswith("_")]
        display_df = results_df.drop(columns=internal_cols)

        n_error = (display_df["Decision (Eligibility Recommendation)"] == "ERROR").sum()
        n_layak = display_df["Decision (Eligibility Recommendation)"].isin(["Layak", "Layak Bersyarat"]).sum()
        belum_narasi_idx = [i for i, r in enumerate(st.session_state["_batch_results"]) if r.get("_result") is not None and not r.get("Alasan")]
        n_override = sum(1 for r in st.session_state["_batch_results"] if r.get("_manual_override"))
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Diproses", len(display_df))
        m2.metric("Layak / Layak Bersyarat", int(n_layak))
        m3.metric("Gagal Diproses", int(n_error))
        m4.metric("Belum Ada Narasi LLM", len(belum_narasi_idx))
        m5.metric("Diedit Manual", int(n_override))

        st.caption(
            "Kolom Decision/Credit Eligibility Score/Zone/Jenis Kredit/Nominal/Tenor/Bunga bisa diedit "
            "langsung di tabel kalau Anda tidak setuju dengan rekomendasi AI — baris yang diubah otomatis "
            "ditandai \"Ya\" di kolom Diedit Manual. Klik \"💾 Simpan Perubahan Manual\" untuk menyimpan."
        )
        editable_df = display_df.copy()
        editable_df["Diedit Manual"] = [
            "Ya" if r.get("_manual_override") else "Tidak" for r in st.session_state["_batch_results"]
        ]
        editor_key = f"_batch_editor_{st.session_state.get('_batch_edit_version', 0)}"
        edited_df = st.data_editor(
            editable_df, use_container_width=True, hide_index=True, key=editor_key,
            disabled=[c for c in editable_df.columns if c not in MANUAL_EDIT_FIELDS],
            column_config={
                "Decision (Eligibility Recommendation)": st.column_config.SelectboxColumn(options=DECISION_OPTIONS),
                "Zone": st.column_config.SelectboxColumn(options=ZONE_OPTIONS),
                "Jenis Kredit": st.column_config.SelectboxColumn(options=JENIS_KREDIT_OPTIONS),
                "Credit Eligibility Score": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.3f"),
                "Nominal Disetujui": st.column_config.NumberColumn(min_value=0, step=1_000_000),
                "Jangka Waktu (bulan)": st.column_config.NumberColumn(min_value=0, step=1),
                "Bunga (% p.a.)": st.column_config.NumberColumn(min_value=0.0, step=0.1, format="%.1f"),
            },
        )

        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            if st.button("💾 Simpan Perubahan Manual", type="primary", use_container_width=True, key="save_batch_manual_changes"):
                updated = list(st.session_state["_batch_results"])
                for i, row in edited_df.iterrows():
                    orig = updated[i]
                    if orig.get("_result") is None:
                        continue  # baris ERROR - tidak punya hasil AI, tidak bisa dioverride
                    proposed = {c: row[c] for c in MANUAL_EDIT_FIELDS}
                    changed = any(proposed[c] != orig.get(c) for c in MANUAL_EDIT_FIELDS)
                    if changed:
                        if not orig.get("_manual_override"):
                            orig["_ai_original"] = {c: orig.get(c) for c in MANUAL_EDIT_FIELDS}
                        proposed = _cascade_edit(proposed, orig)
                        for c in MANUAL_EDIT_FIELDS:
                            orig[c] = proposed[c]
                        orig["_manual_override"] = True
                    updated[i] = orig
                st.session_state["_batch_results"] = updated
                st.session_state["_batch_edit_version"] = st.session_state.get("_batch_edit_version", 0) + 1
                st.success("Perubahan manual disimpan.")
                st.rerun()
        with ecol2:
            if n_override and st.button("↩️ Kembalikan Semua ke Rekomendasi AI", use_container_width=True):
                updated = list(st.session_state["_batch_results"])
                for i, r in enumerate(updated):
                    if r.get("_manual_override"):
                        r.update(r["_ai_original"])
                        r["_manual_override"] = False
                        del r["_ai_original"]
                    updated[i] = r
                st.session_state["_batch_results"] = updated
                st.session_state["_batch_edit_version"] = st.session_state.get("_batch_edit_version", 0) + 1
                st.rerun()

        st.download_button(
            "⬇️ Download Hasil (CSV)", data=editable_df.to_csv(index=False).encode("utf-8"),
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

        with st.expander("AI Insight per Nasabah"):
            for r in st.session_state["_batch_results"]:
                st.markdown(f"**{r['application_id']} — {r['company_name']}**: {r.get('Alasan') or r['_insight']}")
                if r.get("_planner_summary"):
                    st.caption(f"🧭 Proses verifikasi (Planner): {r['_planner_summary']}")
                if r.get("_fallback_reason"):
                    st.caption(f"⚠️ Fallback: {r['_fallback_reason']}")

        if st.button("🗑️ Bersihkan hasil batch"):
            del st.session_state["_batch_results"]
            st.session_state.pop("_batch_edit_version", None)
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

    st.markdown('<div class="section-title">🏭 Industri</div>', unsafe_allow_html=True)
    industry = st.selectbox("Industri", INDUSTRY_OPTIONS, index=sb_index(INDUSTRY_OPTIONS, "industry", INDUSTRY_OPTIONS[0]), key="f_industry")
    sub_industry_options = INDUSTRY_SUBINDUSTRY.get(industry, [])

    st.markdown('<div class="section-title">💰 Profil Finansial</div>', unsafe_allow_html=True)
    fin_mode = st.radio(
        "Mode Input Profil Finansial", ["Isi Manual", "Upload Kwitansi (ZIP)"],
        horizontal=True, key="_fin_input_mode", label_visibility="collapsed",
    )

    if fin_mode == "Upload Kwitansi (ZIP)":
        st.caption(
            "Upload 1 file **.zip** berisi foto kwitansi penjualan & pembelian usaha yang sedang "
            "diajukan (boleh campur jenisnya, untuk satu nasabah). Hasil ekstraksi bisa dicek/"
            "dikoreksi dulu di tabel sebelum dipakai mengisi Estimasi Omset Bulanan & Frekuensi "
            "Transaksi/Bulan di bawah — Nominal Pinjaman dan Estimasi DSR tetap diisi manual."
        )
        kwitansi_zip = st.file_uploader("File Kwitansi (.zip)", type=["zip"], key="_kwitansi_zip")

        if kwitansi_zip is not None and st.session_state.get("_kwitansi_zip_name") != kwitansi_zip.name:
            st.session_state["_kwitansi_zip_name"] = kwitansi_zip.name
            st.session_state.pop("_kwitansi_extracted", None)
            st.session_state.pop("_kwitansi_confirmed", None)
            with st.spinner(
                "Membaca kwitansi dengan model OCR (LightOnOCR-2-1B) — pemuatan model pertama kali "
                "bisa memakan waktu beberapa menit, proses berikutnya lebih cepat..."
            ):
                try:
                    st.session_state["_kwitansi_extracted"] = extract_zip_bytes(kwitansi_zip.getvalue())
                except Exception as e:
                    st.session_state["_kwitansi_error"] = str(e)
            if "_kwitansi_extracted" in st.session_state:
                st.session_state.pop("_kwitansi_error", None)

        if st.session_state.get("_kwitansi_error"):
            st.error(
                f"Gagal memproses ZIP kwitansi: {st.session_state['_kwitansi_error']}. "
                "Field Profil Finansial TIDAK diubah — silakan upload ulang ZIP yang valid, "
                "atau ganti ke mode **Isi Manual**."
            )

        extracted_df = st.session_state.get("_kwitansi_extracted")
        if extracted_df is not None and not extracted_df.empty:
            n_gagal = extracted_df["total"].isna().sum()
            st.markdown("**Preview Hasil Ekstraksi Kwitansi**")
            if n_gagal:
                st.warning(
                    f"⚠️ {n_gagal} dari {len(extracted_df)} kwitansi gagal terbaca lengkap "
                    "(kolom 'total' kosong) — koreksi manual di tabel di bawah kalau perlu, atau "
                    "biarkan (baris itu tidak akan dihitung ke estimasi)."
                )
            st.download_button(
                "⬇️ Download Teks OCR Mentah (.txt)",
                data=build_raw_text_export(extracted_df),
                file_name=f"raw_ocr_kwitansi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Transkripsi OCR mentah per file, sebelum di-parse regex — dipakai untuk "
                     "cek/koreksi kalau ada field yang salah baca atau kosong.",
            )
            preview_df = extracted_df.drop(columns=["raw_text"]).copy()
            preview_df.insert(0, "status", np.where(preview_df["total"].isna(), "⚠️ Perlu Dicek", "✅ OK"))
            edited_kwitansi = st.data_editor(
                preview_df,
                use_container_width=True, hide_index=True, key="_kwitansi_editor",
                disabled=["status", "source_file", "catatan"],
                column_config={
                    "jenis_kwitansi": st.column_config.SelectboxColumn(options=["penjualan", "pembelian"]),
                    "total": st.column_config.NumberColumn(min_value=0, step=1000),
                    "catatan": st.column_config.TextColumn("Catatan / Teks OCR Mentah (potongan)", width="large"),
                },
            )

            if st.button("✅ Gunakan hasil ini untuk isi Profil Finansial", type="primary"):
                estimates = compute_monthly_estimates(edited_kwitansi)
                if estimates["monthly_turnover_est"] is None:
                    st.error(
                        "Tidak ada kwitansi 'penjualan' dengan total & tanggal lengkap untuk dihitung. "
                        "Field Profil Finansial TIDAK diubah — lengkapi tabel di atas, atau gunakan "
                        "mode **Isi Manual**."
                    )
                else:
                    st.session_state["f_monthly_turnover_est"] = estimates["monthly_turnover_est"]
                    st.session_state["f_transaction_frequency_monthly"] = estimates["transaction_frequency_monthly"]
                    st.session_state["_kwitansi_confirmed"] = True
                    st.success(
                        f"Terisi dari {estimates['n_months']} bulan data kwitansi tahun {estimates['year_used']}: "
                        f"Estimasi Omset Bulanan ≈ Rp {estimates['monthly_turnover_est']:,.0f}, "
                        f"Frekuensi Transaksi/Bulan ≈ {estimates['transaction_frequency_monthly']}. "
                        "Cek nilainya di form Profil Finansial di bawah."
                    )
                    st.rerun()
        elif extracted_df is not None and extracted_df.empty:
            st.error(
                "ZIP tidak berisi kwitansi yang bisa diproses. Field Profil Finansial TIDAK diubah "
                "— silakan gunakan mode **Isi Manual**."
            )

        if st.session_state.get("_kwitansi_confirmed"):
            st.info(
                "✔️ Estimasi Omset Bulanan & Frekuensi Transaksi/Bulan sudah terisi dari kwitansi "
                "di form di bawah — Nominal Pinjaman & Estimasi DSR tetap perlu diisi manual."
            )

    with st.form("manual_form"):
        st.markdown('<div class="section-title">👤 Identitas & Profil Usaha</div>', unsafe_allow_html=True)
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

        st.divider()
        st.caption("💰 **Profil Finansial** (mode input dipilih di atas, sebelum form ini).")
        c10, c11 = st.columns(2)
        with c10:
            monthly_turnover_est = st.number_input("Estimasi Omset Bulanan (Rp)", min_value=1_000_000, value=int(sv("monthly_turnover_est", 50_000_000)), step=1_000_000)
        with c11:
            transaction_frequency_monthly = st.number_input("Frekuensi Transaksi/Bulan", min_value=0, max_value=2000, value=int(sv("transaction_frequency_monthly", 80)))
        estimated_dsr = st.number_input("Estimasi DSR (Debt Service Ratio)", min_value=0.0, max_value=3.0, value=float(sv("estimated_dsr", 1.0)), step=0.1,
                                         help="Dibatasi maks 3.0 (dsr_capped) sesuai skema data training.")

        st.markdown('<div class="section-title">🏠 Agunan & Kredit</div>', unsafe_allow_html=True)
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

        st.divider()
        st.markdown('<div class="section-title">💵 Nominal Pinjaman Diajukan</div>', unsafe_allow_html=True)
        with st.container(border=True):
            loan_requested = st.number_input("Nominal Pinjaman Diajukan (Rp)", min_value=1_000_000, value=int(sv("loan_requested", 200_000_000)), step=10_000_000)
            jd1, jd2 = st.columns(2)
            with jd1:
                jenis_kredit_diajukan = st.selectbox(
                    "Jenis Kredit Diajukan", JENIS_KREDIT_DIAJUKAN_OPTIONS,
                    index=sb_index(JENIS_KREDIT_DIAJUKAN_OPTIONS, "jenis_kredit_diajukan", JENIS_KREDIT_DIAJUKAN_OPTIONS[0]),
                    help="Dipakai untuk validasi kesesuaian jenis & tenor terhadap kemampuan bayar (DSR) - lihat kartu 'Kesesuaian Jenis Kredit' di hasil.",
                )
            with jd2:
                tenor_diajukan_bulan = st.number_input(
                    "Tenor Diajukan (bulan)", min_value=1, max_value=120,
                    value=int(sv("tenor_diajukan_bulan", 24) or 24), step=6,
                )
            tujuan_penggunaan_kredit = st.text_area(
                "Tujuan Penggunaan Kredit", value=sv("tujuan_penggunaan_kredit", ""),
                placeholder="Contoh: Tambahan modal kerja operasional usaha", height=80,
            )

        submitted = st.form_submit_button("🤖 Jalankan AI Screening (ML)", type="primary", use_container_width=True)

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
            "jenis_kredit_diajukan": jenis_kredit_diajukan, "tenor_diajukan_bulan": tenor_diajukan_bulan,
            "tujuan_penggunaan_kredit": tujuan_penggunaan_kredit,
            "collateral_type": collateral_type, "certificate_type": certificate_type,
            "collateral_market_value": collateral_market_value, "collateral_liquidation_value": collateral_liquidation_value,
            "collateral_size_m2": collateral_size_m2, "ownership_match": ownership_match,
            "collateral_location": collateral_location, "collateral_province": collateral_province, "collateral_city": collateral_city,
        }
        application_id = f"SIM{datetime.now().strftime('%Y%m%d%H%M%S')}"
        with st.spinner("Menjalankan model screening (ML)..."):
            st.session_state["_manual_result"] = run_ml_screening(user_fields, application_id)
        for k in ("_edit_decision", "_edit_zone", "_edit_risk_score", "_edit_jenis", "_edit_nominal", "_edit_tenor", "_edit_bunga"):
            st.session_state.pop(k, None)

    if st.session_state.get("_manual_result"):
        result_row = st.session_state["_manual_result"]

        st.divider()
        st.markdown('<div class="section-title">📊 Hasil AI Screening (ML)</div>', unsafe_allow_html=True)
        if result_row.get("_manual_override"):
            st.caption("✏️ Hasil di bawah SUDAH diedit manual dari rekomendasi AI.")
        display_row = {k: v for k, v in result_row.items() if not k.startswith("_") and k not in ("NIK", "Alasan")}
        st.dataframe(pd.DataFrame([display_row]), hide_index=True, use_container_width=True)

        with st.expander("✏️ Sesuaikan Hasil Screening (Manual Override)", expanded=False):
            st.caption("Tidak setuju dengan rekomendasi AI? Ubah field di bawah lalu simpan — nilai AI asli tetap tersimpan dan bisa dikembalikan kapan saja.")
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                edit_decision = st.selectbox(
                    "Decision (Eligibility Recommendation)", DECISION_OPTIONS,
                    index=DECISION_OPTIONS.index(result_row["Decision (Eligibility Recommendation)"]) if result_row["Decision (Eligibility Recommendation)"] in DECISION_OPTIONS else 0,
                    key="_edit_decision",
                )
                edit_zone = st.selectbox(
                    "Zone", ZONE_OPTIONS,
                    index=ZONE_OPTIONS.index(result_row["Zone"]) if result_row["Zone"] in ZONE_OPTIONS else 0,
                    key="_edit_zone",
                )
            with ec2:
                edit_risk_score = st.number_input(
                    "Credit Eligibility Score", min_value=0.0, max_value=1.0, step=0.01,
                    value=float(result_row["Credit Eligibility Score"] or 0.0), key="_edit_risk_score",
                )
                edit_jenis = st.selectbox(
                    "Jenis Kredit", JENIS_KREDIT_OPTIONS,
                    index=JENIS_KREDIT_OPTIONS.index(result_row["Jenis Kredit"]) if result_row["Jenis Kredit"] in JENIS_KREDIT_OPTIONS else 0,
                    key="_edit_jenis",
                )
            with ec3:
                edit_nominal = st.number_input(
                    "Nominal Disetujui (Rp)", min_value=0, step=1_000_000,
                    value=int(result_row["Nominal Disetujui"] or 0), key="_edit_nominal",
                )
                edit_tenor = st.number_input(
                    "Jangka Waktu (bulan)", min_value=0, step=1,
                    value=int(result_row["Jangka Waktu (bulan)"] or 0), key="_edit_tenor",
                )
                edit_bunga = st.number_input(
                    "Bunga (% p.a.)", min_value=0.0, step=0.1,
                    value=float(result_row["Bunga (% p.a.)"] or 0.0), key="_edit_bunga",
                )

            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button("💾 Simpan Perubahan Manual", type="primary", use_container_width=True, key="save_single_manual_changes"):
                    if not result_row.get("_manual_override"):
                        result_row["_ai_original"] = {c: result_row[c] for c in MANUAL_EDIT_FIELDS}
                    proposed = {
                        "Decision (Eligibility Recommendation)": edit_decision, "Zone": edit_zone,
                        "Credit Eligibility Score": edit_risk_score, "Jenis Kredit": edit_jenis,
                        "Nominal Disetujui": edit_nominal, "Jangka Waktu (bulan)": edit_tenor,
                        "Bunga (% p.a.)": edit_bunga,
                    }
                    proposed = _cascade_edit(proposed, result_row)
                    for c in MANUAL_EDIT_FIELDS:
                        result_row[c] = proposed[c]
                    result_row["_manual_override"] = True
                    st.session_state["_manual_result"] = result_row
                    for k in ("_edit_decision", "_edit_zone", "_edit_risk_score", "_edit_jenis", "_edit_nominal", "_edit_tenor", "_edit_bunga"):
                        st.session_state.pop(k, None)
                    st.success("Perubahan manual disimpan (field turunan Credit Eligibility Score ikut disesuaikan otomatis).")
                    st.rerun()
            with bcol2:
                if result_row.get("_manual_override") and st.button("↩️ Kembalikan ke Rekomendasi AI", use_container_width=True):
                    result_row.update(result_row["_ai_original"])
                    result_row["_manual_override"] = False
                    del result_row["_ai_original"]
                    st.session_state["_manual_result"] = result_row
                    for k in ("_edit_decision", "_edit_zone", "_edit_risk_score", "_edit_jenis", "_edit_nominal", "_edit_tenor", "_edit_bunga"):
                        st.session_state.pop(k, None)
                    st.rerun()

        if result_row.get("_jenis_kredit_sesuai") is not None:
            sesuai = result_row["_jenis_kredit_sesuai"]
            sesuai_color = "#16a34a" if sesuai else "#d97706"
            st.markdown('<div class="section-title">📑 Kesesuaian Jenis Kredit</div>', unsafe_allow_html=True)
            with st.container(border=True):
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Jenis Diajukan", result_row.get("_jenis_kredit_diajukan") or "-")
                tenor_d = result_row.get("_tenor_diajukan_bulan")
                q2.metric("Tenor Diajukan", f"{tenor_d} bulan" if tenor_d else "-")
                q3.metric("Jenis Direkomendasikan", result_row["Jenis Kredit"])
                dsr = result_row.get("_dsr_pada_pengajuan")
                q4.metric("DSR Pengajuan", f"{dsr*100:.0f}%" if dsr is not None else "-")
                st.markdown(
                    f'<span style="background:{sesuai_color}22;color:{sesuai_color};border:1px solid {sesuai_color};'
                    f'padding:3px 10px;border-radius:99px;font-size:12px;font-weight:600;">'
                    f'{"Sesuai" if sesuai else "Perlu Penyesuaian"}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(result_row.get("_catatan_kesesuaian_kredit") or "")

        with st.expander("Insight teknis dari model (rule-based, tersedia instan)"):
            st.write(result_row["_insight"])
            if result_row["_shap"]:
                st.caption("Faktor paling berpengaruh terhadap skor (SHAP):")
                st.dataframe(pd.DataFrame(result_row["_shap"]), hide_index=True, use_container_width=True)

        st.markdown('<div class="section-title">🗣️ Narasi (LLM)</div>', unsafe_allow_html=True)
        if result_row.get("Alasan"):
            st.info(f"🗣️ **Alasan**: {result_row["Alasan"]}")
            if result_row.get("_fallback_reason"):
                st.warning(
                    f"⚠️ Narasi di atas adalah **fallback** ke insight rule-based — narasi LLM (Gemma) "
                    f"gagal dihasilkan. Alasan teknis: `{result_row['_fallback_reason']}`"
                )
            if result_row.get("_planner_summary"):
                st.write(f"🧭 **Proses Verifikasi (Planner Summary):** {result_row['_planner_summary']}")
        else:
            st.caption("Belum digenerate — klik tombol di bawah kalau perlu narasi natural untuk laporan (opsional, Credit Eligibility Score di atas sudah final).")
            if st.button("🤖 Generate Narasi (LLM)"):
                with st.spinner("Menyusun narasi dengan LLM..."):
                    st.session_state["_manual_result"] = run_llm_narrative(result_row)
                st.rerun()
