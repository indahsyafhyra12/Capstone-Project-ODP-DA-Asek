
"""Premium Detail Nasabah V5"""
import streamlit as st
import plotly.graph_objects as go
from utils.data_loader import load_master_data
from utils.ui_components import apply_logo
from utils.ui_premium import inject_css, hero_banner

st.set_page_config(page_title="Detail Nasabah", page_icon="👤", layout="wide")
apply_logo()
inject_css()

st.markdown("""
<style>
html,body,[class*="css"]{font-family:Inter,sans-serif;}
.section-title{font-size:24px;font-weight:700;color:#1F2937;margin:26px 0 12px;}
[data-testid="stMetricLabel"]{font-size:13px;color:#64748B;}
[data-testid="stMetricValue"]{font-size:22px;font-weight:600;color:#1F2937;}
.agent-card{background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:16px;height:160px;}
</style>
""", unsafe_allow_html=True)

hero_banner("👤 Detail Nasabah",
            "Customer 360° View • Ringkasan kelayakan kredit berdasarkan evaluasi 7-Agent AI.")

df = load_master_data()
df["label"] = df["application_id"] + " — " + df["company_name"]
selected = st.selectbox("Pilih ID Pengajuan", df["label"])
row = df[df["label"] == selected].iloc[0]

def pick(*cols, default="-"):
    for c in cols:
        if c in row.index:
            return row[c]
    return default

score = float(pick("risk_score", default=0))
loan = float(pick("pinjaman_diajukan","loan_amount","nominal_pengajuan", default=0))
approved = float(pick("nominal_disetujui","approved_amount", default=0))
tenor = pick("loan_term_month","tenor_bulan", default="-")
bunga = pick("interest_rate_pct","bunga_pct", default="-")

if score >= .8:
    level,color = "Highly Eligible","#16A34A"
elif score >= .65:
    level,color = "Eligible with Conditions","#F59E0B"
elif score >= .5:
    level,color = "Manual Review","#EA580C"
else:
    level,color = "Not Eligible","#DC2626"

left,right = st.columns([3.3,1])
with left:
    with st.container(border=True):
        st.markdown(f"### 🏢 {row['company_name']}")
        st.caption(f"{row['application_id']} • {pick('industry')} • {pick('sub_industry','subindustry')} • {pick('legal_entity')}")
with right:
    st.markdown(f"""
    <div style="background:{color}12;border:1px solid {color};border-radius:18px;padding:18px;text-align:center;">
      <div style="font-size:13px;color:#64748B;">Credit Eligibility</div>
      <div style="font-size:36px;font-weight:700;color:{color};margin:8px 0;">{score:.2f}</div>
      <span style="background:{color};color:white;padding:5px 12px;border-radius:999px;font-size:12px;">{level}</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Pemilik", pick("owner_name","owner"))
k2.metric("Cabang", pick("branch_name"))
k3.metric("Pinjaman", f"Rp {loan:,.0f}".replace(",","."))
k4.metric("Tanggal", str(pick("application_date")))

st.markdown('<div class="section-title">Credit Eligibility Summary</div>', unsafe_allow_html=True)
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    number={"font":{"size":32}},
    gauge={
        "axis":{"range":[0,1]},
        "bar":{"color":color},
        "steps":[
            {"range":[0,.5],"color":"#FDE2E2"},
            {"range":[.5,.65],"color":"#FEEBC8"},
            {"range":[.65,.8],"color":"#FEF3C7"},
            {"range":[.8,1],"color":"#DCFCE7"}
        ]
    }))
gauge.update_layout(height=230, margin=dict(l=5,r=5,t=20,b=5))
g1,g2 = st.columns([1.1,1])
g1.plotly_chart(gauge, use_container_width=True)
with g2:
    with st.container(border=True):
        st.markdown("#### Keputusan AI")
        st.write(f"**Decision:** {pick('decision')}")
        st.write(f"**Kategori:** {level}")
        st.write(f"**Jenis Kredit:** {pick('loan_type')}")
        st.write(f"**Nominal Disetujui:** Rp {approved:,.0f}".replace(",","."))
        st.write(f"**Tenor:** {tenor} bulan")
        st.write(f"**Bunga:** {bunga}% p.a.")

st.markdown('<div class="section-title">AI Recommendation</div>', unsafe_allow_html=True)
st.info(f"Nasabah direkomendasikan **{pick('decision')}** dengan kategori **{level}** berdasarkan evaluasi multi-agent.")

st.markdown('<div class="section-title">7-Agent Assessment</div>', unsafe_allow_html=True)
agents=[
("Identity","identity_score"),
("Credit History","credit_history_score"),
("DHN","dhn_score"),
("Collateral","collateral_score"),
("Financial","financial_score"),
("Cashflow","cashflow_score"),
("Final Score","risk_score")]
cols=st.columns(3)
for i,(name,col) in enumerate(agents):
    v=float(pick(col, default=0))
    c="#16A34A" if v>=.8 else "#F59E0B" if v>=.65 else "#DC2626"
    with cols[i%3]:
        st.markdown(f"""
        <div class="agent-card">
          <div style="font-size:15px;font-weight:600;">{name}</div>
          <div style="font-size:28px;font-weight:700;color:{c};margin:10px 0;">{v:.2f}</div>
          <div style="background:#E5E7EB;height:8px;border-radius:99px;">
            <div style="width:{v*100:.0f}%;height:8px;background:{c};border-radius:99px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-title">Financial Snapshot</div>', unsafe_allow_html=True)
f1,f2,f3 = st.columns(3)
f1.metric("Omzet Bulanan", f"Rp {float(pick('monthly_turnover_est',default=0)):,.0f}".replace(",","."))
f2.metric("Revenue Growth", f"{pick('revenue_growth_pct',default=0)}%")
f3.metric("Jumlah Karyawan", int(float(pick('employee_count',default=0))))

st.markdown('<div class="section-title">Supporting Information</div>', unsafe_allow_html=True)
s1,s2,s3 = st.columns(3)
s1.metric("Status DHN", pick("status_dhn"))
s2.metric("Badan Usaha", pick("legal_entity"))
s3.metric("Pendidikan", pick("owner_education"))

st.markdown('<div class="section-title">Next Action</div>', unsafe_allow_html=True)
a,b,c = st.columns(3)
a.success("Lanjut ke Analisis Kredit")
b.info("Review Dokumen")
c.warning("Monitoring Berkala")
