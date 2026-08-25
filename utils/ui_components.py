"""Shared rendering helpers so Detail Nasabah and Simulasi look identical."""
import streamlit as st

ZONE_COLORS = {"Hijau": "#16a34a", "Kuning": "#d97706", "Merah": "#dc2626"}
DECISION_ZONE = {
    "Layak": "Hijau",
    "Layak Bersyarat": "Kuning",
    "Perlu Review Ulang": "Kuning",
    "Tidak Layak": "Merah",
}

AGENT_META = {
    "identity": ("🪪", "Identity Agent"),
    "credit_history": ("📊", "Credit History Agent"),
    "dhn": ("🚫", "DHN Agent"),
    "collateral": ("🏠", "Collateral Agent"),
    "financial": ("💰", "Financial Agent"),
    "cashflow": ("💳", "Cashflow Agent"),
}


def zone_color(zone: str) -> str:
    return ZONE_COLORS.get(zone, "#6b7280")


def badge_html(text: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color};'
        f'padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:600;'
        f'white-space:nowrap;">{text}</span>'
    )


def decision_badge_html(decision: str) -> str:
    return badge_html(decision, zone_color(DECISION_ZONE.get(decision, "")))


def zone_badge_html(zone: str) -> str:
    return badge_html(zone, zone_color(zone))


def render_agent_card(agent_key: str, result: dict):
    """One agent's result (score/status/notes) inside a bordered container."""
    icon, title = AGENT_META[agent_key]
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**")
        color = "#16a34a" if result["score"] >= 0.75 else ("#d97706" if result["score"] >= 0.5 else "#dc2626")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-size:1.4em;font-weight:700;color:{color};">{result["score"]:.2f}</span>'
            f'{badge_html(result["status"], color)}</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(max(result["score"], 0.0), 1.0))
        for note in result["notes"]:
            st.caption(f"• {note}")


def render_risk_card(risk: dict):
    """Big final-decision card for the Risk Agent (orchestrator)."""
    color = zone_color(risk["zone"])
    with st.container(border=True):
        st.markdown("**🧭 Risk Agent — Keputusan Akhir**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div style="font-size:2em;font-weight:800;color:{color};">{risk["combined_score"]:.2f}</div>', unsafe_allow_html=True)
            st.caption("Skor gabungan")
        with c2:
            st.markdown(decision_badge_html(risk["decision"]), unsafe_allow_html=True)
            st.caption("Keputusan")
        with c3:
            st.markdown(zone_badge_html(risk["zone"]), unsafe_allow_html=True)
            st.caption("Zona risiko")

        if risk["decision"] != "Tidak Layak":
            c4, c5, c6, c7 = st.columns(4)
            c4.metric("Jenis Kredit", risk["jenis_kredit_rekomendasi"] or "-")
            c5.metric("Nominal Disetujui", f'Rp {risk["nominal_disetujui"]:,.0f}'.replace(",", "."))
            c6.metric("Jangka Waktu", f'{risk["jangka_waktu_bulan"]} bulan')
            c7.metric("Bunga", f'{risk["bunga_persen"]:.1f}% p.a.' if risk["bunga_persen"] is not None else "-")

        st.markdown("**Insight**")
        st.info(risk["insight"])


def render_full_result(result: dict):
    """Render the 6 agent cards (3x2 grid) + the big risk card, given the
    dict returned by agent_pipeline.score_application()."""
    render_risk_card(result["risk"])
    
    keys = ["identity", "credit_history", "dhn", "collateral", "financial", "cashflow"]
    row1 = st.columns(3)
    for col, key in zip(row1, keys[:3]):
        with col:
            render_agent_card(key, result[key])
    row2 = st.columns(3)
    for col, key in zip(row2, keys[3:]):
        with col:
            render_agent_card(key, result[key])
    st.write("")
