# Layer 1: Machien Learning Model
eligibility_score = model.predict(X_features)  # output 0-1

# Layer 2: Policy Engine - deterministic rules
decision, zone = score_to_decision(risk_score)
nominal_disetujui = 0 if decision == "Tidak Layak" else int(min(loan_requested, collateral_market_value * 0.70))
jenis_kredit = "-" if decision == "Tidak Layak" else "KUR" if loan_requested <= 500_000_000 and loan_requested / monthly_turnover_est <= 6 else "KMK" if loan_requested < 200_000_000 else "KI"
tenor = 0 if decision == "Tidak Layak" else 36 if jenis_kredit == "KUR" else 12 if jenis_kredit == "KMK" else 36
bunga = None if decision == "Tidak Layak" else 6.0 if jenis_kredit == "KUR" else 9.5 if zone == "Hijau" else 12.0