## Honors Contract Progress Snapshot

### Project Goal
Build a reproducible derivatives analysis pipeline: implement Black–Scholes Greeks, visualize sensitivities, and demonstrate hedging plus optional PCA/SVD risk decomposition for the MAT 343 honors credit.

### Completed Milestones
- **Week 4 – Foundation & Δ heatmap**
  - Set up repo, data pulls, and pricing notebook.
  - Implemented closed-form Δ/Γ/Θ/Vega for configurable strikes (0.8–1.2 × spot) and expiries (30, 90 days).
  - Produced `fig_delta_heatmap.png` and drafted a 1-page note linking Greeks to Jacobian/Hessian interpretations.
- **Week 5 – Higher-order plots & approximations**
  - Generated full Δ/Γ/Θ(+Vega) heatmaps and surfaces; compared true pricing vs linear (Δ) and quadratic (Δ+Γ) approximations.
  - Captured comparisons in `fig_gamma_heatmap.png` and `fig_linear_vs_quadratic.png`, plus a short narrative on approximation error drivers (strike skew, time-to-maturity).
- **Week 6 – Portfolio sensitivity & hedging**
  - Constructed a sensitivity matrix (rows = Greeks, columns = positions/hedges) covering three core positions and ATM call/put/forward hedges.
  - Solved least-squares hedge weights via QR, generating `table_weights.csv`, `hedge_weights.json`, and `fig_delta_before_after.png`.
  - Result: Δ exposure reduced by >95% while keeping Γ/Θ drift minimal; forward leg handled most Δ while options balanced curvature.

### In-Progress / Upcoming
- **Week 7 (current):** Polish repo, tighten notebook narrative, insert key figures, and assemble a 5–8 page `report_v1.pdf` plus an updated `README.md`.
- **Week 8 (buffer/enhancement):** Optional PCA/SVD on sensitivities or historical covariance to highlight principal risk modes; planned deliverable `fig_pca_sensitivity.png` with a concise explanation.

### Meeting & Support Plan
- Weeks 4–6 sessions covered scope, Δ heatmap checks, approximation review, and hedging deep dive (total ≈5.5 hrs).
- Remaining touchpoints: report v1 review (week 7, 2 hrs) and optional PCA check (week 8, 0.5 hr).

### Risks & Mitigations
- **Reproducibility:** All notebooks export figures/tables directly; next action is to verify paths and pin dependencies before report v1.
- **Data variety:** Exploring divergent symbols (e.g., MCD vs TSLA vs rapidly falling stock) to evidence robustness; queued for week 7 narrative polish.
- **Optional PCA scope:** Preliminary literature scan underway; will either include a concrete PCA demo or document rationale for deferring.

### Completion Criteria
Finishing weeks 4–7 with reproducible code grants honors credit. Delivering the optional week 8 PCA/SVD analysis counts as an enhancement and strengthens the final presentation.

