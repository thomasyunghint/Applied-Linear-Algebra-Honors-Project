# Week 4 Progress

## Focus
- Generated Δ, Γ, Θ heatmaps across spot and maturity grids.
- Compared exact price profile to Δ linear and Δ+Γ quadratic approximations.

## Key Parameters
- Underlying spot range: 80 to 120.
- Strike: 100; risk-free rate: 2%; dividend yield: 1%; volatility: 25%.
- Maturities from 0.05 to 1.5 years.

## Outputs
- `fig_delta_heatmap.png`
- `fig_gamma_heatmap.png`
- `fig_theta_heatmap.png`
- `fig_linear_vs_quadratic.png`
- `greeks_surface.csv`

## Observations
- Δ increases with spot and short maturity; Γ concentrates near at-the-money short-dated points.
- Quadratic approximation markedly reduces error relative to pure Δ hedge for ±20% spot moves.


