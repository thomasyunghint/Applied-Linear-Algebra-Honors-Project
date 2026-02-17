# Week 5 Progress

## Focus
- Built Greeks sensitivity matrix (Δ, Γ, Θ) across portfolio and hedge instruments.
- Solved least-squares hedge weights to minimize net Greeks.
- Visualized pre/post hedge sensitivities.

## Portfolio Setup
- Three core positions combining long/short calls and puts around spot 100.
- Hedge instruments: at-the-money call, put, and forward.

## Outputs
- `table_weights.csv`: combined sensitivity matrix.
- `hedge_weights.json`: optimized hedge exposure.
- `fig_delta_before_after.png`: net Greeks before vs after hedge.

## Observations
- Forward position dominates Δ adjustment; options balance higher-order Greeks.
- Solution reduces Δ exposure by >95% with modest Γ/Θ impact.


