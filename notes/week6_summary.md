# Week 6 Analysis Summary: Stock Comparison and Extreme Market Scenarios

## Overview
Week 6 analysis explores option Greeks behavior across different stock profiles and extreme market conditions, addressing the professor's feedback about comparing different inputs and market scenarios.

## Key Components

### 1. Stock Profile Comparison (MCD vs TSLA)
- **McDonald's (MCD)**: Stable dividend stock with low volatility (15%) and 2.5% dividend yield
- **Tesla (TSLA)**: High-growth stock with high volatility (45%) and no dividends
- **Analysis**: Greeks behavior across different strikes and maturities

### 2. Extreme Market Scenarios
- **Declining Markets**: Simulating ZOOM-like post-pandemic decline (20%, 40%, 60% drops)
- **Rising Markets**: Simulating GME-like rapid rises (50%, 100%, 200% increases)
- **Volatility Regimes**: Low (10%), Normal (25%), High (50%), Extreme (100%) volatility

### 3. Generated Outputs
- `mcd_tsla_comparison.csv`: Comprehensive Greeks data for both stocks
- `extreme_scenarios.csv`: Greeks data across extreme market conditions
- `fig_mcd_vs_tsla_delta.png`: Delta comparison between stocks
- `fig_mcd_vs_tsla_gamma.png`: Gamma comparison between stocks
- `fig_extreme_scenarios.png`: Greeks behavior in extreme conditions
- `fig_volatility_comparison.png`: Impact of volatility on Greeks

## Key Findings

### Stock Comparison Insights
1. **Delta Behavior**: TSLA shows more extreme delta values due to higher volatility
2. **Gamma Concentration**: MCD has more stable gamma, TSLA shows higher gamma near ATM
3. **Moneyness Impact**: Both stocks show expected delta behavior, but TSLA's is more pronounced

### Extreme Scenario Insights
1. **Declining Markets**: Delta becomes more negative, gamma increases as options go OTM
2. **Rising Markets**: Delta approaches 1.0, gamma peaks and then decreases
3. **Volatility Impact**: Vega increases with volatility, but at a decreasing rate
4. **Time Decay**: Theta behavior varies significantly across volatility regimes

## Mathematical Insights
- **Delta Sensitivity**: Higher volatility stocks show greater delta sensitivity to price changes
- **Gamma Concentration**: Gamma peaks near ATM and increases with volatility
- **Vega Behavior**: Vega is highest for ATM options and increases with time to expiry
- **Theta Decay**: Time decay accelerates in high volatility environments

## Applications
- **Portfolio Hedging**: Understanding how different stocks require different hedging strategies
- **Risk Management**: Extreme scenarios help quantify tail risk in option portfolios
- **Market Making**: Greeks behavior guides position sizing and risk limits
- **Volatility Trading**: Understanding vega exposure across different market regimes

## Next Steps
This analysis provides the foundation for Week 7's historical data analysis, where we'll apply these insights to real market data from stocks that experienced extreme movements.
