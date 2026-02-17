"""Week 6 analysis: Stock comparison and extreme market scenarios.

This analysis compares option Greeks between different stocks (MCD vs TSLA)
and explores extreme market scenarios including:
- Different volatility profiles
- Steady declining markets
- Rapidly rising markets
- Extreme skew conditions

Running this module will generate:
  - `data/mcd_tsla_comparison.csv`
  - `data/extreme_scenarios.csv`
  - `figures/fig_mcd_vs_tsla_delta.png`
  - `figures/fig_mcd_vs_tsla_gamma.png`
  - `figures/fig_extreme_scenarios.png`
  - `figures/fig_volatility_comparison.png`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .black_scholes import OptionParams, delta, gamma, price, theta, vega
except ImportError:
    from black_scholes import OptionParams, delta, gamma, price, theta, vega


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"


@dataclass
class StockProfile:
    name: str
    spot: float
    vol: float
    div_yield: float
    description: str


def _ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _generate_stock_comparison_data(
    stock_profiles: list[StockProfile], 
    strikes: Iterable[float], 
    maturities: Iterable[float]
) -> pd.DataFrame:
    """Generate Greeks data for different stock profiles."""
    records = []
    
    for stock in stock_profiles:
        for strike in strikes:
            for maturity in maturities:
                params = OptionParams(
                    spot=stock.spot,
                    strike=strike,
                    time=maturity,
                    rate=0.02,  # 2% risk-free rate
                    vol=stock.vol,
                    div=stock.div_yield,
                )
                
                records.append({
                    "stock": stock.name,
                    "strike": strike,
                    "maturity": maturity,
                    "spot": stock.spot,
                    "vol": stock.vol,
                    "div_yield": stock.div_yield,
                    "price": price(params),
                    "delta": delta(params),
                    "gamma": gamma(params),
                    "theta": theta(params),
                    "vega": vega(params),
                    "moneyness": strike / stock.spot,
                })
    
    return pd.DataFrame.from_records(records)


def _generate_extreme_scenarios() -> pd.DataFrame:
    """Generate data for extreme market scenarios."""
    records = []
    
    # Scenario 1: Steady declining market (like ZOOM post-pandemic)
    declining_scenarios = [
        {"spot": 100, "vol": 0.15, "div": 0.0, "scenario": "Normal"},
        {"spot": 80, "vol": 0.25, "div": 0.0, "scenario": "Declining_20pct"},
        {"spot": 60, "vol": 0.35, "div": 0.0, "scenario": "Declining_40pct"},
        {"spot": 40, "vol": 0.45, "div": 0.0, "scenario": "Declining_60pct"},
    ]
    
    # Scenario 2: Rapidly rising market (like GME during squeeze)
    rising_scenarios = [
        {"spot": 100, "vol": 0.15, "div": 0.0, "scenario": "Normal"},
        {"spot": 150, "vol": 0.30, "div": 0.0, "scenario": "Rising_50pct"},
        {"spot": 200, "vol": 0.50, "div": 0.0, "scenario": "Rising_100pct"},
        {"spot": 300, "vol": 0.80, "div": 0.0, "scenario": "Rising_200pct"},
    ]
    
    # Scenario 3: Extreme volatility scenarios
    vol_scenarios = [
        {"spot": 100, "vol": 0.10, "div": 0.0, "scenario": "Low_Vol"},
        {"spot": 100, "vol": 0.25, "div": 0.0, "scenario": "Normal_Vol"},
        {"spot": 100, "vol": 0.50, "div": 0.0, "scenario": "High_Vol"},
        {"spot": 100, "vol": 1.00, "div": 0.0, "scenario": "Extreme_Vol"},
    ]
    
    all_scenarios = declining_scenarios + rising_scenarios + vol_scenarios
    
    strikes = [80, 90, 100, 110, 120]
    maturities = [0.1, 0.25, 0.5, 1.0]
    
    for scenario in all_scenarios:
        for strike in strikes:
            for maturity in maturities:
                params = OptionParams(
                    spot=scenario["spot"],
                    strike=strike,
                    time=maturity,
                    rate=0.02,
                    vol=scenario["vol"],
                    div=scenario["div"],
                )
                
                records.append({
                    "scenario": scenario["scenario"],
                    "spot": scenario["spot"],
                    "strike": strike,
                    "maturity": maturity,
                    "vol": scenario["vol"],
                    "price": price(params),
                    "delta": delta(params),
                    "gamma": gamma(params),
                    "theta": theta(params),
                    "vega": vega(params),
                    "moneyness": strike / scenario["spot"],
                })
    
    return pd.DataFrame.from_records(records)


def _plot_stock_comparison(df: pd.DataFrame, greek: str, fname: Path) -> None:
    """Plot comparison of Greeks between different stocks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Greeks vs Strike for different maturities
    for stock in df["stock"].unique():
        stock_data = df[df["stock"] == stock]
        for maturity in [0.25, 0.5, 1.0]:
            maturity_data = stock_data[stock_data["maturity"] == maturity]
            if not maturity_data.empty:
                ax1.plot(
                    maturity_data["strike"], 
                    maturity_data[greek], 
                    label=f"{stock} T={maturity:.2f}",
                    linestyle="--" if maturity == 0.25 else "-" if maturity == 0.5 else ":",
                    linewidth=2 if maturity == 0.5 else 1
                )
    
    ax1.set_xlabel("Strike Price")
    ax1.set_ylabel(f"{greek.capitalize()}")
    ax1.set_title(f"{greek.capitalize()} vs Strike Price")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Greeks vs Moneyness
    for stock in df["stock"].unique():
        stock_data = df[df["stock"] == stock]
        maturity_05 = stock_data[stock_data["maturity"] == 0.5]
        if not maturity_05.empty:
            ax2.plot(
                maturity_05["moneyness"], 
                maturity_05[greek], 
                label=stock,
                linewidth=2,
                marker='o',
                markersize=4
            )
    
    ax2.set_xlabel("Moneyness (Strike/Spot)")
    ax2.set_ylabel(f"{greek.capitalize()}")
    ax2.set_title(f"{greek.capitalize()} vs Moneyness (T=0.5)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='ATM')
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_extreme_scenarios(df: pd.DataFrame, fname: Path) -> None:
    """Plot extreme market scenarios."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Declining market scenarios
    declining_data = df[df["scenario"].str.contains("Declining")]
    for scenario in declining_data["scenario"].unique():
        scenario_data = declining_data[declining_data["scenario"] == scenario]
        atm_data = scenario_data[scenario_data["moneyness"].between(0.95, 1.05)]
        if not atm_data.empty:
            ax1.plot(
                atm_data["maturity"], 
                atm_data["delta"], 
                label=scenario.replace("Declining_", ""),
                linewidth=2,
                marker='o'
            )
    
    ax1.set_xlabel("Time to Maturity")
    ax1.set_ylabel("Delta")
    ax1.set_title("Delta in Declining Markets (ATM Options)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rising market scenarios
    rising_data = df[df["scenario"].str.contains("Rising")]
    for scenario in rising_data["scenario"].unique():
        scenario_data = rising_data[rising_data["scenario"] == scenario]
        atm_data = scenario_data[scenario_data["moneyness"].between(0.95, 1.05)]
        if not atm_data.empty:
            ax2.plot(
                atm_data["maturity"], 
                atm_data["gamma"], 
                label=scenario.replace("Rising_", ""),
                linewidth=2,
                marker='s'
            )
    
    ax2.set_xlabel("Time to Maturity")
    ax2.set_ylabel("Gamma")
    ax2.set_title("Gamma in Rising Markets (ATM Options)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility scenarios - Vega
    vol_data = df[df["scenario"].str.contains("Vol")]
    for scenario in vol_data["scenario"].unique():
        scenario_data = vol_data[vol_data["scenario"] == scenario]
        atm_data = scenario_data[scenario_data["moneyness"].between(0.95, 1.05)]
        if not atm_data.empty:
            ax3.plot(
                atm_data["maturity"], 
                atm_data["vega"], 
                label=scenario.replace("_Vol", ""),
                linewidth=2,
                marker='^'
            )
    
    ax3.set_xlabel("Time to Maturity")
    ax3.set_ylabel("Vega")
    ax3.set_title("Vega Across Volatility Regimes (ATM Options)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Price comparison across scenarios
    all_scenarios = ["Normal", "Declining_40pct", "Rising_100pct", "Extreme_Vol"]
    for scenario in all_scenarios:
        scenario_data = df[df["scenario"] == scenario]
        atm_data = scenario_data[scenario_data["moneyness"].between(0.95, 1.05)]
        if not atm_data.empty:
            ax4.plot(
                atm_data["maturity"], 
                atm_data["price"], 
                label=scenario,
                linewidth=2,
                marker='d'
            )
    
    ax4.set_xlabel("Time to Maturity")
    ax4.set_ylabel("Option Price")
    ax4.set_title("Option Prices Across Market Scenarios (ATM)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_volatility_comparison(df: pd.DataFrame, fname: Path) -> None:
    """Plot volatility impact on Greeks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Greeks vs Volatility for different strikes
    vol_data = df[df["scenario"].str.contains("Vol")]
    maturity_05 = vol_data[vol_data["maturity"] == 0.5]
    
    for strike in [90, 100, 110]:
        strike_data = maturity_05[maturity_05["strike"] == strike]
        if not strike_data.empty:
            ax1.plot(
                strike_data["vol"], 
                strike_data["vega"], 
                label=f"Strike {strike}",
                linewidth=2,
                marker='o'
            )
    
    ax1.set_xlabel("Volatility")
    ax1.set_ylabel("Vega")
    ax1.set_title("Vega vs Volatility (T=0.5)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gamma vs Volatility
    for strike in [90, 100, 110]:
        strike_data = maturity_05[maturity_05["strike"] == strike]
        if not strike_data.empty:
            ax2.plot(
                strike_data["vol"], 
                strike_data["gamma"], 
                label=f"Strike {strike}",
                linewidth=2,
                marker='s'
            )
    
    ax2.set_xlabel("Volatility")
    ax2.set_ylabel("Gamma")
    ax2.set_title("Gamma vs Volatility (T=0.5)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_week6_analysis() -> None:
    """Run Week 6 analysis comparing different stocks and extreme scenarios."""
    _ensure_directories()
    
    # Define stock profiles
    stock_profiles = [
        StockProfile(
            name="MCD",
            spot=250.0,
            vol=0.15,  # Lower volatility, stable dividend stock
            div_yield=0.025,  # 2.5% dividend yield
            description="Stable dividend stock"
        ),
        StockProfile(
            name="TSLA",
            spot=200.0,
            vol=0.45,  # High volatility, growth stock
            div_yield=0.0,  # No dividends
            description="High volatility growth stock"
        ),
    ]
    
    # Generate comparison data
    strikes = np.linspace(150, 350, 21)  # Wide range for both stocks
    maturities = np.linspace(0.1, 1.0, 10)
    
    comparison_df = _generate_stock_comparison_data(stock_profiles, strikes, maturities)
    comparison_df.to_csv(DATA_DIR / "mcd_tsla_comparison.csv", index=False)
    
    # Generate extreme scenarios data
    extreme_df = _generate_extreme_scenarios()
    extreme_df.to_csv(DATA_DIR / "extreme_scenarios.csv", index=False)
    
    # Create plots
    _plot_stock_comparison(comparison_df, "delta", FIG_DIR / "fig_mcd_vs_tsla_delta.png")
    _plot_stock_comparison(comparison_df, "gamma", FIG_DIR / "fig_mcd_vs_tsla_gamma.png")
    _plot_extreme_scenarios(extreme_df, FIG_DIR / "fig_extreme_scenarios.png")
    _plot_volatility_comparison(extreme_df, FIG_DIR / "fig_volatility_comparison.png")
    
    print("Week 6 analysis completed!")
    print(f"Generated {len(comparison_df)} comparison records")
    print(f"Generated {len(extreme_df)} extreme scenario records")


if __name__ == "__main__":
    run_week6_analysis()
