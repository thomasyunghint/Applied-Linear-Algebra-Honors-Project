"""Week 7 analysis: Historical data analysis with simulated market scenarios.

This analysis uses simulated historical data to analyze option Greeks for stocks
with extreme movements, including:
- GameStop (GME) during the 2021 squeeze
- Zoom (ZM) post-pandemic decline
- Tesla (TSLA) volatility patterns
- Simulated realistic market data

Running this module will generate:
  - `data/historical_gme_data.csv`
  - `data/historical_zoom_data.csv`
  - `data/historical_tsla_data.csv`
  - `figures/fig_historical_gme_analysis.png`
  - `figures/fig_historical_zoom_analysis.png`
  - `figures/fig_historical_tsla_analysis.png`
  - `figures/fig_historical_comparison.png`
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import time

try:
    from .black_scholes import OptionParams, delta, gamma, price as option_price, theta, vega
    from .data_fetcher import fetch_stock_data, StockDataRequest
except ImportError:
    from black_scholes import OptionParams, delta, gamma, price as option_price, theta, vega
    from data_fetcher import fetch_stock_data, StockDataRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"


@dataclass
class HistoricalStockData:
    symbol: str
    dates: List[str]
    prices: List[float]
    volumes: List[float]
    volatilities: List[float]
    description: str


def _ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_historical_gme_data() -> HistoricalStockData:
    """Generate simulated GME data during the 2021 squeeze period."""
    print("Generating simulated GME data for 2021 squeeze period...")
    
    # Create date range for GME squeeze period (Jan-Feb 2021)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2021, 2, 28)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Only weekdays
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    # Simulate GME squeeze pattern: gradual rise, then explosive growth, then crash
    np.random.seed(42)  # For reproducible results
    n_days = len(dates)
    prices = []
    base_price = 20.0
    
    for i in range(n_days):
        if i < 20:  # Early January: gradual rise
            price = base_price + i * 0.5 + np.random.normal(0, 1)
        elif i < 25:  # Late January: explosive squeeze
            squeeze_factor = (i - 20) * 50
            price = base_price + 10 + squeeze_factor + np.random.normal(0, 5)
        else:  # February: crash and volatility
            crash_factor = (i - 25) * -10
            price = base_price + 250 + crash_factor + np.random.normal(0, 20)
        
        prices.append(max(price, 1.0))  # Ensure positive prices
    
    # Calculate realistic volumes (higher during squeeze)
    volumes = []
    for i, price in enumerate(prices):
        if i < 20:
            base_vol = 10_000_000
        elif i < 25:
            base_vol = 100_000_000  # Massive volume during squeeze
        else:
            base_vol = 50_000_000
        volumes.append(int(base_vol * (1 + np.random.normal(0, 0.3))))
    
    # Calculate volatilities based on price movements
    volatilities = []
    for i in range(len(prices)):
        if i == 0:
            vol = 0.5
        else:
            daily_return = abs(np.log(prices[i] / prices[i-1]))
            vol = min(daily_return * np.sqrt(252), 3.0)  # Annualized volatility, capped at 300%
        volatilities.append(vol)
    
    print(f"Generated {len(prices)} GME data points (simulated)")
    return HistoricalStockData(
        symbol="GME",
        dates=dates,
        prices=prices,
        volumes=volumes,
        volatilities=volatilities,
        description="GameStop during 2021 short squeeze (Simulated Data)"
    )


def _fetch_historical_zoom_data() -> HistoricalStockData:
    """Generate simulated ZOOM data during post-pandemic decline."""
    print("Generating simulated ZM data for post-pandemic decline...")
    
    # Create date range for ZOOM decline period (Oct-Dec 2021)
    start_date = datetime(2021, 10, 1)
    end_date = datetime(2021, 12, 31)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Only weekdays
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    # Simulate ZOOM decline pattern: gradual decline with some volatility
    np.random.seed(123)  # Different seed for variety
    n_days = len(dates)
    prices = []
    base_price = 300.0  # Starting high from pandemic peak
    
    for i in range(n_days):
        # Gradual decline with some recovery attempts
        decline_factor = i * 2.0  # Steady decline
        recovery_attempts = 10 * np.sin(i * 0.1)  # Some volatility
        noise = np.random.normal(0, 5)
        
        price = base_price - decline_factor + recovery_attempts + noise
        prices.append(max(price, 50.0))  # Ensure positive prices
    
    # Calculate realistic volumes (moderate during decline)
    volumes = []
    for i, price in enumerate(prices):
        base_vol = 5_000_000
        # Higher volume on big moves
        if i > 0:
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            volume_multiplier = 1 + price_change * 2
        else:
            volume_multiplier = 1
        volumes.append(int(base_vol * volume_multiplier * (1 + np.random.normal(0, 0.2))))
    
    # Calculate volatilities based on price movements
    volatilities = []
    for i in range(len(prices)):
        if i == 0:
            vol = 0.3
        else:
            daily_return = abs(np.log(prices[i] / prices[i-1]))
            vol = min(daily_return * np.sqrt(252), 1.5)  # Annualized volatility, capped at 150%
        volatilities.append(vol)
    
    print(f"Generated {len(prices)} ZM data points (simulated)")
    return HistoricalStockData(
        symbol="ZM",
        dates=dates,
        prices=prices,
        volumes=volumes,
        volatilities=volatilities,
        description="Zoom post-pandemic decline (Simulated Data)"
    )


def _fetch_historical_tsla_data() -> HistoricalStockData:
    """Generate simulated TSLA data with high volatility patterns."""
    print("Generating simulated TSLA data for high volatility period...")
    
    # Create date range for TSLA volatility period (Jan-Apr 2022)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 4, 30)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Only weekdays
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    # Simulate TSLA high volatility pattern: erratic movements with trends
    np.random.seed(456)  # Different seed for variety
    n_days = len(dates)
    prices = []
    base_price = 1000.0  # Starting price
    
    for i in range(n_days):
        # High volatility with some trending periods
        if i < 30:  # Early period: high volatility, slight uptrend
            trend = i * 2
            volatility = 50
        elif i < 60:  # Middle period: extreme volatility, choppy
            trend = 60 - (i - 30) * 1
            volatility = 80
        else:  # Later period: high volatility, downtrend
            trend = 30 - (i - 60) * 3
            volatility = 60
        
        price = base_price + trend + np.random.normal(0, volatility)
        prices.append(max(price, 100.0))  # Ensure positive prices
    
    # Calculate realistic volumes (high during volatile periods)
    volumes = []
    for i, price in enumerate(prices):
        base_vol = 20_000_000
        # Higher volume on big moves
        if i > 0:
            price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
            volume_multiplier = 1 + price_change * 3
        else:
            volume_multiplier = 1
        volumes.append(int(base_vol * volume_multiplier * (1 + np.random.normal(0, 0.4))))
    
    # Calculate volatilities based on price movements
    volatilities = []
    for i in range(len(prices)):
        if i == 0:
            vol = 0.6
        else:
            daily_return = abs(np.log(prices[i] / prices[i-1]))
            vol = min(daily_return * np.sqrt(252), 2.0)  # Annualized volatility, capped at 200%
        volatilities.append(vol)
    
    print(f"Generated {len(prices)} TSLA data points (simulated)")
    return HistoricalStockData(
        symbol="TSLA",
        dates=dates,
        prices=prices,
        volumes=volumes,
        volatilities=volatilities,
        description="Tesla high volatility period (Simulated Data)"
    )


def _analyze_historical_greeks(stock_data: HistoricalStockData) -> pd.DataFrame:
    """Analyze Greeks for historical stock data."""
    records = []
    
    for i, (date, price, vol) in enumerate(zip(stock_data.dates, stock_data.prices, stock_data.volatilities)):
        # Create option parameters for ATM call
        params = OptionParams(
            spot=price,
            strike=price,  # ATM
            time=0.25,  # 3 months to expiry
            rate=0.02,  # 2% risk-free rate
            vol=vol,
            div=0.0,  # No dividends for these stocks
        )
        
        records.append({
            "date": date,
            "symbol": stock_data.symbol,
            "spot_price": price,
            "volatility": vol,
            "option_price": option_price(params),
            "delta": delta(params),
            "gamma": gamma(params),
            "theta": theta(params),
            "vega": vega(params),
            "day": i,
        })
    
    return pd.DataFrame.from_records(records)


def _plot_historical_analysis(df: pd.DataFrame, stock_symbol: str, fname: Path) -> None:
    """Plot historical analysis for a specific stock."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Convert dates for plotting
    dates = pd.to_datetime(df["date"])
    
    # Plot 1: Price and Volatility
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(dates, df["spot_price"], 'b-', linewidth=2, label='Stock Price')
    line2 = ax1_twin.plot(dates, df["volatility"], 'r--', linewidth=2, label='Volatility')
    
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price ($)", color='b')
    ax1_twin.set_ylabel("Volatility", color='r')
    ax1.set_title(f"{stock_symbol} - Price and Volatility Over Time")
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Greeks over time
    ax2.plot(dates, df["delta"], label='Delta', linewidth=2)
    ax2.plot(dates, df["gamma"], label='Gamma', linewidth=2)
    ax2.plot(dates, df["theta"] * 100, label='Theta (×100)', linewidth=2)
    ax2.plot(dates, df["vega"], label='Vega', linewidth=2)
    
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Greek Value")
    ax2.set_title(f"{stock_symbol} - Greeks Over Time (ATM Options)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Greeks vs Stock Price
    ax3.scatter(df["spot_price"], df["delta"], alpha=0.6, label='Delta', s=20)
    ax3.scatter(df["spot_price"], df["gamma"] * 100, alpha=0.6, label='Gamma (×100)', s=20)
    ax3.scatter(df["spot_price"], df["vega"], alpha=0.6, label='Vega', s=20)
    
    ax3.set_xlabel("Stock Price ($)")
    ax3.set_ylabel("Greek Value")
    ax3.set_title(f"{stock_symbol} - Greeks vs Stock Price")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Greeks vs Volatility
    ax4.scatter(df["volatility"], df["delta"], alpha=0.6, label='Delta', s=20)
    ax4.scatter(df["volatility"], df["gamma"] * 100, alpha=0.6, label='Gamma (×100)', s=20)
    ax4.scatter(df["volatility"], df["vega"], alpha=0.6, label='Vega', s=20)
    
    ax4.set_xlabel("Volatility")
    ax4.set_ylabel("Greek Value")
    ax4.set_title(f"{stock_symbol} - Greeks vs Volatility")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_historical_comparison(all_data: Dict[str, pd.DataFrame], fname: Path) -> None:
    """Plot comparison across all historical stocks."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = {'GME': 'red', 'ZM': 'blue', 'TSLA': 'green'}
    
    # Plot 1: Price evolution (normalized to start at 100)
    for symbol, df in all_data.items():
        normalized_prices = (df["spot_price"] / df["spot_price"].iloc[0]) * 100
        dates = pd.to_datetime(df["date"])
        ax1.plot(dates, normalized_prices, color=colors[symbol], 
                linewidth=2, label=symbol, marker='o', markersize=3)
    
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Normalized Price (Start = 100)")
    ax1.set_title("Stock Price Evolution Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility comparison
    for symbol, df in all_data.items():
        dates = pd.to_datetime(df["date"])
        ax2.plot(dates, df["volatility"], color=colors[symbol], 
                linewidth=2, label=symbol, marker='s', markersize=3)
    
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volatility")
    ax2.set_title("Volatility Evolution Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Delta comparison
    for symbol, df in all_data.items():
        dates = pd.to_datetime(df["date"])
        ax3.plot(dates, df["delta"], color=colors[symbol], 
                linewidth=2, label=symbol, marker='^', markersize=3)
    
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Delta")
    ax3.set_title("Delta Evolution Comparison (ATM Options)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Vega comparison
    for symbol, df in all_data.items():
        dates = pd.to_datetime(df["date"])
        ax4.plot(dates, df["vega"], color=colors[symbol], 
                linewidth=2, label=symbol, marker='d', markersize=3)
    
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Vega")
    ax4.set_title("Vega Evolution Comparison (ATM Options)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_week7_analysis() -> None:
    """Run Week 7 analysis with simulated historical data."""
    _ensure_directories()
    
    print("Generating simulated historical data for analysis...")
    
    # Generate simulated historical data for different stocks
    gme_data = _fetch_historical_gme_data()
    zoom_data = _fetch_historical_zoom_data()
    tsla_data = _fetch_historical_tsla_data()
    
    # Analyze Greeks for each stock
    print("Analyzing Greeks for historical data...")
    gme_df = _analyze_historical_greeks(gme_data)
    zoom_df = _analyze_historical_greeks(zoom_data)
    tsla_df = _analyze_historical_greeks(tsla_data)
    
    # Save data
    gme_df.to_csv(DATA_DIR / "historical_gme_data.csv", index=False)
    zoom_df.to_csv(DATA_DIR / "historical_zoom_data.csv", index=False)
    tsla_df.to_csv(DATA_DIR / "historical_tsla_data.csv", index=False)
    
    # Create individual analysis plots
    print("Creating individual analysis plots...")
    _plot_historical_analysis(gme_df, "GME", FIG_DIR / "fig_historical_gme_analysis.png")
    _plot_historical_analysis(zoom_df, "ZM", FIG_DIR / "fig_historical_zoom_analysis.png")
    _plot_historical_analysis(tsla_df, "TSLA", FIG_DIR / "fig_historical_tsla_analysis.png")
    
    # Create comparison plot
    print("Creating comparison plots...")
    all_data = {
        "GME": gme_df,
        "ZM": zoom_df,
        "TSLA": tsla_df
    }
    _plot_historical_comparison(all_data, FIG_DIR / "fig_historical_comparison.png")
    
    # Print summary statistics
    print("\n=== Historical Analysis Summary ===")
    for symbol, df in all_data.items():
        print(f"\n{symbol}:")
        print(f"  Price range: ${df['spot_price'].min():.2f} - ${df['spot_price'].max():.2f}")
        print(f"  Volatility range: {df['volatility'].min():.3f} - {df['volatility'].max():.3f}")
        print(f"  Delta range: {df['delta'].min():.3f} - {df['delta'].max():.3f}")
        print(f"  Gamma range: {df['gamma'].min():.6f} - {df['gamma'].max():.6f}")
        print(f"  Vega range: {df['vega'].min():.2f} - {df['vega'].max():.2f}")
    
    print("\nWeek 7 analysis completed!")
    print(f"Generated {len(gme_df)} GME records")
    print(f"Generated {len(zoom_df)} ZM records")
    print(f"Generated {len(tsla_df)} TSLA records")


if __name__ == "__main__":
    run_week7_analysis()
