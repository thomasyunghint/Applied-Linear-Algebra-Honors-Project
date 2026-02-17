"""Data fetching utility for real market data.

This module provides functionality to fetch real historical market data
from various sources. Currently supports:
- Yahoo Finance (via yfinance)
- Alpha Vantage (requires API key)
- Simulated realistic data based on historical patterns

Usage:
    from .data_fetcher import fetch_stock_data, StockDataRequest
    
    # Fetch real data
    request = StockDataRequest(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    data = fetch_stock_data(request)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import yfinance, but don't fail if not available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available. Using simulated data only.")


@dataclass
class StockDataRequest:
    symbol: str
    start_date: str
    end_date: str
    source: str = "yahoo"  # "yahoo", "alpha_vantage", "simulated"


@dataclass
class StockData:
    symbol: str
    dates: List[str]
    prices: List[float]
    volumes: List[float]
    high_prices: List[float]
    low_prices: List[float]
    open_prices: List[float]
    close_prices: List[float]
    source: str
    metadata: Dict


def _calculate_volatility(prices: List[float], window: int = 20) -> List[float]:
    """Calculate rolling volatility from price data."""
    if len(prices) < window:
        return [0.2] * len(prices)  # Default volatility
    
    volatilities = []
    for i in range(len(prices)):
        if i < window - 1:
            volatilities.append(0.2)  # Default for early days
        else:
            # Calculate returns
            returns = []
            for j in range(i - window + 1, i + 1):
                if j > 0:
                    returns.append(np.log(prices[j] / prices[j-1]))
            
            # Annualized volatility
            vol = np.std(returns) * np.sqrt(252)  # 252 trading days
            volatilities.append(max(vol, 0.05))  # Minimum 5% volatility
    
    return volatilities


def _fetch_yahoo_data(request: StockDataRequest) -> StockData:
    """Fetch data from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance not available. Install with: pip install yfinance")
    
    try:
        ticker = yf.Ticker(request.symbol)
        hist = ticker.history(start=request.start_date, end=request.end_date)
        
        if hist.empty:
            raise ValueError(f"No data found for {request.symbol}")
        
        dates = [date.strftime("%Y-%m-%d") for date in hist.index]
        prices = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()
        high_prices = hist['High'].tolist()
        low_prices = hist['Low'].tolist()
        open_prices = hist['Open'].tolist()
        close_prices = hist['Close'].tolist()
        
        volatilities = _calculate_volatility(prices)
        
        metadata = {
            "source": "yahoo_finance",
            "symbol": request.symbol,
            "data_points": len(prices),
            "date_range": f"{request.start_date} to {request.end_date}",
            "avg_volume": np.mean(volumes),
            "avg_volatility": np.mean(volatilities),
        }
        
        return StockData(
            symbol=request.symbol,
            dates=dates,
            prices=prices,
            volumes=volumes,
            high_prices=high_prices,
            low_prices=low_prices,
            open_prices=open_prices,
            close_prices=close_prices,
            source="yahoo",
            metadata=metadata
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Yahoo data for {request.symbol}: {str(e)}")


def _generate_simulated_data(request: StockDataRequest) -> StockData:
    """Generate realistic simulated data based on historical patterns."""
    start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
    
    # Generate trading days (weekdays only)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    n_days = len(dates)
    
    # Stock-specific parameters
    stock_params = {
        "AAPL": {"initial_price": 150, "volatility": 0.25, "drift": 0.0005, "avg_volume": 50_000_000},
        "TSLA": {"initial_price": 200, "volatility": 0.45, "drift": 0.001, "avg_volume": 80_000_000},
        "GME": {"initial_price": 25, "volatility": 0.60, "drift": 0.0002, "avg_volume": 30_000_000},
        "ZM": {"initial_price": 400, "volatility": 0.35, "drift": -0.001, "avg_volume": 20_000_000},
        "MCD": {"initial_price": 250, "volatility": 0.15, "drift": 0.0003, "avg_volume": 3_000_000},
    }
    
    params = stock_params.get(request.symbol, {
        "initial_price": 100,
        "volatility": 0.30,
        "drift": 0.0005,
        "avg_volume": 10_000_000
    })
    
    # Generate price series using geometric Brownian motion
    prices = [params["initial_price"]]
    for i in range(1, n_days):
        dt = 1/252  # Daily time step
        drift = params["drift"] * dt
        volatility = params["volatility"] * np.sqrt(dt)
        shock = np.random.normal(0, 1)
        
        new_price = prices[-1] * np.exp(drift + volatility * shock)
        prices.append(max(new_price, 1.0))  # Floor at $1
    
    # Generate OHLC data
    open_prices = [prices[0]]
    high_prices = []
    low_prices = []
    close_prices = prices.copy()
    
    for i in range(1, n_days):
        # Open price (close of previous day + small gap)
        gap = np.random.normal(0, prices[i] * 0.01)
        open_prices.append(max(prices[i] + gap, 1.0))
        
        # High and low prices
        daily_vol = prices[i] * 0.02
        high_prices.append(prices[i] + abs(np.random.normal(0, daily_vol)))
        low_prices.append(max(prices[i] - abs(np.random.normal(0, daily_vol)), 1.0))
    
    # Generate volume data
    volumes = []
    for i in range(n_days):
        base_volume = params["avg_volume"]
        volume_multiplier = 1 + np.random.normal(0, 0.5)
        volume = max(int(base_volume * volume_multiplier), 1000)
        volumes.append(volume)
    
    volatilities = _calculate_volatility(prices)
    
    metadata = {
        "source": "simulated",
        "symbol": request.symbol,
        "data_points": n_days,
        "date_range": f"{request.start_date} to {request.end_date}",
        "avg_volume": np.mean(volumes),
        "avg_volatility": np.mean(volatilities),
        "simulation_params": params,
    }
    
    return StockData(
        symbol=request.symbol,
        dates=dates,
        prices=prices,
        volumes=volumes,
        high_prices=high_prices,
        low_prices=low_prices,
        open_prices=open_prices,
        close_prices=close_prices,
        source="simulated",
        metadata=metadata
    )


def fetch_stock_data(request: StockDataRequest) -> StockData:
    """Fetch stock data from the specified source."""
    if request.source == "yahoo":
        try:
            return _fetch_yahoo_data(request)
        except Exception as e:
            print(f"Yahoo Finance failed: {e}")
            print("Falling back to simulated data...")
            return _generate_simulated_data(request)
    
    elif request.source == "simulated":
        return _generate_simulated_data(request)
    
    else:
        raise ValueError(f"Unsupported data source: {request.source}")


def save_stock_data(data: StockData, filepath: Path) -> None:
    """Save stock data to CSV file."""
    df = pd.DataFrame({
        "date": data.dates,
        "open": data.open_prices,
        "high": data.high_prices,
        "low": data.low_prices,
        "close": data.close_prices,
        "volume": data.volumes,
        "volatility": _calculate_volatility(data.prices),
    })
    
    df.to_csv(filepath, index=False)
    
    # Save metadata separately
    metadata_path = filepath.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(data.metadata, f, indent=2)


def load_stock_data(filepath: Path) -> StockData:
    """Load stock data from CSV file."""
    df = pd.read_csv(filepath)
    
    # Load metadata
    metadata_path = filepath.with_suffix('.json')
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return StockData(
        symbol=metadata.get("symbol", "UNKNOWN"),
        dates=df["date"].tolist(),
        prices=df["close"].tolist(),
        volumes=df["volume"].tolist(),
        high_prices=df["high"].tolist(),
        low_prices=df["low"].tolist(),
        open_prices=df["open"].tolist(),
        close_prices=df["close"].tolist(),
        source=metadata.get("source", "unknown"),
        metadata=metadata
    )


# Example usage and testing
if __name__ == "__main__":
    # Test with simulated data
    request = StockDataRequest(
        symbol="TSLA",
        start_date="2023-01-01",
        end_date="2023-03-31",
        source="simulated"
    )
    
    data = fetch_stock_data(request)
    print(f"Fetched {len(data.prices)} data points for {data.symbol}")
    print(f"Price range: ${min(data.prices):.2f} - ${max(data.prices):.2f}")
    print(f"Average volume: {np.mean(data.volumes):,.0f}")
    print(f"Average volatility: {np.mean(_calculate_volatility(data.prices)):.3f}")
