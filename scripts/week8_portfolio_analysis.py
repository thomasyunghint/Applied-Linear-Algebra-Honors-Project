"""Week 8 analysis: Portfolio construction and risk metrics (no PCA).

Outputs (saved under `data/` and `figures/`):
  - data/returns_daily.csv
  - data/portfolio_metrics.csv
  - data/portfolio_weights.csv
  - figures/fig_cum_returns.png
  - figures/fig_rolling_vol.png
  - figures/fig_corr_heatmap.png
  - figures/fig_efficient_frontier.png
  - figures/fig_portfolio_weights.png

Tickers: GME, ZM, TSLA, MCD (simulated by default to avoid rate limits).
Switch SOURCE to "yahoo" later if real data is desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .data_fetcher import StockDataRequest, fetch_stock_data
except ImportError:
    from data_fetcher import StockDataRequest, fetch_stock_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_close_series(tickers: List[str], start: str, end: str, source: str = "simulated") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for symbol in tickers:
        req = StockDataRequest(symbol=symbol, start_date=start, end_date=end, source=source)
        data = fetch_stock_data(req)
        df = pd.DataFrame({"date": data.dates, symbol: data.prices})
        df["date"] = pd.to_datetime(df["date"])  # ensure proper dtype
        frames.append(df.set_index("date"))
    # Inner join on dates to keep common trading days
    closes = pd.concat(frames, axis=1, join="inner").sort_index()
    return closes


def _compute_returns(closes: pd.DataFrame) -> pd.DataFrame:
    # Use arithmetic returns for Sharpe consistency
    returns = closes.pct_change().dropna(how="any")
    return returns


def _annualize_return_from_series(series: pd.Series) -> float:
    # CAGR from start/end
    if len(series) < 2:
        return 0.0
    cumulative = (1.0 + series).prod()
    years = len(series) / 252.0
    return cumulative ** (1.0 / years) - 1.0


def _risk_metrics(returns: pd.DataFrame, rf_annual: float = 0.03) -> pd.DataFrame:
    rf_daily = rf_annual / 252.0
    metrics = []
    for col in returns.columns:
        r = returns[col].dropna()
        mean_d = r.mean()
        vol_d = r.std(ddof=1)
        ann_vol = vol_d * np.sqrt(252)
        ann_ret = _annualize_return_from_series(r)
        excess = r - rf_daily
        sharpe = (excess.mean() / (vol_d if vol_d > 0 else np.nan)) * np.sqrt(252)
        downside = r[r < 0]
        downside_std = downside.std(ddof=1)
        denom = downside_std if (downside_std is not None and not np.isnan(downside_std) and downside_std > 0) else np.nan
        sortino = (excess.mean() / denom) * np.sqrt(252)
        # Max drawdown
        cum = (1.0 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown = cum / rolling_max - 1.0
        max_dd = drawdown.min()
        metrics.append({
            "asset": col,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
        })
    return pd.DataFrame(metrics).set_index("asset")


def _min_variance_weights(cov: np.ndarray, allow_short: bool = False) -> np.ndarray:
    n = cov.shape[0]
    ones = np.ones(n)
    # Unconstrained global min-variance solution
    try:
        inv = np.linalg.pinv(cov)
        w = inv @ ones
        w = w / (ones @ inv @ ones)
        if not allow_short:
            # Simple non-negativity fix: clip and renormalize
            w = np.clip(w, 0.0, None)
            s = w.sum()
            w = w / s if s > 0 else np.ones(n) / n
        return w
    except Exception:
        return np.ones(n) / n


def _efficient_frontier(mean_d: np.ndarray, cov: np.ndarray, samples: int = 4000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(mean_d)
    rets = []
    vols = []
    weights = []
    for _ in range(samples):
        w = np.random.rand(n)
        w /= w.sum()
        port_ret_d = w @ mean_d
        port_vol_d = np.sqrt(w @ cov @ w)
        rets.append((1 + port_ret_d) ** 252 - 1)  # convert to annualized (approx via compounding)
        vols.append(port_vol_d * np.sqrt(252))
        weights.append(w)
    return np.array(rets), np.array(vols), np.array(weights)


def _plot_cumulative(returns: pd.DataFrame, extra: Dict[str, pd.Series], fname: Path) -> None:
    plt.figure(figsize=(10, 5))
    for col in returns.columns:
        cum = (1.0 + returns[col]).cumprod()
        plt.plot(cum.index, cum.values, label=col)
    for name, series in extra.items():
        cum = (1.0 + series).cumprod()
        plt.plot(cum.index, cum.values, label=name, linewidth=2.5)
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_rolling(returns: pd.DataFrame, series: Dict[str, pd.Series], fname: Path, window: int = 20) -> None:
    plt.figure(figsize=(10, 5))
    for name, s in series.items():
        roll_vol = s.rolling(window).std() * np.sqrt(252)
        plt.plot(roll_vol.index, roll_vol.values, label=f"{name} roll vol")
    plt.title(f"Rolling {window}-day Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_corr_heatmap(returns: pd.DataFrame, fname: Path) -> None:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(cax, ax=ax, label="Correlation")
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_efficient_frontier(rets_a: np.ndarray, vols_a: np.ndarray, ew_point: Tuple[float, float], mv_point: Tuple[float, float], fname: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(vols_a, rets_a, s=6, alpha=0.3, label="Random Portfolios")
    plt.scatter([ew_point[0]], [ew_point[1]], color="green", label="Equal Weight", zorder=5)
    plt.scatter([mv_point[0]], [mv_point[1]], color="red", label="Min Variance", zorder=6)
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier (sampled)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_weights_bar(weights: Dict[str, float], fname: Path) -> None:
    names = list(weights.keys())
    vals = [weights[k] for k in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, vals)
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_week8_portfolio_analysis() -> None:
    _ensure_dirs()

    # Parameters
    tickers = ["GME", "ZM", "TSLA", "MCD"]
    START = "2021-01-01"
    END = "2022-12-31"
    SOURCE = "simulated"  # change to "yahoo" when rate limits are not an issue

    closes = _fetch_close_series(tickers, START, END, source=SOURCE)
    returns = _compute_returns(closes)

    # Save daily returns
    returns.to_csv(DATA_DIR / "returns_daily.csv")

    # Basic metrics per asset
    metrics_assets = _risk_metrics(returns)

    # Portfolios
    n = len(tickers)
    ew = np.ones(n) / n
    mean_d = returns.mean().values
    cov_d = returns.cov().values

    mv = _min_variance_weights(cov_d, allow_short=False)

    # Series
    ret_ew = returns @ ew
    ret_mv = returns @ mv

    # Vol-targeted EW at 15% annual
    realized_vol_ew = ret_ew.std(ddof=1) * np.sqrt(252)
    target_vol = 0.15
    scale = target_vol / realized_vol_ew if realized_vol_ew > 1e-12 else 1.0
    ret_vt = ret_ew * scale

    # Portfolio metrics
    metrics_port = _risk_metrics(pd.DataFrame({
        "EW": ret_ew,
        "MinVar": ret_mv,
        "VolTargetEW": ret_vt,
    }))

    # Save metrics
    out_metrics = pd.concat([metrics_assets, metrics_port], axis=0)
    out_metrics.to_csv(DATA_DIR / "portfolio_metrics.csv")

    # Save weights
    weights_df = pd.DataFrame({
        "asset": tickers,
        "EqualWeight": ew,
        "MinVariance": mv,
    })
    weights_df.to_csv(DATA_DIR / "portfolio_weights.csv", index=False)

    # Efficient frontier sampling
    rets_a, vols_a, _ = _efficient_frontier(mean_d, cov_d, samples=4000)
    ew_point = (realized_vol_ew, (1 + ret_ew.mean()) ** 252 - 1)
    mv_point = (np.sqrt(mv @ cov_d @ mv) * np.sqrt(252), (1 + ret_mv.mean()) ** 252 - 1)

    # Plots
    _plot_cumulative(returns, {"EW": ret_ew, "MinVar": ret_mv, "VolTargetEW": ret_vt}, FIG_DIR / "fig_cum_returns.png")
    _plot_rolling(returns, {"EW": ret_ew, "MinVar": ret_mv, "VolTargetEW": ret_vt}, FIG_DIR / "fig_rolling_vol.png", window=20)
    _plot_corr_heatmap(returns, FIG_DIR / "fig_corr_heatmap.png")
    _plot_efficient_frontier(rets_a, vols_a, ew_point, mv_point, FIG_DIR / "fig_efficient_frontier.png")
    _plot_weights_bar({**{t: w for t, w in zip(tickers, mv)},}, FIG_DIR / "fig_portfolio_weights.png")

    print("Week 8 portfolio analysis completed.")
    print(f"Assets: {tickers}")
    print(f"Equal-weight annual vol: {realized_vol_ew:.2%}; scale to 15% => factor {scale:.2f}")


if __name__ == "__main__":
    run_week8_portfolio_analysis()
