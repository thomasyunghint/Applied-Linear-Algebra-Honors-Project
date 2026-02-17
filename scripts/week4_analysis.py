"""Week 4 analysis: Greeks heatmaps and approximation comparison.

Running this module will generate:
  - `data/greeks_surface.csv`
  - `figures/fig_delta_heatmap.png`
  - `figures/fig_gamma_heatmap.png`
  - `figures/fig_theta_heatmap.png`
  - `figures/fig_linear_vs_quadratic.png`
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .black_scholes import OptionParams, delta, gamma, price, theta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"


def _ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _generate_parameter_grid(
    spots: Iterable[float], maturities: Iterable[float], base_params: OptionParams
) -> pd.DataFrame:
    records = []
    for s in spots:
        for t in maturities:
            params = OptionParams(
                spot=s,
                strike=base_params.strike,
                time=t,
                rate=base_params.rate,
                vol=base_params.vol,
                div=base_params.div,
            )
            records.append({
                "spot": s,
                "time": t,
                "delta": delta(params),
                "gamma": gamma(params),
                "theta": theta(params),
            })
    return pd.DataFrame.from_records(records)


def _plot_heatmap(df: pd.DataFrame, value_col: str, fname: Path) -> None:
    pivot = df.pivot(index="time", columns="spot", values=value_col)
    spots = pivot.columns.values
    times = pivot.index.values
    values = pivot.values

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = "coolwarm"
    mesh = ax.pcolormesh(spots, times, values, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(value_col.capitalize())
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Time to Expiry (years)")
    ax.set_title(f"{value_col.capitalize()} Heatmap")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def _plot_linear_vs_quadratic(base: OptionParams, spots: np.ndarray, fname: Path) -> None:
    base_price = price(base)
    base_delta = delta(base)
    base_gamma = gamma(base)

    actual_prices = np.array([
        price(OptionParams(
            spot=s,
            strike=base.strike,
            time=base.time,
            rate=base.rate,
            vol=base.vol,
            div=base.div,
        ))
        for s in spots
    ])

    d_spot = spots - base.spot
    linear_approx = base_price + base_delta * d_spot
    quadratic_approx = base_price + base_delta * d_spot + 0.5 * base_gamma * d_spot**2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(spots, actual_prices, label="True price", linewidth=2)
    ax.plot(spots, linear_approx, label="Δ linear approx", linestyle="--")
    ax.plot(spots, quadratic_approx, label="Δ+Γ quadratic approx", linestyle=":")
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Option Price")
    ax.set_title("True vs Approximate Pricing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)


def run_week4_analysis() -> None:
    _ensure_directories()

    base_params = OptionParams(
        spot=100.0,
        strike=100.0,
        time=0.5,
        rate=0.02,
        vol=0.25,
        div=0.01,
    )

    spots = np.linspace(80.0, 120.0, 45)
    maturities = np.linspace(0.05, 1.5, 40)

    surface_df = _generate_parameter_grid(spots, maturities, base_params)
    surface_df.to_csv(DATA_DIR / "greeks_surface.csv", index=False)

    _plot_heatmap(surface_df, "delta", FIG_DIR / "fig_delta_heatmap.png")
    _plot_heatmap(surface_df, "gamma", FIG_DIR / "fig_gamma_heatmap.png")
    _plot_heatmap(surface_df, "theta", FIG_DIR / "fig_theta_heatmap.png")

    spot_slice = np.linspace(80.0, 120.0, 90)
    _plot_linear_vs_quadratic(base_params, spot_slice, FIG_DIR / "fig_linear_vs_quadratic.png")


if __name__ == "__main__":
    run_week4_analysis()


