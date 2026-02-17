"""Week 5 analysis: portfolio sensitivity matrix and hedging."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .black_scholes import OptionParams, delta, gamma, theta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"


@dataclass
class Position:
    name: str
    quantity: float
    params: OptionParams


def _ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _greeks_vector(position: Position) -> np.ndarray:
    g_delta = delta(position.params)
    g_gamma = gamma(position.params)
    g_theta = theta(position.params)
    return np.array([g_delta, g_gamma, g_theta]) * position.quantity


def _build_sensitivity_matrix(positions: Sequence[Position]) -> np.ndarray:
    greeks = [_greeks_vector(pos) for pos in positions]
    return np.column_stack(greeks)


def _solve_hedge(A: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Solve min ||Ax - target||_2 via QR (np.linalg.lstsq uses SVD/QR under hood)
    sol, *_ = np.linalg.lstsq(A, target, rcond=None)
    return sol


def _plot_hedge_effect(before: np.ndarray, after: np.ndarray, names: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    index = np.arange(len(before))
    width = 0.35
    ax.bar(index - width / 2, before, width, label="Before Hedge")
    ax.bar(index + width / 2, after, width, label="After Hedge")
    ax.set_xticks(index)
    ax.set_xticklabels(names)
    ax.set_ylabel("Net Greek")
    ax.set_title("Portfolio Sensitivities Before/After Hedge")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_delta_before_after.png", dpi=300)
    plt.close(fig)


def run_week5_analysis() -> None:
    _ensure_directories()

    base_rate = 0.02
    div_yield = 0.01

    portfolio = [
        Position(
            name="Call_90_Jun",
            quantity=50,
            params=OptionParams(spot=100, strike=90, time=0.25, rate=base_rate, vol=0.2, div=div_yield),
        ),
        Position(
            name="Call_110_Sep",
            quantity=-30,
            params=OptionParams(spot=100, strike=110, time=0.5, rate=base_rate, vol=0.22, div=div_yield),
        ),
        Position(
            name="Put_95_Dec",
            quantity=20,
            params=OptionParams(spot=100, strike=95, time=0.75, rate=base_rate, vol=0.18, div=div_yield),
        ),
    ]

    hedge_instruments = [
        Position(
            name="Call_100_Jun",
            quantity=1.0,
            params=OptionParams(spot=100, strike=100, time=0.25, rate=base_rate, vol=0.21, div=div_yield),
        ),
        Position(
            name="Put_100_Sep",
            quantity=1.0,
            params=OptionParams(spot=100, strike=100, time=0.5, rate=base_rate, vol=0.23, div=div_yield),
        ),
        Position(
            name="Forward",
            quantity=1.0,
            params=OptionParams(spot=100, strike=100, time=0.5, rate=base_rate, vol=1e-8, div=div_yield),
        ),
    ]

    portfolio_matrix = _build_sensitivity_matrix(portfolio)
    hedge_matrix = _build_sensitivity_matrix(hedge_instruments)

    net_portfolio = portfolio_matrix.sum(axis=1)
    solution = _solve_hedge(hedge_matrix, -net_portfolio)
    net_after = net_portfolio + hedge_matrix @ solution

    sensitivity_df = pd.DataFrame(
        np.column_stack([portfolio_matrix, hedge_matrix]),
        index=["Delta", "Gamma", "Theta"],
        columns=[pos.name for pos in portfolio + hedge_instruments],
    )
    sensitivity_df.to_csv(DATA_DIR / "table_weights.csv")

    hedge_weights = pd.DataFrame(
        {
            "Instrument": [pos.name for pos in hedge_instruments],
            "Optimal Weight": solution,
        }
    )
    hedge_weights.to_csv(DATA_DIR / "hedge_weights.csv", index=False)

    _plot_hedge_effect(net_portfolio, net_after, ["Delta", "Gamma", "Theta"])


if __name__ == "__main__":
    run_week5_analysis()


