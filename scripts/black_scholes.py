"""Convenience functions for Black-Scholes option pricing and Greeks.

All functions assume continuously compounded rates and annualized volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionParams:
    """Container for standard Black-Scholes inputs."""

    spot: float
    strike: float
    time: float  # time to expiry in years
    rate: float  # continuously compounded risk-free rate
    vol: float  # annualized volatility
    div: float = 0.0  # continuously compounded dividend / carry yield


def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal."""

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Probability density function for the standard normal."""

    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1(params: OptionParams) -> float:
    """Compute d1 appearing in Black-Scholes formulas."""

    if params.time <= 0 or params.vol <= 0:
        raise ValueError("Time to expiry and volatility must be positive.")

    numerator = log(params.spot / params.strike) + (
        params.rate - params.div + 0.5 * params.vol**2
    ) * params.time
    denominator = params.vol * sqrt(params.time)
    return numerator / denominator


def _d2(params: OptionParams) -> float:
    """Compute d2 appearing in Black-Scholes formulas."""

    return _d1(params) - params.vol * sqrt(params.time)


def price(params: OptionParams, option_type: OptionType = "call") -> float:
    """Return the Black-Scholes price for a European option."""

    d1 = _d1(params)
    d2 = _d2(params)
    spot = params.spot
    strike = params.strike
    disc = exp(-params.rate * params.time)
    carry = exp(-params.div * params.time)

    if option_type == "call":
        return carry * spot * _norm_cdf(d1) - disc * strike * _norm_cdf(d2)

    if option_type == "put":
        return disc * strike * _norm_cdf(-d2) - carry * spot * _norm_cdf(-d1)

    raise ValueError(f"Unsupported option type: {option_type}")


def delta(params: OptionParams, option_type: OptionType = "call") -> float:
    d1 = _d1(params)
    carry = exp(-params.div * params.time)

    if option_type == "call":
        return carry * _norm_cdf(d1)
    if option_type == "put":
        return carry * (_norm_cdf(d1) - 1.0)
    raise ValueError(f"Unsupported option type: {option_type}")


def gamma(params: OptionParams) -> float:
    d1 = _d1(params)
    carry = exp(-params.div * params.time)
    return carry * _norm_pdf(d1) / (params.spot * params.vol * sqrt(params.time))


def vega(params: OptionParams) -> float:
    d1 = _d1(params)
    carry = exp(-params.div * params.time)
    return carry * params.spot * _norm_pdf(d1) * sqrt(params.time) / 100.0


def theta(params: OptionParams, option_type: OptionType = "call") -> float:
    d1 = _d1(params)
    d2 = _d2(params)
    carry = exp(-params.div * params.time)
    disc = exp(-params.rate * params.time)

    first_term = -(
        carry * params.spot * _norm_pdf(d1) * params.vol / (2.0 * sqrt(params.time))
    )

    if option_type == "call":
        return (
            first_term
            + params.div * carry * params.spot * _norm_cdf(d1)
            - params.rate * disc * params.strike * _norm_cdf(d2)
        ) / 365.0

    if option_type == "put":
        return (
            first_term
            - params.div * carry * params.spot * _norm_cdf(-d1)
            + params.rate * disc * params.strike * _norm_cdf(-d2)
        ) / 365.0

    raise ValueError(f"Unsupported option type: {option_type}")


def greeks(params: OptionParams, option_type: OptionType = "call") -> dict[str, float]:
    """Return all primary Greeks for convenience."""

    return {
        "price": price(params, option_type),
        "delta": delta(params, option_type),
        "gamma": gamma(params),
        "theta": theta(params, option_type),
        "vega": vega(params),
    }


