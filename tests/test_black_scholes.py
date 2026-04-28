"""
Sanity tests for the closed-form Black-Scholes reference.

We test the *one* non-trivial identity it should satisfy (put-call parity) on a few
diverse parameter sets, plus the input validation branch.
"""

import math

import pytest

from src.core import bs_price


def test_put_call_parity(parity_case):
    """C - P = S - K e^{-rT} is a pure no-arbitrage identity.

    Failing this means signs, discounting, or one of the d1/d2 branches
    is wrong — and the failing parameter set will tell you which.
    """
    p = parity_case
    C = bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call")
    P = bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put")
    expected = p["S"] - p["K"] * math.exp(-p["r"] * p["T"])
    assert C - P == pytest.approx(expected, abs=1e-10)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        bs_price(100, 100, 1.0, 0.05, 0.2, "banana")
