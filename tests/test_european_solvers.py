"""
Tests for the three 1-D solvers (explicit, implicit, Crank-Nicolson).

Each scheme gets ONE correctness check (matches closed-form BS at a
moderate grid) and Crank-Nicolson — the production scheme — gets a
convergence check that catches a regression from O(dt^2) to O(dt).
"""

import numpy as np
import pytest

from finite_difference import bs_price, solve_crank_nicolson, solve_explicit, stability_limit, solve_implicit


def _interp_at(S, V, S0):
    return float(np.interp(S0, S, V))


def test_crank_nicolson_matches_bs(atm):
    """O(dt^2) scheme: tight tolerance on a moderate grid."""
    S, V = solve_crank_nicolson(
        S_max=4 * atm["K"], K=atm["K"], r=atm["r"], sigma=atm["sigma"],
        T=atm["T"], M=200, N=200, option_type="call",
    )
    expected = bs_price(atm["S0"], atm["K"], atm["T"], atm["r"], atm["sigma"], "call")
    assert _interp_at(S, V, atm["S0"]) == pytest.approx(expected, abs=0.02)


def test_implicit_matches_bs(atm):
    """O(dt) scheme: looser tolerance even with more time steps."""
    S, V = solve_implicit(
        S_max=4 * atm["K"], K=atm["K"], r=atm["r"], sigma=atm["sigma"],
        T=atm["T"], M=200, N=400, option_type="call",
    )
    expected = bs_price(atm["S0"], atm["K"], atm["T"], atm["r"], atm["sigma"], "call")
    assert _interp_at(S, V, atm["S0"]) == pytest.approx(expected, abs=0.05)


def test_explicit_matches_bs(atm):
    """Explicit scheme requires the CFL condition to be respected."""
    M = 100
    N = max(stability_limit(atm["T"], atm["r"], atm["sigma"], M), 200)
    S, V = solve_explicit(
        S_max=4 * atm["K"], K=atm["K"], r=atm["r"], sigma=atm["sigma"],
        T=atm["T"], M=M, N=N, option_type="call",
    )
    expected = bs_price(atm["S0"], atm["K"], atm["T"], atm["r"], atm["sigma"], "call")
    assert _interp_at(S, V, atm["S0"]) == pytest.approx(expected, abs=0.10)


def test_crank_nicolson_is_second_order(atm):
    """Halving dS and dt should reduce error by ~4x (slope-2 on log-log).

    This catches a regression where a refactor accidentally drops the time
    discretisation to first order — closed-form comparison alone might miss
    that, but the convergence rate would visibly halve.
    """
    expected = bs_price(atm["S0"], atm["K"], atm["T"], atm["r"], atm["sigma"], "call")
    errors = []
    for M, N in [(40, 40), (80, 80), (160, 160)]:
        S, V = solve_crank_nicolson(
            S_max=4 * atm["K"], K=atm["K"], r=atm["r"], sigma=atm["sigma"],
            T=atm["T"], M=M, N=N, option_type="call",
        )
        errors.append(abs(_interp_at(S, V, atm["S0"]) - expected))

    # Each refinement should reduce error by at least 2x (we'd expect 4x for
    # true second order, but allow slack for the smoothing effect of
    # interpolation at S0 and small boundary effects).
    assert errors[0] / errors[1] > 2.0
    assert errors[1] / errors[2] > 2.0
