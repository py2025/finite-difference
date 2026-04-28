"""
Tests for the American option solver.

Two genuinely strong tests here:

  1. American >= European  — early exercise can never reduce value.
     A sign error in the projection step or a missing max() call
     fails this immediately.

  2. American put = binomial tree — two completely independent
     algorithms (Crank-Nicolson + projection vs. Cox-Ross-Rubinstein
     lattice) on the same model.  They only agree if both are right.

Plus the textbook result that an American CALL on a non-dividend stock
equals the European call — an easy thing to break with a buggy projection.
"""

import numpy as np
import pytest

from src.solvers.american import binomial_american, solve_american
from src.solvers.crank_nicolson import solve_crank_nicolson


def _interp_at(S, V, S0):
    return float(np.interp(S0, S, V))


def test_american_put_dominates_european_put(atm):
    """V_american >= V_european pointwise — the right to exercise can
    only add value."""
    common = dict(S_max=4 * atm["K"], K=atm["K"], r=atm["r"], sigma=atm["sigma"],
                  T=atm["T"], M=200, N=200)
    S_a, V_a = solve_american(**common, option_type="put")
    S_e, V_e = solve_crank_nicolson(**common, option_type="put")
    # Same grid (same M, S_max), so we can compare pointwise.
    assert np.all(V_a + 1e-9 >= V_e)


@pytest.mark.parametrize("S0,sigma", [
    (100, 0.20),    # ATM
    (90, 0.30),     # ITM, high vol
    (110, 0.15),    # OTM, low vol
])
def test_american_put_matches_binomial(atm, S0, sigma):
    """The PDE solver and a binomial lattice are independent algorithms
    discretising the same underlying model.  Disagreement signals a real
    bug in one of them.

    S0 and sigma are swept here; K, r, T come from the shared ATM fixture.
    """
    K, r, T = atm["K"], atm["r"], atm["T"]

    # Crank-Nicolson + early-exercise projection
    S, V = solve_american(S_max=4 * K, K=K, r=r, sigma=sigma, T=T,
                          M=400, N=400, option_type="put")
    pde_price = _interp_at(S, V, S0)

    # CRR binomial tree as an independent reference
    binomial_price = binomial_american(S0, K, r, sigma, T, N=2000,
                                       option_type="put")

    assert pde_price == pytest.approx(binomial_price, abs=0.05)


def test_american_call_equals_european_call(atm):
    """Textbook result: with no dividends, it's never optimal to exercise
    an American call early, so its value equals the European call's."""
    common = dict(S_max=4 * atm["K"], K=atm["K"], r=atm["r"], sigma=atm["sigma"],
                  T=atm["T"], M=200, N=200)
    S_a, V_a = solve_american(**common, option_type="call")
    S_e, V_e = solve_crank_nicolson(**common, option_type="call")
    assert _interp_at(S_a, V_a, atm["S0"]) == pytest.approx(
        _interp_at(S_e, V_e, atm["S0"]), abs=0.02
    )
