import numpy as np
import pytest

from finite_difference import price_extended_heston_option


def test_extended_heston_constant_parameters_returns_positive_call_price():
    price, V, S, v = price_extended_heston_option(
        S0=100.0,
        K=100.0,
        T=1.0,
        v0=0.04,
        kappa=2.0,
        theta_v=0.04,
        sigma=0.30,
        rho=-0.70,
        r=0.03,
        q=0.00,
        M=80,
        L=50,
        N=80,
        S_max=400.0,
        v_max=1.0,
        option_type="call",
    )

    assert np.isfinite(price)
    assert price > 0.0
    assert V.shape == (81, 51)
    assert S.shape == (81,)
    assert v.shape == (51,)


def test_extended_heston_accepts_time_dependent_parameters():
    price, _, _, _ = price_extended_heston_option(
        S0=100.0,
        K=100.0,
        T=1.0,
        v0=0.04,
        kappa=lambda tau: 2.0 + 0.2 * tau,
        theta_v=lambda tau: 0.04 + 0.01 * tau,
        sigma=lambda tau: 0.30,
        rho=lambda tau: -0.70 + 0.05 * tau,
        r=lambda tau: 0.03,
        q=lambda tau: 0.00,
        M=80,
        L=50,
        N=80,
        S_max=400.0,
        v_max=1.0,
        option_type="call",
    )

    assert np.isfinite(price)
    assert price > 0.0


def test_extended_heston_invalid_option_type_raises():
    with pytest.raises(ValueError):
        price_extended_heston_option(
            S0=100.0,
            K=100.0,
            T=1.0,
            v0=0.04,
            kappa=2.0,
            theta_v=0.04,
            sigma=0.30,
            rho=-0.70,
            option_type="bad",
        )