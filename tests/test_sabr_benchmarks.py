import pytest

from finite_difference import price_sabr_option


BENCHOP_SET_I = {
    "S0": 0.50,
    "T": 2.0,
    "alpha": 0.50,
    "beta": 0.50,
    "rho": 0.00,
    "nu": 0.40,
    "rows": [
        (0.434062, 0.221383),
        (0.500000, 0.193837),
        (0.575955, 0.166241),
    ],
}


BENCHOP_SET_II = {
    "S0": 0.07,
    "T": 10.0,
    "alpha": 0.40,
    "beta": 0.50,
    "rho": -0.60,
    "nu": 0.80,
    "rows": [
        (0.051023, 0.052450),
        (0.070000, 0.046586),
        (0.096036, 0.039291),
    ],
}


LEWIS_BETA_ONE_SET_28 = {
    "S0": 1000.0,
    "T": 10.0,
    "alpha": 0.20,
    "beta": 1.00,
    "rho": -0.75,
    "nu": 1.00,
    "S_max": 4000.0,
    "v_max": 2.0,
    "rows": [
        (500.0, 544.323),
        (750.0, 333.923),
        (1000.0, 149.925),
        (1250.0, 44.860),
        (1500.0, 19.108),
        (1750.0, 10.937),
        (2000.0, 7.200),
    ],
}


CHOI_SEO_BETA_ZERO_SET_32 = {
    "S0": 350.0,
    "T": 30.0,
    "alpha": 100.0,
    "beta": 0.00,
    "rho": -0.60,
    "nu": 0.50,
    "rows": [
        (0.0, 569.447800),
        (100.0, 481.519899),
        (200.0, 397.027964),
        (300.0, 318.228180),
        (350.0, 282.240986),
        (400.0, 249.614694),
        (500.0, 198.020488),
        (600.0, 165.130689),
        (700.0, 144.451013),
    ],
}


def _price_case(case, strike):
    return price_sabr_option(
        S0=case["S0"],
        K=strike,
        T=case["T"],
        r=0.0,
        alpha=case["alpha"],
        beta=case["beta"],
        rho=case["rho"],
        nu=case["nu"],
        M=80,
        L=40,
        N=80,
        S_max=case.get("S_max"),
        v_max=case.get("v_max"),
        option_type="call",
    )[0]


@pytest.mark.parametrize("case", [BENCHOP_SET_I, BENCHOP_SET_II])
def test_sabr_adi_matches_benchop_sets_i_and_ii(case):
    """Validate against the SABR ADI paper's parameter sets I and II."""
    for strike, expected in case["rows"]:
        actual = _price_case(case, strike)
        assert actual == pytest.approx(expected, abs=3e-3)


def test_sabr_adi_matches_pyfeng_beta_one_set_28():
    """Validate the beta = 1 benchmark smile from PyFENG sheet 28.

    A fixed S_max is important here. If S_max is chosen separately as 4*K for
    each strike, deep OTM strikes get an unnecessarily huge and coarse stock
    grid, which makes interpolation error dominate the test.
    """
    for strike, expected in LEWIS_BETA_ONE_SET_28["rows"]:
        actual = _price_case(LEWIS_BETA_ONE_SET_28, strike)
        assert actual == pytest.approx(expected, abs=1.5)


@pytest.mark.xfail(reason="beta=0 / normal SABR needs a stock grid allowing S < 0")
def test_sabr_adi_beta_zero_set_32_requires_negative_stock_grid():
    """Regression target for the suggested beta = 0 benchmark dataset.

    The current solver uses S in [0, S_max] with call boundary V(0, v, t)=0,
    which is incompatible with normal SABR. For example, the benchmark call
    value at K=0 is greater than spot because the process can become negative,
    but the current boundary forces that value toward spot.
    """
    for strike, expected in CHOI_SEO_BETA_ZERO_SET_32["rows"]:
        actual = _price_case(CHOI_SEO_BETA_ZERO_SET_32, strike)
        assert actual == pytest.approx(expected, abs=2.0)