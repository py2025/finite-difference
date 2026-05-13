import pytest

from finite_difference import suggest_sabr_grid_boundaries


def test_sabr_grid_boundaries_keep_old_minimum_rules():
    S_max, v_max = suggest_sabr_grid_boundaries(
        S0=100.0,
        K=100.0,
        T=1.0,
        alpha=0.2,
        beta=1.0,
        nu=0.5,
    )

    assert S_max >= 4.0 * 100.0
    assert v_max >= 5.0 * 0.2


def test_sabr_grid_boundaries_expand_for_long_maturity_and_high_vol_of_vol():
    _, short_v_max = suggest_sabr_grid_boundaries(
        S0=100.0,
        K=100.0,
        T=1.0,
        alpha=0.2,
        beta=1.0,
        nu=0.5,
    )

    _, long_v_max = suggest_sabr_grid_boundaries(
        S0=100.0,
        K=100.0,
        T=10.0,
        alpha=0.2,
        beta=1.0,
        nu=1.0,
    )

    assert long_v_max > short_v_max


def test_sabr_grid_boundaries_respect_beta_zero_price_scale():
    S_max, v_max = suggest_sabr_grid_boundaries(
        S0=350.0,
        K=350.0,
        T=30.0,
        alpha=100.0,
        beta=0.0,
        nu=0.5,
    )

    assert S_max > 4.0 * 350.0
    assert v_max > 5.0 * 100.0


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"S0": 0.0},
        {"T": -1.0},
        {"alpha": 0.0},
        {"nu": 0.0},
        {"beta": -0.1},
        {"beta": 1.1},
    ],
)
def test_sabr_grid_boundaries_validate_inputs(bad_kwargs):
    kwargs = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "alpha": 0.2,
        "beta": 0.5,
        "nu": 0.4,
    }
    kwargs.update(bad_kwargs)

    with pytest.raises(ValueError):
        suggest_sabr_grid_boundaries(**kwargs)