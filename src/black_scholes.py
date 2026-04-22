import numpy as np
from scipy.stats import norm


def bs_call_price(S, K, r, sigma, T):
    """
    Black-Scholes price of a European call option.
    """

    # 1. Maturity case
    if T <= 0:
        return max(S - K, 0.0)

    # 2. Handle S = 0
    if S <= 0:
        return 0.0

    # 3. Handle sigma = 0 (deterministic forward price)
    if sigma <= 0:
        forward = S * np.exp(r * T)
        return np.exp(-r * T) * max(forward - K, 0.0)

    # 4. Standard Black-Scholes
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, r, sigma, T):
    """
    Black-Scholes price of a European put option.
    """

    # 1. Maturity case
    if T <= 0:
        return max(K - S, 0.0)

    # 2. Handle S = 0
    if S <= 0:
        return K * np.exp(-r * T)

    # 3. Handle sigma = 0
    if sigma <= 0:
        forward = S * np.exp(r * T)
        return np.exp(-r * T) * max(K - forward, 0.0)

    # 4. Standard Black-Scholes
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)