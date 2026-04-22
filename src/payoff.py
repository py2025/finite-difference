import numpy as np


def call_payoff(S, K):
    """Terminal payoff for a European call: max(S - K, 0)."""
    return np.maximum(S - K, 0.0)


def put_payoff(S, K):
    """Terminal payoff for a European put: max(K - S, 0)."""
    return np.maximum(K - S, 0.0)
