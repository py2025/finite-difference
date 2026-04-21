import numpy as np


def call_boundaries(S_max, K, r, tau):
    """
    Boundary values for a European call at a given time-to-expiry tau.
    V(0, t)   = 0
    V(S_max, t) = S_max - K * exp(-r * tau)
    """
    lower = 0.0
    upper = S_max - K * np.exp(-r * tau)
    return lower, upper


def put_boundaries(S_max, K, r, tau):
    """
    Boundary values for a European put at a given time-to-expiry tau.
    V(0, t)   = K * exp(-r * tau)
    V(S_max, t) = 0
    """
    lower = K * np.exp(-r * tau)
    upper = 0.0
    return lower, upper
