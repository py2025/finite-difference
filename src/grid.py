import numpy as np


def make_stock_grid(S_max, M):
    """Uniform stock price grid from 0 to S_max with M+1 points."""
    return np.linspace(0, S_max, M + 1)


def make_time_grid(T, N):
    """Uniform time grid from 0 to T with N+1 points."""
    return np.linspace(0, T, N + 1)
