"""Leakage ceiling for interleaved (random or shuffled) validation of a stationary linear Gaussian series.

Setting. A flexible learner is trained on rows (features at t, target y[t+h]). Under random or shuffled
splitting the training set contains rows whose targets y[t+h-1], y[t+h+1], ... neighbour the evaluation target in
time, so the learner can interpolate the target from both sides. Under a chronological split with a purge gap it can
only extrapolate h steps ahead from the past.

For a stationary process with spectral density f(w) (Kolmogorov, Wiener) the two best-linear-error variances are

    one-step prediction   s2_pred = exp( 1/(2 pi) * int log f dw )
    interpolation (all other values known)   s2_int = ( 1/(2 pi) * int dw / f )^-1

and the h-step prediction error variance is s2_pred * sum_{j<h} psi_j^2 with psi the MA(inf) weights.
An ideal learner that uses two-sided context therefore reaches mean squared error s2_int while the honest
h-step forecaster has s2_h; the largest possible optimism of interleaved validation is

    ceiling_mse(h) = 1 - s2_int / s2_h ,    ceiling_mae(h) = 1 - sqrt(s2_int / s2_h)   (Gaussian errors).

For AR(1) with coefficient phi: s2_int = s2/(1+phi^2), s2_h = s2 (1-phi^(2h))/(1-phi^2), so
ceiling_mse(h) = 1 - (1-phi^2) / ((1+phi^2)(1-phi^(2h))). It rises with persistence and with the horizon.
Real learners do not reach the ceiling; the ratio of measured optimism to the ceiling is the fraction exploited.
"""
from __future__ import annotations

import numpy as np


def ma_weights(ar: np.ndarray, n: int) -> np.ndarray:
    """First n MA(inf) weights psi_0..psi_{n-1} of an AR process y_t = sum ar_k y_{t-k} + e_t."""
    ar = np.asarray(ar, float)
    psi = np.zeros(n)
    psi[0] = 1.0
    for j in range(1, n):
        k = min(j, len(ar))
        psi[j] = float(np.dot(ar[:k], psi[j - 1 :: -1][:k]))
    return psi


def spectral_density(ar: np.ndarray, sigma2: float = 1.0, n_freq: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Spectral density f(w) = sigma2 / |1 - sum ar_k e^{-ikw}|^2 on (0, pi]."""
    w = np.linspace(1e-6, np.pi, n_freq)
    k = np.arange(1, len(ar) + 1)
    denom = np.abs(1.0 - (np.asarray(ar)[None, :] * np.exp(-1j * np.outer(w, k))).sum(axis=1)) ** 2
    return w, sigma2 / np.maximum(denom, 1e-12)


def interpolation_variance(ar: np.ndarray, sigma2: float = 1.0) -> float:
    """Interpolation error variance (all other values known) of a stationary AR process."""
    w, f = spectral_density(ar, sigma2)
    return float(1.0 / (np.trapezoid(1.0 / f, w) / np.pi))


def prediction_variance(ar: np.ndarray, h: int, sigma2: float = 1.0) -> float:
    """h-step prediction error variance of an AR process with innovation variance sigma2."""
    return float(sigma2 * np.sum(ma_weights(ar, h) ** 2))


def leakage_ceiling(ar, h: int, sigma2: float = 1.0, metric: str = "mae") -> float:
    """Largest relative optimism (fraction, 0..1) of interleaved validation for horizon h."""
    r = interpolation_variance(np.asarray(ar, float), sigma2) / prediction_variance(np.asarray(ar, float), h, sigma2)
    r = float(np.clip(r, 0.0, 1.0))
    return 1.0 - (r if metric == "mse" else np.sqrt(r))
