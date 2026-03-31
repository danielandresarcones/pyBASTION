"""
Seasonality component for pyBASTION.

Translates from R: build_Q_season, sampleBeta_season, init_Sbeta, fit_Sbeta (Seasonality.R)
"""

import numpy as np
from scipy import sparse

from .evol_params import (
    dsp_initEvol0,
    dsp_initEvolParams,
    dsp_sampleEvol0,
    dsp_sampleEvolParams,
)
from .utils import robust_prod, sample_from_precision

__all__ = [
    "build_Q_season",
    "sampleBeta_season",
    "init_Sbeta",
    "fit_Sbeta",
]


def build_Q_season(obs_sigma_t2, evol_sigma_t2, Td, k):
    """
    Build the precision matrix for a seasonal component with period k.

    Parameters
    ----------
    obs_sigma_t2 : array (Td,) or None
        Observation variance. If None, only prior precision is built.
    evol_sigma_t2 : array (Td,)
        Evolution variance at each time point.
    Td : int
        Length of time series.
    k : int
        Seasonal period.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (Td x Td)
    """
    evol_sigma_t2 = np.asarray(evol_sigma_t2, dtype=np.float64).ravel()
    prec_prior = 1.0 / evol_sigma_t2

    # R indices (1-based) → Python (0-based)
    # i0_1 = 1:(k-3) in R => range(0, k-3) in Python => indices 0..k-4
    # i0_2 = 1:(Td-k) in R => range(0, Td-k)
    # i0_3 = 2:(k-2) in R => range(1, k-2) => indices 1..k-3

    diag0 = prec_prior.copy()

    # i0_1: R 1:(k-3) => Python 0:(k-3)
    for i in range(k - 3):
        diag0[i] += prec_prior[i + 2]

    # i0_2: R 1:(Td-k) => Python 0:(Td-k)
    for i in range(Td - k):
        diag0[i] += prec_prior[i + k]

    # i0_3: R 2:(k-2) => Python 1:(k-2)
    for i in range(1, k - 2):
        diag0[i] += 4 * prec_prior[i + 1]

    # Diagonal 1
    diag1 = np.zeros(Td - 1)
    for i in range(k - 3):
        diag1[i] -= 2 * prec_prior[i + 2]
    for i in range(1, k - 2):
        diag1[i] -= 2 * prec_prior[i + 1]

    # Diagonal 2
    diag2 = np.zeros(Td - 2)
    for i in range(k - 3):
        diag2[i] += prec_prior[i + 2]

    # Diagonal k
    diagk = -prec_prior[k:]  # length Td-k

    # Add obs variance to main diagonal
    if obs_sigma_t2 is not None:
        obs_sigma_t2 = np.asarray(obs_sigma_t2, dtype=np.float64).ravel()
        diag0 += 1.0 / obs_sigma_t2

    # Build the banded part
    diags = [diag0, diag1, diag1, diag2, diag2, diagk, diagk]
    offsets = [0, 1, -1, 2, -2, k, -k]
    Q = sparse.diags(diags, offsets, shape=(Td, Td), format="lil")

    # Add the dense (k-1) x (k-1) + cross blocks for the seasonal constraint
    # R code:
    # row_indices <- c(rep(1:(k-1), each = (k-1)), 1:(k-1), rep(k, k-1))
    # col_indices <- c(rep(1:(k-1), (k-1)), rep(k, k-1), 1:(k-1))
    # values <- rep(prec_prior[k], length(row_indices))
    # This is 1-indexed in R, so k means index k-1 in Python
    val = prec_prior[k - 1]  # prec_prior[k] in R (1-based)

    # (k-1)x(k-1) block + cross terms with row/col k
    for i in range(k - 1):
        for j in range(k - 1):
            Q[i, j] += val
        # cross with row k (0-indexed: k-1)
        Q[i, k - 1] += val
        Q[k - 1, i] += val

    Q = Q.tocsc()
    return Q


def sampleBeta_season(data, obs_sigma_t2, evol_sigma_t2, Td, k, rng=None):
    """
    Sample seasonal component from its full conditional.

    Parameters
    ----------
    data : array (Td,)
    obs_sigma_t2 : array (Td,)
    evol_sigma_t2 : array (Td,)
    Td : int
    k : int
    rng : numpy.random.Generator, optional

    Returns
    -------
    mu : array (Td,)
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    obs_sigma_t2 = np.asarray(obs_sigma_t2, dtype=np.float64).ravel()

    Q = build_Q_season(obs_sigma_t2, evol_sigma_t2, Td, k)
    linht = data / obs_sigma_t2
    mu = sample_from_precision(Q, linht, rng=rng)
    return mu


def init_Sbeta(data, obserror, evol_error, k, rng=None):
    """
    Initialize seasonal parameters.

    Parameters
    ----------
    data : array (T,)
    obserror : dict with 'sigma_e', 'sigma_et'
    evol_error : str
    k : int (seasonal period)
    rng : numpy.random.Generator, optional

    Returns
    -------
    sParam : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    Td = len(data)
    sigma_e = obserror["sigma_e"]

    s_mu = sampleBeta_season(
        data,
        obs_sigma_t2=obserror["sigma_et"] ** 2,
        evol_sigma_t2=sigma_e**2 * np.full(Td, 0.01),
        Td=Td,
        k=k,
        rng=rng,
    )

    # S 1:2  (first two values)
    s_evolParams23 = dsp_initEvol0(s_mu[:2] / sigma_e, commonSD=True)

    # S(3:k-1) : second differences of s_mu[0:k-1]
    # R: i1_km1 = 1:(k-1) => Python: 0:k-1
    omega_2 = np.diff(s_mu[: k - 1], n=2)
    s_evolParams3k = dsp_initEvolParams(omega_2 / sigma_e, evol_error="HS")

    # S k+1,..,T
    # R: c((s_mu[k] + sum(s_mu[i1_km1])), diff(s_mu, lag=k))
    # s_mu[k] in R is s_mu[k-1] in Python (0-indexed)
    # sum(s_mu[i1_km1]) = sum(s_mu[1:(k-1)]) in R = sum(s_mu[0:k-1]) in Python
    first_val = s_mu[k - 1] + np.sum(s_mu[: k - 1])
    lag_diff = s_mu[k:] - s_mu[: Td - k]  # diff with lag k
    omega_kT = np.concatenate([[first_val], lag_diff])
    s_evolParamskT = dsp_initEvolParams(omega_kT / sigma_e, evol_error="HS")

    # Normalized squared sum
    all_vals = np.concatenate([s_mu[:2], omega_2, omega_kT])
    all_sigmas = np.concatenate(
        [
            s_evolParams23["sigma_w0"],
            s_evolParams3k["sigma_wt"],
            s_evolParamskT["sigma_wt"],
        ]
    )
    n_squared_sum = np.sum((all_vals / all_sigmas) ** 2)

    return {
        "s_mu": s_mu,
        "s_evolParams23": s_evolParams23,
        "s_evolParams3k": s_evolParams3k,
        "s_evolParamskT": s_evolParamskT,
        "Td": Td,
        "n_squared_sum": n_squared_sum,
        "colname": f"Seasonal{k}",
    }


def fit_Sbeta(data, sParam, obserror, evol_error, k, rng=None):
    """
    Sample seasonal parameters (one Gibbs step).

    Parameters
    ----------
    data : array (T,)
    sParam : dict
    obserror : dict
    evol_error : str
    k : int
    rng : numpy.random.Generator, optional

    Returns
    -------
    sParam : dict (updated)
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    Td = sParam["Td"]
    obs_sigma_e = obserror["sigma_e"]

    evol_sigmas = np.concatenate(
        [
            sParam["s_evolParams23"]["sigma_w0"],
            sParam["s_evolParams3k"]["sigma_wt"],
            sParam["s_evolParamskT"]["sigma_wt"],
        ]
    )
    evol_sigma_t2 = robust_prod(obs_sigma_e**2, evol_sigmas**2)

    s_mu = sampleBeta_season(
        data,
        obs_sigma_t2=obserror["sigma_et"] ** 2,
        evol_sigma_t2=evol_sigma_t2,
        Td=Td,
        k=k,
        rng=rng,
    )

    # S 1:2
    s_evolParams23 = dsp_sampleEvol0(
        s_mu[:2] / obs_sigma_e,
        sParam["s_evolParams23"],
        commonSD=True,
        rng=rng,
    )

    # S(3:k-1)
    omega_2 = np.diff(s_mu[: k - 1], n=2)
    s_evolParams3k = dsp_sampleEvolParams(
        omega_2 / obs_sigma_e,
        evolParams=sParam["s_evolParams3k"],
        evol_error="HS",
        rng=rng,
    )

    # S k+1,..,T
    first_val = s_mu[k - 1] + np.sum(s_mu[: k - 1])
    lag_diff = s_mu[k:] - s_mu[: Td - k]
    omega_kT = np.concatenate([[first_val], lag_diff])
    s_evolParamskT = dsp_sampleEvolParams(
        omega_kT / obs_sigma_e,
        evolParams=sParam["s_evolParamskT"],
        evol_error="HS",
        rng=rng,
    )

    # Normalized squared sum
    all_vals = np.concatenate([s_mu[:2], omega_2, omega_kT])
    all_sigmas = np.concatenate(
        [
            s_evolParams23["sigma_w0"],
            s_evolParams3k["sigma_wt"],
            s_evolParamskT["sigma_wt"],
        ]
    )
    n_squared_sum = np.sum((all_vals / all_sigmas) ** 2)

    sParam["s_mu"] = s_mu
    sParam["s_evolParams23"] = s_evolParams23
    sParam["s_evolParams3k"] = s_evolParams3k
    sParam["s_evolParamskT"] = s_evolParamskT
    sParam["n_squared_sum"] = n_squared_sum
    return sParam
