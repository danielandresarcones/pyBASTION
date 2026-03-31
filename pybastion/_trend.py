"""
Trend component for pyBASTION.

Translates from R: build_Q_trend, sampleTrend, init_Tbeta, fit_Tbeta (Trend.R)
"""

import numpy as np
from scipy import sparse
from ._utils import sample_from_precision, robust_prod
from ._evol_params import (
    dsp_initEvol0,
    dsp_sampleEvol0,
    dsp_initEvolParams,
    dsp_sampleEvolParams,
    initEvolParams_HS_sparse,
    sampleEvolParams_HS_sparse,
)

__all__ = [
    "build_Q_trend",
    "sampleTrend",
    "init_Tbeta",
    "fit_Tbeta",
]


def build_Q_trend(obs_sigma_t2, evol_sigma_t2, D, Td):
    """
    Build the precision matrix for the trend component.

    Parameters
    ----------
    obs_sigma_t2 : array (Td,)
        Observation variance at each time point.
    evol_sigma_t2 : array (Td,)
        Evolution variance at each time point.
    D : int
        Differencing order (1 or 2).
    Td : int
        Length of time series.

    Returns
    -------
    Q : scipy.sparse.csc_matrix (Td x Td)
        Symmetric positive-definite precision matrix.
    """
    obs_sigma_t2 = np.asarray(obs_sigma_t2, dtype=np.float64)
    evol_sigma_t2 = np.asarray(evol_sigma_t2, dtype=np.float64)

    if D == 1:
        # Main diagonal
        d0 = (1.0 / obs_sigma_t2
              + 1.0 / evol_sigma_t2
              + np.append(1.0 / evol_sigma_t2[1:], 0.0))
        # Super-diagonal (k=1)
        d1 = -1.0 / evol_sigma_t2[1:]

        Q = sparse.diags([d0, d1, d1], [0, 1, -1], shape=(Td, Td), format="csc")

    elif D == 2:
        # evol_sigma_t2[2:Td] = evol_sigma_t2[-(1:2)] in R (removing first two)
        ev_3on = evol_sigma_t2[2:]  # length Td-2

        d0 = (1.0 / obs_sigma_t2
              + 1.0 / evol_sigma_t2
              + np.concatenate([[0.0], 4.0 / ev_3on, [0.0]])
              + np.concatenate([1.0 / ev_3on, [0.0, 0.0]]))
        # d0 has length Td

        # Sub/super diagonal (k=1), length Td-1
        # R: c(-2/evol_sigma_t2[3], -2*(1/evol_sigma_t2[-(1:2)] + c(1/evol_sigma_t2[-(1:3)], 0)))
        # evol_sigma_t2[3] in R is evol_sigma_t2[2] in Python (0-indexed)
        d1 = np.concatenate([
            [-2.0 / evol_sigma_t2[2]],
            -2.0 * (1.0 / ev_3on + np.append(1.0 / evol_sigma_t2[3:], 0.0))
        ])
        # d1 has length Td-1

        # k=2 diagonal, length Td-2
        d2 = 1.0 / ev_3on

        Q = sparse.diags(
            [d2, d1, d0, d1, d2],
            [-2, -1, 0, 1, 2],
            shape=(Td, Td),
            format="csc",
        )

    elif D == 3:
        ev_4on = evol_sigma_t2[3:]  # evol_sigma_t2[4:Td] in R (1-based)

        d0 = (1.0 / evol_sigma_t2
              + np.concatenate([[0.0, 0.0], 9.0 / ev_4on, [0.0]])
              + np.concatenate([[0.0], 9.0 / ev_4on, [0.0, 0.0]])
              + np.concatenate([1.0 / ev_4on, [0.0, 0.0, 0.0]])
              + 1.0 / obs_sigma_t2)
        d1 = -(np.concatenate([3.0 / ev_4on, [0.0, 0.0]])
               + np.concatenate([[0.0], 9.0 / ev_4on, [0.0]])
               + np.concatenate([[0.0, 0.0], 3.0 / ev_4on]))
        d2 = (np.concatenate([3.0 / ev_4on, [0.0]])
              + np.concatenate([[0.0], 3.0 / ev_4on]))
        d3 = -1.0 / ev_4on

        Q = sparse.diags(
            [d3, d2, d1, d0, d1, d2, d3],
            [-3, -2, -1, 0, 1, 2, 3],
            shape=(Td, Td),
            format="csc",
        )
    else:
        raise ValueError("build_Q_trend requires D = 1, 2, or 3")

    return Q


def sampleTrend(data, obs_sigma_t2, evol_sigma_t2, D, Td, rng=None):
    """
    Sample trend from its full conditional (Gaussian, sparse precision).

    Parameters
    ----------
    data : array (Td,)
    obs_sigma_t2 : array (Td,)
    evol_sigma_t2 : array (Td,)
    D : int
    Td : int
    rng : numpy.random.Generator, optional

    Returns
    -------
    mu : array (Td,)
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    obs_sigma_t2 = np.asarray(obs_sigma_t2, dtype=np.float64).ravel()
    evol_sigma_t2 = np.asarray(evol_sigma_t2, dtype=np.float64).ravel()

    if D < 0 or D != int(D):
        raise ValueError("D must be a positive integer")
    if np.any(np.isnan(data)):
        raise ValueError("data cannot contain NAs")

    linht = data / obs_sigma_t2
    Q = build_Q_trend(obs_sigma_t2, evol_sigma_t2, D, Td)
    mu = sample_from_precision(Q, linht, rng=rng)
    return mu


def init_Tbeta(data, obserror, evol_error, D, sparse_flag, rng=None):
    """
    Initialize trend parameters.

    Parameters
    ----------
    data : array (T,)
    obserror : dict with 'sigma_e', 'sigma_et'
    evol_error : str
    D : int
    sparse_flag : bool
    rng : numpy.random.Generator, optional

    Returns
    -------
    tParam : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    Td = len(data)

    s_mu = sampleTrend(
        data,
        obs_sigma_t2=obserror["sigma_et"] ** 2,
        evol_sigma_t2=robust_prod(0.01, np.ones(Td)),
        D=D,
        Td=Td,
        rng=rng,
    )

    s_omega = np.diff(s_mu, n=D)
    s_mu0 = s_mu[:D]
    s_evolParams0 = dsp_initEvol0(s_mu0 / obserror["sigma_et"][:D])

    if sparse_flag:
        s_evolParams = initEvolParams_HS_sparse(
            s_omega / obserror["sigma_et"][D:], Td - D, rng=rng
        )
    else:
        s_evolParams = dsp_initEvolParams(
            s_omega / obserror["sigma_et"][D:], evol_error="HS"
        )

    n_squared_sum = np.sum(
        np.concatenate([s_mu[:D], s_omega]) ** 2
        / np.concatenate([s_evolParams0["sigma_w0"] ** 2, s_evolParams["sigma_wt"] ** 2])
    )

    return {
        "s_mu": s_mu,
        "s_evolParams0": s_evolParams0,
        "s_evolParams": s_evolParams,
        "Td": Td,
        "n_squared_sum": n_squared_sum,
        "colname": "Trend",
    }


def fit_Tbeta(data, tParam, obserror, evol_error, D, sparse_flag, rng=None):
    """
    Sample trend parameters (one Gibbs step).

    Parameters
    ----------
    data : array (T,)
    tParam : dict
    obserror : dict
    evol_error : str
    D : int
    sparse_flag : bool
    rng : numpy.random.Generator, optional

    Returns
    -------
    tParam : dict (updated)
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    obs_sigma_e = obserror["sigma_e"]

    evol_sigmas = np.concatenate([
        tParam["s_evolParams0"]["sigma_w0"],
        tParam["s_evolParams"]["sigma_wt"],
    ])
    evol_sigma_t2 = robust_prod(obs_sigma_e ** 2, evol_sigmas ** 2)

    s_mu = sampleTrend(
        data,
        obs_sigma_t2=obserror["sigma_et"] ** 2,
        evol_sigma_t2=evol_sigma_t2,
        D=D,
        Td=tParam["Td"],
        rng=rng,
    )

    s_omega = np.diff(s_mu, n=D)
    s_mu0 = s_mu[:D]

    s_evolParams0 = dsp_sampleEvol0(s_mu0 / obs_sigma_e, tParam["s_evolParams0"], rng=rng)

    if sparse_flag:
        s_evolParams = sampleEvolParams_HS_sparse(
            s_omega / obs_sigma_e, tParam["Td"] - D, tParam["s_evolParams"], rng=rng
        )
    else:
        s_evolParams = dsp_sampleEvolParams(
            omega=s_omega / obs_sigma_e,
            evolParams=tParam["s_evolParams"],
            sigma_e=1.0,
            evol_error=evol_error,
            rng=rng,
        )

    n_squared_sum = np.sum(
        np.concatenate([s_mu[:D], s_omega]) ** 2
        / np.concatenate([s_evolParams0["sigma_w0"] ** 2, s_evolParams["sigma_wt"] ** 2])
    )

    tParam["s_mu"] = s_mu
    tParam["s_evolParams0"] = s_evolParams0
    tParam["s_evolParams"] = s_evolParams
    tParam["n_squared_sum"] = n_squared_sum
    return tParam
