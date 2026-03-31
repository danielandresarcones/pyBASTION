"""
Regression component for pyBASTION.

Translates from R: sample_RC, init_Regression, fit_Regression (Regression.R)
"""

import numpy as np
from scipy.linalg import cho_factor, solve_triangular

from ._evol_params import dsp_initEvol0, dsp_sampleEvol0

__all__ = [
    "sample_RC",
    "init_Regression",
    "fit_Regression",
]


def sample_RC(y, X, beta_sigma_2, sigma_e, Td, rng=None):
    """
    Sample regression coefficients from their full conditional.

    Parameters
    ----------
    y : array (n,)
    X : array (n, Td)
    beta_sigma_2 : array (Td,)
        Prior variance for each coefficient.
    sigma_e : float
        Observation error scale.
    Td : int
        Number of regression coefficients.
    rng : numpy.random.Generator, optional

    Returns
    -------
    beta : array (Td,)
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    beta_sigma_2 = np.asarray(beta_sigma_2, dtype=np.float64).ravel()

    # A = X^T X + diag(1/beta_sigma_2)
    A = X.T @ X
    A[np.diag_indices_from(A)] += 1.0 / beta_sigma_2

    # linht = X^T (y / sigma_e)
    linht = X.T @ (y / sigma_e)

    # Cholesky: A = R^T R (upper triangular R)
    # beta = R^{-1} (R^{-T} linht + z) * sigma_e
    c, low = cho_factor(A)
    z = rng.standard_normal(Td)
    # solve_triangular with R^T (lower) then R (upper)
    v1 = solve_triangular(c, linht, lower=low, trans="T")
    beta = solve_triangular(c, v1 + z, lower=low) * sigma_e
    return beta


def init_Regression(data, X, obserror, rng=None):
    """
    Initialize regression parameters.

    Parameters
    ----------
    data : array (n,)
    X : array (n, p)
    obserror : dict with 'sigma_e'
    rng : numpy.random.Generator, optional

    Returns
    -------
    bParam : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    Td = X.shape[1]

    beta = sample_RC(data, X, np.ones(Td), obserror["sigma_e"], Td, rng=rng)
    beta_params = dsp_initEvol0(beta / obserror["sigma_e"])
    n_squared_sum = float(np.sum((beta / beta_params["sigma_w0"]) ** 2))

    return {
        "s_mu": X @ beta,
        "beta": beta,
        "Td": Td,
        "beta_params": beta_params,
        "n_squared_sum": n_squared_sum,
    }


def fit_Regression(data, X, bParam, obserror, rng=None):
    """
    Sample regression parameters (one Gibbs step).

    Parameters
    ----------
    data : array (n,)
    X : array (n, p)
    bParam : dict
    obserror : dict
    rng : numpy.random.Generator, optional

    Returns
    -------
    bParam : dict (updated)
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)

    beta = sample_RC(
        data,
        X,
        bParam["beta_params"]["sigma_w0"] ** 2,
        obserror["sigma_e"],
        bParam["Td"],
        rng=rng,
    )
    beta_params = dsp_sampleEvol0(
        beta / obserror["sigma_e"],
        bParam["beta_params"],
        commonSD=False,
        rng=rng,
    )
    n_squared_sum = float(np.sum((beta / beta_params["sigma_w0"]) ** 2))

    bParam["s_mu"] = X @ beta
    bParam["beta"] = beta
    bParam["beta_params"] = beta_params
    bParam["n_squared_sum"] = n_squared_sum
    return bParam
