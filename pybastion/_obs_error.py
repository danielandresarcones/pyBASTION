"""
Observation error samplers for pyBASTION.

Translates from R: init_sigmaE_0, fit_sigmaE_0_m, fit_sigmaE_0_m_SV (Remainder.R)
"""

import numpy as np
from ._utils import rinvgamma

__all__ = [
    "init_sigmaE_0",
    "fit_sigmaE_0_m",
    "fit_sigmaE_0_m_SV",
]


def init_sigmaE_0(data):
    """
    Initialize observation error parameters.

    Parameters
    ----------
    data : array (T,)

    Returns
    -------
    obserror : dict with 'sigma_e' (float) and 'sigma_et' (array)
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    T = len(data)
    sigma_e = float(np.std(data, ddof=1))
    sigma_et = np.full(T, sigma_e)
    return {"sigma_e": sigma_e, "sigma_et": sigma_et}


def fit_sigmaE_0_m(data, params_list, TT, a=0.0, b=0.0, rng=None):
    """
    Sample observation error variance (constant across time).

    Marginalizes over all components. Uses conjugate inverse-gamma posterior.

    Parameters
    ----------
    data : array (T,)
    params_list : dict of dicts
        Each value has 's_mu', 'Td', 'n_squared_sum'.
    TT : int
    a, b : float
        Prior hyperparameters.
    rng : numpy.random.Generator, optional

    Returns
    -------
    obserror : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    offset = data.copy()

    for param in params_list.values():
        offset = offset - param["s_mu"]
        a = a + param["Td"] / 2
        b = b + param["n_squared_sum"] / 2

    a = a + TT / 2
    b = b + np.sum(offset ** 2) / 2

    sigma_e = np.sqrt(rinvgamma(1, a, b, rng=rng)[0])
    sigma_et = np.full(TT, sigma_e)
    return {"sigma_e": sigma_e, "sigma_et": sigma_et}


def fit_sigmaE_0_m_SV(data, params_list, TT, svParam, a=0.0, b=0.0, rng=None):
    """
    Sample observation error variance when using SV model (scales by SV sigma_wt).

    Parameters
    ----------
    data : array (T,)
    params_list : dict of dicts
    TT : int
    svParam : dict with 'sigma_wt'
    a, b : float
    rng : numpy.random.Generator, optional

    Returns
    -------
    obserror : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    offset = data.copy()

    for param in params_list.values():
        offset = offset - param["s_mu"]
        a = a + param["Td"] / 2
        b = b + param["n_squared_sum"] / 2

    a = a + TT / 2
    b = b + np.sum((offset / svParam["sigma_wt"]) ** 2) / 2

    sigma_e = np.sqrt(rinvgamma(1, a, b, rng=rng)[0])
    sigma_et = np.full(TT, sigma_e)
    return {"sigma_e": sigma_e, "sigma_et": sigma_et}
