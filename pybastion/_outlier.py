"""
Outlier component for pyBASTION.

Translates from R: sampleOutlier, init_Outlier, fit_Outlier (Outlier.R)
"""

import numpy as np

from ._evol_params import t_initEvolZeta_ps, t_sampleEvolZeta_ps

__all__ = [
    "sampleOutlier",
    "init_Outlier",
    "fit_Outlier",
]


def sampleOutlier(data, obs_sigma_t2, evol_sigma_t2, Td, rng=None):
    """
    Sample outlier component from its full conditional (element-wise Gaussian).

    Parameters
    ----------
    data : array (Td,)
    obs_sigma_t2 : array (Td,)
    evol_sigma_t2 : array (Td,)
    Td : int
    rng : numpy.random.Generator, optional

    Returns
    -------
    sample : array (Td,)
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    obs_sigma_t2 = np.asarray(obs_sigma_t2, dtype=np.float64).ravel()
    evol_sigma_t2 = np.asarray(evol_sigma_t2, dtype=np.float64).ravel()

    linht = data / obs_sigma_t2
    postSD = 1.0 / np.sqrt(1.0 / obs_sigma_t2 + 1.0 / evol_sigma_t2)
    postMean = linht * postSD**2
    return rng.normal(loc=postMean, scale=postSD)


def init_Outlier(data, obserror, rng=None):
    """
    Initialize outlier parameters. First 4 time points are fixed to zero.

    Parameters
    ----------
    data : array (T,)
    obserror : dict with 'sigma_e', 'sigma_et'
    rng : numpy.random.Generator, optional

    Returns
    -------
    zParam : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    Td = len(data) - 4  # outlier effective length
    data_5T = data[4:]  # skip first 4

    zeta_5T = sampleOutlier(
        data_5T,
        obs_sigma_t2=obserror["sigma_et"][4:] ** 2,
        evol_sigma_t2=0.01 * np.ones(Td),
        Td=Td,
        rng=rng,
    )

    s_evolParams = t_initEvolZeta_ps(zeta_5T / obserror["sigma_et"][4:], Td, rng=rng)
    n_squared_sum = np.sum((zeta_5T / s_evolParams["sigma_wt"]) ** 2)
    s_mu = np.concatenate([np.zeros(4), zeta_5T])

    return {
        "s_mu": s_mu,
        "s_evolParams": s_evolParams,
        "Td": Td,
        "n_squared_sum": n_squared_sum,
        "colname": "Outlier",
    }


def fit_Outlier(data, zParam, obserror, rng=None):
    """
    Sample outlier parameters (one Gibbs step).

    Parameters
    ----------
    data : array (T,)
    zParam : dict
    obserror : dict
    rng : numpy.random.Generator, optional

    Returns
    -------
    zParam : dict (updated)
    """
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float64).ravel()
    Td = zParam["Td"]
    obs_sigma_e = obserror["sigma_e"]

    zeta_5T = sampleOutlier(
        data[4:],
        obs_sigma_t2=obserror["sigma_et"][4:] ** 2,
        evol_sigma_t2=obs_sigma_e**2 * zParam["s_evolParams"]["sigma_wt"] ** 2,
        Td=Td,
        rng=rng,
    )

    s_evolParams = t_sampleEvolZeta_ps(
        zeta_5T / obs_sigma_e, Td, zParam["s_evolParams"], rng=rng
    )
    n_squared_sum = np.sum((zeta_5T / s_evolParams["sigma_wt"]) ** 2)
    s_mu = np.concatenate([np.zeros(4), zeta_5T])

    zParam["s_mu"] = s_mu
    zParam["s_evolParams"] = s_evolParams
    zParam["n_squared_sum"] = n_squared_sum
    return zParam
