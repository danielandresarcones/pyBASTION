"""
pyBASTION — Python implementation of BASTION
(Bayesian Adaptive Seasonality Trend Decomposition Incorporating Outliers and Noise)

Usage
-----
>>> from pybastion import fit_BASTION
>>> result = fit_BASTION(y, Ks=[7, 365], Outlier=True)
>>> result["summary"]["Trend_sum"]
"""

import numpy as np
from ._mcmc import fit_ASD_SV
from ._summary import summarize_output
from .datasets import load_airtraffic, load_NYelectricity

__version__ = "0.1.0"

__all__ = [
    "fit_BASTION",
    "load_airtraffic",
    "load_NYelectricity",
]


def fit_BASTION(
    y,
    Ks,
    X=None,
    Outlier=False,
    cl=0.95,
    sparse=False,
    obsSV="const",
    nchains=2,
    nsave=1000,
    nburn=1000,
    nskip=4,
    verbose=True,
    save_samples=False,
    seed=None,
):
    """
    Decompose a time series using BASTION.

    Parameters
    ----------
    y : array-like (T,)
        Time series observations.
    Ks : list of int
        Seasonal periods.
    X : array-like (T, p) or None
        Regression covariate matrix.
    Outlier : bool
        Whether to model outliers (default False).
    cl : float
        Credible level for intervals (default 0.95).
    sparse : bool
        Extra shrinkage on trend (default False).
    obsSV : str
        Observation error model: "const" or "SV" (default "const").
    nchains : int
        Number of MCMC chains (default 2).
    nsave : int
        Iterations to save per chain (default 1000).
    nburn : int
        Burn-in iterations (default 1000).
    nskip : int
        Thinning interval (default 4).
    verbose : bool
        Show progress bar (default True).
    save_samples : bool
        Return full posterior samples (default False).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Always contains 'summary'. If save_samples=True, also contains 'samples'.

    Notes
    -----
    The summary dict contains:
    - p_means : DataFrame with posterior means of y and all components
    - Trend_sum : DataFrame with Mean, CR_lower, CR_upper
    - SeasonalK_sum : DataFrame for each seasonal period K
    - Outlier_sum : DataFrame (if Outlier=True)
    - Volatility : DataFrame with Mean, CR_lower, CR_upper
    """
    y = np.asarray(y, dtype=np.float64).ravel()

    if not isinstance(Ks, list):
        raise ValueError("Ks needs to be a list")
    if obsSV not in ("const", "SV"):
        raise ValueError("obsSV needs to be either 'SV' or 'const'")

    reg = X is not None
    if reg:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X needs to be a 2D array")

    # Manage RNG for reproducibility across chains
    if seed is not None:
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(nchains)
        rngs = [np.random.default_rng(cs) for cs in child_seeds]
    else:
        ss = np.random.SeedSequence()
        child_seeds = ss.spawn(nchains)
        rngs = [np.random.default_rng(cs) for cs in child_seeds]

    combined_samples = None

    for i in range(nchains):
        if verbose:
            print(f"Chain {i + 1}")
        model = fit_ASD_SV(
            y=y,
            Ks=Ks,
            X=X,
            Outlier=Outlier,
            sparse=sparse,
            obsSV=obsSV,
            nsave=nsave,
            nburn=nburn,
            nskip=nskip,
            verbose=verbose,
            rng=rngs[i],
        )
        if i == 0:
            combined_samples = model["samples"]
        else:
            # Concatenate along the first (sample) axis
            for key in ["beta_combined", "obs_sigma_t2", "remainder", "yhat"]:
                combined_samples[key] = np.concatenate(
                    [combined_samples[key], model["samples"][key]], axis=0
                )
            # 3D arrays
            for key in ["beta", "evol_sigma_t2"]:
                combined_samples[key] = np.concatenate(
                    [combined_samples[key], model["samples"][key]], axis=0
                )
            if reg and "reg_coef" in model["samples"]:
                combined_samples["reg_coef"] = np.concatenate(
                    [combined_samples["reg_coef"], model["samples"]["reg_coef"]], axis=0
                )

    summary = summarize_output(
        mcmc_output=combined_samples,
        y=y,
        Ks=Ks,
        cl=cl,
        reg=reg,
        outlier=Outlier,
    )

    if save_samples:
        return {"summary": summary, "samples": combined_samples}
    else:
        return {"summary": summary}
