"""
Summary utilities for pyBASTION posterior output.

Translates from R: summarize_output (main_functions.R)
"""

import numpy as np
import pandas as pd

__all__ = ["summarize_output"]


def summarize_output(mcmc_output, y, Ks, cl, reg, outlier):
    """
    Compute posterior summary statistics from MCMC samples.

    Parameters
    ----------
    mcmc_output : dict
        Must contain 'beta' (nsave x T x ncols), 'remainder', 'obs_sigma_t2'.
    y : array (T,)
    Ks : list of int
    cl : float
        Credible level (e.g. 0.95).
    reg : bool
    outlier : bool

    Returns
    -------
    posterior_summary : dict
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    nKs = len(Ks)
    beta = mcmc_output["beta"]            # (nsave, T, ncols)
    remainder = mcmc_output["remainder"]  # (nsave, T)

    posterior_summary = {}

    # Posterior means
    beta_mean = np.mean(beta, axis=0)  # (T, ncols)
    remainder_mean = np.mean(remainder, axis=0)  # (T,)

    # Build column names
    colnames = [f"Seasonal{k}" for k in Ks] + ["Trend"]
    if reg and outlier:
        colnames += ["Regression", "Outlier"]
    elif outlier:
        colnames += ["Outlier"]
    elif reg:
        colnames += ["Regression"]

    p_means = {"y": y}
    for j, cn in enumerate(colnames):
        p_means[cn] = beta_mean[:, j]
    p_means["Remainder"] = remainder_mean
    posterior_summary["p_means"] = pd.DataFrame(p_means)

    alpha_d2 = (1 - cl) / 2
    lower = np.quantile(beta, alpha_d2, axis=0)       # (T, ncols)
    upper = np.quantile(beta, 1 - alpha_d2, axis=0)   # (T, ncols)

    # Trend summary
    trend_idx = nKs  # Trend is at column nKs (0-indexed)
    posterior_summary["Trend_sum"] = pd.DataFrame({
        "Mean": beta_mean[:, trend_idx],
        "CR_lower": lower[:, trend_idx],
        "CR_upper": upper[:, trend_idx],
    })

    # Seasonal summaries
    for ik, k in enumerate(Ks):
        nm = f"Seasonal{k}"
        posterior_summary[f"{nm}_sum"] = pd.DataFrame({
            "Mean": beta_mean[:, ik],
            "CR_lower": lower[:, ik],
            "CR_upper": upper[:, ik],
        })

    # Regression summary
    if reg:
        reg_idx = colnames.index("Regression")
        posterior_summary["Regression_sum"] = pd.DataFrame({
            "Mean": beta_mean[:, reg_idx],
            "CR_lower": lower[:, reg_idx],
            "CR_upper": upper[:, reg_idx],
        })

    # Outlier summary
    if outlier:
        out_idx = colnames.index("Outlier")
        posterior_summary["Outlier_sum"] = pd.DataFrame({
            "Mean": beta_mean[:, out_idx],
            "CR_lower": lower[:, out_idx],
            "CR_upper": upper[:, out_idx],
        })

    # Signal summary: Trend + all Seasonalities (joint posterior quantiles)
    # This correctly accounts for correlations between components,
    # unlike summing individual CIs.
    trend_samples = beta[:, :, nKs]         # (nsave, T)
    seasonal_samples = beta[:, :, :nKs].sum(axis=2)  # (nsave, T)
    signal_samples = trend_samples + seasonal_samples  # (nsave, T)
    posterior_summary["Signal_sum"] = pd.DataFrame({
        "Mean": np.mean(signal_samples, axis=0),
        "CR_lower": np.quantile(signal_samples, alpha_d2, axis=0),
        "CR_upper": np.quantile(signal_samples, 1 - alpha_d2, axis=0),
    })

    # Volatility summary
    obs_sigma_t = np.sqrt(mcmc_output["obs_sigma_t2"])  # (nsave, T)
    posterior_summary["Volatility"] = pd.DataFrame({
        "Mean": np.mean(obs_sigma_t, axis=0),
        "CR_lower": np.quantile(obs_sigma_t, alpha_d2, axis=0),
        "CR_upper": np.quantile(obs_sigma_t, 1 - alpha_d2, axis=0),
    })

    return posterior_summary
