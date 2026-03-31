"""
MCMC samplers for pyBASTION.

Translates from R: fit_ASD, fit_ASD_SV (main_functions.R)

fit_ASD_SV is the main engine called by fit_BASTION.
It supports obsSV in {"const", "SV", "ASV"}.
"""

import numpy as np
from tqdm import trange

from ._obs_error import init_sigmaE_0, fit_sigmaE_0_m, fit_sigmaE_0_m_SV
from ._trend import init_Tbeta, fit_Tbeta
from ._seasonality import init_Sbeta, fit_Sbeta
from ._outlier import init_Outlier, fit_Outlier
from ._regression import init_Regression, fit_Regression
from ._evol_params import dsp_initSV, dsp_sampleSVparams, init_paramsASV, fit_paramsASV
from ._utils import robust_prod

__all__ = ["fit_ASD_SV"]


def fit_ASD_SV(
    y,
    Ks,
    X=None,
    Outlier=False,
    sparse=False,
    obsSV="const",
    nsave=1000,
    nburn=1000,
    nskip=4,
    verbose=True,
    rng=None,
):
    """
    Run one chain of the BASTION MCMC sampler (with SV normalization).

    This is the core MCMC engine. It standardizes y by mean and sd,
    runs the Gibbs sampler, and returns posterior samples.

    Parameters
    ----------
    y : array (T,)
    Ks : list of int
    X : array (T, p) or None
    Outlier : bool
    sparse : bool
    obsSV : str in {"const", "SV", "ASV"}
    nsave, nburn, nskip : int
    verbose : bool
    rng : numpy.random.Generator, optional

    Returns
    -------
    mcmc_output : dict with 'samples' key
    """
    if rng is None:
        rng = np.random.default_rng()

    D = 2
    evol_error = "HS"
    SVm = obsSV == "SV"
    ASVm = obsSV == "ASV"
    reg = X is not None

    if reg and X.ndim != 2:
        raise ValueError("X needs to be a 2D array")

    nKs = len(Ks)
    TT = len(y)

    # Interpolate NAs
    t01 = np.linspace(0, 1, TT)
    missing = np.isnan(y)
    if np.any(missing):
        valid = ~missing
        y = np.interp(t01, t01[valid], y[valid])

    # Standardize
    offset_y = np.mean(y)
    norm_y = np.std(y, ddof=1)
    y = (y - offset_y) / norm_y

    # ── Column layout ──
    # beta_mat columns: [Seasonal_1, ..., Seasonal_nKs, Trend, (Regression), (Outlier)]
    # error_mat columns: [Seasonal_1, ..., Seasonal_nKs, Trend, (Outlier)]
    n_beta_cols = nKs + 1
    n_err_cols = nKs + 1
    col_trend = nKs  # 0-indexed
    col_reg = None
    col_out_b = None

    colnames_b = [f"Seasonal{k}" for k in Ks] + ["Trend"]
    colnames_er = [f"Seasonal{k}" for k in Ks] + ["Trend"]

    if reg and Outlier:
        col_reg = nKs + 1
        col_out_b = nKs + 2
        n_beta_cols += 2
        n_err_cols += 1
        colnames_b += ["Regression", "Outlier"]
        colnames_er += ["Regression"]
    elif Outlier:
        col_out_b = nKs + 1
        n_beta_cols += 1
        n_err_cols += 1
        colnames_b += ["Outlier"]
        colnames_er += ["Outlier"]
    elif reg:
        col_reg = nKs + 1
        n_beta_cols += 1
        colnames_b += ["Regression"]

    beta_mat = np.zeros((TT, n_beta_cols))
    error_mat = np.full((TT, n_err_cols), np.nan)

    # ── Initialize parameters ──
    params_list = {}

    # Observation error
    obserror = init_sigmaE_0(y)

    # Seasonality
    for ik, k in enumerate(Ks):
        sParam = init_Sbeta(y, obserror, evol_error="HS", k=k, rng=rng)
        cn = sParam["colname"]
        params_list[cn] = sParam
        beta_mat[:, ik] = sParam["s_mu"]
        error_mat[:, ik] = np.concatenate([
            sParam["s_evolParams23"]["sigma_w0"],
            sParam["s_evolParams3k"]["sigma_wt"],
            sParam["s_evolParamskT"]["sigma_wt"],
        ]) ** 2

    # Trend
    tParam = init_Tbeta(y, obserror, evol_error, D, sparse, rng=rng)
    params_list["Trend"] = tParam
    beta_mat[:, col_trend] = tParam["s_mu"]
    error_mat[:, col_trend] = np.concatenate([
        tParam["s_evolParams0"]["sigma_w0"] ** 2,
        tParam["s_evolParams"]["sigma_wt"] ** 2,
    ])

    # Regression
    bParam = None
    if reg:
        bParam = init_Regression(y, X, obserror, rng=rng)
        params_list["Regression"] = bParam
        beta_mat[:, col_reg] = bParam["s_mu"]

    # Outlier
    zParam = None
    if Outlier:
        zParam = init_Outlier(y, obserror, rng=rng)
        params_list["Outlier"] = zParam
        beta_mat[:, col_out_b] = zParam["s_mu"]
        # error_mat for outlier: skip first 4 rows
        err_out_col = colnames_er.index("Outlier") if "Outlier" in colnames_er else None
        if err_out_col is not None:
            error_mat[4:, err_out_col] = zParam["s_evolParams"]["sigma_wt"] ** 2

    # SV initialization
    svParam = None
    if SVm:
        svParam = dsp_initSV(y - beta_mat.sum(axis=1), rng=rng)
        obserror["sigma_et"] = svParam["sigma_wt"]
        obserror["sigma_e"] = 1.0
    elif ASVm:
        residuals = y - beta_mat.sum(axis=1)
        svParam = init_paramsASV(np.log(residuals ** 2 + 1e-300), rng=rng)
        obserror["sigma_et"] = svParam["sigma_wt"]
        obserror["sigma_e"] = 1.0

    # ── Storage arrays ──
    post_obs_sigma_t2 = np.empty((nsave, TT))
    post_s_beta = np.empty((nsave, TT, n_beta_cols))
    post_s_evol_sigma_t2 = np.empty((nsave, TT, n_err_cols))
    post_remainder = np.empty((nsave, TT))
    post_beta_combined = np.empty((nsave, TT))
    post_yhat = np.empty((nsave, TT))
    post_reg = np.empty((nsave, X.shape[1])) if reg else None

    # ── MCMC loop ──
    nstot = nburn + (nskip + 1) * nsave
    skipcount = 0
    isave = 0

    iterator = trange(nstot, desc="MCMC", disable=not verbose)
    for nsi in iterator:
        # ── Sample observation error ──
        if SVm:
            svParam = dsp_sampleSVparams(y - beta_mat.sum(axis=1), svParam, rng=rng)
            obserror["sigma_et"] = svParam["sigma_wt"]
        elif ASVm:
            residuals = y - beta_mat.sum(axis=1)
            svParam = fit_paramsASV(np.log(residuals ** 2 + 1e-300), svParam, rng=rng)
            obserror["sigma_et"] = svParam["sigma_wt"]
        else:
            obserror = fit_sigmaE_0_m(y, params_list, TT, rng=rng)

        # ── Regression ──
        if reg:
            # Exclude regression column from sum
            mask = np.ones(n_beta_cols, dtype=bool)
            mask[col_reg] = False
            resid = y - beta_mat[:, mask].sum(axis=1)
            bParam = fit_Regression(resid, X, params_list["Regression"], obserror, rng=rng)
            params_list["Regression"] = bParam
            beta_mat[:, col_reg] = bParam["s_mu"]

        # ── Seasonality ──
        for ik, k in enumerate(Ks):
            # Exclude this seasonal column
            mask = np.ones(n_beta_cols, dtype=bool)
            mask[ik] = False
            resid = y - beta_mat[:, mask].sum(axis=1)

            cn = f"Seasonal{k}"
            sParam = fit_Sbeta(resid, params_list[cn], obserror, evol_error, k, rng=rng)
            params_list[cn] = sParam
            beta_mat[:, ik] = sParam["s_mu"]
            error_mat[:, ik] = np.concatenate([
                sParam["s_evolParams23"]["sigma_w0"],
                sParam["s_evolParams3k"]["sigma_wt"],
                sParam["s_evolParamskT"]["sigma_wt"],
            ]) ** 2

        # ── Trend ──
        mask = np.ones(n_beta_cols, dtype=bool)
        mask[col_trend] = False
        tParam_data = y - beta_mat[:, mask].sum(axis=1)

        tParam = fit_Tbeta(tParam_data, tParam, obserror, evol_error, D, sparse, rng=rng)
        params_list["Trend"] = tParam
        beta_mat[:, col_trend] = tParam["s_mu"]
        error_mat[:, col_trend] = np.concatenate([
            tParam["s_evolParams0"]["sigma_w0"] ** 2,
            tParam["s_evolParams"]["sigma_wt"] ** 2,
        ])

        # ── Outlier ──
        if Outlier:
            mask = np.ones(n_beta_cols, dtype=bool)
            mask[col_out_b] = False
            resid = y - beta_mat[:, mask].sum(axis=1)

            zParam = fit_Outlier(resid, zParam, obserror, rng=rng)
            params_list["Outlier"] = zParam
            beta_mat[:, col_out_b] = zParam["s_mu"]
            if err_out_col is not None:
                error_mat[4:, err_out_col] = zParam["s_evolParams"]["sigma_wt"] ** 2

        # ── Save MCMC samples ──
        if nsi >= nburn:
            skipcount += 1
            if skipcount > nskip:
                post_obs_sigma_t2[isave] = obserror["sigma_et"] ** 2
                post_s_beta[isave] = beta_mat
                post_s_evol_sigma_t2[isave] = error_mat
                beta_combined = beta_mat.sum(axis=1)
                post_beta_combined[isave] = beta_combined
                if reg:
                    post_reg[isave] = bParam["beta"]
                post_remainder[isave] = y - beta_combined
                post_yhat[isave] = beta_combined + obserror["sigma_et"] * rng.standard_normal(TT)
                isave += 1
                skipcount = 0

    # ── Post-process: undo standardization ──
    posterior_samples = {}
    posterior_samples["beta_combined"] = post_beta_combined * norm_y
    post_s_beta *= norm_y
    post_s_beta[:, :, col_trend] += offset_y
    posterior_samples["beta"] = post_s_beta
    posterior_samples["evol_sigma_t2"] = post_s_evol_sigma_t2
    posterior_samples["obs_sigma_t2"] = robust_prod(post_obs_sigma_t2, norm_y ** 2)
    posterior_samples["remainder"] = post_remainder
    if reg:
        posterior_samples["reg_coef"] = post_reg
    posterior_samples["yhat"] = post_yhat

    return {"samples": posterior_samples}
