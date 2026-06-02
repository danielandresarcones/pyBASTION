"""
MCMC samplers for pyBASTION.

Translates from R: fit_ASD, fit_ASD_SV (main_functions.R)

fit_ASD_SV is the main engine called by fit_BASTION.
It supports obsSV in {"const", "SV", "ASV"}.
"""

import copy
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import trange

from .evol_params import dsp_initSV, dsp_sampleSVparams, fit_paramsASV, init_paramsASV
from .obs_error import fit_sigmaE_0_m, fit_sigmaE_0_m_SV, init_sigmaE_0
from .outlier import fit_Outlier, init_Outlier
from .regression import fit_Regression, init_Regression
from .seasonality import fit_Sbeta, init_Sbeta
from .trend import fit_Tbeta, init_Tbeta
from .utils import robust_prod

__all__ = ["fit_ASD_SV"]


def _spawn_child_rngs(rng, n_children):
    """Create independent child RNGs from a parent generator."""
    entropy = rng.integers(0, np.iinfo(np.uint32).max, size=4, dtype=np.uint32)
    seed_sequence = np.random.SeedSequence(entropy)
    return [np.random.default_rng(child) for child in seed_sequence.spawn(n_children)]


def _sample_component(task):
    """Sample one model component from its full conditional."""
    kind = task["kind"]

    if kind == "Regression":
        param = fit_Regression(
            task["data"],
            task["X"],
            copy.deepcopy(task["param"]),
            task["obserror"],
            rng=task["rng"],
        )
        return {"kind": kind, "param": param, "beta": param["s_mu"]}

    if kind == "Seasonal":
        param = fit_Sbeta(
            task["data"],
            copy.deepcopy(task["param"]),
            task["obserror"],
            task["evol_error"],
            task["k"],
            rng=task["rng"],
        )
        error = (
            np.concatenate(
                [
                    param["s_evolParams23"]["sigma_w0"],
                    param["s_evolParams3k"]["sigma_wt"],
                    param["s_evolParamskT"]["sigma_wt"],
                ]
            )
            ** 2
        )
        return {"kind": kind, "param": param, "beta": param["s_mu"], "error": error}

    if kind == "Trend":
        param = fit_Tbeta(
            task["data"],
            copy.deepcopy(task["param"]),
            task["obserror"],
            task["evol_error"],
            task["D"],
            task["sparse"],
            rng=task["rng"],
        )
        error = np.concatenate(
            [
                param["s_evolParams0"]["sigma_w0"] ** 2,
                param["s_evolParams"]["sigma_wt"] ** 2,
            ]
        )
        return {"kind": kind, "param": param, "beta": param["s_mu"], "error": error}

    if kind == "Outlier":
        param = fit_Outlier(
            task["data"],
            copy.deepcopy(task["param"]),
            task["obserror"],
            rng=task["rng"],
        )
        return {
            "kind": kind,
            "param": param,
            "beta": param["s_mu"],
            "error": param["s_evolParams"]["sigma_wt"] ** 2,
        }

    raise ValueError(f"Unsupported component kind: {kind}")


def _init_streaming_stats(TT, n_beta_cols, n_err_cols, n_reg_cols=None):
    """Initialize streaming posterior statistics."""
    stats = {
        "count": 0,
        "sum": {
            "beta_combined": np.zeros(TT),
            "beta": np.zeros((TT, n_beta_cols)),
            "evol_sigma_t2": np.zeros((TT, n_err_cols)),
            "obs_sigma_t2": np.zeros(TT),
            "remainder": np.zeros(TT),
            "yhat": np.zeros(TT),
        },
        "sum_sq": {
            "beta_combined": np.zeros(TT),
            "beta": np.zeros((TT, n_beta_cols)),
            "evol_sigma_t2": np.zeros((TT, n_err_cols)),
            "obs_sigma_t2": np.zeros(TT),
            "remainder": np.zeros(TT),
            "yhat": np.zeros(TT),
        },
    }
    if n_reg_cols is not None:
        stats["sum"]["reg_coef"] = np.zeros(n_reg_cols)
        stats["sum_sq"]["reg_coef"] = np.zeros(n_reg_cols)
    return stats


def _accumulate_streaming_stats(stats, sample):
    """Accumulate sufficient statistics for posterior moments."""
    stats["count"] += 1
    for key, value in sample.items():
        stats["sum"][key] += value
        stats["sum_sq"][key] += value**2


def _streaming_moments(stats):
    """Convert streaming sufficient statistics into posterior moments."""
    count = stats["count"]
    moments = {}
    for key, total in stats["sum"].items():
        mean = total / count
        var = stats["sum_sq"][key] / count - mean**2
        moments[key] = {"mean": mean, "var": np.maximum(var, 0.0)}
    return moments


def _transform_posterior_moments(moments, norm_y, offset_y, col_trend):
    """Undo standardization for posterior moments."""
    transformed = copy.deepcopy(moments)
    transformed["beta_combined"]["mean"] *= norm_y
    transformed["beta_combined"]["var"] *= norm_y**2

    transformed["beta"]["mean"] *= norm_y
    transformed["beta"]["mean"][:, col_trend] += offset_y
    transformed["beta"]["var"] *= norm_y**2

    transformed["obs_sigma_t2"]["mean"] *= norm_y**2
    transformed["obs_sigma_t2"]["var"] *= norm_y**4
    return transformed


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
    parallel=False,
    rao_blackwell=False,
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
    parallel : bool
        If True, dispatch component updates through a worker pool while
        preserving the observation-error-first Gibbs block order.
    rao_blackwell : bool
        If True, accumulate streaming posterior sums and sums of squares
        and return posterior means/variances instead of full samples.
    verbose : bool
    rng : numpy.random.Generator, optional

    Returns
    -------
    mcmc_output : dict
        Contains ``samples`` by default, or ``moments`` and
        ``streaming_stats`` when ``rao_blackwell=True``.
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
        error_mat[:, ik] = (
            np.concatenate(
                [
                    sParam["s_evolParams23"]["sigma_w0"],
                    sParam["s_evolParams3k"]["sigma_wt"],
                    sParam["s_evolParamskT"]["sigma_wt"],
                ]
            )
            ** 2
        )

    # Trend
    tParam = init_Tbeta(y, obserror, evol_error, D, sparse, rng=rng)
    params_list["Trend"] = tParam
    beta_mat[:, col_trend] = tParam["s_mu"]
    error_mat[:, col_trend] = np.concatenate(
        [
            tParam["s_evolParams0"]["sigma_w0"] ** 2,
            tParam["s_evolParams"]["sigma_wt"] ** 2,
        ]
    )

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
        svParam = init_paramsASV(np.log(residuals**2 + 1e-300), rng=rng)
        obserror["sigma_et"] = svParam["sigma_wt"]
        obserror["sigma_e"] = 1.0

    # ── Storage arrays ──
    streaming_stats = None
    if rao_blackwell:
        streaming_stats = _init_streaming_stats(
            TT, n_beta_cols, n_err_cols, X.shape[1] if reg else None
        )
    else:
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
    max_parallel_jobs = nKs + 1 + int(reg) + int(Outlier)

    pool = (
        ThreadPool(processes=max_parallel_jobs)
        if parallel and max_parallel_jobs > 1
        else None
    )
    try:
        iterator = trange(nstot, desc="MCMC", disable=not verbose)
        for nsi in iterator:
            # ── Sample observation error ──
            if SVm:
                svParam = dsp_sampleSVparams(y - beta_mat.sum(axis=1), svParam, rng=rng)
                obserror["sigma_et"] = svParam["sigma_wt"]
            elif ASVm:
                residuals = y - beta_mat.sum(axis=1)
                svParam = fit_paramsASV(np.log(residuals**2 + 1e-300), svParam, rng=rng)
                obserror["sigma_et"] = svParam["sigma_wt"]
            else:
                obserror = fit_sigmaE_0_m(y, params_list, TT, rng=rng)

            if pool is None:
                # ── Regression ──
                if reg:
                    mask = np.ones(n_beta_cols, dtype=bool)
                    mask[col_reg] = False
                    resid = y - beta_mat[:, mask].sum(axis=1)
                    bParam = fit_Regression(
                        resid, X, params_list["Regression"], obserror, rng=rng
                    )
                    params_list["Regression"] = bParam
                    beta_mat[:, col_reg] = bParam["s_mu"]

                # ── Seasonality ──
                for ik, k in enumerate(Ks):
                    mask = np.ones(n_beta_cols, dtype=bool)
                    mask[ik] = False
                    resid = y - beta_mat[:, mask].sum(axis=1)

                    cn = f"Seasonal{k}"
                    sParam = fit_Sbeta(
                        resid, params_list[cn], obserror, evol_error, k, rng=rng
                    )
                    params_list[cn] = sParam
                    beta_mat[:, ik] = sParam["s_mu"]
                    error_mat[:, ik] = (
                        np.concatenate(
                            [
                                sParam["s_evolParams23"]["sigma_w0"],
                                sParam["s_evolParams3k"]["sigma_wt"],
                                sParam["s_evolParamskT"]["sigma_wt"],
                            ]
                        )
                        ** 2
                    )

                # ── Trend ──
                mask = np.ones(n_beta_cols, dtype=bool)
                mask[col_trend] = False
                tParam_data = y - beta_mat[:, mask].sum(axis=1)

                tParam = fit_Tbeta(
                    tParam_data, tParam, obserror, evol_error, D, sparse, rng=rng
                )
                params_list["Trend"] = tParam
                beta_mat[:, col_trend] = tParam["s_mu"]
                error_mat[:, col_trend] = np.concatenate(
                    [
                        tParam["s_evolParams0"]["sigma_w0"] ** 2,
                        tParam["s_evolParams"]["sigma_wt"] ** 2,
                    ]
                )

                # ── Outlier ──
                if Outlier:
                    mask = np.ones(n_beta_cols, dtype=bool)
                    mask[col_out_b] = False
                    resid = y - beta_mat[:, mask].sum(axis=1)

                    zParam = fit_Outlier(resid, zParam, obserror, rng=rng)
                    params_list["Outlier"] = zParam
                    beta_mat[:, col_out_b] = zParam["s_mu"]
                    if err_out_col is not None:
                        error_mat[4:, err_out_col] = (
                            zParam["s_evolParams"]["sigma_wt"] ** 2
                        )
            else:
                if reg:
                    reg_task = {
                        "kind": "Regression",
                        "data": y - (beta_mat.sum(axis=1) - beta_mat[:, col_reg]),
                        "X": X,
                        "param": params_list["Regression"],
                        "obserror": obserror,
                        "rng": rng,
                    }
                    reg_result = pool.apply(_sample_component, (reg_task,))
                    bParam = reg_result["param"]
                    params_list["Regression"] = bParam
                    beta_mat[:, col_reg] = reg_result["beta"]

                for ik, k in enumerate(Ks):
                    cn = f"Seasonal{k}"
                    seasonal_task = {
                        "kind": "Seasonal",
                        "index": ik,
                        "data": y - (beta_mat.sum(axis=1) - beta_mat[:, ik]),
                        "param": params_list[cn],
                        "obserror": obserror,
                        "evol_error": evol_error,
                        "k": k,
                        "rng": rng,
                    }
                    seasonal_result = pool.apply(_sample_component, (seasonal_task,))
                    params_list[cn] = seasonal_result["param"]
                    beta_mat[:, ik] = seasonal_result["beta"]
                    error_mat[:, ik] = seasonal_result["error"]

                trend_task = {
                    "kind": "Trend",
                    "data": y - (beta_mat.sum(axis=1) - beta_mat[:, col_trend]),
                    "param": tParam,
                    "obserror": obserror,
                    "evol_error": evol_error,
                    "D": D,
                    "sparse": sparse,
                    "rng": rng,
                }
                trend_result = pool.apply(_sample_component, (trend_task,))
                tParam = trend_result["param"]
                params_list["Trend"] = tParam
                beta_mat[:, col_trend] = trend_result["beta"]
                error_mat[:, col_trend] = trend_result["error"]

                if Outlier:
                    outlier_task = {
                        "kind": "Outlier",
                        "data": y - (beta_mat.sum(axis=1) - beta_mat[:, col_out_b]),
                        "param": zParam,
                        "obserror": obserror,
                        "rng": rng,
                    }
                    outlier_result = pool.apply(_sample_component, (outlier_task,))
                    zParam = outlier_result["param"]
                    params_list["Outlier"] = zParam
                    beta_mat[:, col_out_b] = outlier_result["beta"]
                    if err_out_col is not None:
                        error_mat[4:, err_out_col] = outlier_result["error"]

            # ── Save MCMC samples ──
            if nsi >= nburn:
                skipcount += 1
                if skipcount > nskip:
                    beta_combined = beta_mat.sum(axis=1)
                    sample = {
                        "obs_sigma_t2": obserror["sigma_et"] ** 2,
                        "beta": beta_mat.copy(),
                        "evol_sigma_t2": error_mat.copy(),
                        "beta_combined": beta_combined,
                        "remainder": y - beta_combined,
                        "yhat": beta_combined
                        + obserror["sigma_et"] * rng.standard_normal(TT),
                    }
                    if reg:
                        sample["reg_coef"] = bParam["beta"].copy()

                    if rao_blackwell:
                        _accumulate_streaming_stats(streaming_stats, sample)
                    else:
                        post_obs_sigma_t2[isave] = sample["obs_sigma_t2"]
                        post_s_beta[isave] = sample["beta"]
                        post_s_evol_sigma_t2[isave] = sample["evol_sigma_t2"]
                        post_beta_combined[isave] = sample["beta_combined"]
                        if reg:
                            post_reg[isave] = sample["reg_coef"]
                        post_remainder[isave] = sample["remainder"]
                        post_yhat[isave] = sample["yhat"]
                    isave += 1
                    skipcount = 0
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # ── Post-process: undo standardization ──
    metadata = {
        "parallel": parallel,
        "rao_blackwell": rao_blackwell,
        "nsave": isave,
    }
    if rao_blackwell:
        moments = _transform_posterior_moments(
            _streaming_moments(streaming_stats), norm_y, offset_y, col_trend
        )
        return {
            "moments": moments,
            "streaming_stats": streaming_stats,
            "metadata": metadata,
        }

    posterior_samples = {}
    posterior_samples["beta_combined"] = post_beta_combined * norm_y
    post_s_beta *= norm_y
    post_s_beta[:, :, col_trend] += offset_y
    posterior_samples["beta"] = post_s_beta
    posterior_samples["evol_sigma_t2"] = post_s_evol_sigma_t2
    posterior_samples["obs_sigma_t2"] = robust_prod(post_obs_sigma_t2, norm_y**2)
    posterior_samples["remainder"] = post_remainder
    if reg:
        posterior_samples["reg_coef"] = post_reg
    posterior_samples["yhat"] = post_yhat

    return {"samples": posterior_samples, "metadata": metadata}
