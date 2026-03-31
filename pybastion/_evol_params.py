"""
Evolution parameter samplers for pyBASTION.

Translates from R:
  - dsp_initEvol0, dsp_sampleEvol0           (dsp_package.R)
  - dsp_initEvolParams, dsp_sampleEvolParams (dsp_package.R)
  - initEvolParams_HS, sampleEvolParams_HS   (Trend.R)
  - initEvolParams_HS_sparse, sampleEvolParams_HS_sparse (Trend.R)
  - dsp_initSV, dsp_sampleSVparams           (dsp_package.R)
  - sample_jfast, ncind, init_paramsASV, fit_paramsASV (dsp_package.R / Remainder.R)
"""

import numpy as np
from scipy.stats import median_abs_deviation, norm

from ._utils import rinvgamma

__all__ = [
    "dsp_initEvol0",
    "dsp_sampleEvol0",
    "dsp_initEvolParams",
    "dsp_sampleEvolParams",
    "initEvolParams_HS_sparse",
    "sampleEvolParams_HS_sparse",
    "t_initEvolZeta_ps",
    "t_sampleEvolZeta_ps",
    "dsp_initSV",
    "dsp_sampleSVparams",
    "sample_jfast",
    "init_paramsASV",
    "fit_paramsASV",
]


# ──────────────────────────────────────────────────────────────────────────────
# Evol0: initial-value variance parameters (half-Cauchy / parameter expansion)
# ──────────────────────────────────────────────────────────────────────────────


def dsp_initEvol0(mu0, commonSD=True):
    """Initialize evolution parameters for initial values."""
    mu0 = np.atleast_1d(np.asarray(mu0, dtype=np.float64)).ravel()
    p = len(mu0)

    if commonSD:
        sigma_w0 = np.full(p, np.mean(np.abs(mu0)))
    else:
        sigma_w0 = np.abs(mu0).copy()

    px_sigma_w0 = np.ones(p)
    sigma_00 = 1.0
    px_sigma_00 = 1.0

    return {
        "sigma_w0": sigma_w0,
        "px_sigma_w0": px_sigma_w0,
        "sigma_00": sigma_00,
        "px_sigma_00": px_sigma_00,
    }


def dsp_sampleEvol0(mu0, evolParams0, commonSD=False, A=1.0, rng=None):
    """Sample evolution parameters for initial values (Gibbs step)."""
    if rng is None:
        rng = np.random.default_rng()

    mu0 = np.atleast_1d(np.asarray(mu0, dtype=np.float64)).ravel()
    p = len(mu0)

    # Numerical stability offset
    mu02_small = np.any(mu0**2 < 1e-16)
    offset = mu02_small * max(1e-8, median_abs_deviation(mu0, scale=1.0) / 1e6)
    mu02 = mu0**2 + offset

    if commonSD:
        shape = p / 2 + 0.5
        rate = np.sum(mu02) / 2 + evolParams0["px_sigma_w0"][0]
        evolParams0["sigma_w0"] = np.full(
            p, 1.0 / np.sqrt(rng.gamma(shape, 1.0 / rate))
        )
        shape_px = 0.5 + 0.5
        rate_px = 1.0 / evolParams0["sigma_w0"][0] ** 2 + 1.0 / A**2
        evolParams0["px_sigma_w0"] = np.full(p, rng.gamma(shape_px, 1.0 / rate_px))
    else:
        # Distinct standard deviations
        shape = 0.5 + 0.5
        rate = mu02 / 2 + evolParams0["px_sigma_w0"]
        evolParams0["sigma_w0"] = 1.0 / np.sqrt(rng.gamma(shape, 1.0 / rate, size=p))
        # Distinct parameter expansion
        rate_px = (
            1.0 / evolParams0["sigma_w0"] ** 2 + 1.0 / evolParams0["sigma_00"] ** 2
        )
        evolParams0["px_sigma_w0"] = rng.gamma(0.5 + 0.5, 1.0 / rate_px, size=p)

        # Global standard deviation
        shape_g = p / 2 + 0.5
        rate_g = np.sum(evolParams0["px_sigma_w0"]) + evolParams0["px_sigma_00"]
        evolParams0["sigma_00"] = 1.0 / np.sqrt(rng.gamma(shape_g, 1.0 / rate_g))

        # Global parameter expansion
        rate_g_px = 1.0 / evolParams0["sigma_00"] ** 2 + 1.0 / A**2
        evolParams0["px_sigma_00"] = rng.gamma(0.5 + 0.5, 1.0 / rate_g_px)

    return evolParams0


# ──────────────────────────────────────────────────────────────────────────────
# Horseshoe (HS) evolution parameters for trend / seasonality
# ──────────────────────────────────────────────────────────────────────────────


def dsp_initEvolParams(omega, evol_error="HS"):
    """Initialize evolution error parameters (Horseshoe)."""
    omega = np.atleast_2d(np.asarray(omega, dtype=np.float64))
    if omega.ndim == 1:
        omega = omega.reshape(-1, 1)
    n, p = omega.shape

    if evol_error == "HS":
        tauLambdaj = 1.0 / (omega**2 + 1e-300)
        xiLambdaj = 1.0 / (2.0 * tauLambdaj)
        tauLambda = 1.0 / (2.0 * np.mean(xiLambdaj, axis=0))
        xiLambda = 1.0 / (tauLambda + 1.0)
        return {
            "sigma_wt": 1.0 / np.sqrt(tauLambdaj).ravel(),
            "tauLambdaj": tauLambdaj,
            "xiLambdaj": xiLambdaj,
            "tauLambda": np.atleast_1d(tauLambda),
            "xiLambda": np.atleast_1d(xiLambda),
        }
    elif evol_error == "NIG":
        sd_val = np.std(omega, axis=0, ddof=0)
        sigma_wt = np.outer(np.ones(n), sd_val)
        return {"sigma_wt": sigma_wt.ravel()}
    else:
        raise ValueError(f"Unsupported evol_error: {evol_error}")


def dsp_sampleEvolParams(omega, evolParams, sigma_e=1.0, evol_error="HS", rng=None):
    """Sample evolution error parameters (Gibbs step, Horseshoe)."""
    if rng is None:
        rng = np.random.default_rng()

    omega = np.atleast_2d(np.asarray(omega, dtype=np.float64))
    if omega.ndim == 1:
        omega = omega.reshape(-1, 1)
    n, p = omega.shape

    if evol_error == "HS":
        # Numerical offset
        hsOffset = np.zeros_like(omega)
        for j in range(p):
            col = omega[:, j]
            if np.any(col**2 < 1e-16):
                hsOffset[:, j] = max(1e-8, median_abs_deviation(col, scale=1.0) / 1e6)
        hsInput2 = omega**2 + hsOffset

        # Local scale parameters
        evolParams["tauLambdaj"] = rng.gamma(
            1.0, 1.0 / (evolParams["xiLambdaj"] + hsInput2 / 2)
        ).reshape(n, p)

        tauLambda_broadcast = np.outer(np.ones(n), evolParams["tauLambda"])
        evolParams["xiLambdaj"] = rng.gamma(
            1.0, 1.0 / (evolParams["tauLambdaj"] + tauLambda_broadcast)
        ).reshape(n, p)

        # Global scale parameters
        for j in range(p):
            shape = 0.5 + n / 2
            rate = np.sum(evolParams["xiLambdaj"][:, j]) + evolParams["xiLambda"][j]
            evolParams["tauLambda"][j] = rng.gamma(shape, 1.0 / rate)
            evolParams["xiLambda"][j] = rng.gamma(
                1.0, 1.0 / (evolParams["tauLambda"][j] + 1.0)
            )

        evolParams["sigma_wt"] = (1.0 / np.sqrt(evolParams["tauLambdaj"])).ravel()
        return evolParams

    elif evol_error == "NIG":
        for j in range(p):
            col = omega[:, j]
            shape = n / 2 + 0.01
            rate = np.sum(col**2) / 2 + 0.01
            sd_j = 1.0 / np.sqrt(rng.gamma(shape, 1.0 / rate))
            evolParams["sigma_wt"] = np.full(n, sd_j)
        return evolParams

    else:
        raise ValueError(f"Unsupported evol_error: {evol_error}")


# ──────────────────────────────────────────────────────────────────────────────
# Sparse Horseshoe (regularized) for trend with sparsity
# ──────────────────────────────────────────────────────────────────────────────


def initEvolParams_HS_sparse(omega, Td, tau=None, rng=None):
    """Initialize regularized (sparse) horseshoe parameters."""
    if rng is None:
        rng = np.random.default_rng()
    omega = np.asarray(omega, dtype=np.float64).ravel()
    if tau is None:
        tau = 1.0 / (1000 * Td)
    omega_norm = omega / tau
    x_lambda_t = np.full(Td, 100.0)
    lambda_2 = rinvgamma(Td, 1.0, 1.0 / x_lambda_t + omega_norm**2 / 2, rng=rng)
    sigma_wt = np.maximum(np.sqrt(lambda_2) * tau, 1e-8)
    return {"sigma_wt": sigma_wt, "tau": tau, "lambda_2": lambda_2}


def sampleEvolParams_HS_sparse(omega, Td, evolParams, rng=None):
    """Sample regularized (sparse) horseshoe parameters."""
    if rng is None:
        rng = np.random.default_rng()
    omega = np.asarray(omega, dtype=np.float64).ravel()
    tau = evolParams["tau"]
    lambda_2 = evolParams["lambda_2"]
    omega_norm = omega / tau
    x_lambda_t = rinvgamma(Td, 1.0, 1.0 + 1.0 / lambda_2, rng=rng)
    lambda_2 = rinvgamma(Td, 1.0, 1.0 / x_lambda_t + omega_norm**2 / 2, rng=rng)
    sigma_wt = np.maximum(np.sqrt(lambda_2) * tau, 1e-8)
    return {"sigma_wt": sigma_wt, "tau": tau, "lambda_2": lambda_2}


# ──────────────────────────────────────────────────────────────────────────────
# Outlier evolution parameters (heavy-tailed / parameter-expanded horseshoe)
# ──────────────────────────────────────────────────────────────────────────────


def t_initEvolZeta_ps(zeta, Td, rng=None):
    """Initialize outlier evolution parameters (parameter-expanded)."""
    if rng is None:
        rng = np.random.default_rng()
    zeta = np.asarray(zeta, dtype=np.float64).ravel()
    xi = rinvgamma(1, 0.5, 1.0, rng=rng)[0]
    tau_t2 = rinvgamma(1, 0.5, 1.0 / xi, rng=rng)[0]
    v = rinvgamma(Td, 0.5, 1.0 / tau_t2, rng=rng)
    lambda_t2 = rinvgamma(Td, 0.5, 1.0 / v, rng=rng)
    return {
        "xi": xi,
        "v": v,
        "tau_t2": tau_t2,
        "lambda_t2": lambda_t2,
        "sigma_wt": np.sqrt(lambda_t2),
    }


def t_sampleEvolZeta_ps(zeta, Td, evolParams, rng=None):
    """Sample outlier evolution parameters (Gibbs step)."""
    if rng is None:
        rng = np.random.default_rng()
    zeta = np.asarray(zeta, dtype=np.float64).ravel()
    hsInput2 = zeta**2

    evolParams["lambda_t2"] = rinvgamma(
        Td, 1.0, 1.0 / evolParams["v"] + hsInput2 / 2, rng=rng
    )
    evolParams["v"] = rinvgamma(
        Td, 1.0, 1.0 / evolParams["lambda_t2"] + 1.0 / evolParams["tau_t2"], rng=rng
    )
    evolParams["tau_t2"] = rinvgamma(
        1, (Td + 1) / 2, 1.0 / evolParams["xi"] + np.sum(1.0 / evolParams["v"]), rng=rng
    )[0]
    evolParams["xi"] = rinvgamma(1, 1.0, 1.0 + 1.0 / evolParams["tau_t2"], rng=rng)[0]
    evolParams["sigma_wt"] = np.maximum(np.sqrt(evolParams["lambda_t2"]), 1e-6)
    return evolParams


# ──────────────────────────────────────────────────────────────────────────────
# Stochastic Volatility (SV) model
# ──────────────────────────────────────────────────────────────────────────────


def dsp_initSV(omega, rng=None):
    """
    Initialize stochastic volatility parameters.
    Uses log-volatility AR(1) model.
    """
    if rng is None:
        rng = np.random.default_rng()
    omega = np.asarray(omega, dtype=np.float64).ravel()
    n = len(omega)

    # log-volatility
    ht = np.log(omega**2 + 0.0001)

    # Simple AR(1) initialization
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(ht, order=(1, 0, 0))
        res = model.fit()
        intercept = res.params[0]
        ar1 = res.params[1]
        sig = np.sqrt(res.sigma2)
    except Exception:
        intercept = np.mean(ht) / (1 - 0.8)
        ar1 = 0.8
        sig = 1.0

    svParams = np.array([intercept, ar1, sig])

    return {
        "sigma_wt": np.exp(ht / 2),
        "ht": ht,
        "svParams": svParams,
    }


def _sv_sample_one_step(omega_j, svParams_j, ht_j, rng):
    """
    Single-site Gibbs sampler for stochastic volatility.

    Model: y_t = exp(h_t/2) * eps_t,  eps_t ~ N(0,1)
           h_t = mu + phi*(h_{t-1} - mu) + sigma*eta_t,  eta_t ~ N(0,1)

    Uses Kim-Shephard-Chib (1998) mixture approximation.
    """
    n = len(omega_j)
    mu, phi, sigma = svParams_j

    # Ensure phi is in (-1, 1)
    phi = np.clip(phi, -0.999, 0.999)
    sigma = max(sigma, 1e-6)

    # ---- Sample h_t using mixture-of-normals approximation ----
    # log(y_t^2) = h_t + log(eps_t^2), where log(eps_t^2) ~ mixture of normals
    y_star = np.log(omega_j**2 + 1e-8)

    # 10-component mixture from Omori et al. (2007)
    m_st = np.array(
        [
            1.92677,
            1.34744,
            0.73504,
            0.02266,
            -0.85173,
            -1.97278,
            -3.46788,
            -5.55246,
            -8.68384,
            -14.65000,
        ]
    )
    v_st2 = np.array(
        [
            0.11265,
            0.17788,
            0.26768,
            0.40611,
            0.62699,
            0.98583,
            1.57469,
            2.54498,
            4.16591,
            7.33342,
        ]
    )
    q = np.array(
        [
            0.00609,
            0.04775,
            0.13057,
            0.20674,
            0.22715,
            0.18842,
            0.12047,
            0.05591,
            0.01575,
            0.00115,
        ]
    )

    # Sample mixture indicators
    residuals = y_star - ht_j
    log_probs = np.log(q[np.newaxis, :]) + norm.logpdf(
        residuals[:, np.newaxis],
        loc=m_st[np.newaxis, :],
        scale=np.sqrt(v_st2[np.newaxis, :]),
    )
    # Normalize
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)

    z = np.array([rng.choice(10, p=p_row) for p_row in probs])
    d_mean = m_st[z]
    d_var = v_st2[z]

    # ---- FFBS for h_t (Forward Filtering Backward Sampling) ----
    # y_star_t - d_mean_t = h_t + noise(d_var_t)
    obs = y_star - d_mean

    # Forward filter
    h_filt = np.zeros(n)
    P_filt = np.zeros(n)
    h_pred = np.zeros(n)
    P_pred = np.zeros(n)

    # Initial state
    h_pred[0] = mu  # unconditional mean
    P_pred[0] = sigma**2 / (1 - phi**2)

    for t in range(n):
        if t > 0:
            h_pred[t] = mu + phi * (h_filt[t - 1] - mu)
            P_pred[t] = phi**2 * P_filt[t - 1] + sigma**2

        # Update
        K = P_pred[t] / (P_pred[t] + d_var[t])
        h_filt[t] = h_pred[t] + K * (obs[t] - h_pred[t])
        P_filt[t] = (1 - K) * P_pred[t]

    # Backward sample
    ht_new = np.zeros(n)
    ht_new[n - 1] = rng.normal(h_filt[n - 1], np.sqrt(P_filt[n - 1]))

    for t in range(n - 2, -1, -1):
        # Backward smoothing
        J = P_filt[t] * phi / (phi**2 * P_filt[t] + sigma**2)
        h_smooth = h_filt[t] + J * (ht_new[t + 1] - mu - phi * (h_filt[t] - mu))
        P_smooth = P_filt[t] - J**2 * (phi**2 * P_filt[t] + sigma**2)
        P_smooth = max(P_smooth, 1e-12)
        ht_new[t] = rng.normal(h_smooth, np.sqrt(P_smooth))

    # ---- Sample mu, phi, sigma ----
    # Sample sigma^2 | h, mu, phi
    eta = ht_new[1:] - mu - phi * (ht_new[:-1] - mu)
    # sigma^2 ~ InvGamma
    shape_s = (n - 1) / 2 + 0.01
    rate_s = np.sum(eta**2) / 2 + 0.01
    sigma_new = np.sqrt(rinvgamma(1, shape_s, rate_s, rng=rng)[0])
    sigma_new = max(sigma_new, 1e-6)

    # Sample mu | h, phi, sigma
    hbar = ht_new[0] * (1 - phi) + np.sum(ht_new[1:] - phi * ht_new[:-1])
    # Prior: mu ~ N(0, 100)
    prior_var_mu = 100.0
    post_var = 1.0 / (
        (1 - phi) ** 2 / sigma_new**2 * 1
        + (n - 1) * (1 - phi) ** 2 / sigma_new**2
        + 1.0 / prior_var_mu
    )
    # Actually more careful:
    numer = (1 - phi) * (ht_new[0] + np.sum(ht_new[1:] - phi * ht_new[:-1]))
    denom = n * (1 - phi) ** 2 / sigma_new**2 + 1.0 / prior_var_mu
    post_var_mu = 1.0 / denom
    post_mean_mu = numer / sigma_new**2 * post_var_mu
    mu_new = rng.normal(post_mean_mu, np.sqrt(max(post_var_mu, 1e-12)))

    # Sample phi | h, mu, sigma using griddy Gibbs or Metropolis
    # Simple random walk Metropolis for phi
    phi_prop = phi + rng.normal(0, 0.05)
    if abs(phi_prop) < 0.999:
        # Log-likelihood ratio
        ll_curr = -np.sum((ht_new[1:] - mu_new - phi * (ht_new[:-1] - mu_new)) ** 2) / (
            2 * sigma_new**2
        )
        ll_prop = -np.sum(
            (ht_new[1:] - mu_new - phi_prop * (ht_new[:-1] - mu_new)) ** 2
        ) / (2 * sigma_new**2)
        # Prior on phi: Beta((1+phi)/2; 20, 1.5) => favor phi near 1
        # Simple uniform prior
        log_alpha = ll_prop - ll_curr
        if np.log(rng.uniform()) < log_alpha:
            phi_new = phi_prop
        else:
            phi_new = phi
    else:
        phi_new = phi

    return ht_new, np.array([mu_new, phi_new, sigma_new])


def dsp_sampleSVparams(omega, svParams_dict, rng=None):
    """Sample stochastic volatility parameters (full Gibbs step)."""
    if rng is None:
        rng = np.random.default_rng()

    omega = np.asarray(omega, dtype=np.float64).ravel()
    n = len(omega)

    ht_new, params_new = _sv_sample_one_step(
        omega, svParams_dict["svParams"], svParams_dict["ht"], rng
    )

    svParams_dict["svParams"] = params_new
    svParams_dict["ht"] = ht_new
    svParams_dict["sigma_wt"] = np.exp(ht_new / 2)
    # Cap at 1e3
    svParams_dict["sigma_wt"] = np.minimum(svParams_dict["sigma_wt"], 1e3)

    return svParams_dict


# ──────────────────────────────────────────────────────────────────────────────
# Approximate Stochastic Volatility (ASV) helpers
# ──────────────────────────────────────────────────────────────────────────────

# 10-component mixture from Omori, Chib, Shephard, Nakajima (2007)
_M_ST = np.array(
    [
        1.92677,
        1.34744,
        0.73504,
        0.02266,
        -0.85173,
        -1.97278,
        -3.46788,
        -5.55246,
        -8.68384,
        -14.65000,
    ]
)
_V_ST2 = np.array(
    [
        0.11265,
        0.17788,
        0.26768,
        0.40611,
        0.62699,
        0.98583,
        1.57469,
        2.54498,
        4.16591,
        7.33342,
    ]
)
_Q_MIX = np.array(
    [
        0.00609,
        0.04775,
        0.13057,
        0.20674,
        0.22715,
        0.18842,
        0.12047,
        0.05591,
        0.01575,
        0.00115,
    ]
)


def _ncind(y, mu, sig, q, rng):
    """Sample mixture indicator for a single observation."""
    log_w = np.log(q + 1e-300) + norm.logpdf(y, loc=mu, scale=sig)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.argmax(q * norm.pdf(y, loc=mu, scale=sig))
    w /= w_sum
    return rng.choice(len(q), p=w)


def sample_jfast(T, obs=None, rng=None):
    """
    Sample mixture indicators for log-chi-squared approximation.
    Returns dict with 'mean' and 'var' arrays of length T.
    """
    if rng is None:
        rng = np.random.default_rng()

    if obs is None:
        z = rng.choice(10, size=T, replace=True, p=_Q_MIX)
    else:
        obs = np.asarray(obs, dtype=np.float64).ravel()
        sig = np.sqrt(_V_ST2)
        z = np.array([_ncind(obs[i], _M_ST, sig, _Q_MIX, rng) for i in range(T)])

    return {"mean": _M_ST[z], "var": _V_ST2[z]}


def init_paramsASV(data, D=2, rng=None):
    """Initialize approximate stochastic volatility parameters."""
    if rng is None:
        rng = np.random.default_rng()

    # Import here to avoid circular imports
    from ._trend import sampleTrend

    data = np.asarray(data, dtype=np.float64).ravel()
    Td = len(data)
    s_p_error_term = sample_jfast(Td, rng=rng)
    s_mu = sampleTrend(
        data - s_p_error_term["mean"],
        obs_sigma_t2=s_p_error_term["var"],
        evol_sigma_t2=0.01 * np.ones(Td),
        D=D,
        Td=Td,
        rng=rng,
    )
    s_omega = np.diff(s_mu, n=D)
    s_mu0 = s_mu[:D]
    s_evolParams0 = dsp_initEvol0(s_mu0)
    s_evolParams = dsp_initEvolParams(s_omega, "HS")

    return {
        "s_p_error_term": s_p_error_term,
        "s_mu": s_mu,
        "sigma_wt": np.exp(s_mu / 2),
        "s_evolParams0": s_evolParams0,
        "s_evolParams": s_evolParams,
        "Td": Td,
    }


def fit_paramsASV(data, sParams, D=2, rng=None):
    """Sample approximate stochastic volatility parameters (Gibbs step)."""
    if rng is None:
        rng = np.random.default_rng()

    from ._trend import sampleTrend

    data = np.asarray(data, dtype=np.float64).ravel()
    Td = sParams["Td"]

    s_p_error_term = sample_jfast(Td, obs=data - sParams["s_mu"], rng=rng)
    s_mu = sampleTrend(
        data - s_p_error_term["mean"],
        obs_sigma_t2=s_p_error_term["var"],
        evol_sigma_t2=np.concatenate(
            [
                sParams["s_evolParams0"]["sigma_w0"] ** 2,
                sParams["s_evolParams"]["sigma_wt"] ** 2,
            ]
        ),
        D=D,
        Td=Td,
        rng=rng,
    )
    s_omega = np.diff(s_mu, n=D)
    s_mu0 = s_mu[:D]
    s_evolParams0 = dsp_sampleEvol0(s_mu0, sParams["s_evolParams0"], rng=rng)
    s_evolParams = dsp_sampleEvolParams(
        omega=s_omega,
        evolParams=sParams["s_evolParams"],
        sigma_e=1.0,
        evol_error="HS",
        rng=rng,
    )

    sParams["s_p_error_term"] = s_p_error_term
    sParams["s_mu"] = s_mu
    sParams["s_evolParams0"] = s_evolParams0
    sParams["s_evolParams"] = s_evolParams
    sParams["sigma_wt"] = np.exp(s_mu / 2)
    return sParams
