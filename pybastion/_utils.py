"""
Utility functions for pyBASTION: robust arithmetic, sparse Cholesky, etc.

Translates from R: robust_cholesky, robust_prod, robust_div (main_functions.R)
"""

import numpy as np
import warnings
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import cho_factor, cho_solve

__all__ = [
    "robust_cholesky_solve",
    "robust_prod",
    "robust_div",
    "sample_from_precision",
]


def robust_prod(a, b):
    """Numerically stable product via log-space: exp(log(a) + log(b))."""
    return np.exp(np.log(a) + np.log(b))


def robust_div(a, b):
    """Numerically stable division via log-space: exp(log(a) - log(b))."""
    return np.exp(np.log(a) - np.log(b))


def _stabilize_precision(Q_dense):
    """
    Fix rows with negative row-sums (numerically unstable precision matrix).
    Mirrors R's robust_cholesky error-handler.
    """
    rowsums = Q_dense.sum(axis=1)
    sus = np.where(rowsums < 0)[0]
    for i in sus:
        rs = rowsums[i]
        while rs < 0:
            Q_dense[i, i] -= rs * 100
            rs = Q_dense[i, :].sum()
    return Q_dense


def sample_from_precision(Q, linht, rng=None):
    """
    Sample from N(mu, Q^{-1}) where Q is a sparse precision matrix.

    Uses Cholesky: Q = L L^T  (L lower-triangular)
    mu = Q^{-1} linht = L^{-T} L^{-1} linht
    sample = mu + L^{-T} z,  z ~ N(0,I)

    This mirrors the R pattern:
        chQ = chol(Q)   # upper triangular R s.t. Q = R^T R
        mu = solve(chQ, solve(t(chQ), linht) + rnorm(n))

    Parameters
    ----------
    Q : scipy.sparse matrix or dense array
        Symmetric positive-definite precision matrix (T x T).
    linht : array (T,)
        The linear term (e.g. data / obs_sigma_t2).
    rng : numpy.random.Generator, optional

    Returns
    -------
    sample : array (T,)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = Q.shape[0]
    z = rng.standard_normal(n)

    # Try sparse Cholesky via scikit-sparse if available
    try:
        from sksparse.cholmod import cholesky as cholmod_cholesky
        if not sparse.issparse(Q):
            Q = sparse.csc_matrix(Q)
        factor = cholmod_cholesky(Q)
        # mu = Q^{-1} linht
        # sample = Q^{-1} linht + L^{-T} z
        # factor.solve_Lt(z, use_LDLt_decomposition=False) solves L^T x = z
        mu = factor.solve_A(linht)
        # L^{-T} z:  solve L^T x = z => x = L^{-T} z
        Ltinv_z = factor.solve_Lt(z, use_LDLt_decomposition=False)
        # Apply permutation
        P = factor.P()
        Ltinv_z_perm = np.empty_like(Ltinv_z)
        Ltinv_z_perm[P] = Ltinv_z
        return mu + Ltinv_z_perm
    except (ImportError, Exception):
        pass

    # Fallback: convert to dense, Cholesky
    if sparse.issparse(Q):
        Q_dense = Q.toarray()
    else:
        Q_dense = np.array(Q, dtype=np.float64)

    # Ensure symmetry
    Q_dense = (Q_dense + Q_dense.T) / 2.0

    try:
        L = np.linalg.cholesky(Q_dense)
    except np.linalg.LinAlgError:
        warnings.warn("Numerically unstable precision matrix — stabilizing...")
        Q_dense = _stabilize_precision(Q_dense)
        Q_dense = (Q_dense + Q_dense.T) / 2.0
        L = np.linalg.cholesky(Q_dense)

    # L is lower triangular: Q = L L^T
    # Solve L v = linht  =>  v = L^{-1} linht
    v = np.linalg.solve(L, linht)
    # Solve L^T mu = v  =>  mu = L^{-T} v = Q^{-1} linht
    mu = np.linalg.solve(L.T, v)
    # Solve L^T w = z  => w = L^{-T} z
    w = np.linalg.solve(L.T, z)

    return mu + w


def rinvgamma(n, shape, rate, rng=None):
    """
    Sample from Inverse-Gamma(shape, rate) distribution.

    In R: extraDistr::rinvgamma(n, alpha, beta) where beta is rate.
    X ~ InvGamma(alpha, beta) <=> 1/X ~ Gamma(alpha, rate=beta)
    <=> X = 1 / Gamma(alpha, scale=1/beta)

    Parameters
    ----------
    n : int
        Number of samples.
    shape : float or array
        Shape parameter (alpha).
    rate : float or array
        Rate parameter (beta). Note: this is the rate, not the scale.
    rng : numpy.random.Generator, optional

    Returns
    -------
    samples : float or array
    """
    if rng is None:
        rng = np.random.default_rng()
    # Gamma(shape, scale=1/rate) then invert
    shape = np.asarray(shape, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    # Ensure rate > 0
    rate = np.maximum(rate, 1e-300)
    g = rng.gamma(shape=shape, scale=1.0 / rate, size=n)
    g = np.maximum(g, 1e-300)  # prevent division by zero
    return 1.0 / g
