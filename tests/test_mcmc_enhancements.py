import time
import tracemalloc

import numpy as np
from scipy.stats import ks_2samp

from pybastion.mcmc import _spawn_child_rngs, fit_ASD_SV


def _synthetic_series(seed=0, T=90, missing=False):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    X = np.column_stack(
        [
            np.sin(2 * np.pi * t / 11.0),
            np.cos(2 * np.pi * t / 17.0),
        ]
    )
    trend = 0.03 * t
    seasonal_weekly = 1.4 * np.sin(2 * np.pi * t / 7.0)
    seasonal_monthly = 0.9 * np.cos(2 * np.pi * t / 30.0)
    outliers = np.zeros(T)
    outliers[[T // 4, (3 * T) // 5]] = [2.5, -2.0]
    noise = 0.35 * rng.standard_normal(T)
    y = (
        4.0
        + trend
        + seasonal_weekly
        + seasonal_monthly
        + X @ np.array([0.8, -0.5])
        + outliers
        + noise
    )
    if missing:
        y[[5, T // 2, T - 7]] = np.nan
    return y, X


def _sample_moments(samples):
    return {
        key: {
            "mean": np.mean(value, axis=0),
            "var": np.var(value, axis=0),
        }
        for key, value in samples.items()
    }


def _assert_moment_match(actual, expected, atol=1e-10, rtol=1e-10):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(
            actual[key]["mean"], expected[key]["mean"], atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            actual[key]["var"], expected[key]["var"], atol=atol, rtol=rtol
        )


def _deep_nbytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_deep_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_deep_nbytes(item) for item in value)
    return 0


def _profile_sampler(**kwargs):
    tracemalloc.start()
    start = time.perf_counter()
    result = fit_ASD_SV(**kwargs)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "elapsed": elapsed,
        "peak": peak,
        "output_bytes": _deep_nbytes(result),
        "result": result,
    }


def test_parallel_child_rngs_are_distinct_and_weakly_correlated():
    children = _spawn_child_rngs(np.random.default_rng(1234), 4)
    draws = np.vstack([child.standard_normal(256) for child in children])
    corr = np.corrcoef(draws)
    off_diag = corr[np.triu_indices_from(corr, 1)]

    assert len({tuple(np.round(draw[:8], 8)) for draw in draws}) == len(children)
    assert np.all(np.abs(off_diag) < 0.2)


def test_rao_blackwell_matches_sample_moments_sequential():
    y, X = _synthetic_series(seed=10, T=80)
    common_kwargs = dict(
        y=y,
        Ks=[7, 30],
        X=X,
        Outlier=True,
        nsave=12,
        nburn=12,
        nskip=1,
        verbose=False,
    )

    sampled = fit_ASD_SV(**common_kwargs, rng=np.random.default_rng(2024))
    rao_blackwellized = fit_ASD_SV(
        **common_kwargs,
        rao_blackwell=True,
        rng=np.random.default_rng(2024),
    )

    _assert_moment_match(
        rao_blackwellized["moments"],
        _sample_moments(sampled["samples"]),
    )
    assert rao_blackwellized["streaming_stats"]["count"] == common_kwargs["nsave"]


def test_parallel_rao_blackwell_matches_parallel_sample_moments():
    y, X = _synthetic_series(seed=12, T=80)
    common_kwargs = dict(
        y=y,
        Ks=[7, 14, 30],
        X=X,
        Outlier=True,
        nsave=10,
        nburn=10,
        nskip=1,
        parallel=True,
        verbose=False,
    )

    sampled = fit_ASD_SV(**common_kwargs, rng=np.random.default_rng(99))
    rao_blackwellized = fit_ASD_SV(
        **common_kwargs,
        rao_blackwell=True,
        rng=np.random.default_rng(99),
    )

    _assert_moment_match(
        rao_blackwellized["moments"],
        _sample_moments(sampled["samples"]),
    )


def test_parallel_sampler_is_statistically_close_to_sequential_sampler():
    for seed in (7, 19):
        y, X = _synthetic_series(seed=seed, T=90)
        common_kwargs = dict(
            y=y,
            Ks=[7, 14, 30],
            X=X,
            Outlier=True,
            nsave=24,
            nburn=24,
            nskip=1,
            verbose=False,
        )

        sequential = fit_ASD_SV(**common_kwargs, rng=np.random.default_rng(seed))
        parallel = fit_ASD_SV(
            **common_kwargs,
            parallel=True,
            rng=np.random.default_rng(seed),
        )

        seq_moments = _sample_moments(sequential["samples"])
        par_moments = _sample_moments(parallel["samples"])

        assert (
            np.mean(
                np.abs(
                    seq_moments["beta_combined"]["mean"]
                    - par_moments["beta_combined"]["mean"]
                )
            )
            < 0.35
        )
        assert (
            np.mean(
                np.abs(
                    seq_moments["obs_sigma_t2"]["mean"]
                    - par_moments["obs_sigma_t2"]["mean"]
                )
            )
            < 0.35
        )
        assert (
            np.mean(
                np.abs(
                    seq_moments["beta_combined"]["var"]
                    - par_moments["beta_combined"]["var"]
                )
            )
            < 0.5
        )

        probes = [15, 45, 75]
        beta_ks = ks_2samp(
            sequential["samples"]["beta_combined"][:, probes].ravel(),
            parallel["samples"]["beta_combined"][:, probes].ravel(),
            method="asymp",
        )
        sigma_ks = ks_2samp(
            sequential["samples"]["obs_sigma_t2"][:, probes].ravel(),
            parallel["samples"]["obs_sigma_t2"][:, probes].ravel(),
            method="asymp",
        )

        assert beta_ks.pvalue > 0.05
        assert sigma_ks.pvalue > 0.05


def test_single_component_missing_data_parallel_and_rao_blackwell():
    y, _ = _synthetic_series(seed=21, T=60, missing=True)
    result = fit_ASD_SV(
        y=y,
        Ks=[7],
        nsave=8,
        nburn=8,
        nskip=1,
        parallel=True,
        rao_blackwell=True,
        verbose=False,
        rng=np.random.default_rng(321),
    )

    assert result["moments"]["beta"]["mean"].shape == (60, 2)
    assert result["moments"]["obs_sigma_t2"]["mean"].shape == (60,)
    assert result["streaming_stats"]["count"] == 8
    assert not np.isnan(result["moments"]["beta"]["mean"]).any()


def test_memory_reduction_and_performance_smoke():
    y, X = _synthetic_series(seed=33, T=140)
    common_kwargs = dict(
        y=y,
        Ks=[7, 14, 30],
        X=X,
        Outlier=True,
        nsave=18,
        nburn=18,
        nskip=1,
        verbose=False,
    )

    baseline = _profile_sampler(**common_kwargs, rng=np.random.default_rng(501))
    parallel = _profile_sampler(
        **common_kwargs,
        parallel=True,
        rng=np.random.default_rng(501),
    )
    rao_blackwellized = _profile_sampler(
        **common_kwargs,
        rao_blackwell=True,
        rng=np.random.default_rng(501),
    )
    combined = _profile_sampler(
        **common_kwargs,
        parallel=True,
        rao_blackwell=True,
        rng=np.random.default_rng(501),
    )

    assert rao_blackwellized["output_bytes"] < baseline["output_bytes"]
    assert combined["output_bytes"] < parallel["output_bytes"]

    timing_table = {
        "baseline": baseline["elapsed"],
        "parallel": parallel["elapsed"],
        "rao_blackwell": rao_blackwellized["elapsed"],
        "combined": combined["elapsed"],
    }
    peak_table = {
        "baseline": baseline["peak"],
        "parallel": parallel["peak"],
        "rao_blackwell": rao_blackwellized["peak"],
        "combined": combined["peak"],
    }

    assert parallel["elapsed"] < baseline["elapsed"] * 3.0, timing_table
    assert combined["elapsed"] < baseline["elapsed"] * 3.0, timing_table
    assert peak_table["rao_blackwell"] <= peak_table["baseline"] * 1.2, peak_table
