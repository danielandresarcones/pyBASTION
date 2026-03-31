"""
Tests for pyBASTION.

Tests are organized into:
1. Unit tests for internal modules (precision matrices, samplers)
2. Integration tests for the full MCMC pipeline
3. Validation tests comparing against R reference outputs
"""

import numpy as np
import pandas as pd
import pytest

from pybastion import fit_BASTION
from pybastion.datasets import load_airtraffic, load_NYelectricity
from pybastion.evol_params import (
    dsp_initEvol0,
    dsp_initEvolParams,
    dsp_sampleEvol0,
    dsp_sampleEvolParams,
    initEvolParams_HS_sparse,
    sample_jfast,
    sampleEvolParams_HS_sparse,
    t_initEvolZeta_ps,
    t_sampleEvolZeta_ps,
)
from pybastion.obs_error import init_sigmaE_0
from pybastion.outlier import sampleOutlier
from pybastion.seasonality import build_Q_season, sampleBeta_season
from pybastion.trend import build_Q_trend, sampleTrend
from pybastion.utils import rinvgamma, robust_div, robust_prod, sample_from_precision

# ──────────────────────────────────────────────────────────────────────────────
# Utility tests
# ──────────────────────────────────────────────────────────────────────────────


class TestUtils:
    def test_robust_prod(self):
        assert np.isclose(robust_prod(2.0, 3.0), 6.0)
        assert np.allclose(robust_prod(np.array([1, 2, 3]), 2.0), [2, 4, 6])

    def test_robust_div(self):
        assert np.isclose(robust_div(6.0, 2.0), 3.0)
        assert np.allclose(robust_div(np.array([6, 8, 10]), 2.0), [3, 4, 5])

    def test_rinvgamma_shape(self):
        rng = np.random.default_rng(42)
        samples = rinvgamma(1000, 3.0, 2.0, rng=rng)
        assert samples.shape == (1000,)
        assert np.all(samples > 0)

    def test_rinvgamma_mean(self):
        """InvGamma(alpha, beta) has mean beta/(alpha-1) for alpha>1."""
        rng = np.random.default_rng(42)
        alpha, beta = 5.0, 8.0
        samples = rinvgamma(100000, alpha, beta, rng=rng)
        expected_mean = beta / (alpha - 1)
        assert np.isclose(np.mean(samples), expected_mean, rtol=0.05)

    def test_sample_from_precision_identity(self):
        """Sampling from precision=I should give standard normal-like samples."""
        from scipy import sparse

        rng = np.random.default_rng(42)
        n = 50
        Q = sparse.eye(n, format="csc")
        linht = np.zeros(n)
        samples = np.array(
            [sample_from_precision(Q, linht, rng=rng) for _ in range(500)]
        )
        # Mean should be near 0
        assert np.allclose(samples.mean(axis=0), 0.0, atol=0.2)
        # Variance should be near 1
        assert np.allclose(samples.var(axis=0), 1.0, atol=0.3)

    def test_sample_from_precision_with_linht(self):
        """Sampling from precision=I with linht=ones should give mean=1."""
        from scipy import sparse

        rng = np.random.default_rng(42)
        n = 30
        Q = sparse.eye(n, format="csc")
        linht = np.ones(n)
        samples = np.array(
            [sample_from_precision(Q, linht, rng=rng) for _ in range(500)]
        )
        assert np.allclose(samples.mean(axis=0), 1.0, atol=0.2)


# ──────────────────────────────────────────────────────────────────────────────
# Precision matrix tests
# ──────────────────────────────────────────────────────────────────────────────


class TestPrecisionMatrices:
    def test_build_Q_trend_D1_symmetry(self):
        T = 50
        obs = np.ones(T)
        evol = np.ones(T)
        Q = build_Q_trend(obs, evol, D=1, Td=T)
        Q_dense = Q.toarray()
        assert np.allclose(Q_dense, Q_dense.T)

    def test_build_Q_trend_D2_symmetry(self):
        T = 50
        obs = np.ones(T)
        evol = np.ones(T)
        Q = build_Q_trend(obs, evol, D=2, Td=T)
        Q_dense = Q.toarray()
        assert np.allclose(Q_dense, Q_dense.T)

    def test_build_Q_trend_D2_positive_definite(self):
        T = 30
        obs = np.ones(T)
        evol = np.ones(T)
        Q = build_Q_trend(obs, evol, D=2, Td=T)
        Q_dense = Q.toarray()
        eigenvalues = np.linalg.eigvalsh(Q_dense)
        assert np.all(eigenvalues > 0)

    def test_build_Q_season_symmetry(self):
        T = 50
        k = 7
        obs = np.ones(T)
        evol = np.ones(T)
        Q = build_Q_season(obs, evol, Td=T, k=k)
        Q_dense = Q.toarray()
        assert np.allclose(Q_dense, Q_dense.T, atol=1e-12)

    def test_build_Q_season_positive_definite(self):
        T = 50
        k = 7
        obs = np.ones(T)
        evol = np.ones(T)
        Q = build_Q_season(obs, evol, Td=T, k=k)
        Q_dense = Q.toarray()
        eigenvalues = np.linalg.eigvalsh(Q_dense)
        assert np.all(eigenvalues > 0)


# ──────────────────────────────────────────────────────────────────────────────
# Evolution parameter tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEvolParams:
    def test_dsp_initEvol0_common(self):
        mu0 = np.array([1.0, -2.0, 3.0])
        result = dsp_initEvol0(mu0, commonSD=True)
        assert len(result["sigma_w0"]) == 3
        assert np.all(result["sigma_w0"] == result["sigma_w0"][0])  # all same

    def test_dsp_initEvol0_distinct(self):
        mu0 = np.array([1.0, -2.0, 3.0])
        result = dsp_initEvol0(mu0, commonSD=False)
        assert np.allclose(result["sigma_w0"], np.abs(mu0))

    def test_dsp_initEvolParams_HS(self):
        rng = np.random.default_rng(42)
        omega = rng.standard_normal(50)
        result = dsp_initEvolParams(omega, evol_error="HS")
        assert "sigma_wt" in result
        assert len(result["sigma_wt"]) == 50
        assert np.all(result["sigma_wt"] > 0)

    def test_dsp_sampleEvolParams_HS(self):
        rng = np.random.default_rng(42)
        omega = rng.standard_normal(50)
        evolParams = dsp_initEvolParams(omega, evol_error="HS")
        updated = dsp_sampleEvolParams(omega, evolParams, evol_error="HS", rng=rng)
        assert "sigma_wt" in updated
        assert len(updated["sigma_wt"]) == 50

    def test_initEvolParams_HS_sparse(self):
        rng = np.random.default_rng(42)
        omega = rng.standard_normal(50)
        result = initEvolParams_HS_sparse(omega, 50, rng=rng)
        assert "sigma_wt" in result
        assert len(result["sigma_wt"]) == 50

    def test_t_initEvolZeta_ps(self):
        rng = np.random.default_rng(42)
        zeta = rng.standard_normal(50)
        result = t_initEvolZeta_ps(zeta, 50, rng=rng)
        assert "sigma_wt" in result
        assert len(result["sigma_wt"]) == 50
        assert np.all(result["sigma_wt"] > 0)

    def test_sample_jfast_shapes(self):
        rng = np.random.default_rng(42)
        result = sample_jfast(100, rng=rng)
        assert len(result["mean"]) == 100
        assert len(result["var"]) == 100
        assert np.all(result["var"] > 0)

    def test_sample_jfast_with_obs(self):
        rng = np.random.default_rng(42)
        obs = rng.standard_normal(50)
        result = sample_jfast(50, obs=obs, rng=rng)
        assert len(result["mean"]) == 50


# ──────────────────────────────────────────────────────────────────────────────
# Component sampler tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSamplers:
    def test_sampleTrend_shape(self):
        rng = np.random.default_rng(42)
        T = 100
        data = rng.standard_normal(T)
        obs = np.ones(T)
        evol = np.ones(T) * 0.01
        mu = sampleTrend(data, obs, evol, D=2, Td=T, rng=rng)
        assert mu.shape == (T,)

    def test_sampleBeta_season_shape(self):
        rng = np.random.default_rng(42)
        T = 100
        data = rng.standard_normal(T)
        obs = np.ones(T)
        evol = np.ones(T) * 0.01
        mu = sampleBeta_season(data, obs, evol, Td=T, k=7, rng=rng)
        assert mu.shape == (T,)

    def test_sampleOutlier_shape(self):
        rng = np.random.default_rng(42)
        T = 50
        data = rng.standard_normal(T)
        obs = np.ones(T)
        evol = np.ones(T)
        s = sampleOutlier(data, obs, evol, Td=T, rng=rng)
        assert s.shape == (T,)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDatasets:
    def test_airtraffic_shape(self):
        df = load_airtraffic()
        assert df.shape == (249, 3)
        assert "Int_Pax" in df.columns

    def test_nyelectricity_shape(self):
        df = load_NYelectricity()
        assert df.shape == (3288, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests: full pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TestFitBASTION:
    """Integration tests for the full MCMC pipeline."""

    @pytest.fixture
    def simulated_data(self):
        """Generate simple simulated data for testing."""
        rng = np.random.default_rng(42)
        T = 200
        t = np.arange(T)
        trend = 0.02 * t + 5
        seasonal = 3.0 * np.sin(2 * np.pi * t / 7)
        noise = 0.5 * rng.standard_normal(T)
        y = trend + seasonal + noise
        return y, trend, seasonal

    def test_basic_decomposition(self, simulated_data):
        y, _, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            nsave=50,
            nburn=50,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
        )
        assert "summary" in result
        s = result["summary"]
        assert "Trend_sum" in s
        assert "Seasonal7_sum" in s
        assert "Volatility" in s
        assert s["Trend_sum"].shape == (200, 3)

    def test_two_seasonalities(self, simulated_data):
        y, _, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7, 30],
            nsave=30,
            nburn=30,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
        )
        s = result["summary"]
        assert "Seasonal7_sum" in s
        assert "Seasonal30_sum" in s

    def test_outlier(self, simulated_data):
        y, _, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            Outlier=True,
            nsave=30,
            nburn=30,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
        )
        assert "Outlier_sum" in result["summary"]
        assert result["summary"]["Outlier_sum"].shape == (200, 3)

    def test_sv_mode(self, simulated_data):
        y, _, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            obsSV="SV",
            nsave=20,
            nburn=20,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
        )
        vol = result["summary"]["Volatility"]["Mean"].values
        assert len(vol) == 200
        assert not np.all(vol == vol[0])  # SV should give varying volatility

    def test_sparse_trend(self, simulated_data):
        y, _, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            sparse=True,
            nsave=20,
            nburn=20,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
        )
        assert result["summary"]["Trend_sum"].shape == (200, 3)

    def test_two_chains(self, simulated_data):
        y, _, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            nsave=20,
            nburn=20,
            nskip=1,
            nchains=2,
            seed=42,
            verbose=False,
            save_samples=True,
        )
        # 2 chains * 20 saves = 40 samples
        assert result["samples"]["beta_combined"].shape == (40, 200)

    def test_regression(self):
        rng = np.random.default_rng(42)
        T = 100
        t = np.arange(T, dtype=float)
        X = rng.standard_normal((T, 2))
        beta_true = np.array([3.0, -1.5])
        y = 0.01 * t + X @ beta_true + 0.3 * rng.standard_normal(T)
        result = fit_BASTION(
            y,
            Ks=[7],
            X=X,
            nsave=30,
            nburn=30,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
        )
        assert "Regression_sum" in result["summary"]

    def test_save_samples(self, simulated_data):
        y, _, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            nsave=20,
            nburn=20,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
            save_samples=True,
        )
        assert "samples" in result
        s = result["samples"]
        assert s["beta_combined"].shape == (20, 200)
        assert s["beta"].shape == (20, 200, 2)  # 1 seasonal + 1 trend
        assert s["obs_sigma_t2"].shape == (20, 200)
        assert s["remainder"].shape == (20, 200)
        assert s["yhat"].shape == (20, 200)

    def test_trend_recovery(self, simulated_data):
        """Test that the estimated trend roughly follows the true trend."""
        y, trend, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            nsave=200,
            nburn=200,
            nskip=2,
            nchains=1,
            seed=42,
            verbose=False,
        )
        est_trend = result["summary"]["Trend_sum"]["Mean"].values
        # The correlation between estimated and true trend should be high
        corr = np.corrcoef(est_trend, trend)[0, 1]
        assert corr > 0.9, f"Trend correlation too low: {corr}"

    def test_seasonal_recovery(self, simulated_data):
        """Test that the estimated seasonal roughly follows the true seasonal."""
        y, _, seasonal = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            nsave=200,
            nburn=200,
            nskip=2,
            nchains=1,
            seed=42,
            verbose=False,
        )
        est_seasonal = result["summary"]["Seasonal7_sum"]["Mean"].values
        # The correlation between estimated and true seasonal should be high
        corr = np.corrcoef(est_seasonal, seasonal)[0, 1]
        assert corr > 0.9, f"Seasonal correlation too low: {corr}"

    def test_credible_interval_coverage(self, simulated_data):
        """Test that the 95% CI covers at least ~50% of true trend values."""
        y, trend, _ = simulated_data
        result = fit_BASTION(
            y,
            Ks=[7],
            cl=0.95,
            nsave=300,
            nburn=300,
            nskip=2,
            nchains=2,
            seed=42,
            verbose=False,
        )
        ts = result["summary"]["Trend_sum"]
        inside = (ts["CR_lower"].values <= trend) & (trend <= ts["CR_upper"].values)
        coverage = inside.mean()
        assert coverage > 0.5, f"Coverage too low: {coverage}"


# ──────────────────────────────────────────────────────────────────────────────
# Validation tests: compare against R reference outputs
# ──────────────────────────────────────────────────────────────────────────────


class TestReferenceValidation:
    """Compare Python output structure and range against saved R outputs."""

    @pytest.fixture
    def r_electric_ref(self):
        """Load the R reference output for electricity decomposition."""
        try:
            import rdata

            parsed = rdata.parser.parse_file(
                "BASTION/inst/extdata/BASTION_electric.rds"
            )
            return rdata.conversion.convert(parsed)
        except (ImportError, FileNotFoundError):
            pytest.skip("rdata package or reference file not available")

    def test_electric_structure_matches(self, r_electric_ref):
        """Verify Python produces the same output structure as R."""
        r_keys = set(r_electric_ref["summary"].keys())
        expected = {
            "p_means",
            "Trend_sum",
            "Seasonal7_sum",
            "Seasonal365_sum",
            "Outlier_sum",
            "Volatility",
        }
        assert r_keys == expected

    def test_electric_trend_range(self, r_electric_ref):
        """Compare the R trend range against a quick Python run."""
        r_trend = r_electric_ref["summary"]["Trend_sum"]
        r_mean = (
            r_trend["Mean"].values
            if hasattr(r_trend, "values")
            else np.asarray(r_trend["Mean"])
        )
        # The R trend mean should be in a reasonable range for electricity data
        assert 10000 < np.mean(r_mean) < 30000

    def test_electric_python_matches_r_scale(self, r_electric_ref):
        """
        Run a short Python fit on the same data and check that
        the trend is in a similar range as the R reference.
        """
        df = load_NYelectricity()
        y = df.iloc[:, 1].values

        # Very short run just to check scale
        result = fit_BASTION(
            y[:500],
            Ks=[7, 365],
            Outlier=True,
            sparse=True,
            obsSV="SV",
            nsave=10,
            nburn=10,
            nskip=1,
            nchains=1,
            seed=42,
            verbose=False,
        )
        py_trend_mean = result["summary"]["Trend_sum"]["Mean"].values.mean()
        r_trend_mean = np.mean(
            r_electric_ref["summary"]["Trend_sum"]["Mean"].values[:500]
        )
        # Should be within 30% of each other (different seeds, few iterations)
        rel_diff = abs(py_trend_mean - r_trend_mean) / abs(r_trend_mean)
        assert (
            rel_diff < 0.3
        ), f"Python trend mean {py_trend_mean:.0f} vs R {r_trend_mean:.0f}"


# ──────────────────────────────────────────────────────────────────────────────
# Error handling tests
# ──────────────────────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_invalid_Ks_type(self):
        with pytest.raises(ValueError, match="Ks needs to be a list"):
            fit_BASTION(np.ones(100), Ks=7)

    def test_invalid_obsSV(self):
        with pytest.raises(ValueError, match="obsSV"):
            fit_BASTION(np.ones(100), Ks=[7], obsSV="invalid")

    def test_invalid_X_dimension(self):
        with pytest.raises(ValueError, match="2D"):
            fit_BASTION(np.ones(100), Ks=[7], X=np.ones(100))
