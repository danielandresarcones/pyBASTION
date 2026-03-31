# pyBASTION

[![Lint](https://github.com/danielandresarcones/pybastion/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/danielandresarcones/pybastion/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/danielandresarcones/pybastion/ci.yml?label=tests)](https://github.com/danielandresarcones/pybastion/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Python implementation of **BASTION** (Bayesian Adaptive Seasonality and Trend Decomposition Incorporating Outliers and Noise).

BASTION is a flexible Bayesian framework for decomposing time series into trend and multiple seasonality components. It offers three key advantages over existing decomposition methods:

1. **Locally adaptive estimation** of trend and seasonality for enhanced accuracy
2. **Explicit modeling of outliers and time-varying volatility** for greater robustness
3. **Rigorous uncertainty quantification** through credible intervals

This package is a Python translation of the [R package BASTION](https://github.com/jasonbcho/BASTION) by Jason B. Cho and David S. Matteson.

## Installation

```bash
pip install pybastion
```

For development:

```bash
git clone https://github.com/danielandresarcones/pybastion.git
cd pybastion
pip install -e ".[dev]"
```

### Optional: Sparse Cholesky acceleration

For large time series, install [scikit-sparse](https://scikit-sparse.readthedocs.io/) to enable sparse Cholesky factorization (requires SuiteSparse system library):

```bash
pip install "pybastion[fast]"
```

## Quick Start

```python
from pybastion import fit_BASTION, load_airtraffic

# Load example dataset
df = load_airtraffic()
y = df["Passengers"].values

# Decompose with weekly (7-day) seasonality
result = fit_BASTION(y, Ks=[7], Outlier=True, seed=42)

# Access results
summary = result["summary"]
trend = summary["Trend_sum"]          # Mean, CR_lower, CR_upper
seasonal = summary["Seasonal7_sum"]   # Seasonal component for period 7
```

### Multiple Seasonalities

```python
from pybastion import fit_BASTION, load_NYelectricity

df = load_NYelectricity()
y = df["demand"].values

# Weekly (7) and yearly (365) seasonal periods
result = fit_BASTION(y, Ks=[7, 365], Outlier=True, seed=42)

summary = result["summary"]
print(summary["Trend_sum"].head())
print(summary["Seasonal7_sum"].head())
print(summary["Seasonal365_sum"].head())
```

### Stochastic Volatility

```python
result = fit_BASTION(y, Ks=[7], obsSV="SV", seed=42)
volatility = result["summary"]["Volatility"]
```

### Regression Covariates

```python
import numpy as np

X = np.column_stack([covariate1, covariate2])  # (T, p) matrix
result = fit_BASTION(y, Ks=[7], X=X, seed=42)
```

## API Reference

### `fit_BASTION(y, Ks, ...)`

Decompose a time series using BASTION.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array-like (T,) | *required* | Time series observations |
| `Ks` | list of int | *required* | Seasonal periods |
| `X` | array-like (T, p) or None | `None` | Regression covariate matrix |
| `Outlier` | bool | `False` | Whether to model outliers |
| `cl` | float | `0.95` | Credible level for intervals |
| `sparse` | bool | `False` | Extra shrinkage on trend |
| `obsSV` | str | `"const"` | Observation error model: `"const"` or `"SV"` |
| `nchains` | int | `2` | Number of MCMC chains |
| `nsave` | int | `1000` | Iterations to save per chain |
| `nburn` | int | `1000` | Burn-in iterations |
| `nskip` | int | `4` | Thinning interval |
| `verbose` | bool | `True` | Show progress bar |
| `save_samples` | bool | `False` | Return full posterior samples |
| `seed` | int or None | `None` | Random seed for reproducibility |

**Returns:** A dict with key `"summary"` (and `"samples"` if `save_samples=True`).

The summary dict contains:
- `p_means` — DataFrame with posterior means of y and all components
- `Trend_sum` — DataFrame with `Mean`, `CR_lower`, `CR_upper`
- `SeasonalK_sum` — DataFrame for each seasonal period K
- `Outlier_sum` — DataFrame (if `Outlier=True`)
- `Volatility` — DataFrame with `Mean`, `CR_lower`, `CR_upper`

### Data Loaders

```python
from pybastion import load_airtraffic, load_NYelectricity

df_air = load_airtraffic()      # 249 × 3 DataFrame
df_elec = load_NYelectricity()  # 3288 × 2 DataFrame
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.22
- SciPy ≥ 1.8
- pandas ≥ 1.4
- tqdm ≥ 4.0

## Citation

If you use pyBASTION in your research, please cite the original paper:

```bibtex
@article{cho2025bastion,
  title={BASTION: Bayesian Adaptive Seasonality and Trend Decomposition Incorporating Outliers and Noise},
  author={Cho, Jason B. and Matteson, David S.},
  year={2025},
  eprint={2601.18052},
  archivePrefix={arXiv},
  primaryClass={stat.ME}
}
```

## License

MIT
