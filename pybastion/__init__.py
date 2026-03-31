"""
pyBASTION — Python implementation of BASTION
(Bayesian Adaptive Seasonality Trend Decomposition Incorporating Outliers and Noise)

Usage
-----
>>> from pybastion import fit_BASTION
>>> result = fit_BASTION(y, Ks=[7, 365], Outlier=True)
>>> result["summary"]["Trend_sum"]
"""

from .model import fit_BASTION

__version__ = "0.1.0"

__all__ = [
    "fit_BASTION",
]
