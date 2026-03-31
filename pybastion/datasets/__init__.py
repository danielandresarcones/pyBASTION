"""
Example datasets bundled with pyBASTION.

This subpackage provides convenience loaders for the datasets included
with the package.  These are **not** required by the core BASTION model —
they are provided purely for demonstration and reproducibility of the
examples in the documentation.

Functions
---------
load_airtraffic
    US airline international passenger traffic (2003–2023).
load_NYelectricity
    New York electricity demand (2015–2024).
"""

import os

import pandas as pd

__all__ = ["load_airtraffic", "load_NYelectricity"]

_DATA_DIR = os.path.dirname(__file__)


def load_airtraffic():
    """
    Load the US airline traffic dataset (2003–2023).

    Returns
    -------
    df : pd.DataFrame
        Columns: Year, Month, Int_Pax
    """
    return pd.read_csv(os.path.join(_DATA_DIR, "airtraffic.csv"))


def load_NYelectricity():
    """
    Load the NY electricity demand dataset (2015–2024).

    Returns
    -------
    df : pd.DataFrame
        Columns: Data.Date, Demand..MW.
    """
    df = pd.read_csv(os.path.join(_DATA_DIR, "NYelectricity.csv"))
    df["Data.Date"] = pd.to_datetime(df["Data.Date"])
    return df
