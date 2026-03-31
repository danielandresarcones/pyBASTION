"""
Dataset loaders for pyBASTION.
"""

import os
import pandas as pd

__all__ = ["load_airtraffic", "load_NYelectricity"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


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
