"""Scripts for manipulating, formatting and re-structuring data."""

import pandas as pd
import numpy.typing as npt


def create_df_from_uneven_data(data_dict: dict[str, npt.NDArray]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from a dictionary of lists with uneven lengths.

    Parameters
    ----------
    data_dict : dict[str, npt.NDArray]
        A dictionary where keys are column names and values are lists of data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with blank spaces for missing values due to uneven lengths.
    """
    return pd.DataFrame({k: pd.Series(v) for k, v in data_dict.items()})
