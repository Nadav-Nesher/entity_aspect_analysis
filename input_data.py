"""
This module contains a function for loading input data for the
entity and aspect extraction and evaluation project.

Imports:
- pandas: A data manipulation and analysis library.

Function:
- `load_data`: Loads data from a CSV file into a Pandas DataFrame.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Loads data from a specified CSV file path into a Pandas DataFrame.

    Parameters:
        path (str): The file path to the CSV file to be loaded.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """
    data_df = pd.read_csv(path)

    return data_df
